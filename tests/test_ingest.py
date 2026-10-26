"""Tests for dataset ingestion, using a tiny fake FLEURS built in a temporary folder."""

import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asr_lab.data.ingest import (
    build_fleurs_manifest,
    ingest_fleurs,
    prepare_audio,
    read_fleurs_tsv,
)
from asr_lab.data.manifest import read_manifest

FAKE_ROWS = [
    ("1987", "111.wav", "L'uomo è qui.", "l'uomo è qui", "l ' u", "16000", "FEMALE"),
    ("1987", "222.wav", "L'uomo è qui.", "l'uomo è qui", "l ' u", "16000", "MALE"),
    ("2001", "333.wav", "Buongiorno a tutti.", "buongiorno a tutti", "b u o", "16000", "FEMALE"),
]


def _sine(seconds: float, sample_rate: int, channels: int = 1) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    wave = 0.3 * np.sin(2 * np.pi * 440 * t).astype("float32")
    return np.stack([wave] * channels, axis=1)


def _fake_fleurs(tmp_path: Path, rows=FAKE_ROWS, missing: str | None = None) -> tuple[Path, Path]:
    """Write a FLEURS-like TSV and a tar.gz archive with one-second 16 kHz float WAV files."""
    tsv_path = tmp_path / "test.tsv"
    tsv_path.write_text("".join("\t".join(row) + "\n" for row in rows), encoding="utf-8")
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    archive_path = tmp_path / "test.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for row in rows:
            if row[1] == missing:
                continue
            wav = wav_dir / row[1]
            sf.write(wav, _sine(1.0, 16_000), 16_000, subtype="FLOAT")
            archive.add(wav, arcname=f"test/{row[1]}")
    return tsv_path, archive_path


# --- read_fleurs_tsv ------------------------------------------------------------------


def test_read_fleurs_tsv(tmp_path: Path) -> None:
    tsv_path, _ = _fake_fleurs(tmp_path)
    rows = read_fleurs_tsv(tsv_path)
    assert len(rows) == 3
    assert rows[0].sentence_id == "1987"
    assert rows[0].raw_transcription == "L'uomo è qui."
    assert rows[0].num_samples == 16000
    assert rows[1].gender == "MALE"


def test_read_fleurs_tsv_rejects_wrong_column_count(tmp_path: Path) -> None:
    path = tmp_path / "bad.tsv"
    path.write_text("1\t2\t3\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.tsv:1: expected 7 tab-separated columns, got 3"):
        read_fleurs_tsv(path)


def test_read_fleurs_tsv_rejects_non_numeric_samples(tmp_path: Path) -> None:
    path = tmp_path / "bad.tsv"
    path.write_text(
        "\t".join(["1", "a.wav", "x", "x", "x", "many", "MALE"]) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="num_samples is not a number"):
        read_fleurs_tsv(path)


# --- prepare_audio --------------------------------------------------------------------


def test_prepare_audio_converts_to_16k_mono_pcm16(tmp_path: Path) -> None:
    source = tmp_path / "stereo_44k.wav"
    sf.write(source, _sine(2.0, 44_100, channels=2), 44_100, subtype="FLOAT")
    destination = tmp_path / "out" / "converted.wav"
    duration = prepare_audio(source, destination)
    info = sf.info(destination)
    assert info.samplerate == 16_000
    assert info.channels == 1
    assert info.subtype == "PCM_16"
    assert duration == pytest.approx(2.0, abs=0.01)
    assert info.duration == pytest.approx(duration)


def test_prepare_audio_keeps_16k_audio_length(tmp_path: Path) -> None:
    source = tmp_path / "mono_16k.wav"
    sf.write(source, _sine(1.5, 16_000), 16_000)
    assert prepare_audio(source, tmp_path / "out.wav") == 1.5


# --- build_fleurs_manifest ------------------------------------------------------------


def test_build_fleurs_manifest(tmp_path: Path) -> None:
    tsv_path, archive_path = _fake_fleurs(tmp_path)
    output_dir = tmp_path / "fleurs_it"
    manifest_path = build_fleurs_manifest(
        tsv_path, archive_path, output_dir, lang="it", split="test"
    )
    assert manifest_path == output_dir / "test.jsonl"
    entries = read_manifest(manifest_path)
    assert [e.utterance_id for e in entries] == ["fleurs_it_111", "fleurs_it_222", "fleurs_it_333"]

    first = entries[0]
    assert first.text == "L'uomo è qui."  # original text, not normalized
    assert first.lang == "it"
    assert first.source == "fleurs"
    assert first.duration == pytest.approx(1.0)
    assert first.metadata == {
        "sentence_id": "1987",
        "gender": "female",
        "fleurs_transcription": "l'uomo è qui",
    }
    # No speaker ids in FLEURS: every utterance gets its own placeholder.
    assert len({e.speaker_id for e in entries}) == 3

    audio = sf.info(first.audio_path(manifest_path))
    assert (audio.samplerate, audio.channels, audio.subtype) == (16_000, 1, "PCM_16")


def test_max_utterances_selects_a_reproducible_subset(tmp_path: Path) -> None:
    tsv_path, archive_path = _fake_fleurs(tmp_path)
    ids = []
    for run in ("a", "b"):
        manifest_path = build_fleurs_manifest(
            tsv_path, archive_path, tmp_path / run, lang="it", split="test", max_utterances=2
        )
        ids.append([e.utterance_id for e in read_manifest(manifest_path)])
    assert len(ids[0]) == 2
    assert ids[0] == ids[1]


def test_missing_audio_raises(tmp_path: Path) -> None:
    tsv_path, archive_path = _fake_fleurs(tmp_path, missing="222.wav")
    with pytest.raises(ValueError, match=r"1 files listed .* are missing"):
        build_fleurs_manifest(tsv_path, archive_path, tmp_path / "out", lang="it", split="test")


@pytest.mark.parametrize(
    ("lang", "split", "message"),
    [("fr", "test", "Unsupported language 'fr'"), ("it", "validation", "Unsupported split")],
)
def test_ingest_fleurs_rejects_unknown_options(lang: str, split: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ingest_fleurs(lang, split, Path("unused"))
