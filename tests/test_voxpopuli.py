"""Tests for the VoxPopuli ingestion, on a small parquet file with the dataset's schema."""

import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from asr_lab.data.manifest import read_manifest, write_manifest
from asr_lab.data.voxpopuli import build_entries, load_data_spec, read_row_group


def wav_bytes(seconds: float, value: float = 0.1) -> bytes:
    """Float32 WAV, the format of VoxPopuli's parquet files."""
    buffer = io.BytesIO()
    sf.write(buffer, np.full(int(16_000 * seconds), value, dtype=np.float32), 16_000,
             format="WAV", subtype="FLOAT")  # fmt: skip
    return buffer.getvalue()


ROWS = [
    {
        "audio_id": "20130522-0900-PLENARY-15-it_20130522-17:29:48_2",
        "language": 9,
        "audio": {"bytes": wav_bytes(1.5), "path": "a.wav"},
        "raw_text": "Signor Presidente, onorevoli colleghi.",
        "normalized_text": "signor presidente onorevoli colleghi.",
        "gender": "female",
        "speaker_id": "97229",
        "is_gold_transcript": True,
        "accent": "None",
    },
    {
        "audio_id": "20170601-0900-PLENARY-4-it_20170601-10:03:05_1",
        "language": 9,
        "audio": {"bytes": wav_bytes(0.5), "path": "b.wav"},
        "raw_text": "",
        "normalized_text": "",
        "gender": "male",
        "speaker_id": "124851",
        "is_gold_transcript": True,
        "accent": "None",
    },
]


def test_row_groups_are_read_one_at_a_time(tmp_path: Path) -> None:
    path = tmp_path / "train-00000-of-00001.parquet"
    pq.write_table(pa.Table.from_pylist(ROWS), path, row_group_size=1)
    second = read_row_group(path, 1)
    assert [row["audio_id"] for row in second] == [ROWS[1]["audio_id"]]
    assert set(second[0]) == {
        "audio_id", "audio", "raw_text", "normalized_text", "gender", "speaker_id",
    }  # fmt: skip


def test_entries_keep_empty_transcripts_and_safe_file_names(tmp_path: Path) -> None:
    entries = build_entries(ROWS, tmp_path, lang="it", origin="train-00000#0")
    first, empty = entries
    assert first.utterance_id == "voxpopuli_it_20130522-0900-PLENARY-15-it_20130522-17-29-48_2"
    assert ":" not in first.audio_filepath
    assert first.speaker_id == "voxpopuli_97229" and first.source == "voxpopuli"
    assert first.metadata["session"] == "20130522-0900-PLENARY-15"
    assert first.metadata["origin"] == "train-00000#0"
    assert first.duration == pytest.approx(1.5)
    audio, rate = sf.read(tmp_path / first.audio_filepath)
    assert rate == 16_000 and sf.info(tmp_path / first.audio_filepath).subtype == "PCM_16"
    assert audio.mean() == pytest.approx(0.1, abs=1e-3)
    assert empty.text == ""  # kept: the quality checks flag it
    write_manifest(tmp_path / "train.jsonl", entries)
    assert read_manifest(tmp_path / "train.jsonl") == entries


def test_the_e6_spec_pins_a_revision() -> None:
    spec = load_data_spec(Path("experiments/E6/quality.toml"))
    assert len(spec.revision) == 40  # a commit hash, not a branch name
    assert spec.manifest == Path("data/voxpopuli_it/train.jsonl")
    assert all(len(pair) == 2 for pair in spec.row_groups)
