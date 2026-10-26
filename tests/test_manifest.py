"""Tests for JSONL manifests."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from asr_lab.data.manifest import ManifestEntry, read_manifest, write_manifest

ENTRY = ManifestEntry(
    utterance_id="fleurs_it_0001",
    audio_filepath="audio/fleurs_it_0001.wav",
    duration=4.2,
    text="Perché l'uomo è arrivato?",
    lang="it",
    speaker_id="fleurs_it_spk12",
    source="fleurs",
)


def _write_lines(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_round_trip(tmp_path: Path) -> None:
    second = replace(ENTRY, utterance_id="fleurs_it_0002", metadata={"gender": "female"})
    path = tmp_path / "test.jsonl"
    assert write_manifest(path, [ENTRY, second]) == 2
    assert read_manifest(path) == [ENTRY, second]


def test_file_format_is_utf8_jsonl_with_lf(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    write_manifest(path, [ENTRY, replace(ENTRY, utterance_id="fleurs_it_0002")])
    raw = path.read_bytes()
    assert b"\r\n" not in raw  # LF line endings even on Windows
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 2  # one line per utterance
    assert "Perché" in raw.decode("utf-8")  # accents stored as-is, not as é


def test_write_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "data" / "fleurs_it" / "test.jsonl"
    write_manifest(path, [ENTRY])
    assert path.exists()


def test_audio_path_is_resolved_from_the_manifest_location(tmp_path: Path) -> None:
    manifest_path = tmp_path / "fleurs_it" / "test.jsonl"
    assert (
        ENTRY.audio_path(manifest_path) == tmp_path / "fleurs_it" / "audio" / "fleurs_it_0001.wav"
    )


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = _write_lines(tmp_path / "m.jsonl", [json.dumps(asdict(ENTRY)), "", "   "])
    assert read_manifest(path) == [ENTRY]


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate utterance_id 'fleurs_it_0001'"):
        write_manifest(tmp_path / "m.jsonl", [ENTRY, ENTRY])


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"utterance_id": ""}, "empty utterance_id"),
        ({"duration": 0.0}, "duration must be positive"),
        ({"duration": -1.0}, "duration must be positive"),
        ({"duration": "4.2"}, "duration must be a number"),
        ({"audio_filepath": "/home/pc/audio.wav"}, "must be relative"),
        ({"audio_filepath": "audio\\x.wav"}, "forward slashes"),
        ({"text": None}, "text must be a string"),
    ],
)
def test_invalid_entries_are_rejected(tmp_path: Path, changes: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        write_manifest(tmp_path / "m.jsonl", [replace(ENTRY, **changes)])


def test_read_reports_file_and_line_number(tmp_path: Path) -> None:
    good = json.dumps(asdict(ENTRY))
    path = _write_lines(tmp_path / "m.jsonl", [good, "{not json"])
    with pytest.raises(ValueError, match=r"m\.jsonl:2: invalid JSON"):
        read_manifest(path)


def test_missing_and_unknown_fields_are_rejected(tmp_path: Path) -> None:
    record = asdict(ENTRY)
    del record["speaker_id"]
    path = _write_lines(tmp_path / "missing.jsonl", [json.dumps(record)])
    with pytest.raises(ValueError, match=r"missing fields \['speaker_id'\]"):
        read_manifest(path)

    record = dict(asdict(ENTRY), gender="female")
    path = _write_lines(tmp_path / "unknown.jsonl", [json.dumps(record)])
    with pytest.raises(ValueError, match=r"unknown fields.*metadata"):
        read_manifest(path)


def test_non_object_line_is_rejected(tmp_path: Path) -> None:
    path = _write_lines(tmp_path / "m.jsonl", ["[1, 2, 3]"])
    with pytest.raises(ValueError, match="expected a JSON object"):
        read_manifest(path)
