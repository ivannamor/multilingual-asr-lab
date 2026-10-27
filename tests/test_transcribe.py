"""Tests for predictions files, resumable manifest transcription and the CLI."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from asr_lab import cli
from asr_lab.data.manifest import ManifestEntry, write_manifest
from asr_lab.inference.base import SAMPLE_RATE, ASRModel, AudioSample, Hypothesis
from asr_lab.inference.predictions import append_predictions, read_predictions
from asr_lab.inference.run import transcribe_manifest


class FakeModel(ASRModel):
    """Echoes utterance ids; can fail after a number of calls to simulate an interruption."""

    def __init__(self, fail_after_calls: int | None = None, beam_size: int = 5) -> None:
        self.fail_after_calls = fail_after_calls
        self.beam_size = beam_size
        self.transcribed: list[str] = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def config(self) -> dict[str, Any]:
        return {"beam_size": self.beam_size}

    def _transcribe_batch(self, samples: Sequence[AudioSample]) -> list[Hypothesis]:
        if self.fail_after_calls is not None and len(self.transcribed) >= self.fail_after_calls:
            raise KeyboardInterrupt
        self.transcribed += [s.utterance_id for s in samples]
        return [Hypothesis(s.utterance_id, f"text {s.utterance_id}", score=-0.5) for s in samples]


def _manifest(tmp_path: Path, n: int = 5) -> Path:
    audio_dir = tmp_path / "data" / "audio"
    audio_dir.mkdir(parents=True)
    entries = []
    for i in range(n):
        sf.write(audio_dir / f"u{i}.wav", np.zeros(SAMPLE_RATE), SAMPLE_RATE, subtype="PCM_16")
        entries.append(ManifestEntry(f"u{i}", f"audio/u{i}.wav", 1.0, "ciao", "it", "s", "test"))
    path = tmp_path / "data" / "test.jsonl"
    write_manifest(path, entries)
    return path


# --- predictions files ----------------------------------------------------------------


def test_predictions_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    first = [Hypothesis("a", "Perché sì", score=-0.1), Hypothesis("b", "", nbest=("x", "y"))]
    append_predictions(path, first[:1])
    append_predictions(path, first[1:])
    assert read_predictions(path) == first
    assert "Perché" in path.read_text(encoding="utf-8")


def test_predictions_reject_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"utterance_id": "a"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"predictions\.jsonl:1: invalid prediction"):
        read_predictions(path)


def test_predictions_reject_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    append_predictions(path, [Hypothesis("a", "x"), Hypothesis("a", "y")])
    with pytest.raises(ValueError, match="duplicate utterance_id 'a'"):
        read_predictions(path)


# --- transcribe_manifest --------------------------------------------------------------


def test_transcribe_manifest_writes_predictions_and_run_metadata(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    out = tmp_path / "runs" / "r1"
    path = transcribe_manifest(FakeModel(), manifest, out, batch_size=2, progress=False)
    assert [h.utterance_id for h in read_predictions(path)] == ["u0", "u1", "u2", "u3", "u4"]

    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["model"] == "fake"
    assert run["model_config"] == {"beam_size": 5}
    assert run["n_utterances"] == 5
    assert run["audio_seconds"] == 5.0
    assert len(run["manifest_sha256"]) == 64
    assert run["finished_at"] is not None
    assert run["rtf"] == pytest.approx(run["processing_seconds"] / 5.0)


def test_interrupted_run_resumes_where_it_stopped(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    out = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        transcribe_manifest(
            FakeModel(fail_after_calls=2), manifest, out, batch_size=2, progress=False
        )
    partial = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert partial["finished_at"] is None
    assert len(read_predictions(out / "predictions.jsonl")) == 2

    resumed = FakeModel()
    transcribe_manifest(resumed, manifest, out, batch_size=2, progress=False)
    assert resumed.transcribed == ["u2", "u3", "u4"]  # only what was missing
    assert len(read_predictions(out / "predictions.jsonl")) == 5
    assert json.loads((out / "run.json").read_text(encoding="utf-8"))["finished_at"] is not None


def test_running_again_when_complete_does_nothing(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    out = tmp_path / "run"
    transcribe_manifest(FakeModel(), manifest, out, progress=False)
    run_json = (out / "run.json").read_text(encoding="utf-8")
    again = FakeModel()
    transcribe_manifest(again, manifest, out, progress=False)
    assert again.transcribed == []
    assert (out / "run.json").read_text(encoding="utf-8") == run_json  # timing kept


def test_different_settings_in_the_same_directory_are_rejected(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    out = tmp_path / "run"
    transcribe_manifest(FakeModel(beam_size=5), manifest, out, progress=False)
    with pytest.raises(ValueError, match=r"different \['model_config'\]"):
        transcribe_manifest(FakeModel(beam_size=1), manifest, out, progress=False)


def test_predictions_for_unknown_utterances_are_rejected(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    out = tmp_path / "run"
    append_predictions(out / "predictions.jsonl", [Hypothesis("not-in-manifest", "x")])
    with pytest.raises(ValueError, match="not in the manifest"):
        transcribe_manifest(FakeModel(), manifest, out, progress=False)


def test_limit_transcribes_only_the_first_utterances(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    path = transcribe_manifest(FakeModel(), manifest, tmp_path / "run", limit=2, progress=False)
    assert [h.utterance_id for h in read_predictions(path)] == ["u0", "u1"]


def test_invalid_batch_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        transcribe_manifest(FakeModel(), _manifest(tmp_path), tmp_path / "run", batch_size=0)


# --- CLI ------------------------------------------------------------------------------


def test_cli_transcribe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, Any] = {}

    def fake_whisper(model_size: str, **settings: Any) -> FakeModel:
        created.update(model_size=model_size, **settings)
        return FakeModel(beam_size=settings["beam_size"])

    monkeypatch.setattr(cli, "FasterWhisperModel", fake_whisper)
    manifest = _manifest(tmp_path)
    out = tmp_path / "run"
    cli.main(["transcribe", "--manifest", str(manifest), "--output", str(out), "--beam-size", "1"])
    assert created == {
        "model_size": "small",
        "device": "cpu",
        "compute_type": None,
        "beam_size": 1,
        "cpu_threads": 4,
    }
    assert len(read_predictions(out / "predictions.jsonl")) == 5
