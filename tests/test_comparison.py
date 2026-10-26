"""Tests for paired comparisons of two scorings."""

import json
from pathlib import Path

import pytest

from asr_lab import cli
from asr_lab.data.manifest import ManifestEntry, write_manifest
from asr_lab.evaluation.comparison import compare_runs, format_comparison
from asr_lab.evaluation.normalize import BASIC, IDENTITY
from asr_lab.inference.base import Hypothesis
from asr_lab.inference.predictions import append_predictions
from asr_lab.provenance import file_sha256


def _manifest(tmp_path: Path, n: int = 40) -> Path:
    entries = [
        ManifestEntry(
            f"u{i:02d}",
            f"audio/u{i}.wav",
            1.0,
            "Il gatto dorme.",
            "it",
            f"s{i}",
            "test",
            metadata={"sentence_id": str(i // 2)},
        )
        for i in range(n)
    ]
    path = tmp_path / "data" / "test.jsonl"
    write_manifest(path, entries)
    return path


def _run(tmp_path: Path, manifest: Path, name: str, texts: list[str]) -> Path:
    run_dir = tmp_path / name
    hypotheses = [Hypothesis(f"u{i:02d}", text) for i, text in enumerate(texts)]
    append_predictions(run_dir / "predictions.jsonl", hypotheses)
    run = {"model": name, "manifest": manifest.as_posix(), "manifest_sha256": file_sha256(manifest)}
    (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
    return run_dir


def test_better_model_is_significantly_better(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    worse = _run(tmp_path, manifest, "worse", ["il cane dorme"] * 20 + ["il gatto dorme"] * 20)
    better = _run(tmp_path, manifest, "better", ["il gatto dorme"] * 40)
    result = compare_runs(worse, better, n_resamples=200)
    assert result["a"]["rate"] == pytest.approx(20 / 120)
    assert result["b"]["rate"] == 0.0
    assert result["delta"] == pytest.approx(-20 / 120)
    assert result["significant"]
    assert set(result["confidence_intervals"]) == {"utterance", "sentence_id"}


def test_same_run_with_two_normalizers(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    run = _run(
        tmp_path, manifest, "run", ["il gatto dorme"] * 40
    )  # differs only in case/punctuation
    result = compare_runs(run, run, normalizer_a=IDENTITY, normalizer_b=BASIC, n_resamples=200)
    assert result["a"]["normalizer"] == ["none"]
    assert result["b"]["normalizer"] == ["basic-v1"]
    assert result["a"]["rate"] == pytest.approx(2 / 3)  # "Il" and "dorme." differ
    assert result["b"]["rate"] == 0.0
    assert result["confidence_intervals"]["utterance"]["prob_b_better"] == 1.0


def test_identical_scorings_are_not_significantly_different(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    run = _run(tmp_path, manifest, "run", ["il cane dorme"] * 40)
    result = compare_runs(run, run, n_resamples=100)
    assert result["delta"] == 0.0
    assert not result["significant"]


def test_runs_on_different_utterances_are_rejected(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    full = _run(tmp_path, manifest, "full", ["x"] * 40)
    other_manifest = _manifest(tmp_path / "other", n=10)
    short = _run(tmp_path, other_manifest, "short", ["x"] * 10)
    with pytest.raises(ValueError, match="do not cover the same utterances"):
        compare_runs(full, short)


def test_unknown_metric(tmp_path: Path) -> None:
    run = _run(tmp_path, _manifest(tmp_path), "run", ["x"] * 40)
    with pytest.raises(ValueError, match="Unknown metric 'ber'"):
        compare_runs(run, run, metric="ber")


def test_cli_compare_saves_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest = _manifest(tmp_path)
    run = _run(tmp_path, manifest, "run", ["il gatto dorme"] * 40)
    output = tmp_path / "results" / "cmp.json"
    cli.main(
        ["compare", "--a", str(run), "--b", str(run), "--normalizer-a", "none",
         "--normalizer-b", "it-v1", "--metric", "cer", "--n-resamples", "50",
         "--output", str(output)]
    )  # fmt: skip
    printed = capsys.readouterr().out
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["metric"] == "cer"
    assert format_comparison(saved) in printed
    assert "B - A =" in printed
