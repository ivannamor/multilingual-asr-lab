"""Tests for the MLflow mirror of scores and comparisons.

Skipped when the optional ``tracking`` extra is not installed. Each test uses its
own SQLite database and artifact folder in a temporary directory.
"""

import json
import sys
from pathlib import Path

import pytest

mlflow = pytest.importorskip("mlflow")

from asr_lab import cli  # noqa: E402 (after the skip)
from asr_lab.data.manifest import ManifestEntry, write_manifest  # noqa: E402
from asr_lab.inference.base import Hypothesis  # noqa: E402
from asr_lab.inference.predictions import append_predictions  # noqa: E402
from asr_lab.provenance import file_sha256  # noqa: E402
from asr_lab.tracking import KEY_TAG  # noqa: E402

EXPERIMENT = "test-experiment"


@pytest.fixture(autouse=True)
def tracking_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # artifacts go to ./mlruns, inside the temporary directory
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")


def _run(name: str, texts: list[str]) -> Path:
    manifest = Path("data/test.jsonl")
    if not manifest.exists():
        entries = [
            ManifestEntry(
                f"u{i}", f"audio/u{i}.wav", 2.0, "il gatto dorme", "it", f"s{i}", "test",
                metadata={"sentence_id": str(i)},
            )
            for i in range(len(texts))
        ]  # fmt: skip
        write_manifest(manifest, entries)
    run_dir = Path("runs") / name
    append_predictions(
        run_dir / "predictions.jsonl", [Hypothesis(f"u{i}", t) for i, t in enumerate(texts)]
    )
    run = {
        "model": name,
        "manifest": manifest.as_posix(),
        "manifest_sha256": file_sha256(manifest),
        "rtf": 0.1,
        "audio_seconds": 2.0 * len(texts),
        "git_commit": "abc123",
    }
    (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
    return run_dir


def _runs(kind: str) -> list:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    runs = mlflow.MlflowClient().search_runs([experiment.experiment_id])
    return [run for run in runs if run.data.tags["kind"] == kind]


def test_score_is_logged_with_metrics_params_and_artifact() -> None:
    run_dir = _run("good", ["il gatto dorme"] * 9 + ["il cane dorme"])
    cli.main(["score", "--run", str(run_dir), "--n-resamples", "50", "--mlflow", EXPERIMENT])
    (logged,) = _runs("scores")
    assert logged.data.metrics["wer"] == pytest.approx(1 / 30)
    assert logged.data.metrics["wer_ci_low"] <= logged.data.metrics["wer"]
    assert logged.data.metrics["rtf"] == 0.1
    assert logged.data.params["normalizer"] == "it-v1"
    assert logged.data.tags["transcription_git_commit"] == "abc123"
    assert logged.data.tags[KEY_TAG] == "runs/good/scores_it-v1_r50.json"
    artifacts = mlflow.MlflowClient().list_artifacts(logged.info.run_id)
    assert [artifact.path for artifact in artifacts] == ["scores_it-v1_r50.json"]
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    assert experiment.tags["mlflow.experimentKind"] == "custom_model_development"


def test_logging_the_same_file_again_replaces_its_run() -> None:
    run_dir = _run("good", ["il gatto dorme"] * 10)
    for _ in range(2):
        cli.main(["score", "--run", str(run_dir), "--n-resamples", "50", "--mlflow", EXPERIMENT])
    assert len(_runs("scores")) == 1


def test_comparison_is_logged() -> None:
    worse = _run("worse", ["il cane dorme"] * 5 + ["il gatto dorme"] * 5)
    better = _run("better", ["il gatto dorme"] * 10)
    output = Path("results/cmp.json")
    cli.main(["compare", "--a", str(worse), "--b", str(better), "--n-resamples", "50",
              "--output", str(output), "--mlflow", EXPERIMENT])  # fmt: skip
    (logged,) = _runs("comparison")
    assert logged.data.metrics["delta"] == pytest.approx(-5 / 30)
    assert logged.data.params["a.run"] == "runs/worse"
    assert logged.info.run_name == "cmp"


def test_comparison_needs_an_output_file_to_be_logged() -> None:
    run_dir = _run("good", ["il gatto dorme"] * 10)
    with pytest.raises(SystemExit):
        cli.main(["compare", "--a", str(run_dir), "--b", str(run_dir), "--mlflow", EXPERIMENT])


def test_clear_error_when_mlflow_is_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlflow", None)  # makes `import mlflow` fail
    run_dir = _run("good", ["il gatto dorme"] * 10)
    with pytest.raises(RuntimeError, match="uv sync --extra tracking"):
        cli.main(["score", "--run", str(run_dir), "--n-resamples", "50", "--mlflow", EXPERIMENT])
