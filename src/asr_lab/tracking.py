"""Mirror scores and comparisons in MLflow, to browse and compare them in its UI.

The JSON files written by ``score_run`` and ``save_comparison`` remain the source
of truth: tables and documents are generated from them, never from MLflow. Each
result file maps to exactly one MLflow run (identified by the tag
``asr_lab.key``), so logging the same file again replaces its MLflow run instead
of adding a duplicate.

MLflow is an optional dependency (``uv sync --extra tracking``). It stores runs
where ``MLFLOW_TRACKING_URI`` points, by default in ``mlflow.db`` in the current
directory; ``uv run mlflow ui`` opens the browser interface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

KEY_TAG = "asr_lab.key"
# Opens the experiment in the UI's "Model training" view (runs table) instead of the
# GenAI one (traces), which would look empty.
_EXPERIMENT_KIND = {"mlflow.experimentKind": "custom_model_development"}


def log_scores(scores: dict[str, Any], scores_path: Path, experiment: str) -> str:
    """Log a scores file (and its run's speed and provenance); returns the MLflow run id."""
    run_info = _read_json(scores_path.parent / "run.json")
    params = {
        "run_dir": scores_path.parent.as_posix(),
        "model": scores.get("model"),
        "normalizer": ",".join(scores["normalizer"]),
        "manifest": scores["manifest"],
        "manifest_sha256": scores["manifest_sha256"],
        **{f"model.{key}": value for key, value in (scores.get("model_config") or {}).items()},
    }
    metrics = {"n_utterances": scores["n_utterances"]}
    for metric in ("wer", "cer"):
        report = scores[metric]
        ci = report["confidence_intervals"][report["headline_unit"]]
        metrics |= {
            metric: report["estimate"],
            f"{metric}_ci_low": ci["low"],
            f"{metric}_ci_high": ci["high"],
            f"{metric}_substitutions": report["substitutions"],
            f"{metric}_deletions": report["deletions"],
            f"{metric}_insertions": report["insertions"],
            f"{metric}_reference_tokens": report["n_reference_tokens"],
        }
    for field in ("rtf", "processing_seconds", "audio_seconds"):
        if run_info.get(field) is not None:
            metrics[field] = run_info[field]
    tags = {
        "kind": "scores",
        "scoring_git_commit": scores.get("scoring_git_commit"),
        "transcription_git_commit": run_info.get("git_commit"),
    }
    name = f"{scores_path.parent.name} [{params['normalizer']}]"
    return _replace_run(experiment, scores_path, name, params, metrics, tags)


def log_comparison(result: dict[str, Any], path: Path, experiment: str) -> str:
    """Log a saved paired comparison; returns the MLflow run id."""
    ci = result["confidence_intervals"][result["headline_unit"]]
    params = {"metric": result["metric"], "manifest": result["manifest"]}
    for side in ("a", "b"):
        params |= {
            f"{side}.run": result[side]["run"],
            f"{side}.model": result[side]["model"],
            f"{side}.normalizer": ",".join(result[side]["normalizer"]),
        }
    metrics = {
        "rate_a": result["a"]["rate"],
        "rate_b": result["b"]["rate"],
        "delta": result["delta"],
        "delta_ci_low": ci["low"],
        "delta_ci_high": ci["high"],
        "prob_b_better": ci["prob_b_better"],
        "significant": float(result["significant"]),
        "n_utterances": result["n_utterances"],
    }
    tags = {"kind": "comparison"}
    return _replace_run(experiment, path, path.stem, params, metrics, tags)


def _replace_run(
    experiment: str,
    source: Path,
    name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, Any],
) -> str:
    mlflow = _import_mlflow()
    key = source.as_posix()
    experiment_id = mlflow.set_experiment(experiment).experiment_id
    client = mlflow.MlflowClient()
    for tag, value in _EXPERIMENT_KIND.items():
        client.set_experiment_tag(experiment_id, tag, value)
    for old in client.search_runs([experiment_id], filter_string=f"tags.`{KEY_TAG}` = '{key}'"):
        client.delete_run(old.info.run_id)
    run_tags = {KEY_TAG: key, **{k: str(v) for k, v in tags.items() if v is not None}}
    with mlflow.start_run(run_name=name, tags=run_tags) as run:
        # MLflow stores parameters as strings; None becomes "None" so that it stays visible.
        mlflow.log_params({key: str(value) for key, value in params.items()})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(source))
    return run.info.run_id


def _import_mlflow() -> Any:
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError(
            "MLflow is not installed: run `uv sync --extra tracking` to enable tracking"
        ) from error
    return mlflow


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
