"""Transcribe a whole manifest with a model, saving predictions and run metadata.

A run directory contains:

- ``predictions.jsonl``: one raw hypothesis per utterance;
- ``run.json``: model name and decoding config, manifest path and SHA-256, git
  commit, environment, audio duration, processing time and real-time factor.

Runs are resumable: running again with the same model, settings and manifest
skips the utterances already in ``predictions.jsonl``. Running with different
settings in the same directory is an error, so one directory is always one
configuration.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from tqdm import tqdm

from asr_lab.data.manifest import ManifestEntry, read_manifest
from asr_lab.inference.base import ASRModel, load_sample
from asr_lab.inference.predictions import append_predictions, read_predictions
from asr_lab.provenance import environment, file_sha256, git_commit, utc_now

PREDICTIONS_FILE = "predictions.jsonl"
RUN_FILE = "run.json"
# Fields that identify a configuration: a run can only be resumed if they all match.
_IDENTITY_FIELDS = ("model", "model_config", "manifest_sha256", "limit")


def transcribe_manifest(
    model: ASRModel,
    manifest_path: Path,
    output_dir: Path,
    *,
    batch_size: int = 8,
    limit: int | None = None,
    progress: bool = True,
) -> Path:
    """Transcribe the manifest (or its first ``limit`` utterances) into ``output_dir``.

    Returns:
        The path of ``predictions.jsonl``.

    Raises:
        ValueError: if ``output_dir`` holds a run with different settings, or
            predictions for utterances that are not in the manifest.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    entries = read_manifest(manifest_path)[:limit]
    predictions_path = output_dir / PREDICTIONS_FILE
    run_path = output_dir / RUN_FILE

    run = _start_or_resume(run_path, model, manifest_path, entries, limit)
    done = _done_ids(predictions_path, {entry.utterance_id for entry in entries})
    todo = [entry for entry in entries if entry.utterance_id not in done]

    started = time.perf_counter()
    try:
        with tqdm(total=len(entries), initial=len(done), unit="utt", disable=not progress) as bar:
            for batch in _batches(todo, batch_size):
                samples = [load_sample(entry, manifest_path) for entry in batch]
                append_predictions(predictions_path, model.transcribe(samples))
                bar.update(len(batch))
    finally:
        # Save the elapsed time even if the run is interrupted, so the RTF stays correct.
        run["processing_seconds"] += time.perf_counter() - started
        complete = _all_written(predictions_path, entries)
        run["finished_at"] = utc_now() if complete else None
        run["rtf"] = run["processing_seconds"] / run["audio_seconds"] if complete else None
        _write_json(run_path, run)
    return predictions_path


def _start_or_resume(
    run_path: Path,
    model: ASRModel,
    manifest_path: Path,
    entries: Sequence[ManifestEntry],
    limit: int | None,
) -> dict[str, Any]:
    identity = {
        "model": model.name,
        "model_config": model.config,
        "manifest_sha256": file_sha256(manifest_path),
        "limit": limit,
    }
    if run_path.exists():
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        changed = [name for name in _IDENTITY_FIELDS if previous.get(name) != identity[name]]
        if changed:
            raise ValueError(
                f"{run_path.parent} already holds a run with different {changed}; "
                "use a new output directory"
            )
        return previous
    return {
        **identity,
        "manifest": manifest_path.as_posix(),
        "n_utterances": len(entries),
        "audio_seconds": sum(entry.duration for entry in entries),
        "git_commit": git_commit(),
        "git_dirty": "unknown",  # cannot be checked without running git
        "environment": environment(),
        "started_at": utc_now(),
        "finished_at": None,
        "processing_seconds": 0.0,
        "rtf": None,
    }


def _done_ids(predictions_path: Path, manifest_ids: set[str]) -> set[str]:
    if not predictions_path.exists():
        return set()
    done = {hypothesis.utterance_id for hypothesis in read_predictions(predictions_path)}
    unknown = done - manifest_ids
    if unknown:
        raise ValueError(
            f"{predictions_path} has {len(unknown)} predictions for utterances "
            f"not in the manifest, e.g. {sorted(unknown)[:3]}"
        )
    return done


def _all_written(predictions_path: Path, entries: Sequence[ManifestEntry]) -> bool:
    if not predictions_path.exists():
        return not entries
    written = {hypothesis.utterance_id for hypothesis in read_predictions(predictions_path)}
    return written == {entry.utterance_id for entry in entries}


def _batches(items: Sequence[ManifestEntry], size: int) -> Iterator[Sequence[ManifestEntry]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
