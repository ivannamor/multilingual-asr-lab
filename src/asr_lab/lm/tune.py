"""Choosing the language model's weights on a dev set, then decoding a test set once.

``tune_test_set`` reads only the dev log probabilities and the dev references. ``decode_test_set``
reads the test log probabilities with the weights already chosen and writes ordinary run
directories, which the rest of the project scores without knowing where they came from.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from asr_lab.data.manifest import read_manifest
from asr_lab.evaluation.metrics import ErrorCounts, error_rate, utterance_counts, word_tokens
from asr_lab.evaluation.normalize import get_normalizer_by_name
from asr_lab.inference.base import Hypothesis
from asr_lab.inference.predictions import append_predictions
from asr_lab.lm.decoder import DecodingOptions, KenLMModel, beam_search
from asr_lab.lm.logprobs import iter_log_probs, read_metadata
from asr_lab.provenance import environment, file_sha256, git_commit, utc_now

_worker: dict[str, Any] = {}


def lm_name(corpus: str, order: int) -> str:
    """Name of a language model, as in the comparisons of the spec.

    >>> lm_name("europarl", 5)
    'europarl-5'
    """
    return f"{corpus}-{order}"


def lm_paths(models_dir: Path, corpus: str, lang: str, order: int) -> tuple[Path, Path]:
    """The text of a language model and its binary file."""
    text = models_dir / f"{corpus}-{lang}.txt"
    return text, models_dir / f"{corpus}-{lang}-{order}.bin"


def decoding_options(spec: dict[str, Any], alpha: float, beta: float) -> DecodingOptions:
    return replace(DecodingOptions(**spec["decoding"]), alpha=alpha, beta=beta)


def select_weights(grid: list[dict[str, float]]) -> dict[str, float]:
    """The cell with the lowest WER; in a tie, the smaller alpha, then the smaller beta.

    >>> cells = [{"alpha": 0.5, "beta": 1.0, "wer": 0.1}, {"alpha": 0.2, "beta": 3.0, "wer": 0.1}]
    >>> select_weights(cells)
    {'alpha': 0.2, 'beta': 3.0, 'wer': 0.1}
    """
    return min(grid, key=lambda cell: (cell["wer"], cell["alpha"], cell["beta"]))


def _start_worker(
    logprobs: Path, lm_path: Path, references: dict[str, str], normalizer: str
) -> None:
    _worker["utterances"] = dict(iter_log_probs(logprobs))
    _worker["tokens"] = read_metadata(logprobs)["tokens"]
    _worker["lm"] = KenLMModel(str(lm_path))
    _worker["references"] = references
    _worker["normalizer"] = get_normalizer_by_name(normalizer)


def _score_cell(options: DecodingOptions) -> dict[str, float]:
    ids = list(_worker["references"])
    hypotheses = [
        _worker["normalizer"](
            beam_search(_worker["utterances"][i], _worker["tokens"], _worker["lm"], options)
        )
        for i in ids
    ]
    references = [_worker["references"][i] for i in ids]
    counts = sum(utterance_counts(references, hypotheses, word_tokens), ErrorCounts())
    return {"alpha": options.alpha, "beta": options.beta, "wer": error_rate(counts)}


def tune_test_set(
    spec: dict[str, Any],
    test_set: dict[str, Any],
    lm_path: Path,
    logprobs: Path,
    *,
    workers: int = 8,
) -> dict[str, Any]:
    """The dev WER of every cell of the grid for one language model, and the chosen cell.

    Raises:
        ValueError: if the log probabilities are not those of the dev manifest.
    """
    manifest = Path(test_set["dev"])
    if read_metadata(logprobs)["manifest_sha256"] != file_sha256(manifest):
        raise ValueError(f"{logprobs} does not come from {manifest}")
    normalize = get_normalizer_by_name(test_set["normalizer"])
    references = {entry.utterance_id: normalize(entry.text) for entry in read_manifest(manifest)}
    cells = [
        decoding_options(spec, alpha, beta)
        for alpha in spec["grid"]["alpha"]
        for beta in spec["grid"]["beta"]
    ]
    with ProcessPoolExecutor(
        workers,
        initializer=_start_worker,
        initargs=(logprobs, lm_path, references, test_set["normalizer"]),
    ) as pool:
        grid = list(pool.map(_score_cell, cells))
    return {"grid": grid, "selected": select_weights(grid)}


def decode_test_set(
    spec: dict[str, Any],
    test_set: dict[str, Any],
    name: str,
    lm_path: Path,
    logprobs: Path,
    weights: dict[str, float],
    output: Path,
) -> Path:
    """Decode a test set with the chosen weights and write a run directory.

    A complete run with the same language model and weights is left as it is.

    Raises:
        ValueError: if the log probabilities are not those of the test manifest.
    """
    manifest = Path(test_set["test"])
    metadata = read_metadata(logprobs)
    sha256 = file_sha256(manifest)
    if metadata["manifest_sha256"] != sha256:
        raise ValueError(f"{logprobs} does not come from {manifest}")
    options = decoding_options(spec, weights["alpha"], weights["beta"])
    config = {
        "acoustic_model": metadata["model_config"],
        "language_model": name,
        "language_model_sha256": file_sha256(lm_path),
        "decoding": options.__dict__,
    }
    run_path = output / "run.json"
    if (
        run_path.exists()
        and json.loads(run_path.read_text(encoding="utf-8")).get("model_config") == config
    ):
        return output / "predictions.jsonl"
    entries = {entry.utterance_id: entry for entry in read_manifest(manifest)}
    lm = KenLMModel(str(lm_path))
    started = time.perf_counter()
    hypotheses = [
        Hypothesis(key, beam_search(values, metadata["tokens"], lm, options))
        for key, values in iter_log_probs(logprobs)
    ]
    seconds = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions.jsonl").unlink(missing_ok=True)
    append_predictions(output / "predictions.jsonl", hypotheses)
    audio_seconds = sum(entries[h.utterance_id].duration for h in hypotheses)
    run = {
        "model": f"{metadata['model']}+{name}",
        "model_config": config,
        "manifest": manifest.as_posix(),
        "manifest_sha256": sha256,
        "limit": None,
        "n_utterances": len(hypotheses),
        "audio_seconds": audio_seconds,
        "git_commit": git_commit(),
        "git_dirty": "unknown",
        "environment": environment(),
        "started_at": None,
        "finished_at": utc_now(),
        "processing_seconds": seconds,  # the decoding only: the acoustic model ran earlier
        "rtf": seconds / audio_seconds,
    }
    text = json.dumps(run, ensure_ascii=False, indent=2) + "\n"
    run_path.write_text(text, encoding="utf-8", newline="\n")
    return output / "predictions.jsonl"
