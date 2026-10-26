"""Paired comparison of two scorings of the same utterances.

The two sides can differ by model (two runs on the same manifest), by
normalizer (one run scored two ways), or both. Every bootstrap replicate draws
the same utterances for both sides, so the interval of the difference excludes
the noise of which utterances happened to be in the test set.

As for single scores, an interval is computed for every meaningful resampling
unit and the widest one is the headline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from asr_lab.evaluation.metrics import (
    ErrorCounts,
    char_tokens,
    error_rate,
    utterance_counts,
    word_tokens,
)
from asr_lab.evaluation.normalize import Normalizer
from asr_lab.evaluation.scoring import (
    DEFAULT_N_RESAMPLES,
    DEFAULT_SEED,
    LoadedRun,
    load_run,
    normalized_pairs,
    resampling_units,
)
from asr_lab.evaluation.stats import paired_bootstrap

METRICS = {"wer": word_tokens, "cer": char_tokens}


def compare_runs(
    run_a: Path,
    run_b: Path,
    *,
    normalizer_a: Normalizer | None = None,
    normalizer_b: Normalizer | None = None,
    metric: str = "wer",
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Compare B against A; ``delta`` = rate(B) - rate(A), negative when B is better.

    ``run_a`` and ``run_b`` may be the same directory (comparing normalizers).

    Raises:
        ValueError: if the two runs do not cover the same utterances.
    """
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}; choose from {sorted(METRICS)}")
    loaded_a, loaded_b = load_run(run_a), load_run(run_b)
    ids_a = [entry.utterance_id for entry in loaded_a.entries]
    if ids_a != [entry.utterance_id for entry in loaded_b.entries]:
        raise ValueError(f"{run_a} and {run_b} do not cover the same utterances")
    counts_a, names_a = _counts(loaded_a, normalizer_a, metric)
    counts_b, names_b = _counts(loaded_b, normalizer_b, metric)

    intervals = {}
    for unit, groups in resampling_units(loaded_a.entries).items():
        result = paired_bootstrap(
            counts_a, counts_b, groups=groups, n_resamples=n_resamples, seed=seed
        )
        intervals[unit] = {
            "low": result.low,
            "high": result.high,
            "prob_b_better": result.prob_b_better,
            "significant": result.significant,
            "n_units": result.n_units,
            "level": result.level,
            "n_resamples": result.n_resamples,
        }
    headline = max(intervals, key=lambda u: intervals[u]["high"] - intervals[u]["low"])
    rate_a = error_rate(sum(counts_a, ErrorCounts()))
    rate_b = error_rate(sum(counts_b, ErrorCounts()))
    return {
        "metric": metric,
        "manifest": loaded_a.manifest_path.as_posix(),
        "n_utterances": len(ids_a),
        "a": _side(run_a, loaded_a, names_a, rate_a),
        "b": _side(run_b, loaded_b, names_b, rate_b),
        "delta": rate_b - rate_a,
        "confidence_intervals": intervals,
        "headline_unit": headline,
        "significant": intervals[headline]["significant"],
    }


def save_comparison(result: dict[str, Any], path: Path) -> None:
    """Write a comparison as JSON (UTF-8, LF)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def format_comparison(result: dict[str, Any]) -> str:
    """Human-readable summary of a comparison for the terminal."""
    metric = result["metric"].upper()
    ci = result["confidence_intervals"][result["headline_unit"]]
    verdict = "significant" if result["significant"] else "not significant"
    return "\n".join(
        (
            f"A: {_describe(result['a'])}  {metric} {result['a']['rate']:.2%}",
            f"B: {_describe(result['b'])}  {metric} {result['b']['rate']:.2%}",
            f"B - A = {result['delta']:+.2%}  {ci['level']:.0%} CI [{ci['low']:+.2%}, "
            f"{ci['high']:+.2%}]  ({verdict}; B better in {ci['prob_b_better']:.1%} of "
            f"replicates; resampling {result['headline_unit']}, n={ci['n_units']})",
        )
    )


def _counts(
    run: LoadedRun, normalizer: Normalizer | None, metric: str
) -> tuple[list[ErrorCounts], list[str]]:
    pairs = normalized_pairs(run.entries, run.hypotheses, normalizer)
    counts = utterance_counts(pairs.references, pairs.hypotheses, METRICS[metric])
    return counts, pairs.normalizer_names


def _side(run_dir: Path, run: LoadedRun, normalizers: list[str], rate: float) -> dict[str, Any]:
    return {
        "run": run_dir.as_posix(),
        "model": run.info.get("model"),
        "normalizer": normalizers,
        "rate": rate,
    }


def _describe(side: dict[str, Any]) -> str:
    return f"{side['model'] or side['run']} [{', '.join(side['normalizer'])}]"
