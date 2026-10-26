"""Score saved predictions against a manifest: WER and CER with confidence intervals.

Every result carries what is needed to interpret it: number of utterances,
reference words and characters, normalizer, resampling unit, and the model and
data of the run.

Confidence intervals are computed for every available resampling unit:
utterances; ``metadata["sentence_id"]`` when present (FLEURS readings of the
same sentence); and speakers when some speaker has several utterances. The
widest interval is reported as the headline: correlated utterances make
narrower intervals over-optimistic.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from asr_lab.data.manifest import ManifestEntry, read_manifest
from asr_lab.evaluation.metrics import (
    ErrorCounts,
    char_tokens,
    error_rate,
    utterance_counts,
    word_tokens,
)
from asr_lab.evaluation.normalize import Normalizer, get_normalizer
from asr_lab.evaluation.stats import bootstrap_ci
from asr_lab.inference.base import Hypothesis
from asr_lab.inference.predictions import read_predictions
from asr_lab.provenance import file_sha256, git_commit, utc_now

SCORES_FILE = "scores.json"


def score_predictions(
    entries: Sequence[ManifestEntry],
    hypotheses: Sequence[Hypothesis],
    *,
    normalizer: Normalizer | None = None,
    n_resamples: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute corpus WER and CER with bootstrap confidence intervals.

    Args:
        normalizer: applied to references and hypotheses; by default the
            normalizer of each utterance's language.

    Raises:
        ValueError: if predictions are missing for some utterances or exist for
            utterances that are not in the manifest.
    """
    by_id = _match(entries, hypotheses)
    normalizers = [normalizer or get_normalizer(entry.lang) for entry in entries]
    references = [norm(entry.text) for norm, entry in zip(normalizers, entries, strict=True)]
    predicted = [
        norm(by_id[entry.utterance_id].text)
        for norm, entry in zip(normalizers, entries, strict=True)
    ]
    units = _resampling_units(entries)
    return {
        "n_utterances": len(entries),
        "normalizer": sorted({norm.name for norm in normalizers}),
        "n_empty_hypotheses": sum(1 for text in predicted if not text),
        "wer": _rate_report(
            utterance_counts(references, predicted, word_tokens),
            units,
            n_resamples,
            level,
            seed,
        ),
        "cer": _rate_report(
            utterance_counts(references, predicted, char_tokens),
            units,
            n_resamples,
            level,
            seed,
        ),
    }


def score_run(
    run_dir: Path,
    manifest_path: Path | None = None,
    *,
    normalizer: Normalizer | None = None,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Score a run directory and write ``scores.json`` next to its predictions.

    The manifest defaults to the one recorded in the run's ``run.json``.
    """
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
    if manifest_path is None:
        if "manifest" not in run:
            raise ValueError(f"No run.json with a manifest in {run_dir}; pass the manifest")
        manifest_path = Path(run["manifest"])
    manifest_sha256 = file_sha256(manifest_path)
    if run.get("manifest_sha256") not in (None, manifest_sha256):
        raise ValueError(f"{manifest_path} has changed since the run was transcribed")
    entries = read_manifest(manifest_path)[: run.get("limit")]
    hypotheses = read_predictions(run_dir / "predictions.jsonl")
    scores = {
        "model": run.get("model"),
        "model_config": run.get("model_config"),
        "manifest": manifest_path.as_posix(),
        "manifest_sha256": manifest_sha256,
        **score_predictions(
            entries, hypotheses, normalizer=normalizer, n_resamples=n_resamples, seed=seed
        ),
        "scored_at": utc_now(),
        "scoring_git_commit": git_commit(),
    }
    text = json.dumps(scores, ensure_ascii=False, indent=2) + "\n"
    (run_dir / SCORES_FILE).write_text(text, encoding="utf-8", newline="\n")
    return scores


def format_summary(scores: dict[str, Any]) -> str:
    """Human-readable summary of ``scores`` for the terminal."""
    wer, cer = scores["wer"], scores["cer"]
    lines = [
        f"{scores.get('model') or 'unknown model'} on {scores.get('manifest', '?')}",
        f"{scores['n_utterances']} utterances, {wer['n_reference_tokens']} words, "
        f"{cer['n_reference_tokens']} characters; normalizer {', '.join(scores['normalizer'])}",
    ]
    for label, report in (("WER", wer), ("CER", cer)):
        headline = report["confidence_intervals"][report["headline_unit"]]
        lines.append(
            f"{label} {report['estimate']:.2%}  "
            f"{headline['level']:.0%} CI [{headline['low']:.2%}, {headline['high']:.2%}]  "
            f"(resampling {report['headline_unit']}, n={headline['n_units']})"
        )
    return "\n".join(lines)


def _match(
    entries: Sequence[ManifestEntry], hypotheses: Sequence[Hypothesis]
) -> dict[str, Hypothesis]:
    by_id = {hypothesis.utterance_id: hypothesis for hypothesis in hypotheses}
    expected = {entry.utterance_id for entry in entries}
    missing = expected - by_id.keys()
    extra = by_id.keys() - expected
    if missing or extra:
        raise ValueError(
            f"Predictions do not match the manifest: {len(missing)} missing "
            f"(e.g. {sorted(missing)[:3]}), {len(extra)} not in the manifest "
            f"(e.g. {sorted(extra)[:3]})"
        )
    return by_id


def _resampling_units(entries: Sequence[ManifestEntry]) -> dict[str, list[str] | None]:
    """Group ids for each resampling unit that is meaningful for these entries."""
    units: dict[str, list[str] | None] = {"utterance": None}
    sentence_ids = [entry.metadata.get("sentence_id") for entry in entries]
    if all(sentence_ids):
        units["sentence_id"] = [str(sentence_id) for sentence_id in sentence_ids]
    speakers = [entry.speaker_id for entry in entries]
    if len(set(speakers)) < len(speakers):
        units["speaker"] = speakers
    return units


def _rate_report(
    counts: Sequence[ErrorCounts],
    units: dict[str, list[str] | None],
    n_resamples: int,
    level: float,
    seed: int,
) -> dict[str, Any]:
    total = sum(counts, ErrorCounts())
    intervals = {}
    for unit, groups in units.items():
        ci = bootstrap_ci(counts, groups=groups, n_resamples=n_resamples, level=level, seed=seed)
        intervals[unit] = {
            "low": ci.low,
            "high": ci.high,
            "level": ci.level,
            "n_units": ci.n_units,
            "n_resamples": ci.n_resamples,
        }
    return {
        "estimate": error_rate(total),
        "n_reference_tokens": total.ref_length,
        "hits": total.hits,
        "substitutions": total.substitutions,
        "deletions": total.deletions,
        "insertions": total.insertions,
        "confidence_intervals": intervals,
        # The widest interval is the most conservative: see the module docstring.
        "headline_unit": max(intervals, key=lambda u: intervals[u]["high"] - intervals[u]["low"]),
    }
