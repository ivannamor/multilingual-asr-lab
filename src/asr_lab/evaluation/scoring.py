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
from dataclasses import dataclass
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

DEFAULT_N_RESAMPLES = 1000
DEFAULT_SEED = 0


def score_predictions(
    entries: Sequence[ManifestEntry],
    hypotheses: Sequence[Hypothesis],
    *,
    normalizer: Normalizer | None = None,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    level: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Compute corpus WER and CER with bootstrap confidence intervals.

    Args:
        normalizer: applied to references and hypotheses; by default the
            normalizer of each utterance's language.

    Raises:
        ValueError: if predictions are missing for some utterances or exist for
            utterances that are not in the manifest.
    """
    pairs = normalized_pairs(entries, hypotheses, normalizer)
    references, predicted = pairs.references, pairs.hypotheses
    units = resampling_units(entries)
    return {
        "n_utterances": len(entries),
        "normalizer": pairs.normalizer_names,
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
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> tuple[dict[str, Any], Path]:
    """Score a run directory and save the scores next to its predictions.

    The manifest defaults to the one recorded in the run's ``run.json``. The
    file name depends on the scoring settings (see ``scores_filename``), so
    scoring the same run with another normalizer never overwrites a result.

    Returns:
        The scores and the path of the file they were written to.
    """
    run = load_run(run_dir, manifest_path)
    scores = {
        "model": run.info.get("model"),
        "model_config": run.info.get("model_config"),
        "manifest": run.manifest_path.as_posix(),
        "manifest_sha256": run.manifest_sha256,
        **score_predictions(
            run.entries, run.hypotheses, normalizer=normalizer, n_resamples=n_resamples, seed=seed
        ),
        "scored_at": utc_now(),
        "scoring_git_commit": git_commit(),
    }
    path = run_dir / scores_filename(scores["normalizer"], n_resamples, seed)
    text = json.dumps(scores, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return scores, path


def scores_filename(normalizers: Sequence[str], n_resamples: int, seed: int) -> str:
    """File name for a set of scoring settings.

    Settings that change the result are part of the name; defaults are left out
    to keep the common case short. Same settings give the same name and, since
    scoring is deterministic, the same content.

    Example:
        >>> scores_filename(["it-v1"], 1000, 0)
        'scores_it-v1.json'
        >>> scores_filename(["de-v1", "it-v1"], 5000, 1)
        'scores_de-v1+it-v1_r5000_s1.json'
    """
    name = "scores_" + "+".join(normalizers)
    if n_resamples != DEFAULT_N_RESAMPLES:
        name += f"_r{n_resamples}"
    if seed != DEFAULT_SEED:
        name += f"_s{seed}"
    return name + ".json"


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


@dataclass(frozen=True)
class LoadedRun:
    """A run directory with its manifest entries and predictions, ready to be scored."""

    info: dict[str, Any]  # content of run.json, empty if there is none
    manifest_path: Path
    manifest_sha256: str
    entries: list[ManifestEntry]  # limited like the run
    hypotheses: list[Hypothesis]


@dataclass(frozen=True)
class NormalizedPairs:
    """References and hypotheses normalized identically, in manifest order."""

    references: list[str]
    hypotheses: list[str]
    normalizer_names: list[str]


def load_run(run_dir: Path, manifest_path: Path | None = None) -> LoadedRun:
    """Read a run directory and its manifest (by default the one recorded in run.json).

    Raises:
        ValueError: if no manifest is known or it changed since the run was transcribed.
    """
    run_path = run_dir / "run.json"
    info = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
    if manifest_path is None:
        if "manifest" not in info:
            raise ValueError(f"No run.json with a manifest in {run_dir}; pass the manifest")
        manifest_path = Path(info["manifest"])
    manifest_sha256 = file_sha256(manifest_path)
    if info.get("manifest_sha256") not in (None, manifest_sha256):
        raise ValueError(f"{manifest_path} has changed since the run was transcribed")
    return LoadedRun(
        info=info,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        entries=read_manifest(manifest_path)[: info.get("limit")],
        hypotheses=read_predictions(run_dir / "predictions.jsonl"),
    )


def normalized_pairs(
    entries: Sequence[ManifestEntry],
    hypotheses: Sequence[Hypothesis],
    normalizer: Normalizer | None = None,
) -> NormalizedPairs:
    """Match predictions to manifest entries and normalize both sides identically.

    Args:
        normalizer: by default the normalizer of each utterance's language.

    Raises:
        ValueError: if predictions are missing for some utterances or exist for
            utterances that are not in the manifest.
    """
    by_id = _match(entries, hypotheses)
    normalizers = [normalizer or get_normalizer(entry.lang) for entry in entries]
    return NormalizedPairs(
        references=[norm(e.text) for norm, e in zip(normalizers, entries, strict=True)],
        hypotheses=[
            norm(by_id[e.utterance_id].text) for norm, e in zip(normalizers, entries, strict=True)
        ],
        normalizer_names=sorted({norm.name for norm in normalizers}),
    )


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


def resampling_units(entries: Sequence[ManifestEntry]) -> dict[str, list[str] | None]:
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
