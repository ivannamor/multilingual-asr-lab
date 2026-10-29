"""Quality features of training utterances, and the filter rules built on them.

Each utterance gets cheap features that describe its audio, its transcript and how well a strong
model agrees with the transcript:

- ``duration`` (seconds) and ``chars_per_second``: letters of the normalized transcript per second
  of audio. Speech has a narrow range of speaking rates, so a rate far outside it means that the
  transcript and the audio do not match (missing or extra words);
- ``snr_db``: a blind estimate of the signal-to-noise ratio (see ``estimate_snr``);
- ``clipping``: the share of samples at full scale (distorted recordings);
- ``duplicate_text`` and ``duplicate_audio``: the utterance repeats an earlier one;
- ``wer`` and ``length_ratio``: word error rate of a strong model's transcript against the
  reference, and its length in words relative to the reference. A high WER suggests label noise,
  but also hard audio, so it is read together with the other features.

Rules (thresholds in the spec) flag the utterances; they are independent, so their counts can
overlap, and the summary also gives the union.

A high WER only says that the model and the transcript disagree, not which one is wrong. After
reading the first results, the utterances flagged by ``high_wer`` are split by the length of the
model's transcript (``high_wer_kind``), and a second model transcribes the collapsed ones
(``second_opinion``). Both are post hoc: they describe the flagged utterances and change no rule.
"""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from asr_lab.data.clustering import cluster_rows, load_cluster_spec
from asr_lab.data.manifest import ManifestEntry, read_manifest, write_manifest
from asr_lab.data.voxpopuli import load_data_spec
from asr_lab.evaluation.error_analysis import UtteranceErrors, analyze
from asr_lab.evaluation.metrics import ErrorCounts, error_rate, word_tokens
from asr_lab.evaluation.normalize import Normalizer, get_normalizer_by_name
from asr_lab.evaluation.scoring import load_run
from asr_lab.evaluation.stats import bootstrap_ci
from asr_lab.provenance import file_sha256

FRAME_SECONDS = 0.025  # frames for the energy statistics of estimate_snr
_ENERGY_FLOOR = 1e-10  # digital silence: caps the estimate at 100 dB
FULL_SCALE = 32_767 / 32_768  # the largest 16-bit sample, as a float
_TEXTS = ("normalized_text", "hypothesis_text")
_DESCRIBED = ("duration", "chars_per_second", "snr_db", "clipping", "wer", "length_ratio")
RULES = (
    "empty_text", "too_short", "too_long", "too_slow", "too_fast", "low_snr", "clipped",
    "high_wer", "duplicate_text", "duplicate_audio",
)  # fmt: skip
# Post hoc split of the high_wer utterances by the model transcript's length (words, relative to
# the reference), set after reading the results
COLLAPSED_BELOW = 0.2  # a fifth of the reference or less: the model dropped most of the speech
SIMILAR_LENGTH = (0.5, 1.5)  # about as long: wrong words, or words of a neighbouring segment
HIGH_WER_KINDS = ("collapsed", "similar_length", "other")


@dataclass(frozen=True)
class Filters:
    """The ``[filters]`` table of a quality spec."""

    min_duration: float
    max_duration: float
    min_chars_per_second: float
    max_chars_per_second: float
    min_snr_db: float
    max_clipping: float
    max_wer: float


def load_filters(path: Path) -> Filters:
    """Read the ``[filters]`` table of a quality spec."""
    return Filters(**tomllib.loads(path.read_text(encoding="utf-8"))["filters"])


def estimate_snr(waveform: np.ndarray, sample_rate: int) -> float:
    """Blind estimate of the signal-to-noise ratio in decibels, without a clean reference.

    The audio is cut into 25 ms frames; the quietest frames (10th percentile of energy) are
    taken as noise alone, the loudest (90th percentile) as speech plus noise. Speech that never
    pauses makes the "noise" frames contain speech, so the estimate is a lower bound in that case.

    >>> rng = np.random.default_rng(0)
    >>> noise = 0.01 * rng.standard_normal(16_000)
    >>> speech = np.concatenate([np.zeros(8_000), 0.3 * np.ones(8_000)])
    >>> print(f"{estimate_snr(speech + noise, 16_000):.0f} dB")
    30 dB
    """
    frame = max(1, round(FRAME_SECONDS * sample_rate))
    n_frames = len(waveform) // frame
    if n_frames == 0:
        return float("nan")
    frames = np.asarray(waveform[: n_frames * frame], dtype=np.float64).reshape(n_frames, frame)
    energy = np.mean(np.square(frames), axis=1)
    noise, loud = np.percentile(energy, [10, 90])
    return float(10 * np.log10(max(loud, _ENERGY_FLOOR) / max(noise, _ENERGY_FLOOR)))


def clipping(waveform: np.ndarray) -> float:
    """Share of samples at (or beyond) full scale."""
    return float(np.mean(np.abs(waveform) >= FULL_SCALE)) if len(waveform) else 0.0


def audio_features(path: Path) -> dict[str, Any]:
    """SNR estimate, clipping and a hash of the samples (to find duplicated audio)."""
    samples, sample_rate = sf.read(path, dtype="int16")
    waveform = samples.astype(np.float32) / 32_768
    return {
        "snr_db": estimate_snr(waveform, sample_rate),
        "clipping": clipping(waveform),
        "audio_sha1": hashlib.sha1(samples.tobytes()).hexdigest(),
    }


def text_features(text: str, duration: float, normalizer: Normalizer) -> dict[str, Any]:
    """Words and speaking rate of a transcript, after normalization."""
    normalized = normalizer(text)
    letters = len(normalized.replace(" ", ""))
    return {
        "empty_text": not normalized,
        "words": len(word_tokens(normalized)),
        "chars_per_second": letters / duration,
        "normalized_text": normalized,
    }


def compute_features(
    manifest_path: Path, normalizer: Normalizer, run_dir: Path | None = None
) -> list[dict[str, Any]]:
    """One row of features per utterance, in manifest order.

    With ``run_dir`` (a transcription of the same manifest), also the model's WER and length
    ratio; they are None for empty references, where a rate is undefined.
    """
    entries = read_manifest(manifest_path)
    model = _model_features(run_dir, entries, normalizer) if run_dir else {}
    rows = []
    for entry in entries:
        row: dict[str, Any] = {
            "utterance_id": entry.utterance_id,
            "speaker_id": entry.speaker_id,
            "session": entry.metadata.get("session", ""),
            "duration": entry.duration,
        }
        row |= text_features(entry.text, entry.duration, normalizer)
        row |= audio_features(entry.audio_path(manifest_path))
        row |= model.get(
            entry.utterance_id, {"wer": None, "length_ratio": None, "hypothesis_text": None}
        )
        rows.append(row)
    _mark_duplicates(rows, "normalized_text", "duplicate_text", skip_empty=True)
    _mark_duplicates(rows, "audio_sha1", "duplicate_audio")
    return rows


def flag(row: dict[str, Any], filters: Filters) -> list[str]:
    """The names of the rules an utterance breaks (a missing feature breaks no rule)."""
    checks = {
        "empty_text": row["empty_text"],
        "too_short": row["duration"] < filters.min_duration,
        "too_long": row["duration"] > filters.max_duration,
        "too_slow": not row["empty_text"]
        and row["chars_per_second"] < filters.min_chars_per_second,
        "too_fast": row["chars_per_second"] > filters.max_chars_per_second,
        "low_snr": _below(row["snr_db"], filters.min_snr_db),
        "clipped": row["clipping"] > filters.max_clipping,
        "high_wer": row["wer"] is not None and row["wer"] > filters.max_wer,
        "duplicate_text": row["duplicate_text"],
        "duplicate_audio": row["duplicate_audio"],
    }
    return [name for name in RULES if checks[name]]


def filter_summary(rows: Sequence[dict[str, Any]], filters: Filters) -> list[dict[str, Any]]:
    """Per rule, and for any rule: utterances flagged, their hours and their share."""
    flags = [set(flag(row, filters)) for row in rows]
    total_hours = sum(row["duration"] for row in rows) / 3600
    summary = []
    for name in (*RULES, "any"):
        flagged = [
            row for row, broken in zip(rows, flags, strict=True)
            if (broken if name == "any" else name in broken)
        ]  # fmt: skip
        hours = sum(row["duration"] for row in flagged) / 3600
        summary.append(
            {
                "rule": name,
                "utterances": len(flagged),
                "share": len(flagged) / len(rows) if rows else 0.0,
                "hours": hours,
                "hour_share": hours / total_hours if total_hours else 0.0,
            }
        )
    return summary


def save_features(rows: Sequence[dict[str, Any]], path: Path) -> None:
    """Write the feature rows as JSONL (UTF-8, LF), without the texts (they are in the manifest
    and the predictions)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({k: v for k, v in row.items() if k not in _TEXTS}, ensure_ascii=False)
        for row in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def analyze_quality(
    spec_path: Path, run_dir: Path | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Features, filter summary and (with a transcription run) clusters of the spec's data.

    Returns the feature rows (with each row's flags and cluster) and a JSON-serializable report.
    """
    data = load_data_spec(spec_path)
    filters = load_filters(spec_path)
    rows = compute_features(data.manifest, get_normalizer_by_name(data.normalizer), run_dir)
    flags = [flag(row, filters) for row in rows]
    for row, broken in zip(rows, flags, strict=True):
        row["flags"] = broken
    clusters = None
    if run_dir is not None:
        clusters = cluster_rows(rows, load_cluster_spec(spec_path), flags)
        labels = clusters.pop("labels")
        for row in rows:
            row["cluster"] = labels.get(row["utterance_id"])
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8")) if run_dir else {}
    report = {
        "spec": spec_path.as_posix(),
        "repo": data.repo,
        "revision": data.revision,
        "row_groups": [list(pair) for pair in data.row_groups],
        "manifest": data.manifest.as_posix(),
        "manifest_sha256": file_sha256(data.manifest),
        "normalizer": data.normalizer,
        "run": run_dir.as_posix() if run_dir else None,
        "model": run.get("model"),
        "utterances": len(rows),
        "hours": sum(row["duration"] for row in rows) / 3600,
        "speakers": len({row["speaker_id"] for row in rows}),
        "sessions": len({row["session"] for row in rows}),
        "features": {name: _distribution([row[name] for row in rows]) for name in _DESCRIBED},
        "filters": asdict(filters),
        "flags": filter_summary(rows, filters),
        "clusters": clusters,
        "high_wer": high_wer_breakdown(rows) if run_dir else None,
    }
    return rows, report


def high_wer_kind(length_ratio: float) -> str:
    """How a high-WER transcript compares in length with the reference (post hoc).

    ``collapsed``: the model wrote a few words for a whole utterance (e.g. "Grazie a tutti.");
    ``similar_length``: about as many words as the reference, so either the model got many words
    wrong or the transcript does not cover the same speech (shifted segment boundaries);
    ``other``: in between, or much longer than the reference.

    >>> [high_wer_kind(r) for r in (0.1, 0.35, 1.0, 2.0)]
    ['collapsed', 'other', 'similar_length', 'other']
    """
    if length_ratio < COLLAPSED_BELOW:
        return "collapsed"
    low, high = SIMILAR_LENGTH
    return "similar_length" if low <= length_ratio < high else "other"


def high_wer_breakdown(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per kind of ``high_wer_kind``: utterances (and share of the flagged), hours, speakers."""
    flagged = [row for row in rows if "high_wer" in row["flags"]]
    breakdown = []
    for kind in HIGH_WER_KINDS:
        members = [row for row in flagged if high_wer_kind(row["length_ratio"]) == kind]
        breakdown.append(
            {
                "kind": kind,
                "utterances": len(members),
                "share": len(members) / len(flagged) if flagged else 0.0,
                "hours": sum(row["duration"] for row in members) / 3600,
                "speakers": len({row["speaker_id"] for row in members}),
                "sessions": len({row["session"] for row in members}),
            }
        )
    return breakdown


def export_collapsed(rows: Sequence[dict[str, Any]], spec_path: Path) -> Path:
    """Write the manifest of the collapsed high-WER utterances, for a second model to transcribe.

    It is written next to the data manifest (``<output>/collapsed.jsonl``), so that the audio paths
    stay valid, and keeps the manifest order.
    """
    data = load_data_spec(spec_path)
    collapsed = {
        row["utterance_id"]
        for row in rows
        if "high_wer" in row["flags"] and high_wer_kind(row["length_ratio"]) == "collapsed"
    }
    path = data.output / "collapsed.jsonl"
    write_manifest(path, [e for e in read_manifest(data.manifest) if e.utterance_id in collapsed])
    return path


def second_opinion(
    spec_path: Path, first_run: Path, second_run: Path, *, seed: int = 0
) -> dict[str, Any]:
    """Both models on the utterances the second one transcribed (the collapsed ones).

    If the second model transcribes the speech where the first collapsed, the first model is at
    fault, not the transcript. Per model: corpus WER with a 95% CI (bootstrap by speaker),
    utterances still collapsed and utterances that would pass the ``high_wer`` rule.
    """
    normalizer_name = load_data_spec(spec_path).normalizer
    normalizer = get_normalizer_by_name(normalizer_name)
    max_wer = load_filters(spec_path).max_wer
    second = load_run(second_run)
    first = load_run(first_run)
    ids = {entry.utterance_id for entry in second.entries}
    if not ids <= {entry.utterance_id for entry in first.entries}:
        raise ValueError(f"{second_run} transcribes utterances that {first_run} does not have")
    first_hypotheses = [h for h in first.hypotheses if h.utterance_id in ids]
    speakers = [entry.speaker_id for entry in second.entries]
    models = []
    for run_dir, run, utterances in (
        (first_run, first, analyze(second.entries, first_hypotheses, normalizer)),
        (second_run, second, analyze(second.entries, second.hypotheses, normalizer)),
    ):
        ci = bootstrap_ci([u.counts for u in utterances], groups=speakers, seed=seed)
        models.append(
            {
                "run": run_dir.as_posix(),
                "model": run.info.get("model"),
                "wer": ci.estimate,
                "ci_low": ci.low,
                "ci_high": ci.high,
                "collapsed": sum(_length_ratio(u) < COLLAPSED_BELOW for u in utterances),
                "within_max_wer": sum(u.wer <= max_wer for u in utterances),
            }
        )
    words = sum((u.counts for u in utterances), ErrorCounts()).ref_length  # same for both models
    return {
        "utterances": len(ids),
        "words": words,
        "speakers": len(set(speakers)),
        "hours": sum(entry.duration for entry in second.entries) / 3600,
        "normalizer": normalizer_name,
        "max_wer": max_wer,
        "models": models,
    }


def save_report(report: dict[str, Any], path: Path) -> None:
    """Write a quality report as JSON (UTF-8, LF)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def quality_table(report: dict[str, Any], section: str) -> str:
    """One Markdown table of a quality report: ``flags`` or ``clusters``."""
    if section == "flags":
        lines = [
            "| Rule | Utterances flagged | Hours flagged |",
            "|---|---|---|",
        ]
        for row in report["flags"]:
            name = "**any rule**" if row["rule"] == "any" else f"`{row['rule']}`"
            lines.append(
                f"| {name} | {row['utterances']:,} ({row['share']:.1%}) "
                f"| {row['hours']:.2f} ({row['hour_share']:.1%}) |"
            )
        return "\n".join(lines)
    if section == "clusters":
        clusters = report["clusters"]
        lines = [
            "| Cluster | Utterances (share) | Hours | Speakers | Stands out for | Median WER "
            "| Median SNR | Median rate (chars/s) | Flagged by a rule |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for c in clusters["clusters"]:
            median = c["median"]
            lines.append(
                f"| {c['cluster']} | {c['utterances']:,} ({c['share']:.0%}) | {c['hours']:.2f} "
                f"| {c['speakers']} | {c['description']} | {median['wer']:.1%} "
                f"| {median['snr_db']:.1f} dB | {median['chars_per_second']:.1f} "
                f"| {c['flagged']:.0%} |"
            )
        return "\n".join(lines)
    if section == "high_wer":
        lines = [
            "| Model transcript length (post hoc) | Utterances (share of `high_wer`) | Hours "
            "| Speakers | Sessions |",
            "|---|---|---|---|---|",
        ]
        names = {
            "collapsed": f"collapsed (< {COLLAPSED_BELOW:.0%} of the reference)",
            "similar_length": f"similar ({SIMILAR_LENGTH[0]:.0%} to {SIMILAR_LENGTH[1]:.0%})",
            "other": "other",
        }
        for row in report["high_wer"]:
            lines.append(
                f"| {names[row['kind']]} | {row['utterances']:,} ({row['share']:.0%}) "
                f"| {row['hours']:.2f} | {row['speakers']} | {row['sessions']} |"
            )
        return "\n".join(lines)
    if section == "second_opinion":
        opinion = report["second_opinion"]
        lines = [
            "| Model | WER [95% CI] | Utterances still collapsed "
            f"| Utterances with WER ≤ {opinion['max_wer']:.0%} |",
            "|---|---|---|---|",
        ]
        n = opinion["utterances"]
        for m in opinion["models"]:
            lines.append(
                f"| {m['model']} | {m['wer']:.2%} [{m['ci_low']:.2%}, {m['ci_high']:.2%}] "
                f"| {m['collapsed']} / {n} | {m['within_max_wer']} / {n} |"
            )
        return "\n".join(lines)
    raise ValueError(
        f"unknown quality table {section!r}; "
        "expected 'flags', 'clusters', 'high_wer' or 'second_opinion'"
    )


def _distribution(values: Sequence[float | None]) -> dict[str, float | int]:
    """5th percentile, median and 95th percentile of the defined values, and how many."""
    defined = [v for v in values if v is not None and not math.isnan(v)]
    if not defined:
        return {"n": 0}
    p5, median, p95 = np.percentile(defined, [5, 50, 95]).tolist()
    return {"n": len(defined), "p5": p5, "median": median, "p95": p95}


def _length_ratio(utterance: UtteranceErrors) -> float:
    return len(word_tokens(utterance.hypothesis)) / len(word_tokens(utterance.reference))


def _below(value: float | None, threshold: float) -> bool:
    return value is not None and not math.isnan(value) and value < threshold


def _mark_duplicates(
    rows: list[dict[str, Any]], key: str, name: str, *, skip_empty: bool = False
) -> None:
    """Flag every row whose ``key`` already appeared in an earlier row (the first one is kept)."""
    seen: Counter[str] = Counter()
    for row in rows:
        value = row[key]
        row[name] = bool(seen[value]) and not (skip_empty and not value)
        seen[value] += 1


def _model_features(
    run_dir: Path, entries: Sequence[ManifestEntry], normalizer: Normalizer
) -> dict[str, dict[str, Any]]:
    run = load_run(run_dir)
    if [e.utterance_id for e in run.entries] != [e.utterance_id for e in entries]:
        raise ValueError(f"{run_dir} was not transcribed from the same manifest")
    features = {}
    for utterance in analyze(run.entries, run.hypotheses, normalizer):
        reference = len(word_tokens(utterance.reference))
        features[utterance.utterance_id] = {
            "wer": error_rate(utterance.counts) if reference else None,
            "length_ratio": len(word_tokens(utterance.hypothesis)) / reference
            if reference
            else None,
            "hypothesis_text": utterance.hypothesis,
        }
    return features
