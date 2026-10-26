"""Where a model goes wrong: most confused word pairs, deleted and inserted words, worst utterances.

Everything is computed on normalized text (the same normalizer as the scores),
from the word alignments of ``metrics.align``. Counts depend on how alignment
ties are broken (see ``metrics.align``); totals match the WER computation.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from asr_lab.data.manifest import ManifestEntry
from asr_lab.evaluation.metrics import (
    AlignedPair,
    EditOp,
    ErrorCounts,
    align,
    count_errors,
    error_rate,
)
from asr_lab.evaluation.normalize import Normalizer
from asr_lab.evaluation.scoring import load_run, normalized_pairs
from asr_lab.inference.base import Hypothesis


@dataclass(frozen=True)
class UtteranceErrors:
    """The word alignment of one utterance after normalization."""

    utterance_id: str
    reference: str
    hypothesis: str
    alignment: list[AlignedPair]

    @property
    def counts(self) -> ErrorCounts:
        return count_errors(self.alignment)

    @property
    def wer(self) -> float:
        return error_rate(self.counts)


def analyze(
    entries: Sequence[ManifestEntry],
    hypotheses: Sequence[Hypothesis],
    normalizer: Normalizer | None = None,
) -> list[UtteranceErrors]:
    """Align every utterance at word level after normalization, in manifest order."""
    pairs = normalized_pairs(entries, hypotheses, normalizer)
    return [
        UtteranceErrors(entry.utterance_id, ref, hyp, align(ref.split(), hyp.split()))
        for entry, ref, hyp in zip(entries, pairs.references, pairs.hypotheses, strict=True)
    ]


def confusion_pairs(utterances: Sequence[UtteranceErrors]) -> Counter[tuple[str, str]]:
    """How often each (reference word, hypothesis word) substitution occurs."""
    return Counter(
        (pair.ref, pair.hyp)
        for utterance in utterances
        for pair in utterance.alignment
        if pair.op is EditOp.SUBSTITUTION
    )


def deleted_words(utterances: Sequence[UtteranceErrors]) -> Counter[str]:
    """Reference words the model left out."""
    return Counter(
        pair.ref
        for utterance in utterances
        for pair in utterance.alignment
        if pair.op is EditOp.DELETION
    )


def inserted_words(utterances: Sequence[UtteranceErrors]) -> Counter[str]:
    """Words the model added that are not in the reference."""
    return Counter(
        pair.hyp
        for utterance in utterances
        for pair in utterance.alignment
        if pair.op is EditOp.INSERTION
    )


def worst_utterances(utterances: Sequence[UtteranceErrors], n: int) -> list[UtteranceErrors]:
    """The ``n`` utterances with the most errors (ties broken by WER, then id)."""
    return sorted(utterances, key=lambda u: (-u.counts.errors, -u.wer, u.utterance_id))[:n]


def involves_digits_or_symbols(pair: tuple[str, str]) -> bool:
    """True if either word contains a character that is not a letter or an apostrophe.

    Such substitutions ("20" vs "venti", "$" vs "dollari") are often
    normalization artefacts rather than recognition errors.
    """
    return any(not (char.isalpha() or char == "'") for word in pair for char in word)


def format_alignment(alignment: Sequence[AlignedPair]) -> str:
    """Three aligned lines (REF, HYP, operations) in the style of NIST sclite.

    Example:
        >>> from asr_lab.evaluation.metrics import align
        >>> print(format_alignment(align("il gatto dorme".split(), "il cane".split())))
        REF: il gatto dorme
        HYP: il ***** cane
                D     S
    """
    marks = {EditOp.SUBSTITUTION: "S", EditOp.DELETION: "D", EditOp.INSERTION: "I"}
    ref_cells, hyp_cells, op_cells = [], [], []
    for pair in alignment:
        width = max(len(pair.ref or ""), len(pair.hyp or ""))
        ref_cells.append((pair.ref or "*" * width).ljust(width))
        hyp_cells.append((pair.hyp or "*" * width).ljust(width))
        op_cells.append(marks.get(pair.op, "").ljust(width))
    lines = (
        "REF: " + " ".join(ref_cells),
        "HYP: " + " ".join(hyp_cells),
        "     " + " ".join(op_cells),
    )
    return "\n".join(line.rstrip() for line in lines)


def error_report(
    utterances: Sequence[UtteranceErrors], *, top: int = 20, worst: int = 10
) -> dict[str, Any]:
    """JSON-serializable summary of the errors, for saving next to the scores."""
    confusions = confusion_pairs(utterances)
    total = sum((u.counts for u in utterances), ErrorCounts())
    return {
        "n_utterances": len(utterances),
        "substitutions": total.substitutions,
        "deletions": total.deletions,
        "insertions": total.insertions,
        "distinct_confusion_pairs": len(confusions),
        "substitutions_with_digits_or_symbols": sum(
            count for pair, count in confusions.items() if involves_digits_or_symbols(pair)
        ),
        "top_confusions": [
            {"reference": ref, "hypothesis": hyp, "count": count}
            for (ref, hyp), count in confusions.most_common(top)
        ],
        "top_deletions": [
            {"word": word, "count": count}
            for word, count in deleted_words(utterances).most_common(top)
        ],
        "top_insertions": [
            {"word": word, "count": count}
            for word, count in inserted_words(utterances).most_common(top)
        ],
        "worst_utterances": [
            {
                "utterance_id": u.utterance_id,
                "errors": u.counts.errors,
                "reference_words": u.counts.ref_length,
                "reference": u.reference,
                "hypothesis": u.hypothesis,
            }
            for u in worst_utterances(utterances, worst)
        ],
    }


def analyze_run(
    run_dir: Path,
    manifest_path: Path | None = None,
    *,
    normalizer: Normalizer | None = None,
    top: int = 20,
    worst: int = 10,
) -> tuple[list[UtteranceErrors], dict[str, Any], Path]:
    """Analyze a run directory and save ``errors_<normalizer>.json`` next to its predictions.

    Returns:
        The per-utterance alignments, the report and the path it was written to.
    """
    run = load_run(run_dir, manifest_path)
    utterances = analyze(run.entries, run.hypotheses, normalizer)
    names = normalized_pairs(run.entries, run.hypotheses, normalizer).normalizer_names
    report = {
        "model": run.info.get("model"),
        "manifest": run.manifest_path.as_posix(),
        "normalizer": names,
        **error_report(utterances, top=top, worst=worst),
    }
    path = run_dir / f"errors_{'+'.join(names)}.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return utterances, report, path


def format_errors(
    utterances: Sequence[UtteranceErrors], report: dict[str, Any], *, show: int = 5
) -> str:
    """Human-readable error report for the terminal."""
    lines = [
        f"{report.get('model') or 'unknown model'}: {report['n_utterances']} utterances, "
        f"normalizer {', '.join(report.get('normalizer', []))}",
        f"S={report['substitutions']} D={report['deletions']} I={report['insertions']}; "
        f"{report['distinct_confusion_pairs']} distinct substitution pairs, "
        f"{report['substitutions_with_digits_or_symbols']} substitutions involve digits or symbols",
        "",
        "Most frequent substitutions (reference -> hypothesis):",
        *(
            f"  {c['count']:>3}  {c['reference']} -> {c['hypothesis']}"
            for c in report["top_confusions"]
        ),
        "",
        "Most deleted: "
        + ", ".join(f"{d['word']} ({d['count']})" for d in report["top_deletions"]),
        "Most inserted: "
        + ", ".join(f"{i['word']} ({i['count']})" for i in report["top_insertions"]),
        "",
        f"Utterances with the most errors (first {show}):",
    ]
    for utterance in worst_utterances(utterances, show):
        counts = utterance.counts
        lines += [
            f"- {utterance.utterance_id}: {counts.errors} errors / {counts.ref_length} words",
            format_alignment(utterance.alignment),
        ]
    return "\n".join(lines)
