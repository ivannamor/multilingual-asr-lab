"""Levenshtein alignment and edit-operation counts for ASR evaluation.

The same functions work at word level (sequences of words) and at character
level (sequences of characters), which is how WER and CER are computed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum


class EditOp(StrEnum):
    """Type of a single aligned position."""

    MATCH = "match"
    SUBSTITUTION = "substitution"
    DELETION = "deletion"
    INSERTION = "insertion"


@dataclass(frozen=True)
class AlignedPair:
    """One position of an alignment.

    ``ref`` is ``None`` for insertions and ``hyp`` is ``None`` for deletions.
    """

    op: EditOp
    ref: str | None
    hyp: str | None


@dataclass(frozen=True)
class ErrorCounts:
    """Edit-operation counts for one utterance or a whole corpus."""

    hits: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    @property
    def errors(self) -> int:
        """Total number of errors (S + D + I)."""
        return self.substitutions + self.deletions + self.insertions

    @property
    def ref_length(self) -> int:
        """Number of reference tokens (H + S + D)."""
        return self.hits + self.substitutions + self.deletions

    def __add__(self, other: ErrorCounts) -> ErrorCounts:
        return ErrorCounts(
            hits=self.hits + other.hits,
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
        )


def align(reference: Sequence[str], hypothesis: Sequence[str]) -> list[AlignedPair]:
    """Align two token sequences with minimum Levenshtein distance.

    Every edit (substitution, deletion, insertion) costs 1. Among alignments
    with the same minimum cost, the backtrace prefers diagonal moves
    (match/substitution), then deletions, then insertions.

    Example:
        >>> pairs = align("il gatto dorme".split(), "il cane".split())
        >>> [(p.op.value, p.ref, p.hyp) for p in pairs]
        [('match', 'il', 'il'), ('deletion', 'gatto', None), ('substitution', 'dorme', 'cane')]
    """
    cost = _cost_matrix(reference, hypothesis)
    return _backtrace(reference, hypothesis, cost)


def count_errors(alignment: Sequence[AlignedPair]) -> ErrorCounts:
    """Count hits, substitutions, deletions and insertions in an alignment.

    Example:
        >>> counts = count_errors(align("il gatto dorme".split(), "il cane".split()))
        >>> counts
        ErrorCounts(hits=1, substitutions=1, deletions=1, insertions=0)
        >>> counts.errors, counts.ref_length
        (2, 3)
    """
    ops = [pair.op for pair in alignment]
    return ErrorCounts(
        hits=ops.count(EditOp.MATCH),
        substitutions=ops.count(EditOp.SUBSTITUTION),
        deletions=ops.count(EditOp.DELETION),
        insertions=ops.count(EditOp.INSERTION),
    )


def word_tokens(text: str) -> list[str]:
    """Split a transcript into words on whitespace."""
    return text.split()


def char_tokens(text: str) -> list[str]:
    """Split a transcript into characters, keeping one space between words.

    Runs of whitespace collapse to a single space and leading/trailing
    whitespace is dropped, so spacing differences are never counted as errors.
    Spaces between words are characters, as in jiwer.
    """
    return list(" ".join(text.split()))


def utterance_counts(
    references: Sequence[str],
    hypotheses: Sequence[str],
    tokenize: Callable[[str], list[str]] = word_tokens,
) -> list[ErrorCounts]:
    """Align every reference/hypothesis pair and return one ``ErrorCounts`` each.

    Keeping per-utterance counts lets later steps (corpus totals, bootstrap
    resampling, slicing) work without re-running the alignment.

    Raises:
        ValueError: if the number of references and hypotheses differs.
    """
    if len(references) != len(hypotheses):
        raise ValueError(f"Got {len(references)} references but {len(hypotheses)} hypotheses")
    return [
        count_errors(align(tokenize(ref), tokenize(hyp)))
        for ref, hyp in zip(references, hypotheses, strict=True)
    ]


def error_rate(counts: ErrorCounts) -> float:
    """Return (S + D + I) / N, where N is the number of reference tokens.

    With an empty reference (N = 0) the rate is 0.0 if there are no errors
    and ``math.inf`` otherwise: any output on an empty reference is infinitely
    wrong relative to its length. At corpus level this only happens if every
    reference is empty.
    """
    if counts.ref_length == 0:
        return 0.0 if counts.errors == 0 else math.inf
    return counts.errors / counts.ref_length


def wer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Corpus-level word error rate: total errors over total reference words.

    This is not the mean of per-utterance WERs, which would give short
    utterances the same weight as long ones.

    Example:
        >>> wer(["sì", "oggi piove molto forte"], ["no", "oggi piove molto forte"])
        0.2
    """
    return error_rate(sum(utterance_counts(references, hypotheses, word_tokens), ErrorCounts()))


def cer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Corpus-level character error rate (spaces between words count as characters).

    Example:
        >>> round(cer(["perché"], ["perche"]), 3)
        0.167
    """
    return error_rate(sum(utterance_counts(references, hypotheses, char_tokens), ErrorCounts()))


def _cost_matrix(reference: Sequence[str], hypothesis: Sequence[str]) -> list[list[int]]:
    """Return the dynamic-programming table of edit distances.

    ``cost[i][j]`` is the minimum number of edits that turn the first ``i``
    reference tokens into the first ``j`` hypothesis tokens.
    """
    n, m = len(reference), len(hypothesis)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i  # delete all i reference tokens
    for j in range(1, m + 1):
        cost[0][j] = j  # insert all j hypothesis tokens

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            mismatch = int(reference[i - 1] != hypothesis[j - 1])
            cost[i][j] = min(
                cost[i - 1][j - 1] + mismatch,  # match or substitution
                cost[i - 1][j] + 1,  # deletion
                cost[i][j - 1] + 1,  # insertion
            )
    return cost


def _backtrace(
    reference: Sequence[str], hypothesis: Sequence[str], cost: list[list[int]]
) -> list[AlignedPair]:
    """Walk the cost table from the bottom-right corner back to the origin."""
    pairs: list[AlignedPair] = []
    i, j = len(reference), len(hypothesis)
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            ref_tok, hyp_tok = reference[i - 1], hypothesis[j - 1]
            mismatch = int(ref_tok != hyp_tok)
            if cost[i][j] == cost[i - 1][j - 1] + mismatch:
                op = EditOp.SUBSTITUTION if mismatch else EditOp.MATCH
                pairs.append(AlignedPair(op, ref_tok, hyp_tok))
                i, j = i - 1, j - 1
                continue
        if i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            pairs.append(AlignedPair(EditOp.DELETION, reference[i - 1], None))
            i -= 1
        else:
            pairs.append(AlignedPair(EditOp.INSERTION, None, hypothesis[j - 1]))
            j -= 1
    pairs.reverse()
    return pairs
