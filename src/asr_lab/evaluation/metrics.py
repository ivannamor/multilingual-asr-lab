"""Levenshtein alignment and edit-operation counts for ASR evaluation.

The same functions work at word level (sequences of words) and at character
level (sequences of characters), which is how WER and CER are computed.
"""

from __future__ import annotations

from collections.abc import Sequence
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
