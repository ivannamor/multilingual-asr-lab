"""Bootstrap confidence intervals and paired system comparisons for error rates.

Resampling works on per-utterance ``ErrorCounts`` (see ``metrics.utterance_counts``),
so the alignment is never recomputed: each bootstrap replicate only sums counts.

Utterances from the same speaker are correlated, so resampling them one by one
gives intervals that are too narrow. Pass ``groups`` (e.g. speaker ids) to
resample whole groups instead (Bisani & Ney, 2004).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from asr_lab.evaluation.metrics import ErrorCounts


@dataclass(frozen=True)
class ConfidenceInterval:
    """Bootstrap percentile interval for a corpus-level error rate."""

    estimate: float  # error rate on the original data
    low: float
    high: float
    level: float  # e.g. 0.95
    n_resamples: int
    n_units: int  # number of resampled units (utterances or groups)


@dataclass(frozen=True)
class PairedComparison:
    """Paired bootstrap comparison of system B against system A.

    ``delta`` is rate(B) - rate(A): negative values mean B makes fewer errors.
    """

    delta: float  # difference on the original data
    low: float
    high: float
    prob_b_better: float  # fraction of resamples where B has the lower error rate
    level: float
    n_resamples: int
    n_units: int
    prob_b_worse: float  # fraction of resamples where B has the higher error rate

    @property
    def significant(self) -> bool:
        """True if the interval for the difference does not contain zero."""
        return self.low > 0 or self.high < 0

    @property
    def p_value(self) -> float:
        """Two-sided bootstrap p-value of "no difference".

        Twice the share of resamples on the less frequent side of zero, ties (a difference of
        exactly zero) counting on both sides. With ``n_resamples`` resamples it cannot go below
        1 / ``n_resamples`` except at 0, which means "below 1 / n_resamples".
        """
        at_most_zero, at_least_zero = 1 - self.prob_b_worse, 1 - self.prob_b_better
        return min(1.0, 2 * min(at_most_zero, at_least_zero))


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm's step-down adjustment of p-values for a family of tests.

    An adjusted p-value below the level means the test is significant with the family-wise
    error rate controlled at that level. Holm is uniformly more powerful than Bonferroni and
    makes no assumption on how the tests depend on each other.

    Example:
        >>> holm([0.01, 0.04, 0.03, 0.2])
        [0.04, 0.09, 0.09, 0.2]
    """
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def bootstrap_ci(
    counts: Sequence[ErrorCounts],
    *,
    groups: Sequence[str] | None = None,
    n_resamples: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    """Confidence interval for the corpus error rate by bootstrap resampling.

    Example:
        >>> counts = [ErrorCounts(hits=9, substitutions=1)] * 50 + [ErrorCounts(hits=10)] * 50
        >>> ci = bootstrap_ci(counts)
        >>> ci.estimate
        0.05
        >>> ci.low < ci.estimate < ci.high
        True
    """
    _check_options(n_resamples, level)
    errors, refs = _unit_totals(counts, groups)
    indices = _resample_indices(len(errors), n_resamples, seed)
    rates = _rates(errors[indices].sum(axis=1), refs[indices].sum(axis=1))
    low, high = _percentiles(rates, level)
    return ConfidenceInterval(
        estimate=float(_rates(errors.sum(), refs.sum())),
        low=low,
        high=high,
        level=level,
        n_resamples=n_resamples,
        n_units=len(errors),
    )


def paired_bootstrap(
    counts_a: Sequence[ErrorCounts],
    counts_b: Sequence[ErrorCounts],
    *,
    groups: Sequence[str] | None = None,
    n_resamples: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> PairedComparison:
    """Compare two systems scored on the same utterances.

    Every replicate draws the same utterances (or groups) for both systems, so
    the comparison is not affected by which utterances happened to be drawn.
    """
    if len(counts_a) != len(counts_b):
        raise ValueError(
            f"Systems must be scored on the same utterances: {len(counts_a)} vs {len(counts_b)}"
        )
    _check_options(n_resamples, level)
    errors_a, refs_a = _unit_totals(counts_a, groups)
    errors_b, refs_b = _unit_totals(counts_b, groups)
    indices = _resample_indices(len(errors_a), n_resamples, seed)
    deltas = _rates(errors_b[indices].sum(axis=1), refs_b[indices].sum(axis=1)) - _rates(
        errors_a[indices].sum(axis=1), refs_a[indices].sum(axis=1)
    )
    low, high = _percentiles(deltas, level)
    delta = _rates(errors_b.sum(), refs_b.sum()) - _rates(errors_a.sum(), refs_a.sum())
    return PairedComparison(
        delta=float(delta),
        low=low,
        high=high,
        prob_b_better=float(np.mean(deltas < 0)),
        prob_b_worse=float(np.mean(deltas > 0)),
        level=level,
        n_resamples=n_resamples,
        n_units=len(errors_a),
    )


def _check_options(n_resamples: int, level: float) -> None:
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be at least 1, got {n_resamples}")
    if not 0 < level < 1:
        raise ValueError(f"level must be between 0 and 1, got {level}")


def _unit_totals(
    counts: Sequence[ErrorCounts], groups: Sequence[str] | None
) -> tuple[np.ndarray, np.ndarray]:
    """Return errors and reference lengths per resampling unit.

    Without groups each utterance is a unit; with groups the counts of all
    utterances sharing a group id are summed into one unit.
    """
    if not counts:
        raise ValueError("Cannot bootstrap an empty list of counts")
    errors = np.array([c.errors for c in counts], dtype=float)
    refs = np.array([c.ref_length for c in counts], dtype=float)
    if groups is None:
        return errors, refs
    if len(groups) != len(counts):
        raise ValueError(f"Got {len(groups)} group ids for {len(counts)} utterances")
    _, unit_of_utterance = np.unique(np.asarray(groups), return_inverse=True)
    return (
        np.bincount(unit_of_utterance, weights=errors),
        np.bincount(unit_of_utterance, weights=refs),
    )


def _resample_indices(n_units: int, n_resamples: int, seed: int) -> np.ndarray:
    """Draw ``n_resamples`` samples of ``n_units`` indices with replacement."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_units, size=(n_resamples, n_units))


def _rates(errors: np.ndarray, refs: np.ndarray) -> np.ndarray:
    """Element-wise errors / refs with the convention of ``metrics.error_rate``.

    An empty reference gives 0.0 without errors and inf with errors.
    """
    errors, refs = np.asarray(errors, dtype=float), np.asarray(refs, dtype=float)
    empty_rate = np.where(errors > 0, np.inf, 0.0)
    return np.where(refs > 0, errors / np.where(refs > 0, refs, 1.0), empty_rate)


def _percentiles(values: np.ndarray, level: float) -> tuple[float, float]:
    """Lower and upper bounds of the central ``level`` fraction of the values."""
    tail = (1 - level) / 2
    # Interpolating between two infinite rates computes inf - inf = nan; the
    # bound is infinite in that case.
    with np.errstate(invalid="ignore"):
        bounds = np.quantile(values, [tail, 1 - tail])
    low, high = np.nan_to_num(bounds, nan=np.inf)
    return float(low), float(high)
