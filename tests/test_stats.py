"""Tests for bootstrap confidence intervals and paired comparisons."""

import math

import pytest

from asr_lab.evaluation.metrics import ErrorCounts, error_rate
from asr_lab.evaluation.stats import bootstrap_ci, paired_bootstrap

PERFECT = ErrorCounts(hits=10)
ONE_ERROR = ErrorCounts(hits=9, substitutions=1)


def _corpus(n_perfect: int, n_with_error: int) -> list[ErrorCounts]:
    return [PERFECT] * n_perfect + [ONE_ERROR] * n_with_error


# --- bootstrap_ci ---------------------------------------------------------------------


def test_estimate_is_the_corpus_error_rate() -> None:
    counts = _corpus(70, 30)
    ci = bootstrap_ci(counts)
    assert ci.estimate == pytest.approx(error_rate(sum(counts, ErrorCounts())))
    assert ci.estimate == pytest.approx(0.03)


def test_interval_contains_the_estimate() -> None:
    ci = bootstrap_ci(_corpus(70, 30))
    assert ci.low < ci.estimate < ci.high


def test_no_variability_gives_a_zero_width_interval() -> None:
    ci = bootstrap_ci([PERFECT] * 20)
    assert ci.low == ci.estimate == ci.high == 0.0


def test_same_seed_gives_same_interval() -> None:
    counts = _corpus(70, 30)
    assert bootstrap_ci(counts, seed=1) == bootstrap_ci(counts, seed=1)


def test_more_data_gives_a_narrower_interval() -> None:
    small = bootstrap_ci(_corpus(70, 30))
    large = bootstrap_ci(_corpus(700, 300))
    assert (large.high - large.low) < (small.high - small.low)


def test_higher_level_gives_a_wider_interval() -> None:
    counts = _corpus(70, 30)
    ci90 = bootstrap_ci(counts, level=0.90)
    ci99 = bootstrap_ci(counts, level=0.99)
    assert (ci99.high - ci99.low) > (ci90.high - ci90.low)


def test_speaker_level_interval_is_wider_when_speakers_differ() -> None:
    # 10 speakers with 20 utterances each: half the speakers make an error in
    # every utterance, the other half never do. Utterances of one speaker are
    # strongly correlated, so resampling utterances underestimates the uncertainty.
    counts, speakers = [], []
    for speaker in range(10):
        utterance = ONE_ERROR if speaker % 2 == 0 else PERFECT
        counts += [utterance] * 20
        speakers += [f"spk{speaker}"] * 20
    by_utterance = bootstrap_ci(counts)
    by_speaker = bootstrap_ci(counts, groups=speakers)
    assert by_speaker.estimate == by_utterance.estimate
    assert by_speaker.n_units == 10
    assert by_utterance.n_units == 200
    assert (by_speaker.high - by_speaker.low) > 2 * (by_utterance.high - by_utterance.low)


def test_empty_references_follow_the_error_rate_convention() -> None:
    hallucination = ErrorCounts(insertions=2)
    assert bootstrap_ci([hallucination] * 5).estimate == math.inf
    mixed = bootstrap_ci([PERFECT] * 9 + [hallucination])
    assert math.isfinite(mixed.estimate)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_resamples": 0}, "n_resamples must be at least 1"),
        ({"level": 1.0}, "level must be between 0 and 1"),
        ({"groups": ["a"]}, "Got 1 group ids for 2 utterances"),
    ],
)
def test_invalid_options_raise(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        bootstrap_ci([PERFECT, ONE_ERROR], **kwargs)


def test_empty_counts_raise() -> None:
    with pytest.raises(ValueError, match="empty list"):
        bootstrap_ci([])


# --- paired_bootstrap -----------------------------------------------------------------


def test_identical_systems_have_no_difference() -> None:
    counts = _corpus(70, 30)
    comparison = paired_bootstrap(counts, counts)
    assert comparison.delta == comparison.low == comparison.high == 0.0
    assert comparison.prob_b_better == 0.0
    assert not comparison.significant


def test_consistent_improvement_is_significant() -> None:
    system_a = _corpus(50, 50)
    system_b = _corpus(90, 10)  # B fixes the error in 40 of the 50 utterances
    comparison = paired_bootstrap(system_a, system_b)
    assert comparison.delta == pytest.approx(-0.04)
    assert comparison.high < 0
    assert comparison.significant
    assert comparison.prob_b_better == 1.0


def test_improvement_on_a_single_utterance_is_not_significant() -> None:
    # B is never worse, but it fixes only one utterance out of 200: whether B
    # wins depends on that utterance being drawn (about 63% of the replicates).
    system_a = _corpus(100, 100)
    system_b = _corpus(101, 99)
    comparison = paired_bootstrap(system_a, system_b)
    assert comparison.delta < 0
    assert comparison.high == 0.0
    assert not comparison.significant
    assert comparison.prob_b_better == pytest.approx(1 - math.exp(-1), abs=0.05)


def test_paired_bootstrap_rejects_different_lengths() -> None:
    with pytest.raises(ValueError, match="same utterances"):
        paired_bootstrap([PERFECT], [PERFECT, PERFECT])
