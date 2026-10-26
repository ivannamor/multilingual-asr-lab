"""Tests for Levenshtein alignment, error counts and corpus-level error rates."""

import math
import random

import jiwer
import pytest

from asr_lab.evaluation.metrics import (
    AlignedPair,
    EditOp,
    ErrorCounts,
    align,
    cer,
    char_tokens,
    count_errors,
    error_rate,
    utterance_counts,
    wer,
)


def _align_words(ref: str, hyp: str) -> list[AlignedPair]:
    return align(ref.split(), hyp.split())


def _counts(ref: str, hyp: str) -> ErrorCounts:
    return count_errors(_align_words(ref, hyp))


def test_identical_sentences_have_no_errors() -> None:
    counts = _counts("il gatto dorme", "il gatto dorme")
    assert counts == ErrorCounts(hits=3)
    assert counts.errors == 0


def test_deletion_and_insertion() -> None:
    alignment = _align_words("il gatto è sul tavolo", "il gatto sul tavolo nero")
    assert alignment == [
        AlignedPair(EditOp.MATCH, "il", "il"),
        AlignedPair(EditOp.MATCH, "gatto", "gatto"),
        AlignedPair(EditOp.DELETION, "è", None),
        AlignedPair(EditOp.MATCH, "sul", "sul"),
        AlignedPair(EditOp.MATCH, "tavolo", "tavolo"),
        AlignedPair(EditOp.INSERTION, None, "nero"),
    ]


def test_substitution() -> None:
    counts = _counts("il gatto dorme", "il cane dorme")
    assert counts == ErrorCounts(hits=2, substitutions=1)


def test_empty_hypothesis_is_all_deletions() -> None:
    counts = _counts("buongiorno a tutti", "")
    assert counts == ErrorCounts(deletions=3)


def test_empty_reference_is_all_insertions() -> None:
    counts = _counts("", "grazie per la visione")
    assert counts == ErrorCounts(insertions=4)
    assert counts.ref_length == 0


def test_both_empty() -> None:
    assert align([], []) == []
    assert count_errors([]) == ErrorCounts()


def test_errors_can_exceed_reference_length() -> None:
    counts = _counts("ciao", "ciao a tutti voi")
    assert counts == ErrorCounts(hits=1, insertions=3)
    assert counts.errors > counts.ref_length


def test_character_level_kitten_sitting() -> None:
    # Classic example: Levenshtein distance between "kitten" and "sitting" is 3.
    counts = count_errors(align(list("kitten"), list("sitting")))
    assert counts == ErrorCounts(hits=4, substitutions=2, insertions=1)


@pytest.mark.parametrize(
    ("ref", "hyp"),
    [
        ("a b c d", "a x c d e"),
        ("the cat sat on the mat", "cat sat on mat"),
        ("", "a b"),
        ("a b", ""),
        ("uno due tre", "tre due uno"),
    ],
)
def test_alignment_preserves_both_sequences(ref: str, hyp: str) -> None:
    alignment = _align_words(ref, hyp)
    assert [p.ref for p in alignment if p.ref is not None] == ref.split()
    assert [p.hyp for p in alignment if p.hyp is not None] == hyp.split()


def test_error_counts_add_up() -> None:
    total = ErrorCounts(hits=2, substitutions=1) + ErrorCounts(deletions=1, insertions=2)
    assert total == ErrorCounts(hits=2, substitutions=1, deletions=1, insertions=2)
    assert total.errors == 4
    assert total.ref_length == 4


# --- Corpus-level error rates ---------------------------------------------------------


def test_wer_single_utterance() -> None:
    assert wer(["il gatto dorme"], ["il cane"]) == pytest.approx(2 / 3)


def test_wer_is_corpus_level_not_mean_of_utterances() -> None:
    references = ["ciao", "oggi piove molto forte"]
    hypotheses = ["addio", "oggi piove molto forte"]
    # Per-utterance WERs are 1.0 and 0.0: their mean would be 0.5.
    assert wer(references, hypotheses) == pytest.approx(1 / 5)


def test_wer_can_exceed_one() -> None:
    assert wer(["ciao"], ["ciao a tutti voi"]) == pytest.approx(3.0)


def test_perfect_transcription_has_zero_wer() -> None:
    assert wer(["uno due", "tre"], ["uno due", "tre"]) == 0.0


def test_whitespace_differences_are_not_errors() -> None:
    assert wer(["il  gatto "], [" il gatto"]) == 0.0
    assert cer(["il  gatto "], [" il gatto"]) == 0.0


def test_empty_reference_insertions_count_at_corpus_level() -> None:
    # A hallucination on a silent utterance adds errors but no reference words.
    assert wer(["", "il gatto"], ["grazie", "il gatto"]) == pytest.approx(1 / 2)


def test_error_rate_with_empty_reference() -> None:
    assert error_rate(ErrorCounts()) == 0.0
    assert error_rate(ErrorCounts(insertions=2)) == math.inf


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="2 references but 1 hypotheses"):
        wer(["a", "b"], ["a"])


def test_utterance_counts_keep_one_entry_per_utterance() -> None:
    counts = utterance_counts(["a b", "c"], ["a", "c d"])
    assert counts == [ErrorCounts(hits=1, deletions=1), ErrorCounts(hits=1, insertions=1)]


def test_char_tokens_collapse_whitespace() -> None:
    assert char_tokens("  ab   c ") == ["a", "b", " ", "c"]


def test_cer_counts_spaces_between_words() -> None:
    assert cer(["ab cd"], ["abcd"]) == pytest.approx(1 / 5)


# --- Cross-check against jiwer --------------------------------------------------------

_VOCAB = ["il", "gatto", "cane", "dorme", "sul", "tavolo", "anno", "hanno", "che", "perché"]


def _random_corpus(seed: int, size: int = 200) -> tuple[list[str], list[str]]:
    """Build non-empty references and noisy hypotheses (deletions, substitutions, insertions)."""
    rng = random.Random(seed)
    references, hypotheses = [], []
    for _ in range(size):
        ref = rng.choices(_VOCAB, k=rng.randint(1, 12))
        hyp = [w for w in ref if rng.random() > 0.1]
        hyp = [rng.choice(_VOCAB) if rng.random() < 0.15 else w for w in hyp]
        for _ in range(rng.randint(0, 2)):
            hyp.insert(rng.randint(0, len(hyp)), rng.choice(_VOCAB))
        references.append(" ".join(ref))
        hypotheses.append(" ".join(hyp))
    return references, hypotheses


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_wer_and_cer_match_jiwer(seed: int) -> None:
    references, hypotheses = _random_corpus(seed)
    assert wer(references, hypotheses) == pytest.approx(jiwer.wer(references, hypotheses))
    assert cer(references, hypotheses) == pytest.approx(jiwer.cer(references, hypotheses))
