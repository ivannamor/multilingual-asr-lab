"""Tests for Levenshtein alignment and error counts."""

import pytest

from asr_lab.evaluation.metrics import (
    AlignedPair,
    EditOp,
    ErrorCounts,
    align,
    count_errors,
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
