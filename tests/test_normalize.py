"""Tests for text normalization."""

import unicodedata

import pytest

from asr_lab.evaluation.metrics import wer
from asr_lab.evaluation.normalize import (
    BASIC,
    ENGLISH,
    GERMAN,
    IDENTITY,
    ITALIAN,
    Normalizer,
    get_normalizer,
)

RIGHT_QUOTE = "\u2019"  # typographic apostrophe, produced by word processors and many models


def test_identity_changes_nothing() -> None:
    assert IDENTITY("  Ciao, Mondo!  ") == "  Ciao, Mondo!  "


def test_basic_lowercases_and_removes_punctuation() -> None:
    assert BASIC("Ciao,   come stai?!") == "ciao come stai"


def test_composed_and_decomposed_accents_are_equal() -> None:
    composed = unicodedata.normalize("NFC", "perché")
    decomposed = unicodedata.normalize("NFD", "perché")
    assert composed != decomposed  # same on screen, different strings
    assert ITALIAN(composed) == ITALIAN(decomposed) == "perché"


def test_accents_are_kept() -> None:
    # "è" (is) and "e" (and) are different words in Italian.
    assert ITALIAN("È vero e falso") == "è vero e falso"


def test_typographic_apostrophe_is_unified() -> None:
    assert ENGLISH(f"don{RIGHT_QUOTE}t") == ENGLISH("don't") == "don't"


def test_italian_elisions_are_split() -> None:
    assert ITALIAN(f"L{RIGHT_QUOTE}uomo dell'anno") == "l uomo dell anno"


def test_italian_truncation_apostrophe_is_removed() -> None:
    assert ITALIAN("Un po' di più") == "un po di più"


def test_english_contractions_are_kept() -> None:
    assert ENGLISH("Don't stop, it's 'fine'.") == "don't stop it's fine"


def test_german_eszett_becomes_ss_and_umlauts_are_kept() -> None:
    assert GERMAN("Grüße aus der Straße") == "grüsse aus der strasse"


def test_hyphenated_words_are_split() -> None:
    assert ITALIAN("vento da nord-est") == "vento da nord est"


@pytest.mark.parametrize("normalizer", [BASIC, ENGLISH, ITALIAN, GERMAN])
@pytest.mark.parametrize("text", ["", "   ", "?!...", "«»"])
def test_empty_or_punctuation_only_gives_empty_string(normalizer: Normalizer, text: str) -> None:
    assert normalizer(text) == ""


@pytest.mark.parametrize("normalizer", [BASIC, ENGLISH, ITALIAN, GERMAN])
@pytest.mark.parametrize(
    "text",
    [
        f"L{RIGHT_QUOTE}uomo, dell'anno: «perché?»",
        "Don't stop -- it's 'fine'.",
        "Grüße aus der Straße!",
        "  spazi   multipli\te\na capo ",
    ],
)
def test_normalizers_are_idempotent(normalizer: Normalizer, text: str) -> None:
    once = normalizer(text)
    assert normalizer(once) == once


def test_get_normalizer_by_language() -> None:
    assert get_normalizer("it") is ITALIAN
    assert get_normalizer("de") is GERMAN
    assert get_normalizer("en") is ENGLISH


def test_get_normalizer_unknown_language() -> None:
    with pytest.raises(ValueError, match="No normalizer for language 'fr'"):
        get_normalizer("fr")


def test_normalizer_names_are_unique() -> None:
    names = [n.name for n in (IDENTITY, BASIC, ENGLISH, ITALIAN, GERMAN)]
    assert len(names) == len(set(names))


def test_normalization_changes_wer() -> None:
    reference = "L'uomo è arrivato."
    hypothesis = f"l{RIGHT_QUOTE}uomo è arrivato"
    assert wer([reference], [hypothesis]) == pytest.approx(2 / 3)
    assert wer([ITALIAN(reference)], [ITALIAN(hypothesis)]) == 0.0
