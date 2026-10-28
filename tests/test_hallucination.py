"""Tests for the detection of invented closing phrases (on normalized text)."""

import pytest

from asr_lab.evaluation.hallucination import (
    find_invented_closing_phrases,
    invented_closing_phrase,
)


@pytest.mark.parametrize(
    ("reference", "hypothesis", "lang", "expected"),
    [
        ("das ist alles", "das ist alles vielen dank", "de", "vielen dank"),
        ("es regnet", "es regnet untertitelung des zdf für funk 2017", "de",
         "untertitelung des zdf für funk 2017"),
        ("piove", "piove sottotitoli e revisione a cura di qtss", "it",
         "sottotitoli e revisione a cura di qtss"),
        ("it rains", "it rains thank you for watching", "en", "thank you for watching"),
        ("grazie a tutti", "grazie a tutti", "it", None),  # said by the speaker
        ("piove grazie", "piove grazie", "it", None),
        ("grazie al sole", "grazie al sole", "it", None),  # not at the end
        ("lo ringrazie", "lo ringrazie", "it", None),
        ("il sole", "il sole ringrazie", "it", None),  # a whole word only
        ("the end", "the end vielen dank", "en", None),  # phrases of the language only
        ("la fine", "la fine grazie", "fr", None),  # no list for the language
        ("", "vielen dank", "de", "vielen dank"),
    ],
)  # fmt: skip
def test_invented_closing_phrase(
    reference: str, hypothesis: str, lang: str, expected: str | None
) -> None:
    assert invented_closing_phrase(reference, hypothesis, lang) == expected


def test_find_keeps_input_order_and_skips_clean_utterances() -> None:
    found = find_invented_closing_phrases(
        ["a", "b", "c"],
        ["uno", "zwei", "three"],
        ["uno grazie", "zwei", "three thank you"],
        ["it", "de", "en"],
    )
    assert found == [
        {"utterance_id": "a", "phrase": "grazie"},
        {"utterance_id": "c", "phrase": "thank you"},
    ]
