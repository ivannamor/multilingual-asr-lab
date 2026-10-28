"""Invented closing phrases: a typical Whisper hallucination.

Whisper was trained on subtitled videos, whose last subtitle is often a credit or
a thank-you ("Sottotitoli a cura di QTSS", "Untertitelung des ZDF", "Thank you for
watching"). At the end of an utterance, especially after a silence, it sometimes
writes such a phrase although nobody said it. Each case costs only a few
insertions, so WER barely shows it, but in a product an invented "Vielen Dank"
at the end of a dictated message is a visible error.

Detection matches a short list of known phrases at the end of the *normalized*
hypothesis, when the normalized reference does not end the same way. Other
invented endings are missed, so the counts are a lower bound.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

# Known closing phrases per language, matched at the end of normalized text.
CLOSING_PHRASES = {
    "it": re.compile(r"\b(grazie( a tutti)?|sottotitoli.* qtss)$"),
    "de": re.compile(r"\b(vielen dank( für ihre aufmerksamkeit)?|untertitel.*)$"),
    "en": re.compile(r"\b(thank you( for watching)?|subtitles.*)$"),
}


def invented_closing_phrase(reference: str, hypothesis: str, lang: str) -> str | None:
    """The closing phrase that ends the hypothesis but not the reference, if any.

    Both texts must be normalized the same way. Languages without a list of
    phrases never match.

    >>> invented_closing_phrase("das ist alles", "das ist alles vielen dank", "de")
    'vielen dank'
    >>> invented_closing_phrase("grazie a tutti", "grazie a tutti", "it") is None
    True
    """
    pattern = CLOSING_PHRASES.get(lang)
    if pattern is None or pattern.search(reference):
        return None
    match = pattern.search(hypothesis)
    return match[0] if match else None


def find_invented_closing_phrases(
    utterance_ids: Sequence[str],
    references: Sequence[str],
    hypotheses: Sequence[str],
    langs: Sequence[str],
) -> list[dict[str, str]]:
    """Every utterance that ends with an invented closing phrase, in input order."""
    found = []
    for utterance_id, reference, hypothesis, lang in zip(
        utterance_ids, references, hypotheses, langs, strict=True
    ):
        phrase = invented_closing_phrase(reference, hypothesis, lang)
        if phrase is not None:
            found.append({"utterance_id": utterance_id, "phrase": phrase})
    return found
