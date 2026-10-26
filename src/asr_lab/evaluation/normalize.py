"""Text normalization applied identically to references and hypotheses before scoring.

Every normalizer has a versioned name (e.g. ``"it-v1"``) that must be reported
together with every error rate: changing the normalizer can move WER by several
points. Numbers are not normalized yet, so "venti" and "20" still count as an error.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

# Characters used in place of the apostrophe: right and left single quotation
# marks, modifier letter apostrophe, acute accent, grave accent.
_APOSTROPHE_VARIANTS = "\u2019\u2018\u02bc\u00b4\u0060"
_APOSTROPHE_TABLE = str.maketrans(dict.fromkeys(_APOSTROPHE_VARIANTS, "'"))
# An apostrophe that is not between two word characters: quotes, or "po'".
_EDGE_APOSTROPHE = re.compile(r"(?<!\w)'|'(?!\w)")
# Text in square or angle brackets ("[Music]", "<unk>") or in parentheses ("(laughs)").
_BRACKETED = re.compile(r"[<\[][^>\]]*[>\]]|\(([^)]+?)\)")


def unicode_nfc(text: str) -> str:
    """Give each accented character a single representation (Unicode NFC).

    "é" can be stored as one code point or as "e" plus a combining accent; the
    two look identical but would otherwise be counted as different words.
    """
    return unicodedata.normalize("NFC", text)


def unify_apostrophes(text: str) -> str:
    """Replace typographic apostrophes and quotes with the ASCII apostrophe."""
    return text.translate(_APOSTROPHE_TABLE)


def lowercase(text: str) -> str:
    """Lowercase the text."""
    return text.lower()


def remove_punctuation(text: str) -> str:
    """Replace punctuation with spaces, keeping apostrophes inside words.

    "don't" and "l'uomo" keep their apostrophe; quotes and trailing apostrophes
    ("po'") are removed. Hyphens are punctuation, so "nord-est" becomes two words.
    """
    text = _EDGE_APOSTROPHE.sub(" ", text)
    return "".join(
        " " if unicodedata.category(char).startswith("P") and char != "'" else char for char in text
    )


def split_elisions(text: str) -> str:
    """Split Italian elisions into two words: "l'uomo" -> "l uomo"."""
    return text.replace("'", " ")


def replace_eszett(text: str) -> str:
    """Spell German "ß" as "ss", as in Swiss German and in many ASR outputs."""
    return text.replace("ß", "ss")


def collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces and trim both ends."""
    return " ".join(text.split())


def remove_bracketed(text: str) -> str:
    """Remove annotations in brackets or parentheses, e.g. "[Music]" or "(laughs)"."""
    return _BRACKETED.sub("", text)


def remove_marks_symbols_punctuation(text: str) -> str:
    """Apply Unicode NFKC, then replace every mark, symbol and punctuation character with a space.

    Unlike ``remove_punctuation`` this also removes apostrophes inside words and
    symbols such as "$" or "%", as OpenAI Whisper's basic normalizer does.
    """
    return "".join(
        " " if unicodedata.category(char)[0] in "MSP" else char
        for char in unicodedata.normalize("NFKC", text)
    )


@dataclass(frozen=True)
class Normalizer:
    """A named, ordered pipeline of text transformations.

    Example:
        >>> ITALIAN("L'uomo, dell'anno!")
        'l uomo dell anno'
        >>> ITALIAN.name
        'it-v1'
    """

    name: str
    steps: tuple[Callable[[str], str], ...]

    def __call__(self, text: str) -> str:
        for step in self.steps:
            text = step(text)
        return text


_COMMON_STEPS = (unicode_nfc, unify_apostrophes, lowercase, remove_punctuation)

IDENTITY = Normalizer("none", ())
BASIC = Normalizer("basic-v1", (*_COMMON_STEPS, collapse_whitespace))
ENGLISH = Normalizer("en-v1", (*_COMMON_STEPS, collapse_whitespace))
ITALIAN = Normalizer("it-v1", (*_COMMON_STEPS, split_elisions, collapse_whitespace))
GERMAN = Normalizer("de-v1", (*_COMMON_STEPS, replace_eszett, collapse_whitespace))
# Reimplementation of ``BasicTextNormalizer`` from OpenAI Whisper (the default for
# non-English text in many papers and leaderboards), written here to avoid
# depending on the openai-whisper package and PyTorch. Tokens are identical to the
# original; leading and trailing spaces are also trimmed.
WHISPER_BASIC = Normalizer(
    "whisper-basic",
    (lowercase, remove_bracketed, remove_marks_symbols_punctuation, collapse_whitespace),
)

_BY_LANGUAGE = {"en": ENGLISH, "it": ITALIAN, "de": GERMAN}
_BY_NAME = {
    normalizer.name: normalizer
    for normalizer in (IDENTITY, BASIC, ENGLISH, ITALIAN, GERMAN, WHISPER_BASIC)
}
NORMALIZER_NAMES = tuple(_BY_NAME)


def get_normalizer(lang: str) -> Normalizer:
    """Return the normalizer for an ISO 639-1 language code ("en", "it", "de").

    Raises:
        ValueError: if there is no normalizer for the language.
    """
    try:
        return _BY_LANGUAGE[lang]
    except KeyError:
        supported = ", ".join(sorted(_BY_LANGUAGE))
        raise ValueError(f"No normalizer for language {lang!r}; supported: {supported}") from None


def get_normalizer_by_name(name: str) -> Normalizer:
    """Return a normalizer by its versioned name, e.g. ``"it-v1"`` or ``"whisper-basic"``.

    Raises:
        ValueError: if no normalizer has that name.
    """
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(
            f"No normalizer named {name!r}; available: {', '.join(NORMALIZER_NAMES)}"
        ) from None
