"""Text normalization applied identically to references and hypotheses before scoring.

Every normalizer has a versioned name (e.g. ``"it-v1"``) that must be reported
together with every error rate: changing the normalizer can move WER by several
points. ``it-v1`` does not normalize numbers ("venti" and "20" count as an error);
``it-v2`` writes numbers, symbols and unambiguous units out in words.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

from num2words import num2words

# Characters used in place of the apostrophe: right and left single quotation
# marks, modifier letter apostrophe, acute accent, grave accent.
_APOSTROPHE_VARIANTS = "\u2019\u2018\u02bc\u00b4\u0060"
_APOSTROPHE_TABLE = str.maketrans(dict.fromkeys(_APOSTROPHE_VARIANTS, "'"))
# An apostrophe that is not between two word characters: quotes, or "po'".
_EDGE_APOSTROPHE = re.compile(r"(?<!\w)'|'(?!\w)")
# Text in square or angle brackets ("[Music]", "<unk>") or in parentheses ("(laughs)").
_BRACKETED = re.compile(r"[<\[][^>\]]*[>\]]|\(([^)]+?)\)")

# Italian number formats: "2.000" (thousands), "3,5" (decimal), "10:30" (time).
_IT_TIME = re.compile(r"(?<![\w:])(\d{1,2}):(\d{2})(?![\w:])")
_IT_NUMBER = re.compile(r"(?<![\w.,])(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)(?![\w])")
_IT_THOUSANDS = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?")
_CURRENCY_BEFORE = re.compile(r"([$€£])\s*(\d[\d.,]*\d|\d)")
# Only units that cannot be confused with Italian words ("ha" is also "has", "m" is ambiguous).
_IT_UNITS = (
    (re.compile(r"(?<!\w)km/h(?!\w)"), "chilometri orari"),
    (re.compile(r"(?<!\w)km(?!\w)"), "chilometri"),
    (re.compile(r"(?<!\w)cm(?!\w)"), "centimetri"),
    (re.compile(r"(?<!\w)mm(?!\w)"), "millimetri"),
    (re.compile(r"(?<!\w)kg(?!\w)"), "chilogrammi"),
)
_IT_SYMBOLS = {"$": " dollari ", "€": " euro ", "£": " sterline ", "%": " per cento "}
# Tens lose their final vowel before a vowel: "vent'anni" = "venti anni".
_IT_ELIDED_TENS = {
    "vent": "venti",
    "trent": "trenta",
    "quarant": "quaranta",
    "cinquant": "cinquanta",
    "sessant": "sessanta",
    "settant": "settanta",
    "ottant": "ottanta",
    "novant": "novanta",
}


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


def expand_symbols_and_units_it(text: str) -> str:
    """Write currency, percent and unambiguous units in Italian words.

    The amount is moved before the currency, as it is spoken: "$5" -> "5 dollari".
    """
    text = _CURRENCY_BEFORE.sub(r"\2 \1", text)
    for pattern, words in _IT_UNITS:
        text = pattern.sub(words, text)
    for symbol, words in _IT_SYMBOLS.items():
        text = text.replace(symbol, words)
    return text


def verbalize_numbers_it(text: str) -> str:
    """Write numbers in Italian words: "20" -> "venti", "2.000" -> "duemila".

    Handles Italian thousands separators ("2.000"), decimal commas ("3,5" ->
    "tre virgola cinque") and times ("10:30" -> "dieci e trenta"). Digits that are
    part of a word or code ("M16", "mp3") are left unchanged.
    """
    text = _IT_TIME.sub(_italian_time, text)
    return _IT_NUMBER.sub(_italian_number, text)


def restore_elided_tens_it(text: str) -> str:
    """Restore elided tens after elisions were split: "vent anni" -> "venti anni"."""
    return " ".join(_IT_ELIDED_TENS.get(word, word) for word in text.split())


def _italian_time(match: re.Match[str]) -> str:
    hours, minutes = match.groups()
    return hours if minutes == "00" else f"{hours} e {int(minutes)}"


def _italian_number(match: re.Match[str]) -> str:
    token = match.group()
    if _IT_THOUSANDS.fullmatch(token):
        token = token.replace(".", "")
    integer, _, decimals = token.replace(".", ",").partition(",")
    words = num2words(int(integer), lang="it")
    if decimals:
        decimal_words = (
            " ".join(num2words(int(digit), lang="it") for digit in decimals)
            if decimals.startswith("0")
            else num2words(int(decimals), lang="it")
        )
        words += " virgola " + decimal_words
    return f" {words} "


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

# it-v1 plus numbers, symbols and units in words (experiment E3). Not yet the default
# for Italian: changing the default would change every reported Italian number.
ITALIAN_V2 = Normalizer(
    "it-v2",
    (
        unicode_nfc,
        unify_apostrophes,
        lowercase,
        expand_symbols_and_units_it,
        verbalize_numbers_it,
        remove_punctuation,
        split_elisions,
        restore_elided_tens_it,
        collapse_whitespace,
    ),
)

_BY_LANGUAGE = {"en": ENGLISH, "it": ITALIAN, "de": GERMAN}
_BY_NAME = {
    normalizer.name: normalizer
    for normalizer in (IDENTITY, BASIC, ENGLISH, ITALIAN, ITALIAN_V2, GERMAN, WHISPER_BASIC)
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
