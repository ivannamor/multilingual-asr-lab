"""Decoding a CTC model's frame probabilities, with or without a language model.

Greedy decoding takes the best token of every frame. Beam search keeps the ``beam_width`` best
partial transcripts instead, and adds the opinion of a language model each time a word ends::

    score = acoustic + alpha * ln P(word | previous words) + beta

``alpha`` says how much to trust the language model and ``beta`` rewards each written word, which
offsets the language model's tendency to prefer fewer, more common words. Both are tuned on a dev
set.

Several alignments give the same text ("aa-b", "a-bb", where "-" is the blank), so hypotheses with
the same text and the same last token are merged by adding their probabilities. The blank and the
last token stay in the hypothesis because they decide what the next token means: a repeated
character adds nothing unless a blank separates the two.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

WORD_DELIMITER = "|"
BLANK = "<pad>"
SPECIAL_TOKENS = ("<s>", "</s>", "<unk>")
LOG10_TO_LN = math.log(10)


class LanguageModel(Protocol):
    """What the beam search needs from a language model."""

    def begin(self) -> Any:
        """The state before the first word."""

    def score(self, state: Any, word: str) -> tuple[float, Any]:
        """ln P(word | state), and the state after the word."""

    def end(self, state: Any) -> float:
        """ln P(end of sentence | state)."""


class KenLMModel:
    """A binary language model read with KenLM. Its scores are converted from log10 to ln."""

    def __init__(self, path: str) -> None:
        import kenlm  # an optional extra, compiled at installation

        self._kenlm = kenlm
        self._model = kenlm.Model(path)

    def begin(self) -> Any:
        state = self._kenlm.State()
        self._model.BeginSentenceWrite(state)
        return state

    def score(self, state: Any, word: str) -> tuple[float, Any]:
        after = self._kenlm.State()
        return self._model.BaseScore(state, word, after) * LOG10_TO_LN, after

    def end(self, state: Any) -> float:
        return self._model.BaseScore(state, "</s>", self._kenlm.State()) * LOG10_TO_LN


@dataclass(frozen=True)
class DecodingOptions:
    """Settings of the beam search. Fixed for a whole experiment, except ``alpha`` and ``beta``."""

    alpha: float = 0.0
    beta: float = 0.0
    beam_width: int = 100
    token_min_logp: float = -10.0  # tokens less likely than this in a frame are not tried
    beam_prune_logp: float = -20.0  # hypotheses this far from the best one are dropped


def greedy_decode(log_probs: np.ndarray, tokens: Sequence[str]) -> str:
    """The best token of each frame, without repeats and blanks.

    >>> tokens = ["<pad>", "|", "a", "b"]
    >>> frames = np.log(np.eye(4)[[2, 2, 0, 2, 1, 3, 3]] * 0.97 + 0.01)
    >>> greedy_decode(frames, tokens)
    'aa b'
    """
    best = log_probs.argmax(axis=1).tolist()
    chars = []
    previous = None
    for index in best:
        if index != previous:
            token = tokens[index]
            if token == WORD_DELIMITER:
                chars.append(" ")
            elif token != BLANK and token not in SPECIAL_TOKENS:
                chars.append(token)
        previous = index
    return " ".join("".join(chars).split())


def beam_search(
    log_probs: np.ndarray,
    tokens: Sequence[str],
    lm: LanguageModel | None = None,
    options: DecodingOptions | None = None,
) -> str:
    """The most probable transcript of one utterance; see the module docstring.

    Args:
        log_probs: natural log probabilities, frames x tokens.
        tokens: the token of each column.
        lm: without one, the search only merges the alignments of the acoustic model.
    """
    options = options or DecodingOptions()
    usable = [i for i, token in enumerate(tokens) if token not in SPECIAL_TOKENS]
    begin = lm.begin() if lm else None
    # (written words, current word, last token) -> [acoustic score, language score, LM state]
    beams: dict[tuple[str, str, str | None], list[Any]] = {("", "", None): [0.0, 0.0, begin]}
    for frame in log_probs:
        candidates = _candidates(frame, tokens, usable, options.token_min_logp)
        extended: dict[tuple[str, str, str | None], list[Any]] = {}
        for (text, word, last), (acoustic, language, state) in beams.items():
            for token, logp in candidates:
                key, new_language, new_state = _extend(
                    text, word, last, token, language, state, lm, options
                )
                score = acoustic + logp
                found = extended.get(key)
                if found is None:
                    extended[key] = [score, new_language, new_state]
                else:
                    found[0] = _log_add(found[0], score)
        beams = _prune(extended, options)
    return _best_transcript(beams, lm, options)


def _candidates(
    frame: np.ndarray, tokens: Sequence[str], usable: Sequence[int], min_logp: float
) -> list[tuple[str, float]]:
    """The tokens worth trying in a frame: the likely ones, and always the best one."""
    values = frame.tolist()
    best = max(usable, key=values.__getitem__)
    return [(tokens[i], values[i]) for i in usable if values[i] >= min_logp or i == best]


def _extend(
    text: str,
    word: str,
    last: str | None,
    token: str,
    language: float,
    state: Any,
    lm: LanguageModel | None,
    options: DecodingOptions,
) -> tuple[tuple[str, str, str | None], float, Any]:
    if token == BLANK:
        return (text, word, None), language, state
    if token == last:  # the same character over several frames
        return (text, word, last), language, state
    if token != WORD_DELIMITER:
        return (text, word + token, token), language, state
    if not word:
        return (text, "", token), language, state
    new_language, new_state = _add_word(word, language, state, lm, options)
    return (f"{text} {word}" if text else word, "", token), new_language, new_state


def _add_word(
    word: str, language: float, state: Any, lm: LanguageModel | None, options: DecodingOptions
) -> tuple[float, Any]:
    if lm is None:
        return language, state
    logp, new_state = lm.score(state, word)
    return language + options.alpha * logp + options.beta, new_state


def _log_add(a: float, b: float) -> float:
    """ln(exp(a) + exp(b)), without leaving the log domain."""
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def _prune(
    beams: dict[tuple[str, str, str | None], list[Any]], options: DecodingOptions
) -> dict[tuple[str, str, str | None], list[Any]]:
    scored = sorted(beams.items(), key=lambda item: -(item[1][0] + item[1][1]))
    best = scored[0][1][0] + scored[0][1][1]
    return {
        key: value
        for key, value in scored[: options.beam_width]
        if value[0] + value[1] >= best + options.beam_prune_logp
    }


def _best_transcript(
    beams: dict[tuple[str, str, str | None], list[Any]],
    lm: LanguageModel | None,
    options: DecodingOptions,
) -> str:
    """Close the last word of each hypothesis, merge equal texts and take the best one."""
    finished: dict[str, list[float]] = {}
    for (text, word, _), (acoustic, language, state) in beams.items():
        full = f"{text} {word}".strip() if word else text
        if word:
            language, state = _add_word(word, language, state, lm, options)
        if lm is not None:
            language += options.alpha * lm.end(state)
        found = finished.get(full)
        if found is None:
            finished[full] = [acoustic, language]
        else:
            found[0] = _log_add(found[0], acoustic)
    return max(finished, key=lambda text: finished[text][0] + finished[text][1])
