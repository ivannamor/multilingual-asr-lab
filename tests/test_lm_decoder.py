"""The CTC beam search, against brute force and with a language model that can be read by hand."""

from __future__ import annotations

import itertools
import math

import numpy as np

from asr_lab.lm.decoder import DecodingOptions, beam_search, greedy_decode

TOKENS = ["<pad>", "|", "a", "b"]


def frames(*rows: list[float]) -> np.ndarray:
    return np.log(np.array(rows))


def collapse(path: tuple[int, ...]) -> str:
    """The text of one alignment, by the CTC rule."""
    chars, previous = [], None
    for index in path:
        if index != previous and index != 0:
            chars.append(" " if TOKENS[index] == "|" else TOKENS[index])
        previous = index
    return " ".join("".join(chars).split())


def brute_force(log_probs: np.ndarray) -> str:
    """The transcript with the highest total probability, summing all its alignments."""
    totals: dict[str, float] = {}
    for path in itertools.product(range(len(TOKENS)), repeat=len(log_probs)):
        logp = sum(log_probs[t, index] for t, index in enumerate(path))
        text = collapse(path)
        totals[text] = totals.get(text, 0.0) + math.exp(logp)
    return max(totals, key=totals.__getitem__)


class FixedLM:
    """A unigram language model: a fixed probability per word, 1 for the end of the sentence."""

    def __init__(self, probabilities: dict[str, float]) -> None:
        self.probabilities = probabilities

    def begin(self) -> None:
        return None

    def score(self, state: None, word: str) -> tuple[float, None]:
        return math.log(self.probabilities.get(word, 1e-6)), state

    def end(self, state: None) -> float:
        return 0.0


def test_greedy_applies_the_ctc_rule() -> None:
    assert greedy_decode(np.log(np.eye(4)[[2, 2, 0, 2]] * 0.97 + 0.01), TOKENS) == "aa"
    assert greedy_decode(np.log(np.eye(4)[[1, 2, 1, 1, 3, 1]] * 0.97 + 0.01), TOKENS) == "a b"
    assert greedy_decode(np.zeros((0, 4)), TOKENS) == ""


def test_greedy_skips_the_special_tokens() -> None:
    tokens = ["<pad>", "|", "a", "<s>"]
    assert greedy_decode(np.log(np.eye(4)[[3, 2, 3]] * 0.97 + 0.01), tokens) == "a"


def test_beam_search_finds_what_greedy_misses() -> None:
    # The best alignment is blank-blank, but "a" has more probability over all its alignments
    log_probs = frames([0.4, 0.0, 0.3, 0.3], [0.4, 0.0, 0.3, 0.3])
    options = DecodingOptions(beam_width=100, token_min_logp=-100, beam_prune_logp=-100)
    assert greedy_decode(log_probs, TOKENS) == ""
    assert beam_search(log_probs, TOKENS, options=options) in {"a", "b"}


def test_beam_search_agrees_with_brute_force() -> None:
    options = DecodingOptions(beam_width=10_000, token_min_logp=-100, beam_prune_logp=-1000)
    rng = np.random.default_rng(0)
    for _ in range(30):
        log_probs = np.log(rng.dirichlet(np.ones(4) * 0.5, size=5) + 1e-9)
        assert beam_search(log_probs, TOKENS, options=options) == brute_force(log_probs)


def test_a_narrow_beam_still_returns_a_transcript() -> None:
    log_probs = frames([0.1, 0.1, 0.7, 0.1], [0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7])
    assert beam_search(log_probs, TOKENS, options=DecodingOptions(beam_width=1)) == "ab"


def test_the_language_model_decides_between_similar_sounds() -> None:
    log_probs = frames([0.05, 0.05, 0.5, 0.4], [0.05, 0.05, 0.5, 0.4])
    options = DecodingOptions(alpha=1.0, token_min_logp=-100, beam_prune_logp=-100)
    assert beam_search(log_probs, TOKENS, options=options) == "a"
    lm = FixedLM({"a": 0.001, "b": 0.9})
    assert beam_search(log_probs, TOKENS, lm, options) == "b"


def test_alpha_zero_ignores_the_language_model() -> None:
    log_probs = frames([0.05, 0.05, 0.5, 0.4], [0.05, 0.05, 0.5, 0.4])
    options = DecodingOptions(alpha=0.0, token_min_logp=-100, beam_prune_logp=-100)
    assert beam_search(log_probs, TOKENS, FixedLM({"a": 1e-5, "b": 0.9}), options) == "a"


def test_beta_rewards_more_words() -> None:
    # "a b" and "ab" differ only in the delimiter, slightly unlikely
    log_probs = frames([0.01, 0.01, 0.97, 0.01], [0.6, 0.38, 0.01, 0.01], [0.01, 0.01, 0.01, 0.97])
    lm = FixedLM({"a": 0.5, "b": 0.5, "ab": 0.5})
    base = {"alpha": 0.0, "token_min_logp": -100, "beam_prune_logp": -100}
    assert beam_search(log_probs, TOKENS, lm, DecodingOptions(beta=0.0, **base)) == "ab"
    assert beam_search(log_probs, TOKENS, lm, DecodingOptions(beta=2.0, **base)) == "a b"


def test_the_end_of_the_sentence_is_scored() -> None:
    class Ending(FixedLM):
        def end(self, state: None) -> float:
            return -5.0

    log_probs = frames([0.05, 0.05, 0.9, 0.0001], [0.05, 0.05, 0.9, 0.0001])
    options = DecodingOptions(alpha=1.0, token_min_logp=-100, beam_prune_logp=-100)
    assert beam_search(log_probs, TOKENS, Ending({"a": 0.9}), options) == "a"


def test_empty_audio_gives_an_empty_transcript() -> None:
    assert beam_search(np.zeros((0, 4)), TOKENS) == ""
