"""The prompts of experiment E8: what the language model reads for each N-best list.

The texts live in a versioned TOML file, so that a result always says which words produced it. A
prompt is a list of chat messages, ``{"role": ..., "content": ...}``, which the model's own chat
template turns into tokens.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from asr_lab.provenance import file_sha256

LANGUAGES = {"it": "Italian", "de": "German", "en": "English"}
Message = dict[str, str]
_NUMBERING = re.compile(r"^\s*(?:\d+\s*[.)]\s+|transcript\s*:\s*)", re.IGNORECASE)
_QUOTES = "\"'“”«»`"


@dataclass(frozen=True)
class Example:
    """A few-shot example: the N-best list of a dev utterance and its reference."""

    utterance_id: str
    candidates: tuple[str, ...]
    target: str


@dataclass(frozen=True)
class Prompts:
    version: str
    sha256: str
    texts: Mapping[str, Mapping[str, str]]

    def score_prefix(self, lang: str) -> str:
        """The text before each transcript when its probability is computed.

        >>> prompts = Prompts("v", "", {"score": {"prefix": "{language}:"}})
        >>> prompts.score_prefix("de")
        'German:'
        """
        return self.texts["score"]["prefix"].format(language=LANGUAGES[lang])

    def choose(self, candidates: Sequence[str], lang: str) -> list[Message]:
        """The messages that ask which candidate is correct."""
        return self._conversation("choose", candidates, lang)

    def correct(
        self, candidates: Sequence[str], lang: str, examples: Sequence[Example] = ()
    ) -> list[Message]:
        """The messages that ask for a corrected transcript, after the examples if any."""
        messages = self._conversation("correct", candidates, lang)
        shots = []
        for example in examples:
            shots.append(self._user("correct", example.candidates, lang))
            shots.append({"role": "assistant", "content": example.target})
        return [messages[0], *shots, messages[1]]

    def _conversation(self, task: str, candidates: Sequence[str], lang: str) -> list[Message]:
        system = {"role": "system", "content": self.texts[task]["system"]}
        return [system, self._user(task, candidates, lang)]

    def _user(self, task: str, candidates: Sequence[str], lang: str) -> Message:
        listed = "\n".join(f"{i}. {text}" for i, text in enumerate(candidates, start=1))
        content = self.texts[task]["user"].format(
            n=len(candidates), language=LANGUAGES[lang], candidates=listed
        )
        return {"role": "user", "content": content}


def load_prompts(path: Path) -> Prompts:
    """The prompts of a TOML file, with its version and SHA-256."""
    texts: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
    version = texts.pop("version")
    return Prompts(version, file_sha256(path), texts)


def clean_answer(text: str) -> str:
    """The transcript in a model's answer: its first line, without a number or quotes around it.

    >>> clean_answer('1. "vado a roma"\\nI changed one word.')
    'vado a roma'
    >>> clean_answer("Transcript: buongiorno")
    'buongiorno'
    >>> clean_answer("")
    ''
    """
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    return _NUMBERING.sub("", lines[0]).strip().strip(_QUOTES).strip()
