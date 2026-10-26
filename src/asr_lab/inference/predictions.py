"""Raw model outputs stored as JSONL, one hypothesis per line.

Predictions are saved before any normalization or scoring, so metrics can be
recomputed (with another normalizer, another statistic) without re-running
inference. The file is appended to batch by batch, which makes runs resumable.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from asr_lab.inference.base import Hypothesis


def append_predictions(path: Path, hypotheses: Iterable[Hypothesis]) -> None:
    """Append hypotheses to a JSONL file (UTF-8, LF line endings), creating it if needed."""
    lines = [
        json.dumps(
            {
                "utterance_id": h.utterance_id,
                "text": h.text,
                "score": h.score,
                "nbest": list(h.nbest),
            },
            ensure_ascii=False,
        )
        + "\n"
        for h in hypotheses
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.writelines(lines)


def read_predictions(path: Path) -> list[Hypothesis]:
    """Read a predictions file.

    Raises:
        ValueError: with file name and line number for invalid lines or repeated ids.
    """
    hypotheses = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            where = f"{path}:{line_number}"
            try:
                record = json.loads(line)
                hypothesis = Hypothesis(
                    utterance_id=record["utterance_id"],
                    text=record["text"],
                    score=record.get("score"),
                    nbest=tuple(record.get("nbest", ())),
                )
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"{where}: invalid prediction ({error})") from None
            if not isinstance(hypothesis.text, str):
                raise ValueError(f"{where}: text must be a string")
            if hypothesis.utterance_id in seen:
                raise ValueError(f"{where}: duplicate utterance_id {hypothesis.utterance_id!r}")
            seen.add(hypothesis.utterance_id)
            hypotheses.append(hypothesis)
    return hypotheses
