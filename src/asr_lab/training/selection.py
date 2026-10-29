"""Choose training data with several strategies under the same budget of audio (experiment E7).

The pool is the sample of E6, with its features and filter rules. Each strategy first decides
which utterances are *candidates*, then draws from them, in a seeded order, until the budget (in
hours) is used up, so every strategy trains on the same amount of audio:

- ``random``: every utterance is a candidate;
- ``naive_filter``: only the utterances that break no E6 rule;
- ``two_source_filter``: as ``naive_filter``, but an utterance broken only by ``high_wer`` stays a
  candidate when a second model agrees with its transcript (WER at most ``max_wer``). In E6, half
  of the high-WER utterances were the strong model's mistakes, not wrong transcripts: dropping
  them throws away hard but correct data;
- ``speaker_balanced``: every utterance is a candidate; the speakers take turns, the one with the
  fewest hours so far going next, so a few talkative speakers cannot fill the budget.

Utterances longer than 30 seconds are never candidates: Whisper sees 30-second windows, so the
rest of their audio would be cut while their transcript would not. This is a limit of the model,
not a quality rule, so it applies to ``random`` too.

Each seed is a new draw of the data (and, in ``finetune``, a new training run). The same seed gives
every strategy the same random order of the pool, so the strategies differ only by their rules.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from asr_lab.data.manifest import ManifestEntry, read_manifest
from asr_lab.data.quality import analyze_quality, load_filters
from asr_lab.data.voxpopuli import load_data_spec
from asr_lab.evaluation.error_analysis import analyze
from asr_lab.evaluation.metrics import error_rate
from asr_lab.evaluation.normalize import get_normalizer_by_name
from asr_lab.evaluation.scoring import load_run
from asr_lab.provenance import file_sha256

STRATEGIES = ("random", "naive_filter", "two_source_filter", "speaker_balanced")
MAX_SECONDS = 30.0  # Whisper's window


@dataclass(frozen=True)
class SelectionSpec:
    """The ``[selection]`` table of a fine-tuning spec (e.g. ``experiments/E7/finetune.toml``)."""

    quality_spec: Path
    first_run: Path
    second_run: Path
    budget_hours: float
    seeds: tuple[int, ...]
    strategies: tuple[str, ...]


def load_selection_spec(path: Path) -> SelectionSpec:
    """Read the ``[selection]`` table of a fine-tuning spec.

    Raises:
        ValueError: for an unknown strategy or a budget that is not positive.
    """
    table = tomllib.loads(path.read_text(encoding="utf-8"))["selection"]
    spec = SelectionSpec(
        quality_spec=Path(table["quality_spec"]),
        first_run=Path(table["first_run"]),
        second_run=Path(table["second_run"]),
        budget_hours=float(table["budget_hours"]),
        seeds=tuple(int(seed) for seed in table["seeds"]),
        strategies=tuple(table["strategies"]),
    )
    unknown = set(spec.strategies) - set(STRATEGIES)
    if unknown:
        raise ValueError(f"unknown strategies {sorted(unknown)}; known: {list(STRATEGIES)}")
    if spec.budget_hours <= 0:
        raise ValueError(f"budget_hours must be positive, got {spec.budget_hours}")
    return spec


def second_model_wer(second_run: Path, normalizer_name: str) -> dict[str, float]:
    """The WER of the second model on each utterance it transcribed."""
    run = load_run(second_run)
    normalizer = get_normalizer_by_name(normalizer_name)
    return {
        utterance.utterance_id: error_rate(utterance.counts)
        for utterance in analyze(run.entries, run.hypotheses, normalizer)
    }


def candidates(
    rows: Sequence[dict[str, Any]],
    strategy: str,
    second_wer: dict[str, float],
    max_wer: float,
) -> list[dict[str, Any]]:
    """The rows a strategy may draw from, in pool order (see the module docstring).

    ``rows`` are feature rows with their ``flags`` (``analyze_quality``).

    Raises:
        ValueError: if the second model did not transcribe a high-WER utterance.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; known: {list(STRATEGIES)}")
    fitting = [row for row in rows if row["duration"] <= MAX_SECONDS]
    if strategy in ("random", "speaker_balanced"):
        return fitting
    if strategy == "naive_filter":
        return [row for row in fitting if not row["flags"]]
    missing = [
        row
        for row in fitting
        if row["flags"] == ["high_wer"] and row["utterance_id"] not in second_wer
    ]
    if missing:
        raise ValueError(f"the second model did not transcribe {len(missing)} high-WER utterances")
    return [
        row
        for row in fitting
        if not row["flags"]
        or (row["flags"] == ["high_wer"] and second_wer[row["utterance_id"]] <= max_wer)
    ]


def draw(
    rows: Sequence[dict[str, Any]], budget_seconds: float, seed: int, *, by_speaker: bool = False
) -> list[dict[str, Any]]:
    """Rows in a seeded random order, each kept if it still fits in the budget.

    With ``by_speaker``, the speaker with the fewest selected seconds goes next (ties: the
    speaker who comes first in the seeded order), taking their next utterance.

    >>> rows = [{"utterance_id": str(i), "speaker_id": "a", "duration": 2.0} for i in range(5)]
    >>> len(draw(rows, budget_seconds=5.0, seed=0))
    2
    """
    rng = np.random.default_rng(seed)
    order = [rows[i] for i in rng.permutation(len(rows))]
    if not by_speaker:
        return _fill(order, budget_seconds)
    queues: dict[str, list[dict[str, Any]]] = {}
    for row in order:  # each speaker's utterances, in the seeded order
        queues.setdefault(row["speaker_id"], []).append(row)
    seconds = dict.fromkeys(queues, 0.0)
    selected, total = [], 0.0
    while queues:
        speaker = min(queues, key=lambda name: seconds[name])  # dicts keep the seeded order
        row = queues[speaker].pop(0)
        if not queues[speaker]:
            del queues[speaker]
        if total + row["duration"] <= budget_seconds:
            selected.append(row)
            total += row["duration"]
            seconds[speaker] += row["duration"]
    return selected


def select_data(spec_path: Path) -> dict[str, Any]:
    """Every (strategy, seed) selection of the spec, with a description of each.

    Returns a JSON-serializable report: the pool, the candidates of each strategy, and the
    selections with their utterance ids (in pool order).
    """
    spec = load_selection_spec(spec_path)
    data = load_data_spec(spec.quality_spec)
    max_wer = load_filters(spec.quality_spec).max_wer
    rows, _ = analyze_quality(spec.quality_spec, spec.first_run)
    second_wer = second_model_wer(spec.second_run, data.normalizer)
    position = {row["utterance_id"]: index for index, row in enumerate(rows)}
    budget = spec.budget_hours * 3600
    pools, selections = [], []
    for strategy in spec.strategies:
        pool = candidates(rows, strategy, second_wer, max_wer)
        pools.append({"strategy": strategy, **_describe(pool)})
        for seed in spec.seeds:
            chosen = draw(pool, budget, seed, by_speaker=strategy == "speaker_balanced")
            chosen.sort(key=lambda row: position[row["utterance_id"]])
            selections.append(
                {
                    "strategy": strategy,
                    "seed": seed,
                    **_describe(chosen),
                    "ids": [row["utterance_id"] for row in chosen],
                }
            )
    return {
        "spec": spec_path.as_posix(),
        "manifest": data.manifest.as_posix(),
        "manifest_sha256": file_sha256(data.manifest),
        "first_run": spec.first_run.as_posix(),
        "second_run": spec.second_run.as_posix(),
        "budget_hours": spec.budget_hours,
        "max_wer": max_wer,
        "pool": _describe(rows),
        "candidates": pools,
        "selections": selections,
    }


def save_selections(report: dict[str, Any], path: Path) -> None:
    """Write the selection report as JSON (UTF-8, LF) and the ids of each selection as a text
    file, one id per line, in a folder named like the report (``selections/random-s0.txt``).

    The ids of twelve selections would make a JSON file of more than a megabyte; apart, each file
    is small and its diff readable. The report records each file with its SHA-256.
    """
    folder = path.parent / path.stem
    folder.mkdir(parents=True, exist_ok=True)
    summary = {**report, "selections": []}
    for selection in report["selections"]:
        ids_file = folder / f"{selection['strategy'].replace('_', '-')}-s{selection['seed']}.txt"
        ids_file.write_text("\n".join(selection["ids"]) + "\n", encoding="utf-8", newline="\n")
        described = {key: value for key, value in selection.items() if key != "ids"}
        summary["selections"].append(
            {**described, "ids_file": ids_file.as_posix(), "ids_sha256": file_sha256(ids_file)}
        )
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def selected_entries(
    selections_path: Path, strategy: str, seed: int
) -> tuple[Path, list[ManifestEntry]]:
    """The manifest of a selection report and the entries of one of its selections.

    Raises:
        ValueError: if the report has no such selection, or the manifest or the selection's ids
            changed since.
    """
    report = json.loads(selections_path.read_text(encoding="utf-8"))
    manifest = Path(report["manifest"])
    if file_sha256(manifest) != report["manifest_sha256"]:
        raise ValueError(f"{manifest} is not the manifest the selections were made from")
    for selection in report["selections"]:
        if (selection["strategy"], selection["seed"]) == (strategy, seed):
            ids_file = Path(selection["ids_file"])
            if file_sha256(ids_file) != selection["ids_sha256"]:
                raise ValueError(f"{ids_file} changed since the selections were made")
            ids = set(ids_file.read_text(encoding="utf-8").split())
            return manifest, [e for e in read_manifest(manifest) if e.utterance_id in ids]
    raise ValueError(f"{selections_path} has no selection {strategy!r} with seed {seed}")


def _fill(order: Sequence[dict[str, Any]], budget_seconds: float) -> list[dict[str, Any]]:
    selected, total = [], 0.0
    for row in order:
        if total + row["duration"] <= budget_seconds:
            selected.append(row)
            total += row["duration"]
    return selected


def _describe(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    flagged = [row for row in rows if row["flags"]]
    return {
        "utterances": len(rows),
        "hours": sum(row["duration"] for row in rows) / 3600,
        "speakers": len({row["speaker_id"] for row in rows}),
        "sessions": len({row["session"] for row in rows}),
        "flagged": len(flagged),
        "high_wer": sum("high_wer" in row["flags"] for row in flagged),
    }
