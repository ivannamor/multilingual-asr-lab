"""Training data selection of E7: candidates of each strategy, budget, seeds, speaker turns."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from asr_lab.data.manifest import ManifestEntry, write_manifest
from asr_lab.training.selection import (
    STRATEGIES,
    candidates,
    draw,
    load_selection_spec,
    save_selections,
    selected_entries,
)


def row(utterance_id: str, *, speaker: str = "s1", duration: float = 10.0, flags=()) -> dict:
    return {
        "utterance_id": utterance_id,
        "speaker_id": speaker,
        "session": "day1",
        "duration": duration,
        "flags": list(flags),
    }


POOL = [
    row("clean"),
    row("long", duration=31.0, flags=["too_long"]),
    row("noisy", flags=["low_snr"]),
    row("model-wrong", flags=["high_wer"]),  # the second model agrees with the transcript
    row("label-wrong", flags=["high_wer"]),  # the second model disagrees too
    row("both", flags=["low_snr", "high_wer"]),
]
SECOND_WER = {"model-wrong": 0.1, "label-wrong": 0.9, "both": 0.1}


def ids(rows: list[dict[str, Any]]) -> list[str]:
    return [r["utterance_id"] for r in rows]


def test_candidates_of_each_strategy() -> None:
    def of(strategy: str) -> list[str]:
        return ids(candidates(POOL, strategy, SECOND_WER, max_wer=0.5))

    everything = ["clean", "noisy", "model-wrong", "label-wrong", "both"]  # never "long"
    assert of("random") == everything
    assert of("speaker_balanced") == everything
    assert of("naive_filter") == ["clean"]
    # Kept only when high_wer is the one rule broken and the second model agrees
    assert of("two_source_filter") == ["clean", "model-wrong"]


def test_two_source_filter_needs_the_second_model_on_every_high_wer_utterance() -> None:
    with pytest.raises(ValueError, match="did not transcribe 1"):
        candidates(POOL, "two_source_filter", {"model-wrong": 0.1}, max_wer=0.5)


def test_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        candidates(POOL, "everything", SECOND_WER, max_wer=0.5)


def test_draw_fills_the_budget_and_depends_on_the_seed() -> None:
    pool = [row(str(i), duration=d) for i, d in enumerate([4.0, 3.0, 2.0, 6.0, 1.0, 5.0])]
    for seed in range(5):
        chosen = draw(pool, budget_seconds=10.0, seed=seed)
        assert sum(r["duration"] for r in chosen) <= 10.0
        # Nothing left out would still fit
        left = [r for r in pool if r not in chosen]
        assert all(sum(r["duration"] for r in chosen) + r["duration"] > 10.0 for r in left)
        assert draw(pool, budget_seconds=10.0, seed=seed) == chosen  # reproducible
    assert len({tuple(ids(draw(pool, 10.0, seed))) for seed in range(5)}) > 1


def test_draw_by_speaker_gives_each_speaker_the_same_hours() -> None:
    talkative = [row(f"a{i}", speaker="a") for i in range(20)]
    quiet = [row(f"b{i}", speaker="b") for i in range(3)] + [row("c0", speaker="c")]
    pool = talkative + quiet

    chosen = draw(pool, budget_seconds=80.0, seed=0, by_speaker=True)
    speakers = [r["speaker_id"] for r in chosen]
    # c runs out after one turn, b after three: a gets the rest of the budget
    assert speakers.count("c") == 1
    assert speakers.count("b") == 3
    assert speakers.count("a") == 4
    # Without turns, the talkative speaker takes most of the budget
    plain = [r["speaker_id"] for r in draw(pool, budget_seconds=80.0, seed=0)]
    assert plain.count("a") > 4


def write_spec(tmp_path: Path, strategies: str = '["random"]', budget: float = 1.0) -> Path:
    spec = tmp_path / "finetune.toml"
    spec.write_text(
        f"""
[selection]
quality_spec = "quality.toml"
first_run = "runs/first"
second_run = "runs/second"
budget_hours = {budget}
seeds = [0, 1]
strategies = {strategies}
""",
        encoding="utf-8",
    )
    return spec


def test_load_selection_spec(tmp_path: Path) -> None:
    spec = load_selection_spec(write_spec(tmp_path, json.dumps(list(STRATEGIES))))
    assert spec.strategies == STRATEGIES
    assert spec.seeds == (0, 1)
    assert spec.second_run == Path("runs/second")
    with pytest.raises(ValueError, match="unknown strategies"):
        load_selection_spec(write_spec(tmp_path, '["random", "best"]'))
    with pytest.raises(ValueError, match="positive"):
        load_selection_spec(write_spec(tmp_path, budget=0))


def test_selected_entries_checks_the_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    entries = [
        ManifestEntry(f"u{i}", f"audio/u{i}.wav", 1.0, f"testo {i}", "it", "s1", "voxpopuli")
        for i in range(3)
    ]
    write_manifest(manifest, entries)
    from asr_lab.provenance import file_sha256

    report = {
        "manifest": manifest.as_posix(),
        "manifest_sha256": file_sha256(manifest),
        "selections": [{"strategy": "random", "seed": 0, "ids": ["u0", "u2"]}],
    }
    path = tmp_path / "selections.json"
    save_selections(report, path)
    ids_file = tmp_path / "selections" / "random-s0.txt"
    assert ids_file.read_text(encoding="utf-8") == "u0\nu2\n"
    assert "ids" not in json.loads(path.read_text(encoding="utf-8"))["selections"][0]

    found, chosen = selected_entries(path, "random", 0)
    assert found == manifest
    assert [e.utterance_id for e in chosen] == ["u0", "u2"]
    with pytest.raises(ValueError, match="no selection"):
        selected_entries(path, "random", 1)
    ids_file.write_text("u0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed since"):
        selected_entries(path, "random", 0)
    ids_file.write_text("u0\nu2\n", encoding="utf-8")
    write_manifest(manifest, entries[:2])
    with pytest.raises(ValueError, match="not the manifest"):
        selected_entries(path, "random", 0)
