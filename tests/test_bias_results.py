"""Results of the entity lists of E9: recall, WER, comparisons, invented names, short outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from asr_lab.biasing.results import bias_results, bias_table, is_short
from asr_lab.data.manifest import ManifestEntry, write_manifest
from asr_lab.evaluation.comparison import save_results
from asr_lab.inference.base import Hypothesis
from asr_lab.inference.predictions import append_predictions
from asr_lab.reporting import build_tables

# Eight utterances by four speakers: the first four name Roma, the others Parigi
ROMA = "vado a roma domani con mario e luca"
PARIGI = "parto per parigi oggi con anna e sara"
REFERENCES = [ROMA] * 4 + [PARIGI] * 4
MISSED = ["vado a rome domani con mario e luca"] * 4 + [PARIGI] * 4
COLLAPSED = [*MISSED[:7], "grazie"]  # the last transcript is a closing phrase for a whole utterance
HYPOTHESES = {
    "whisper": {
        "none": COLLAPSED,
        "oracle": REFERENCES,
        "distractors": COLLAPSED,
        # The prompt prevents the collapse, and writes the name of the list once without reason
        "domain-1": [f"{MISSED[0]} europa", *MISSED[1:]],
    },
    "ctc": {"none": MISSED, "oracle": REFERENCES, "distractors": MISSED, "domain-1": MISSED},
}
SECTIONS = ("selection", "check", "results", "comparisons", "invented", "short")


def run_dir(root: Path, model: str, condition: str) -> Path:
    runs = root / "runs"
    if condition == "none":
        return runs / ("whisper-baseline" if model == "whisper" else "e4-europarl-5-vox")
    prefix = "e9-turbo" if model == "whisper" else "e9-ctc"
    return runs / f"{prefix}-{condition}-vox"


def write_experiment(root: Path, ctc_oracle_weight: float | None = 2.0) -> tuple[Path, ...]:
    """A spec with one test set and a domain list, its runs, and the files the results read."""
    entries = [
        ManifestEntry(f"u{i}", f"u{i}.wav", 2.0, text, "it", f"s{i % 4}", "test")
        for i, text in enumerate(REFERENCES)
    ]
    manifest = root / "test.jsonl"
    write_manifest(manifest, entries)
    for model, conditions in HYPOTHESES.items():
        for condition, texts in conditions.items():
            run = run_dir(root, model, condition)
            run.mkdir(parents=True)
            append_predictions(
                run / "predictions.jsonl",
                [Hypothesis(f"u{i}", text) for i, text in enumerate(texts)],
            )
            info = {"model": model, "model_config": {}, "git_commit": "abc", "rtf": 0.01}
            (run / "run.json").write_text(json.dumps(info), encoding="utf-8")
    e4_grid = root / "e4_grid.json"
    e4_grid.write_text(
        json.dumps({"test_sets": {"vox": {"europarl-5": {"selected": {"wer": 0.1}}}}}),
        encoding="utf-8",
    )
    lm_spec = root / "lm.toml"
    lm_spec.write_text("[decoding]\nbeam_width = 10\n", encoding="utf-8")
    spec = root / "biasing.toml"
    spec.write_text(
        f"""
[entities]
seed = 0

[ctc]
language_model = "europarl"
order = 5
lm_spec = "{lm_spec.as_posix()}"
grid = "{e4_grid.as_posix()}"

[[test_sets]]
name = "vox"
lang = "it"
normalizer = "it-v1"
test = "{manifest.as_posix()}"
whisper_baseline = "{run_dir(root, "whisper", "none").as_posix()}"
has_domain_list = true
""",
        encoding="utf-8",
    )
    found = {f"u{i}": ["Roma"] if i < 4 else ["Parigi"] for i in range(8)}
    entities = root / "entities.json"
    entities.write_text(
        json.dumps(
            {"domain": [{"entity": "Europa", "count": 5}], "test_sets": {"vox": {"test": found}}}
        ),
        encoding="utf-8",
    )
    cell = {"wer": 0.1, "recall": 0.5, "n_entities": 8}
    oracle = {**cell, "weight": ctc_oracle_weight} if ctc_oracle_weight else None
    selection = root / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "whisper": {"baseline": cell, "cells": [{**cell, "size": 1}], "domain_size": 1},
                "ctc": {
                    "vox": {
                        "baseline": cell,
                        "oracle": oracle,
                        "domain": {**cell, "size": 1, "weight": 2.0},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    ctc_grid = root / "ctc_grid.json"
    ctc_grid.write_text(json.dumps({"test_sets": {"vox": {"baseline": cell}}}), encoding="utf-8")
    return spec, entities, selection, ctc_grid, root / "runs"


@pytest.fixture
def report(tmp_path: Path) -> dict:
    return bias_results(*write_experiment(tmp_path), n_resamples=50)


def test_short_needs_a_long_enough_reference() -> None:
    assert is_short(["a", "b", "c", "d", "e", "f"], ["a", "b"])
    assert not is_short(["a", "b", "c", "d", "e", "f"], ["a", "b", "c"])
    assert not is_short(["a", "b", "c", "d"], [])


def test_recall_and_wer_of_every_list(report: dict) -> None:
    whisper = report["test_sets"][0]["models"]["whisper"]["systems"]
    # Roma is missed four times, and Parigi once, in the collapsed transcript
    assert whisper["none"]["recall"] == pytest.approx(3 / 8)
    assert whisper["oracle"]["recall"] == 1.0
    assert whisper["oracle"]["wer"] == 0.0
    assert report["test_sets"][0]["n_entities"] == 8


def test_comparisons_turn_recall_into_a_gain(report: dict) -> None:
    comparisons = report["test_sets"][0]["comparisons"]
    assert len(comparisons) == 12  # two models, three lists, two metrics
    oracle = next(
        c
        for c in comparisons
        if (c["model"], c["list"], c["metric"]) == ("whisper", "oracle", "recall")
    )
    assert oracle["delta"] == pytest.approx(5 / 8)
    assert oracle["low"] <= oracle["delta"] <= oracle["high"]
    same = next(
        c
        for c in comparisons
        if (c["model"], c["list"], c["metric"]) == ("ctc", "distractors", "recall")
    )
    assert str(same["delta"]) == "0.0"
    assert all(c["p_holm"] >= c["p_value"] for c in comparisons)


def test_invented_entities_are_counted_with_and_without_the_list(report: dict) -> None:
    domain = report["test_sets"][0]["models"]["whisper"]["systems"]["domain-1"]
    assert domain["invented"] == 1
    assert domain["invented_without_list"] == 0


def test_short_transcripts_and_the_wer_without_them(report: dict) -> None:
    test_set = report["test_sets"][0]
    whisper = test_set["models"]["whisper"]["systems"]
    assert whisper["none"]["short"] == 1
    assert whisper["none"]["short_missed_entities"] == 1
    assert whisper["domain-1"]["short"] == 0
    row = next(
        r for r in test_set["without_short"] if (r["model"], r["list"]) == ("whisper", "domain-1")
    )
    assert row["n_utterances"] == 7


def test_the_check_compares_with_the_dev_wer_of_e4(report: dict) -> None:
    assert report["check"] == [{"test_set": "vox", "e4_dev_wer": 0.1, "e9_dev_wer": 0.1}]


def test_ctc_without_an_oracle_weight_keeps_only_the_domain_list(tmp_path: Path) -> None:
    paths = write_experiment(tmp_path, ctc_oracle_weight=None)
    report = bias_results(*paths, n_resamples=50)
    assert list(report["test_sets"][0]["models"]["ctc"]["systems"]) == ["none", "domain-1"]
    assert len(report["test_sets"][0]["comparisons"]) == 8


@pytest.mark.parametrize("section", SECTIONS)
def test_every_table_has_a_row_per_item(report: dict, section: str) -> None:
    table = bias_table(report, section)
    assert table.startswith("|")
    assert table.count("\n") >= 2


def test_the_short_table_shows_only_runs_with_short_transcripts(report: dict) -> None:
    table = bias_table(report, "short")
    assert "Whisper large-v3-turbo" in table
    assert "XLS-R CTC" not in table


def test_unknown_section_is_refused(report: dict) -> None:
    with pytest.raises(ValueError, match="unknown section"):
        bias_table(report, "nope")


def test_results_render_through_the_tables_spec(tmp_path: Path, report: dict) -> None:
    results = tmp_path / "results.json"
    save_results(report, results)
    spec = tmp_path / "tables.toml"
    table = (
        f'name = "e9-results"\nkind = "bias"\nfile = "{results.as_posix()}"\nsection = "results"\n'
    )
    spec.write_text(f"[[table]]\n{table}", encoding="utf-8")
    assert "Domain (1)" in build_tables(spec)["e9-results"]
