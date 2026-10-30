"""Results of the fine-tuned models of E7: edge errors, pooled seeds, contrasts, tables."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from asr_lab.data.manifest import ManifestEntry, write_manifest
from asr_lab.evaluation.metrics import ErrorCounts, align
from asr_lab.inference.base import Hypothesis
from asr_lab.inference.predictions import append_predictions
from asr_lab.reporting import build_tables
from asr_lab.training.results import (
    edge_errors,
    finetune_results,
    finetune_table,
    invented_address,
    model_name,
    run_dir,
    save_results,
    sum_counts,
)


def test_edge_errors_are_outside_the_matched_span() -> None:
    # The hypothesis starts one word early and stops one word short: a boundary error
    pairs = align(["a", "b", "c", "d"], ["x", "a", "b", "c"])
    assert edge_errors(pairs) == ErrorCounts(deletions=1, insertions=1)
    # An error between two matches is not at an edge
    assert edge_errors(align(["a", "b", "c"], ["a", "x", "c"])) == ErrorCounts()
    # Without any match every error is at an edge
    assert edge_errors(align(["a", "b"], ["x"])).errors == 2
    assert edge_errors([]) == ErrorCounts()


def test_invented_address_needs_the_phrase_in_the_hypothesis_only() -> None:
    assert invented_address("la città", "onorevoli colleghi la città")
    assert invented_address("la città", "la signora presidente della città")
    assert not invented_address("signor presidente la città", "signor presidente la città")
    assert not invented_address("la città", "la città presidente")


def test_sum_counts_adds_utterance_by_utterance() -> None:
    summed = sum_counts([[ErrorCounts(hits=2), ErrorCounts(deletions=1)]] * 3)
    assert summed == [ErrorCounts(hits=6), ErrorCounts(deletions=3)]
    with pytest.raises(ValueError):
        sum_counts([[ErrorCounts()], [ErrorCounts(), ErrorCounts()]])


def test_model_names_match_the_run_directories() -> None:
    assert model_name("base", 0) == "base"
    assert model_name("two_source_filter", 1) == "two-source-filter-s1"
    assert run_dir(Path("runs"), "base", "fleurs") == Path("runs/e7-base-fleurs")


# A test set of 8 utterances by 4 speakers, two of whom are in the training pool. Every
# reference is "uno due tre quattro"; each model's hypotheses are given as numbers of errors.
SPEAKERS = ["seen1", "seen2", "new1", "new2"]
REFERENCE = "uno due tre quattro"
ERRORS = {
    "base": [0, 1, 0, 1, 0, 1, 0, 1],
    "random-s0": [2, 2, 2, 2, 2, 2, 2, 2],
    "random-s1": [2, 1, 2, 1, 2, 1, 2, 1],
    "naive-filter-s0": [0, 0, 0, 1, 0, 0, 0, 1],
    "naive-filter-s1": [0, 1, 0, 0, 0, 1, 0, 0],
}


def hypothesis(errors: int) -> str:
    """The reference with its last ``errors`` words dropped (deletions at the edge)."""
    words = REFERENCE.split()
    return " ".join(words[: len(words) - errors])


def write_experiment(root: Path) -> Path:
    entries = [
        ManifestEntry(f"u{i}", f"u{i}.wav", 2.0, REFERENCE, "it", SPEAKERS[i % 4], "test")
        for i in range(8)
    ]
    test_manifest = root / "data" / "test.jsonl"
    write_manifest(test_manifest, entries)
    pool = [
        ManifestEntry(f"p{i}", f"p{i}.wav", 2.0, "ciao", "it", s, "pool")
        for i, s in enumerate(SPEAKERS[:2])
    ]
    write_manifest(root / "pool" / "train.jsonl", pool)
    (root / "quality.toml").write_text(
        f"""[data]
repo = "r"
revision = "x"
lang = "it"
split = "train"
row_groups = [[0, 0]]
output = "{(root / "pool").as_posix()}"
normalizer = "it-v2"
""",
        encoding="utf-8",
    )
    spec = root / "finetune.toml"
    spec.write_text(
        f"""[selection]
quality_spec = "{(root / "quality.toml").as_posix()}"
first_run = "first"
second_run = "second"
budget_hours = 1.0
seeds = [0, 1]
strategies = ["random", "naive_filter"]

[evaluation]
test_sets = {{ voxpopuli = "{test_manifest.as_posix()}" }}
normalizer = "it-v2"

[analysis]
contrasts = [["naive_filter", "random"]]
""",
        encoding="utf-8",
    )
    for name, errors in ERRORS.items():
        directory = run_dir(root / "runs", name, "voxpopuli")
        directory.mkdir(parents=True)
        info = {"manifest": test_manifest.as_posix(), "model": name}
        (directory / "run.json").write_text(json.dumps(info), encoding="utf-8")
        append_predictions(
            directory / "predictions.jsonl",
            [Hypothesis(f"u{i}", hypothesis(e)) for i, e in enumerate(errors)],
        )
        losses = [{"step": 10, "loss": 0.8}, {"step": 20, "loss": 0.3}] if name != "base" else []
        record = {
            "utterances": 10,
            "hours": 1.0,
            "steps": 20 if losses else 0,
            "train_seconds": 5.0,
            "losses": losses,
            "gpu": "none",
            "git_commit": "abc",
        }
        (root / "records").mkdir(exist_ok=True)
        (root / "records" / f"{name}.json").write_text(json.dumps(record), encoding="utf-8")
    return spec


def test_finetune_results_on_a_small_experiment(tmp_path: Path) -> None:
    spec = write_experiment(tmp_path)
    report = finetune_results(spec, tmp_path / "runs", tmp_path / "records", n_resamples=200)

    every, unseen = report["subsets"]
    assert (every["name"], unseen["name"]) == ("voxpopuli", "voxpopuli_unseen")
    assert (every["n_utterances"], every["n_words"], every["n_speakers"]) == (8, 32, 4)
    assert (unseen["n_utterances"], unseen["n_speakers_in_pool"]) == (4, 0)

    systems = every["systems"]
    assert systems["base"]["wer"] == pytest.approx(4 / 32)
    # The seeds are pooled: (16 + 12) errors over 2 x 32 words
    assert systems["random"]["wer"] == pytest.approx(28 / 64)
    assert every["runs"]["random"] == {
        "random-s0": pytest.approx(16 / 32),
        "random-s1": pytest.approx(12 / 32),
    }
    assert systems["random"]["unit"] == "speaker"
    # Every error is a deletion at the end of the segment
    assert systems["naive_filter"]["edge_errors"] == systems["naive_filter"]["errors"] == 4

    [contrast] = every["contrasts"]
    assert (contrast["b"], contrast["a"]) == ("naive_filter", "random")
    assert contrast["delta"] == pytest.approx(4 / 64 - 28 / 64)
    assert contrast["p_holm"] == contrast["p_value"]  # a family of one
    assert [c["b"] for c in every["vs_base"]] == ["random", "naive_filter"]
    assert report["training"][0]["model"] == "base"
    assert report["training"][1]["last_loss"] == 0.3


def test_finetune_tables_render_from_the_saved_results(tmp_path: Path) -> None:
    spec = write_experiment(tmp_path)
    report = finetune_results(spec, tmp_path / "runs", tmp_path / "records", n_resamples=200)
    path = tmp_path / "finetune.json"
    save_results(report, path)
    # The same results computed again (another time, another commit) leave the file alone
    saved = path.read_text(encoding="utf-8")
    save_results({**report, "created": "later", "git_commit": "other"}, path)
    assert path.read_text(encoding="utf-8") == saved
    tables_spec = tmp_path / "tables.toml"
    tables_spec.write_text(
        "\n".join(
            f'[[table]]\nname = "{section}"\nkind = "finetune"\nfile = "{path.as_posix()}"\n'
            f'section = "{section}"\n'
            for section in ("training", "wer", "seeds", "contrasts", "vs_base", "edges", "address")
        ),
        encoding="utf-8",
    )
    tables = build_tables(tables_spec)
    assert "| `random-s0` | 10 | 1.00 | 20 | 5 s | 0.80 / 0.30 / 0.80 |" in tables["training"]
    assert "voxpopuli, speakers not in the pool (2)" in tables["wer"]
    assert "`naive_filter` - `random`" in tables["contrasts"]
    assert "base model" in tables["edges"]
    assert "| `random` (seeds 0, 1) | 0 / 0 |" in tables["address"]
    with pytest.raises(ValueError, match="unknown section"):
        finetune_table(report, "everything")
