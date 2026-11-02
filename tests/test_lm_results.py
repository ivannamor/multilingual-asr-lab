"""Results of the language models of E4: scores, comparisons, recall, word changes, tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from asr_lab.data.manifest import ManifestEntry, write_manifest
from asr_lab.evaluation.comparison import save_results
from asr_lab.evaluation.metrics import ErrorCounts, align
from asr_lab.inference.base import Hypothesis
from asr_lab.inference.predictions import append_predictions
from asr_lab.lm.logprobs import METADATA_KEY
from asr_lab.lm.results import greedy_check, lm_results, lm_table, recall_counts, word_changes
from asr_lab.plots import plot_lm_grid
from asr_lab.reporting import build_tables

# Eight utterances by four speakers; the last two references have a digit
REFERENCES = ["uno due tre quattro"] * 6 + ["ho 2 gatti neri"] * 2
HYPOTHESES = {
    "greedy": ["uno due tre cuatro"] * 6 + ["ho due gatti neri"] * 2,  # a misspelling, the digits
    "aula-2": ["uno due tre quattro"] * 6 + ["ho due gatti neri"] * 2,  # repairs the misspelling
    "aula-3": ["uno due tre"] * 6 + ["ho due gatti neri"] * 2,  # drops the word it does not know
}
SECTIONS = ("wer", "comparisons", "no_digit", "weights", "vocabulary", "text")


def write_experiment(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    """A spec with one text source and two orders, its runs, and what the results read."""
    entries = [
        ManifestEntry(f"u{i}", f"u{i}.wav", 2.0, text, "it", f"s{i % 4}", "test")
        for i, text in enumerate(REFERENCES)
    ]
    manifest = root / "test.jsonl"
    write_manifest(manifest, entries)
    for system, texts in HYPOTHESES.items():
        run = root / "runs" / ("e1b-ctc-set" if system == "greedy" else f"e4-{system}-set")
        run.mkdir(parents=True)
        append_predictions(
            run / "predictions.jsonl",
            [Hypothesis(f"u{i}", text) for i, text in enumerate(texts)],
        )
        info = {
            "model": system,
            "model_config": {"model": system},
            "git_commit": "abc",
            "rtf": 0.01,
            "processing_seconds": 1.0,
            "audio_seconds": 100.0,
        }
        (run / "run.json").write_text(json.dumps(info), encoding="utf-8")
    spec = root / "lm.toml"
    spec.write_text(
        f"""
[[corpora]]
name = "aula"

[text]
orders = [2, 3]

[[test_sets]]
name = "set"
lang = "it"
normalizer = "it-v1"
dev = "{manifest.as_posix()}"
test = "{manifest.as_posix()}"
greedy_run = "{(root / "runs" / "e1b-ctc-set").as_posix()}"

[analysis]
comparisons = [["aula-2", "greedy"], ["aula-3", "greedy"], ["aula-3", "aula-2"]]
""",
        encoding="utf-8",
    )
    models = root / "models"
    models.mkdir()
    (models / "aula-it.txt").write_text("uno due tre\nho due gatti neri\n", encoding="utf-8")
    cell = {"alpha": 0.5, "beta": 1.0, "wer": 0.1}
    selection = root / "grid.json"
    selection.write_text(
        json.dumps(
            {
                "test_sets": {
                    "set": {
                        name: {"grid": [cell], "selected": cell} for name in ("aula-2", "aula-3")
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    text = root / "text.json"
    row = {
        "corpus": "europarl",
        "lang": "it",
        "read": 10,
        "not_writable": 2,
        "in_dev_or_test": 1,
        "distinct_in_dev_or_test": 1,
        "duplicates": 1,
        "available_words": 20,
        "sentences": 2,
        "words": 7,
    }
    text.write_text(json.dumps({"text": [row]}), encoding="utf-8")
    return spec, selection, text, models, root / "runs"


@pytest.fixture
def report(tmp_path: Path) -> dict:
    return lm_results(*write_experiment(tmp_path), n_resamples=50)


def test_recall_separates_the_words_of_the_vocabulary() -> None:
    counts = recall_counts(align(["il", "gatto", "zibibbo"], ["il", "cane"]), {"il", "gatto"})
    assert counts["known"] == ErrorCounts(hits=1, substitutions=1)
    assert counts["unknown"] == ErrorCounts(substitutions=1)


def test_recall_ignores_inserted_words() -> None:
    counts = recall_counts(align(["il"], ["il", "gatto"]), {"il"})
    assert counts["known"] == ErrorCounts(hits=1)
    assert counts["unknown"] == ErrorCounts()


def test_word_changes_keep_what_was_written_in_place_of_the_word() -> None:
    reference = ["il", "gatto", "dorme"]
    repaired, broken = word_changes(
        align(reference, ["il", "gato", "dorme"]), align(reference, ["il", "gatto"])
    )
    assert repaired == {("gato", "gatto"): 1}
    assert broken == {("dorme", None): 1}


def test_word_changes_ignore_words_right_or_wrong_in_both() -> None:
    reference = ["il", "gatto"]
    repaired, broken = word_changes(
        align(reference, ["il", "gato"]), align(reference, ["il", "gatta"])
    )
    assert not repaired
    assert not broken


def test_scores_and_comparisons_of_every_system(report: dict) -> None:
    subset = report["test_sets"][0]["all"]
    # Six misspelled words and two digits written in words, on 32 reference words
    assert subset["systems"]["greedy"]["wer"] == pytest.approx(8 / 32)
    assert subset["systems"]["aula-2"]["wer"] == pytest.approx(2 / 32)
    assert subset["systems"]["aula-3"]["deletions"] == 6
    pairs = [(c["b"], c["a"]) for c in subset["comparisons"]]
    assert pairs == [("aula-2", "greedy"), ("aula-3", "greedy"), ("aula-3", "aula-2")]
    assert subset["comparisons"][0]["delta"] == pytest.approx(-6 / 32)
    assert all(c["p_holm"] >= c["p_value"] for c in subset["comparisons"])


def test_the_subset_without_digits_drops_their_references(report: dict) -> None:
    test_set = report["test_sets"][0]
    assert test_set["n_with_digits"] == 2
    assert test_set["no_digit"]["n_utterances"] == 6
    assert test_set["no_digit"]["systems"]["aula-2"]["wer"] == 0.0


def test_recall_shows_what_a_language_model_does_to_unknown_words(report: dict) -> None:
    groups = report["test_sets"][0]["vocabulary"]["aula-3"]
    assert groups["known"]["n_words"] == 18  # "uno due tre", six times
    assert groups["unknown"]["n_words"] == 6  # "quattro"
    assert groups["unknown"]["greedy_recall"] == 0.0
    assert groups["unknown"]["lm_recall"] == 0.0
    assert report["test_sets"][0]["vocabulary"]["aula-2"]["unknown"]["lm_recall"] == 1.0
    assert report["test_sets"][0]["vocabulary"]["aula-2"]["unknown"]["difference"] == 1.0


def test_changes_count_and_list_the_words(report: dict) -> None:
    changes = report["test_sets"][0]["changes"]["aula-2"]
    assert changes["repaired"] == 6
    assert changes["broken"] == 0
    assert changes["most_repaired"] == [["cuatro", "quattro", 6]]


def test_the_weights_come_from_the_selection(report: dict) -> None:
    assert report["test_sets"][0]["weights"]["aula-2"]["beta"] == 1.0


@pytest.mark.parametrize("section", SECTIONS)
def test_every_table_has_a_row_per_item(report: dict, section: str) -> None:
    table = lm_table(_with_known_corpus(report), section)
    assert table.startswith("|")
    assert table.count("\n") >= 2


def test_changes_table_shows_only_the_highest_order() -> None:
    changes = {"repaired": 3, "broken": 1, "most_repaired": [[None, "il", 3]], "most_broken": []}
    report = {
        "test_sets": [
            {"name": "set", "changes": {"europarl-3": changes, "europarl-5": changes}},
        ]
    }
    table = lm_table(report, "changes")
    assert "Europarl 5-gram" in table
    assert "Europarl 3-gram" not in table
    assert "(nothing) → il (3)" in table


def test_unknown_section_is_refused(report: dict) -> None:
    with pytest.raises(ValueError, match="unknown section"):
        lm_table(report, "nope")


def test_results_render_through_the_tables_spec(tmp_path: Path, report: dict) -> None:
    results = tmp_path / "results.json"
    save_results(_with_known_corpus(report), results)
    spec = tmp_path / "tables.toml"
    table = f'name = "e4-wer"\nkind = "lm"\nfile = "{results.as_posix()}"\nsection = "wer"\n'
    spec.write_text(f"[[table]]\n{table}", encoding="utf-8")
    assert "Europarl 2-gram" in build_tables(spec)["e4-wer"]


def test_greedy_check_separates_spaces_and_special_tokens(tmp_path: Path) -> None:
    spec, *_ = write_experiment(tmp_path)
    run = tmp_path / "runs" / "e1b-ctc-set"
    (run / "predictions.jsonl").unlink()
    # The third token is a, the fourth b. E1b wrote the same text, once with a space more, once with
    # a special token, and once with something else.
    originals = ["a", "a  b", "a<unk>", "b"] + ["a"] * 4
    decoded = ["a", "a b", "a", "a"] + ["a"] * 4
    append_predictions(
        run / "predictions.jsonl", [Hypothesis(f"u{i}", t) for i, t in enumerate(originals)]
    )
    tokens = ["<pad>", "|", "a", "b"]
    frames = {
        "a": [2],
        "a b": [2, 1, 3],
    }
    arrays = {
        f"u{i}": np.log(np.eye(4)[frames[text]] * 0.96 + 0.01).astype(np.float16)
        for i, text in enumerate(decoded)
    }
    metadata = np.array(json.dumps({"tokens": tokens}))
    folder = tmp_path / "logprobs"
    folder.mkdir()
    for split in ("dev", "test"):
        np.savez_compressed(folder / f"set-{split}.npz", **arrays, **{METADATA_KEY: metadata})
    row = greedy_check(spec, folder)["test_sets"][0]
    assert row["different"] == 3
    assert row["different_beyond_spaces"] == 2
    assert row["different_beyond_special_tokens"] == 1


def test_plot_lm_grid_writes_a_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    cells = [
        {"alpha": alpha, "beta": beta, "wer": 0.1 + alpha / 10 + beta / 100}
        for alpha in (0.1, 0.5)
        for beta in (0.0, 1.0)
    ]
    models = {"europarl-5": {"grid": cells, "selected": cells[0]}}
    grid = {"test_sets": {"one": models, "two": models}}
    path = plot_lm_grid(grid, "europarl-5", tmp_path / "figures" / "grid.png")
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def _with_known_corpus(report: dict) -> dict:
    """The tables name the real sources; the small experiment calls its source "aula"."""
    text = json.dumps(report).replace("aula-", "europarl-")
    return json.loads(text)
