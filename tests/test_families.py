"""Results of the model families of E1b: subsets, comparisons, post hoc counts, tables."""

from __future__ import annotations

import json
from pathlib import Path

from asr_lab.data.manifest import ManifestEntry, write_manifest
from asr_lab.evaluation.comparison import save_results
from asr_lab.evaluation.families import (
    MODELS,
    families_table,
    family_results,
    family_run_dir,
    has_digit,
    is_runaway,
)
from asr_lab.inference.base import Hypothesis
from asr_lab.inference.predictions import append_predictions
from asr_lab.reporting import build_tables


def test_has_digit_reads_the_raw_reference() -> None:
    assert has_digit("alle 8 di sera")
    assert not has_digit("alle otto di sera")
    assert not has_digit("")


def test_run_dirs_come_from_the_spec_for_whisper_only() -> None:
    test_set = {"name": "fleurs-it", "whisper_small": "e1/small"}
    assert family_run_dir(test_set, "whisper_small", Path("runs")) == Path("e1/small")
    assert family_run_dir(test_set, "tdt", Path("runs")) == Path("runs/e1b-tdt-fleurs-it")


def test_runaway_is_more_words_per_second_than_speech() -> None:
    assert is_runaway("non " * 41, 2.0)
    assert not is_runaway("non " * 40, 2.0)
    assert not is_runaway("", 0.0)


# Eight utterances by four speakers; the last two references have a digit
REFERENCES = ["uno due tre quattro"] * 6 + ["ho 2 gatti neri"] * 2
HYPOTHESES = {
    "whisper_small": ["uno due tre cinque"] * 8,  # one substitution everywhere
    "whisper_turbo": ["uno due tre quattro"] * 6 + ["ho 2 gatti neri"] * 2,  # perfect
    "ctc": ["uno due tre quattro"] * 6 + ["ho due gatti neri"] * 2,  # only the digits are wrong
    "tdt": [""] + ["uno due tre quattro grazie"] + ["uno due tre quattro"] * 6,  # empty, insertion
}


def write_experiment(root: Path) -> Path:
    entries = [
        ManifestEntry(f"u{i}", f"u{i}.wav", 2.0, text, "it", f"s{i % 4}", "test")
        for i, text in enumerate(REFERENCES)
    ]
    manifest = root / "test.jsonl"
    write_manifest(manifest, entries)
    test_set = {}
    for model, texts in HYPOTHESES.items():
        run = root / "runs" / f"e1b-{model}-set"
        run.mkdir(parents=True)
        append_predictions(
            run / "predictions.jsonl",
            [Hypothesis(f"u{i}", text) for i, text in enumerate(texts)],
        )
        info = {
            "model": model,
            "model_config": {"model": model},
            "git_commit": "abc",
            "rtf": 0.01,
            "processing_seconds": 1.0,
            "audio_seconds": 100.0,
        }
        (run / "run.json").write_text(json.dumps(info), encoding="utf-8")
        if model.startswith("whisper"):
            test_set[model] = run.as_posix()
    whisper = "\n".join(f'{name} = "{path}"' for name, path in test_set.items())
    spec = root / "models.toml"
    spec.write_text(
        f"""
[[test_sets]]
name = "set"
manifest = "{manifest.as_posix()}"
lang = "it"
normalizer = "it-v1"
{whisper}

[analysis]
comparisons = [["ctc", "whisper_small"], ["tdt", "ctc"]]

[analysis.parameters]
whisper_small = 1000000
whisper_turbo = 2000000
ctc = 3000000
tdt = 4000000
""",
        encoding="utf-8",
    )
    return spec


def test_family_results(tmp_path: Path) -> None:
    report = family_results(write_experiment(tmp_path), tmp_path / "runs", n_resamples=50)
    test_set = report["test_sets"][0]
    assert test_set["n_with_digits"] == 2
    systems = test_set["all"]["systems"]
    assert systems["whisper_turbo"]["wer"] == 0
    assert systems["whisper_small"]["substitutions"] == 14
    assert systems["ctc"]["substitutions"] == 2
    assert systems["tdt"]["empty"] == 1
    assert systems["tdt"]["deletions"] >= 4
    assert systems["whisper_small"]["cer"] > 0
    assert set(systems) == set(MODELS)


def test_no_digit_subset_removes_the_notation_errors(tmp_path: Path) -> None:
    report = family_results(write_experiment(tmp_path), tmp_path / "runs", n_resamples=50)
    subset = report["test_sets"][0]["no_digit"]
    assert subset["n_utterances"] == 6
    assert subset["n_words"] == 24
    assert subset["systems"]["ctc"]["wer"] == 0


def test_comparisons_are_oriented_and_corrected(tmp_path: Path) -> None:
    report = family_results(write_experiment(tmp_path), tmp_path / "runs", n_resamples=50)
    family = report["test_sets"][0]["all"]["comparisons"]
    assert [(c["b"], c["a"]) for c in family] == [("ctc", "whisper_small"), ("tdt", "ctc")]
    first = family[0]
    assert first["delta"] < 0  # the CTC model has fewer errors than Whisper-small
    assert all(c["p_holm"] >= c["p_value"] for c in family)


def test_silent_utterances_compare_the_others(tmp_path: Path) -> None:
    report = family_results(write_experiment(tmp_path), tmp_path / "runs", n_resamples=50)
    silent = report["test_sets"][0]["silent"]
    assert silent["n_utterances"] == 1
    assert silent["n_words"] == 4
    assert silent["mean_duration"] == 2.0
    assert silent["systems"]["whisper_turbo"]["hits"] == 4
    assert silent["systems"]["tdt"]["errors"] == 4


def test_a_test_set_without_empty_transcripts_has_no_silent_rows(tmp_path: Path) -> None:
    spec = write_experiment(tmp_path)
    run = tmp_path / "runs" / "e1b-tdt-set"
    (run / "predictions.jsonl").unlink()
    append_predictions(
        run / "predictions.jsonl",
        [Hypothesis(f"u{i}", "uno due tre quattro") for i in range(8)],
    )
    report = family_results(spec, tmp_path / "runs", n_resamples=50)
    assert report["test_sets"][0]["silent"]["n_utterances"] == 0
    assert report["test_sets"][0]["silent"]["mean_duration"] == 0.0
    assert families_table(report, "silent").count("\n") == 1  # the header only


def test_runaway_utterances_are_excluded_for_every_model(tmp_path: Path) -> None:
    spec = write_experiment(tmp_path)
    run = tmp_path / "runs" / "e1b-tdt-set"
    (run / "predictions.jsonl").unlink()
    append_predictions(
        run / "predictions.jsonl",
        [Hypothesis("u0", "non " * 100)]
        + [Hypothesis(f"u{i}", "uno due tre quattro") for i in range(1, 8)],
    )
    report = family_results(spec, tmp_path / "runs", n_resamples=50)
    test_set = report["test_sets"][0]
    assert [r["utterance_id"] for r in test_set["runaway"]] == ["u0"]
    assert test_set["runaway"][0]["words"]["tdt"] == 100
    assert test_set["without_runaway"]["n_utterances"] == 7
    # The utterance leaves the score of the others too, so the comparison stays paired
    assert test_set["without_runaway"]["systems"]["whisper_small"]["errors"] == 13
    assert "100 / 4 / 2.0" in families_table(report, "runaway")


def test_tables_are_built_from_the_saved_results(tmp_path: Path) -> None:
    report = family_results(write_experiment(tmp_path), tmp_path / "runs", n_resamples=50)
    results = tmp_path / "families.json"
    save_results(report, results)
    sections = ["wer", "comparisons", "no_digit", "speed", "errors", "silent", "runaway"]
    tables = "\n".join(
        f'[[table]]\nname = "{s}"\nkind = "families"\n'
        f'file = "{results.as_posix()}"\nsection = "{s}"'
        for s in sections
    )
    spec = tmp_path / "tables.toml"
    spec.write_text(tables, encoding="utf-8")
    built = build_tables(spec)
    assert "Parakeet TDT" in built["wer"]
    assert "XLS-R (CTC) - Whisper-small" in built["comparisons"]
    assert "6 of 8" in built["no_digit"]
    assert "4 M" in built["speed"]
    assert "1 |" in built["errors"]
    assert "1 / 4 / 2.0 s" in built["silent"]
