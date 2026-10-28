"""Tests for error analysis."""

import json
from pathlib import Path

from asr_lab import cli
from asr_lab.data.manifest import ManifestEntry, write_manifest
from asr_lab.evaluation.error_analysis import (
    analyze,
    analyze_run,
    confusion_pairs,
    deleted_words,
    error_report,
    format_alignment,
    format_errors,
    inserted_words,
    involves_digits_or_symbols,
    worst_utterances,
)
from asr_lab.evaluation.metrics import align
from asr_lab.inference.base import Hypothesis
from asr_lab.inference.predictions import append_predictions
from asr_lab.provenance import file_sha256


def _entry(i: int, text: str) -> ManifestEntry:
    return ManifestEntry(f"u{i}", f"audio/u{i}.wav", 1.0, text, "it", f"s{i}", "test")


ENTRIES = [
    _entry(0, "Il gatto dorme."),
    _entry(1, "Il gatto mangia."),
    _entry(2, "Costa 20 euro."),
    _entry(3, "Oggi piove molto forte a Roma."),
]
HYPOTHESES = [
    Hypothesis("u0", "il cane dorme"),  # gatto -> cane
    Hypothesis("u1", "il cane mangia"),  # gatto -> cane
    Hypothesis("u2", "costa venti euro"),  # 20 -> venti
    Hypothesis("u3", "oggi piove forte a roma grazie"),  # molto deleted, grazie inserted
]


def test_analyze_normalizes_and_aligns_every_utterance() -> None:
    utterances = analyze(ENTRIES, HYPOTHESES)
    assert [u.utterance_id for u in utterances] == ["u0", "u1", "u2", "u3"]
    assert utterances[0].reference == "il gatto dorme"
    assert utterances[0].counts.substitutions == 1
    assert utterances[3].wer == 2 / 6


def test_counters() -> None:
    utterances = analyze(ENTRIES, HYPOTHESES)
    assert confusion_pairs(utterances) == {("gatto", "cane"): 2, ("20", "venti"): 1}
    assert deleted_words(utterances) == {"molto": 1}
    assert inserted_words(utterances) == {"grazie": 1}


def test_worst_utterances_by_errors_then_wer() -> None:
    utterances = analyze(ENTRIES, HYPOTHESES)
    # u3 has 2 errors; u0, u1 and u2 have 1 error out of 3 words each, so the id breaks the tie.
    assert [u.utterance_id for u in worst_utterances(utterances, 2)] == ["u3", "u0"]


def test_digits_and_symbols_are_flagged() -> None:
    assert involves_digits_or_symbols(("20", "venti"))
    assert involves_digits_or_symbols(("$", "dollari"))
    assert not involves_digits_or_symbols(("gatto", "cane"))
    assert not involves_digits_or_symbols(("dell'anno", "dellanno"))


def test_format_alignment_marks_every_operation() -> None:
    text = format_alignment(align(["a", "bb", "c"], ["a", "x", "c", "d"]))
    assert text.splitlines() == [
        "REF: a bb c *",
        "HYP: a x  c d",
        "       S    I",
    ]


def test_error_report_totals_and_rankings() -> None:
    report = error_report(analyze(ENTRIES, HYPOTHESES), top=1, worst=1)
    assert (report["substitutions"], report["deletions"], report["insertions"]) == (3, 1, 1)
    assert report["distinct_confusion_pairs"] == 2
    assert report["substitutions_with_digits_or_symbols"] == 1
    assert report["top_confusions"] == [{"reference": "gatto", "hypothesis": "cane", "count": 2}]
    assert report["worst_utterances"][0]["utterance_id"] == "u3"
    json.dumps(report)  # must be serializable


def _run_dir(tmp_path: Path) -> Path:
    manifest = tmp_path / "data" / "test.jsonl"
    write_manifest(manifest, ENTRIES)
    run_dir = tmp_path / "run"
    append_predictions(run_dir / "predictions.jsonl", HYPOTHESES)
    run = {
        "model": "fake",
        "manifest": manifest.as_posix(),
        "manifest_sha256": file_sha256(manifest),
    }
    (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
    return run_dir


def test_analyze_run_writes_an_errors_file(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    utterances, report, path = analyze_run(run_dir)
    assert path == run_dir / "errors_it-v1.json"
    assert json.loads(path.read_text(encoding="utf-8")) == report
    assert report["model"] == "fake"
    assert report["invented_closing_phrases"] == [{"utterance_id": "u3", "phrase": "grazie"}]
    assert "gatto -> cane" in format_errors(utterances, report)


def test_cli_errors(tmp_path: Path, capsys) -> None:
    run_dir = _run_dir(tmp_path)
    cli.main(["errors", "--run", str(run_dir), "--worst", "1"])
    output = capsys.readouterr().out
    assert "S=3 D=1 I=1" in output
    assert "1 substitutions involve digits or symbols" in output
    assert "u3: 2 errors / 6 words" in output
    assert "Invented closing phrases: 1 utterance(s), 'grazie' (1)" in output
    assert "Errors:" in output
