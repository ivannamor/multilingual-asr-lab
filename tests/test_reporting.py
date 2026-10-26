"""Tests for generated result tables and the results document."""

import json
from pathlib import Path

import pytest

from asr_lab import cli
from asr_lab.reporting import comparisons_table, render_document, render_results, scores_table

SCORES = {
    "normalizer": ["it-v1"],
    "n_utterances": 865,
    "wer": {
        "estimate": 0.0746,
        "n_reference_tokens": 21255,
        "substitutions": 1223,
        "deletions": 169,
        "insertions": 193,
        "confidence_intervals": {"sentence_id": {"low": 0.0681, "high": 0.0808}},
        "headline_unit": "sentence_id",
    },
    "cer": {
        "estimate": 0.0211,
        "confidence_intervals": {"utterance": {"low": 0.0187, "high": 0.0237}},
        "headline_unit": "utterance",
    },
}
COMPARISON = {
    "a": {"rate": 0.0746},
    "b": {"rate": 0.0715},
    "delta": -0.0031,
    "confidence_intervals": {
        "sentence_id": {"low": -0.0048, "high": -0.0015, "prob_b_better": 1.0}
    },
    "headline_unit": "sentence_id",
    "significant": True,
}


def test_scores_table() -> None:
    table = scores_table([("Whisper-small", SCORES)])
    assert table.splitlines()[2] == (
        "| Whisper-small | `it-v1` | 865 / 21,255 | 7.46% [6.81%, 8.08%] "
        "| 1,223 / 169 / 193 | 2.11% [1.87%, 2.37%] |"
    )


def test_comparisons_table() -> None:
    table = comparisons_table([("it-v2 vs it-v1", COMPARISON)])
    assert table.splitlines()[2] == (
        "| it-v2 vs it-v1 | 7.46% | 7.15% | -0.31% [-0.48%, -0.15%] | 100.0% | yes |"
    )


def test_render_document_replaces_only_between_markers() -> None:
    document = "Intro\n<!-- BEGIN t1 -->\nold table\n<!-- END t1 -->\nOutro\n"
    rendered = render_document(document, {"t1": "| new |"})
    assert rendered == "Intro\n<!-- BEGIN t1 -->\n| new |\n<!-- END t1 -->\nOutro\n"
    assert render_document(rendered, {"t1": "| new |"}) == rendered  # idempotent


def test_render_document_requires_markers() -> None:
    with pytest.raises(ValueError, match="No <!-- BEGIN/END missing --> markers"):
        render_document("no markers here", {"missing": "x"})


def test_render_document_can_skip_missing_markers() -> None:
    document = "<!-- BEGIN t1 -->\n<!-- END t1 -->\n"
    rendered = render_document(document, {"t1": "| a |", "t2": "| b |"}, allow_missing=True)
    assert rendered == "<!-- BEGIN t1 -->\n| a |\n<!-- END t1 -->\n"
    with pytest.raises(ValueError, match="None of the tables has markers"):
        render_document("nothing", {"t1": "x"}, allow_missing=True)


def _spec(tmp_path: Path) -> Path:
    scores = tmp_path / "scores.json"
    scores.write_text(json.dumps(SCORES), encoding="utf-8")
    comparison = tmp_path / "cmp.json"
    comparison.write_text(json.dumps(COMPARISON), encoding="utf-8")
    spec = tmp_path / "tables.toml"
    spec.write_text(
        f"""
[[table]]
name = "scores"
kind = "scores"
rows = [{{ label = "Whisper-small", file = "{scores.as_posix()}" }}]

[[table]]
name = "cmp"
kind = "comparisons"
rows = [{{ label = "it-v2 vs it-v1", file = "{comparison.as_posix()}" }}]
""",
        encoding="utf-8",
    )
    return spec


def test_render_results_and_check(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    document = tmp_path / "RESULTS.md"
    document.write_text(
        "# R\n<!-- BEGIN scores -->\n<!-- END scores -->\n<!-- BEGIN cmp -->\n<!-- END cmp -->\n",
        encoding="utf-8",
    )
    assert render_results([spec], document, check=True)  # out of date, not written
    assert "7.46%" not in document.read_text(encoding="utf-8")
    assert render_results([spec], document)  # written
    text = document.read_text(encoding="utf-8")
    assert "7.46% [6.81%, 8.08%]" in text
    assert "-0.31% [-0.48%, -0.15%]" in text
    assert not render_results([spec], document, check=True)  # now up to date


def test_cli_render_results_check_fails_when_outdated(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    document = tmp_path / "RESULTS.md"
    document.write_text(
        "<!-- BEGIN scores -->\n<!-- END scores -->\n<!-- BEGIN cmp -->\n<!-- END cmp -->\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        cli.main(["render-results", "--spec", str(spec), "--document", str(document), "--check"])
    cli.main(["render-results", "--spec", str(spec), "--document", str(document)])
    cli.main(["render-results", "--spec", str(spec), "--document", str(document), "--check"])


def test_unknown_table_kind(tmp_path: Path) -> None:
    spec = tmp_path / "bad.toml"
    spec.write_text('[[table]]\nname = "x"\nkind = "pie"\nrows = []\n', encoding="utf-8")
    document = tmp_path / "RESULTS.md"
    document.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown table kind 'pie'"):
        render_results([spec], document)
