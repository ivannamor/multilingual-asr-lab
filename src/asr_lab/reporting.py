"""Generate the tables of the results document from saved JSON results.

Tables are never typed by hand: the document contains marker pairs such as::

    <!-- BEGIN e3-normalizers -->
    ...generated table...
    <!-- END e3-normalizers -->

and a TOML spec says which table goes between each pair and which files it is
built from. ``render_results`` rewrites only the text between markers, so the
prose around them is preserved; with ``check=True`` it only reports whether the
document is up to date, which the test suite uses to keep numbers in sync.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def scores_table(rows: Sequence[tuple[str, dict[str, Any]]]) -> str:
    """Markdown table of WER and CER with their headline confidence intervals."""
    lines = [
        "| System | Normalizer | Utterances / words | WER [95% CI] | S / D / I | CER [95% CI] |",
        "|---|---|---|---|---|---|",
    ]
    for label, scores in rows:
        wer, cer = scores["wer"], scores["cer"]
        lines.append(
            f"| {label} | `{', '.join(scores['normalizer'])}` "
            f"| {scores['n_utterances']:,} / {wer['n_reference_tokens']:,} "
            f"| {_rate_with_ci(wer)} "
            f"| {wer['substitutions']:,} / {wer['deletions']:,} / {wer['insertions']:,} "
            f"| {_rate_with_ci(cer)} |"
        )
    return "\n".join(lines)


def comparisons_table(rows: Sequence[tuple[str, dict[str, Any]]]) -> str:
    """Markdown table of paired comparisons (B - A), with interval and verdict."""
    lines = [
        "| Comparison | A | B | B - A [95% CI] | B better in | Significant |",
        "|---|---|---|---|---|---|",
    ]
    for label, result in rows:
        ci = result["confidence_intervals"][result["headline_unit"]]
        lines.append(
            f"| {label} | {result['a']['rate']:.2%} | {result['b']['rate']:.2%} "
            f"| {result['delta']:+.2%} [{ci['low']:+.2%}, {ci['high']:+.2%}] "
            f"| {ci['prob_b_better']:.1%} | {'yes' if result['significant'] else 'no'} |"
        )
    return "\n".join(lines)


def selection_table(selection: dict[str, Any]) -> str:
    """Markdown table of a decoding sweep: WER per language (best in bold), speed, eligibility.

    Speeds are relative to the first configuration, the baseline.
    """
    languages = selection["languages"]
    configs = selection["configs"]
    baseline_seconds = next(iter(configs.values()))["processing_seconds"]
    lines = [
        "| Configuration | Change from the baseline | "
        + " | ".join(f"WER {lang}" for lang in languages)
        + " | RTF | Time vs baseline | Eligible |",
        "|---|---|" + "---|" * len(languages) + "---|---|---|",
    ]
    for name, result in configs.items():
        cells = [
            f"**{result['wer'][lang]:.2%}**"
            if selection["best"][lang] == name
            else f"{result['wer'][lang]:.2%}"
            for lang in languages
        ]
        change = ", ".join(f"`{key} = {value}`" for key, value in result["options"].items())
        eligible = ("yes" if result["eligible"] else "no") if result["selectable"] else "n/a"
        label = f"**{name}** (selected)" if name == selection["selected"] else name
        relative = result["processing_seconds"] / baseline_seconds - 1
        lines.append(
            f"| {label} | {change or 'none'} | {' | '.join(cells)} "
            f"| {result['rtf']:.3f} | {relative:+.1%} | {eligible} |"
        )
    return "\n".join(lines)


def closing_phrases_table(
    columns: Sequence[str], rows: Sequence[tuple[str, list[dict[str, Any]]]]
) -> str:
    """Markdown table of utterances with an invented closing phrase, from error reports."""
    lines = [
        "| Utterances with an invented closing phrase | " + " | ".join(columns) + " |",
        "|---|" + "---|" * len(columns),
    ]
    for label, reports in rows:
        counts = [str(len(report["invented_closing_phrases"])) for report in reports]
        lines.append(f"| {label} | " + " | ".join(counts) + " |")
    return "\n".join(lines)


def build_tables(spec_path: Path) -> dict[str, str]:
    """Build every table described in a TOML spec; returns marker name -> Markdown.

    Table kinds: ``scores`` and ``comparisons`` (one JSON file per row),
    ``closing-phrases`` (one error report per row and column) and ``selection``
    (a single selection file).
    """
    spec = tomllib.loads(spec_path.read_text(encoding="utf-8"))
    row_builders = {"scores": scores_table, "comparisons": comparisons_table}
    tables = {}
    for table in spec.get("table", []):
        kind = table["kind"]
        if kind in row_builders:
            rows = [(row["label"], _read_json(row["file"])) for row in table["rows"]]
            tables[table["name"]] = row_builders[kind](rows)
        elif kind == "closing-phrases":
            rows = [
                (row["label"], [_read_json(file) for file in row["files"]]) for row in table["rows"]
            ]
            tables[table["name"]] = closing_phrases_table(table["columns"], rows)
        elif kind == "selection":
            tables[table["name"]] = selection_table(_read_json(table["file"]))
        else:
            raise ValueError(f"{spec_path}: unknown table kind {kind!r}")
    return tables


def render_document(document: str, tables: dict[str, str], *, allow_missing: bool = False) -> str:
    """Replace the text between each pair of markers with its table.

    With ``allow_missing`` tables without markers are skipped (a document such as
    the README shows only some tables), but at least one table must be found.

    Raises:
        ValueError: if a table has no markers (or, with ``allow_missing``, if none has).
    """
    found = 0
    for name, table in tables.items():
        pattern = re.compile(
            rf"(<!-- BEGIN {re.escape(name)} -->\n).*?(<!-- END {re.escape(name)} -->)",
            re.DOTALL,
        )
        if not pattern.search(document):
            if allow_missing:
                continue
            raise ValueError(f"No <!-- BEGIN/END {name} --> markers in the document")
        found += 1
        # ``table=table`` binds the current value (a lambda would otherwise see the loop
        # variable's last value if it were called later).
        document = pattern.sub(
            lambda match, table=table: f"{match[1]}{table}\n{match[2]}", document
        )
    if tables and not found:
        raise ValueError("None of the tables has markers in the document")
    return document


def render_results(
    spec_paths: Sequence[Path],
    document_path: Path,
    *,
    check: bool = False,
    allow_missing: bool = False,
) -> bool:
    """Regenerate the tables of ``document_path``; return True if it was (or would be) changed.

    With ``check=True`` the file is not written.
    """
    tables: dict[str, str] = {}
    for spec_path in spec_paths:
        tables.update(build_tables(spec_path))
    current = document_path.read_text(encoding="utf-8")
    rendered = render_document(current, tables, allow_missing=allow_missing)
    if rendered != current and not check:
        document_path.write_text(rendered, encoding="utf-8", newline="\n")
    return rendered != current


def _rate_with_ci(report: dict[str, Any]) -> str:
    ci = report["confidence_intervals"][report["headline_unit"]]
    return f"{report['estimate']:.2%} [{ci['low']:.2%}, {ci['high']:.2%}]"


def _read_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
