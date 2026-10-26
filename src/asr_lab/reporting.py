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


def build_tables(spec_path: Path) -> dict[str, str]:
    """Build every table described in a TOML spec; returns marker name -> Markdown."""
    spec = tomllib.loads(spec_path.read_text(encoding="utf-8"))
    builders = {"scores": scores_table, "comparisons": comparisons_table}
    tables = {}
    for table in spec.get("table", []):
        if table["kind"] not in builders:
            raise ValueError(f"{spec_path}: unknown table kind {table['kind']!r}")
        rows = [
            (row["label"], json.loads(Path(row["file"]).read_text(encoding="utf-8")))
            for row in table["rows"]
        ]
        tables[table["name"]] = builders[table["kind"]](rows)
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
