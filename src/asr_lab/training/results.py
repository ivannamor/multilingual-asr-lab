"""Results of the fine-tuned models of experiment E7, from their transcriptions of the test sets.

Every model, the base model included, transcribed the same test sets with the same decoding. For
each test set (and, when some test speakers are also in the training pool, for the test speakers
that are not), this computes:

- the WER of the base model and of every strategy, the errors of a strategy's seeds summed per
  utterance, so that a strategy is judged on all its training runs at once;
- the WER of every single training run, to show how much a new seed alone moves the result;
- the contrasts between strategies fixed in the spec (``[B, A]``: B - A, negative when B is
  better), as paired bootstraps with Holm's correction;
- every strategy against the base model: the gain of fine-tuning in-domain, and the forgetting
  out-of-domain;
- (post hoc) how many errors sit at the edges of the segments, before the first or after the last
  word that the hypothesis gets right, where wrong segment boundaries put them;
- (post hoc) how many hypotheses contain a parliamentary form of address ("signor presidente",
  "onorevoli colleghi") that their reference does not: the training domain written into speech
  that never had it.

As in the rest of the project, an interval is computed for every meaningful resampling unit and the
widest one is reported.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from asr_lab.data.manifest import read_manifest
from asr_lab.data.voxpopuli import load_data_spec
from asr_lab.evaluation.metrics import (
    AlignedPair,
    EditOp,
    ErrorCounts,
    align,
    count_errors,
    error_rate,
    word_tokens,
)
from asr_lab.evaluation.normalize import get_normalizer_by_name
from asr_lab.evaluation.scoring import (
    DEFAULT_N_RESAMPLES,
    DEFAULT_SEED,
    load_run,
    normalized_pairs,
    resampling_units,
)
from asr_lab.evaluation.stats import bootstrap_ci, holm, paired_bootstrap
from asr_lab.provenance import git_commit, utc_now
from asr_lab.training.selection import load_selection_spec

BASE = "base"
LEVEL = 0.95
# Post hoc: the opening of a speech in the European Parliament, matched in normalized text
ADDRESS = re.compile(r"\b(signora? presidente|onorevoli colleghi)\b")


def model_name(strategy: str, seed: int) -> str:
    """Name of a fine-tuned model (``base`` for the base model), as in its run directories.

    >>> model_name("naive_filter", 2)
    'naive-filter-s2'
    """
    return BASE if strategy == BASE else f"{strategy.replace('_', '-')}-s{seed}"


def run_dir(runs_dir: Path, name: str, test_set: str) -> Path:
    """Run directory of a model on a test set, e.g. ``runs/e7-random-s0-voxpopuli``."""
    return runs_dir / f"e7-{name}-{test_set}"


def edge_errors(alignment: Sequence[AlignedPair]) -> ErrorCounts:
    """The errors before the first match and after the last one (all of them without a match).

    When a segment's boundaries are wrong, the audio and the transcript disagree at the start or
    at the end: the model writes words the transcript does not have, or misses words that are not
    in the audio. Those errors fall outside the span that the hypothesis gets right.

    >>> pairs = align("a b c d".split(), "x a b c".split())
    >>> edge_errors(pairs)
    ErrorCounts(hits=0, substitutions=0, deletions=1, insertions=1)
    """
    matches = [i for i, pair in enumerate(alignment) if pair.op == EditOp.MATCH]
    if not matches:
        return count_errors(alignment)
    return count_errors([*alignment[: matches[0]], *alignment[matches[-1] + 1 :]])


def invented_address(reference: str, hypothesis: str) -> bool:
    """True if the hypothesis has a parliamentary form of address that the reference has not.

    Both texts must be normalized the same way.

    >>> invented_address("la città di barcellona", "signor presidente la città di barcellona")
    True
    >>> invented_address("signor presidente grazie", "signor presidente grazie")
    False
    """
    return bool(ADDRESS.search(hypothesis)) and not ADDRESS.search(reference)


def sum_counts(per_run: Sequence[Sequence[ErrorCounts]]) -> list[ErrorCounts]:
    """Add up the counts of several runs on the same utterances, utterance by utterance.

    >>> sum_counts([[ErrorCounts(hits=1)], [ErrorCounts(deletions=1)]])
    [ErrorCounts(hits=1, substitutions=0, deletions=1, insertions=0)]
    """
    return [sum(counts, ErrorCounts()) for counts in zip(*per_run, strict=True)]


def finetune_results(
    spec_path: Path,
    runs_dir: Path,
    records_dir: Path,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Score every model of a fine-tuning spec on its test sets; see the module docstring.

    Raises:
        ValueError: if the runs of a test set do not cover the same utterances.
    """
    spec = tomllib.loads(spec_path.read_text(encoding="utf-8"))
    selection = load_selection_spec(spec_path)
    evaluation = spec["evaluation"]
    normalizer = get_normalizer_by_name(evaluation["normalizer"])
    contrasts = [tuple(pair) for pair in spec["analysis"]["contrasts"]]
    pool = read_manifest(load_data_spec(selection.quality_spec).manifest)
    pool_speakers = {entry.speaker_id for entry in pool}
    names = {
        strategy: [model_name(strategy, s) for s in selection.seeds]
        for strategy in selection.strategies
    }
    models = [BASE, *(name for runs in names.values() for name in runs)]
    options = {"n_resamples": n_resamples, "seed": seed}

    subsets = []
    for test_set in evaluation["test_sets"]:
        scored = _score_test_set(runs_dir, test_set, models, normalizer)
        entries = scored["entries"]
        unseen = [i for i, entry in enumerate(entries) if entry.speaker_id not in pool_speakers]
        masks = {test_set: list(range(len(entries)))}
        if len(unseen) < len(entries):
            masks[f"{test_set}_unseen"] = unseen
        for subset, keep in masks.items():
            subsets.append(
                _subset(subset, test_set, keep, scored, names, contrasts, pool_speakers, options)
            )
    return {
        "spec": spec_path.as_posix(),
        "normalizer": normalizer.name,
        "strategies": list(selection.strategies),
        "seeds": list(selection.seeds),
        "contrasts": [list(pair) for pair in contrasts],
        "level": LEVEL,
        **options,
        "training": [_training(records_dir, name) for name in models],
        "subsets": subsets,
        "created": utc_now(),
        "git_commit": git_commit(),
    }


def save_results(report: dict[str, Any], path: Path) -> None:
    """Write the results as JSON (UTF-8, LF).

    A file with the same results is left as it is (with its original ``created`` and
    ``git_commit``), so running the experiment again does not modify committed results.
    """
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if _results_only(previous) == _results_only(json.loads(text)):
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _results_only(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key not in ("created", "git_commit")}


def finetune_table(report: dict[str, Any], section: str) -> str:
    """One Markdown table of the results: ``training``, ``wer``, ``seeds``, ``contrasts``,
    ``vs_base``, ``edges`` or ``address``."""
    builders = {
        "training": _training_table,
        "wer": _wer_table,
        "seeds": _seeds_table,
        "contrasts": _contrasts_table,
        "vs_base": _vs_base_table,
        "edges": _edges_table,
        "address": _address_table,
    }
    if section not in builders:
        raise ValueError(f"unknown section {section!r}; choose from {sorted(builders)}")
    return builders[section](report)


def _score_test_set(
    runs_dir: Path, test_set: str, models: Sequence[str], normalizer: Any
) -> dict[str, Any]:
    """Per-utterance counts and edge errors of every model on one test set."""
    entries, counts, edges, address = None, {}, {}, {}
    for name in models:
        run = load_run(run_dir(runs_dir, name, test_set))
        ids = [entry.utterance_id for entry in run.entries]
        if entries is None:
            entries = run.entries
        elif ids != [entry.utterance_id for entry in entries]:
            raise ValueError(f"{name} on {test_set} does not cover the same utterances")
        pairs = normalized_pairs(run.entries, run.hypotheses, normalizer)
        alignments = [
            align(word_tokens(ref), word_tokens(hyp))
            for ref, hyp in zip(pairs.references, pairs.hypotheses, strict=True)
        ]
        counts[name] = [count_errors(alignment) for alignment in alignments]
        edges[name] = [edge_errors(alignment) for alignment in alignments]
        address[name] = [
            invented_address(ref, hyp)
            for ref, hyp in zip(pairs.references, pairs.hypotheses, strict=True)
        ]
    return {"entries": entries, "counts": counts, "edges": edges, "address": address}


def _subset(
    subset: str,
    test_set: str,
    keep: Sequence[int],
    scored: dict[str, Any],
    names: dict[str, list[str]],
    contrasts: Sequence[tuple[str, str]],
    pool_speakers: set[str],
    options: dict[str, int],
) -> dict[str, Any]:
    entries = [scored["entries"][i] for i in keep]
    units = resampling_units(entries)

    def pick(per_utterance: Sequence[ErrorCounts]) -> list[ErrorCounts]:
        return [per_utterance[i] for i in keep]

    counts = {name: pick(values) for name, values in scored["counts"].items()}
    edges = {name: pick(values) for name, values in scored["edges"].items()}
    systems = {BASE: counts[BASE]}
    systems |= {s: sum_counts([counts[n] for n in runs]) for s, runs in names.items()}
    system_edges = {BASE: edges[BASE]}
    system_edges |= {s: sum_counts([edges[n] for n in runs]) for s, runs in names.items()}

    speakers = {entry.speaker_id for entry in entries}
    return {
        "name": subset,
        "test_set": test_set,
        "unseen_only": subset != test_set,
        "n_utterances": len(entries),
        "n_words": sum(c.ref_length for c in counts[BASE]),
        "n_speakers": len(speakers),
        "n_speakers_in_pool": len(speakers & pool_speakers),
        "systems": {
            name: {**_interval(values, units, options), **_composition(values, system_edges[name])}
            for name, values in systems.items()
        },
        "runs": {
            strategy: {name: error_rate(sum(counts[name], ErrorCounts())) for name in runs}
            for strategy, runs in names.items()
        },
        "invented_address": {
            name: sum(values[i] for i in keep) for name, values in scored["address"].items()
        },
        "contrasts": _family(
            [(a, b, systems[a], systems[b]) for b, a in contrasts], units, options
        ),
        "vs_base": _family([(BASE, s, systems[BASE], systems[s]) for s in names], units, options),
    }


def _interval(
    counts: Sequence[ErrorCounts], units: dict[str, list[str] | None], options: dict[str, int]
) -> dict[str, Any]:
    intervals = {
        unit: bootstrap_ci(counts, groups=groups, level=LEVEL, **options)
        for unit, groups in units.items()
    }
    unit = max(intervals, key=lambda u: intervals[u].high - intervals[u].low)
    ci = intervals[unit]
    return {
        "wer": ci.estimate,
        "low": ci.low,
        "high": ci.high,
        "unit": unit,
        "n_units": ci.n_units,
    }


def _composition(counts: Sequence[ErrorCounts], edges: Sequence[ErrorCounts]) -> dict[str, Any]:
    total, at_edges = sum(counts, ErrorCounts()), sum(edges, ErrorCounts())
    return {
        "substitutions": total.substitutions,
        "deletions": total.deletions,
        "insertions": total.insertions,
        "errors": total.errors,
        "edge_errors": at_edges.errors,
        "edge_deletions_insertions": at_edges.deletions + at_edges.insertions,
    }


def _family(
    pairs: Sequence[tuple[str, str, Sequence[ErrorCounts], Sequence[ErrorCounts]]],
    units: dict[str, list[str] | None],
    options: dict[str, int],
) -> list[dict[str, Any]]:
    """Paired bootstraps of B against A for a family of comparisons, with Holm's correction."""
    results = []
    for a, b, counts_a, counts_b in pairs:
        by_unit = {
            unit: paired_bootstrap(counts_a, counts_b, groups=groups, level=LEVEL, **options)
            for unit, groups in units.items()
        }
        unit = max(by_unit, key=lambda u: by_unit[u].high - by_unit[u].low)
        result = by_unit[unit]
        results.append(
            {
                "a": a,
                "b": b,
                "delta": result.delta,
                "low": result.low,
                "high": result.high,
                "prob_b_better": result.prob_b_better,
                "p_value": result.p_value,
                "unit": unit,
                "n_units": result.n_units,
            }
        )
    for result, adjusted in zip(results, holm([r["p_value"] for r in results]), strict=True):
        result["p_holm"] = adjusted
        result["significant"] = adjusted < 1 - LEVEL
    return results


def _training(records_dir: Path, name: str) -> dict[str, Any]:
    record = json.loads((records_dir / f"{name}.json").read_text(encoding="utf-8"))
    losses = [point["loss"] for point in record["losses"]]
    return {
        "model": name,
        "utterances": record["utterances"],
        "hours": record["hours"],
        "steps": record["steps"],
        "train_seconds": record["train_seconds"],
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "max_loss": max(losses) if losses else None,
        "gpu": record["gpu"],
        "git_commit": record["git_commit"],
    }


def _model_label(report: dict[str, Any], name: str) -> str:
    return "base model" if name == BASE else f"`{name}` ({len(report['seeds'])} seeds)"


def _subset_label(subset: dict[str, Any]) -> str:
    if subset["unseen_only"]:
        return f"{subset['test_set']}, speakers not in the pool ({subset['n_speakers']})"
    if subset["n_speakers_in_pool"]:
        return f"{subset['test_set']}, all speakers ({subset['n_speakers']})"
    return subset["test_set"]


def _training_table(report: dict[str, Any]) -> str:
    lines = [
        "| Model | Utterances | Hours | Steps | Training time "
        "| Loss per 10 steps: first / last / highest |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["training"]:
        if row["model"] == BASE:
            continue
        lines.append(
            f"| `{row['model']}` | {row['utterances']:,} | {row['hours']:.2f} | {row['steps']} "
            f"| {row['train_seconds']:.0f} s | {row['first_loss']:.2f} / {row['last_loss']:.2f} "
            f"/ {row['max_loss']:.2f} |"
        )
    return "\n".join(lines)


def _wer_table(report: dict[str, Any]) -> str:
    subsets = report["subsets"]
    lines = [
        "| Model | " + " | ".join(_subset_label(s) for s in subsets) + " |",
        "|---|" + "---|" * len(subsets),
    ]
    for name in [BASE, *report["strategies"]]:
        label = _model_label(report, name)
        cells = [
            f"{s['systems'][name]['wer']:.2%} [{s['systems'][name]['low']:.2%}, "
            f"{s['systems'][name]['high']:.2%}]"
            for s in subsets
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    words = [f"{s['n_utterances']:,} / {s['n_words']:,}" for s in subsets]
    lines.append("| *utterances / words* | " + " | ".join(words) + " |")
    return "\n".join(lines)


def _seeds_table(report: dict[str, Any]) -> str:
    subsets = report["subsets"]
    lines = [
        "| Strategy | " + " | ".join(_subset_label(s) for s in subsets) + " |",
        "|---|" + "---|" * len(subsets),
    ]
    for strategy in report["strategies"]:
        cells = []
        for subset in subsets:
            rates = list(subset["runs"][strategy].values())
            cells.append(
                " / ".join(f"{rate:.2%}" for rate in rates)
                + f" (range {max(rates) - min(rates):.2%})"
            )
        lines.append(f"| `{strategy}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _comparison_rows(report: dict[str, Any], key: str) -> list[str]:
    rows = []
    for subset in report["subsets"]:
        family = subset[key]
        for c in family:
            # A p-value of 0 means that no resample crossed zero: below 1 / n_resamples
            p = _p_value(c["p_value"], 1 / report["n_resamples"])
            p_holm = _p_value(c["p_holm"], len(family) / report["n_resamples"])
            rows.append(
                f"| {_subset_label(subset)} | `{c['b']}` - `{c['a']}` "
                f"| {c['delta']:+.2%} [{c['low']:+.2%}, {c['high']:+.2%}] | {p} | {p_holm} "
                f"| {'yes' if c['significant'] else 'no'} |"
            )
    return rows


def _p_value(value: float, bound: float) -> str:
    return f"< {bound:.3f}" if value == 0 else f"{value:.3f}"


def _comparison_header() -> list[str]:
    return [
        "| Test set | B - A | WER difference [95% CI] | p | p (Holm) | Significant |",
        "|---|---|---|---|---|---|",
    ]


def _contrasts_table(report: dict[str, Any]) -> str:
    return "\n".join(_comparison_header() + _comparison_rows(report, "contrasts"))


def _vs_base_table(report: dict[str, Any]) -> str:
    return "\n".join(_comparison_header() + _comparison_rows(report, "vs_base"))


def _edges_table(report: dict[str, Any]) -> str:
    """Errors per training run (a strategy's seeds averaged), at the edges and inside."""
    lines = [
        "| Test set | Model | Errors per run | S / D / I | At the segment edges "
        "| Inside the matched span |",
        "|---|---|---|---|---|---|",
    ]
    for subset in report["subsets"]:
        if subset["unseen_only"]:
            continue
        for name in [BASE, *report["strategies"]]:
            s = subset["systems"][name]
            runs = 1 if name == BASE else len(report["seeds"])
            edges = s["edge_errors"] / runs
            lines.append(
                f"| {subset['test_set']} | {_model_label(report, name)} "
                f"| {s['errors'] / runs:,.0f} "
                f"| {s['substitutions'] / runs:,.0f} / {s['deletions'] / runs:,.0f} "
                f"/ {s['insertions'] / runs:,.0f} "
                f"| {edges:,.0f} ({s['edge_errors'] / s['errors']:.0%}) "
                f"| {s['errors'] / runs - edges:,.0f} |"
            )
    return "\n".join(lines)


def _address_table(report: dict[str, Any]) -> str:
    """Utterances with an invented parliamentary form of address, per training run."""
    subsets = [subset for subset in report["subsets"] if not subset["unseen_only"]]
    lines = [
        "| Model | "
        + " | ".join(f"{s['test_set']} ({s['n_utterances']:,})" for s in subsets)
        + " |",
        "|---|" + "---|" * len(subsets),
    ]
    lines.append(
        "| base model | " + " | ".join(str(s["invented_address"][BASE]) for s in subsets) + " |"
    )
    for strategy in report["strategies"]:
        runs = [model_name(strategy, seed) for seed in report["seeds"]]
        cells = [" / ".join(str(s["invented_address"][run]) for run in runs) for s in subsets]
        lines.append(
            f"| `{strategy}` (seeds {', '.join(map(str, report['seeds']))}) | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines)
