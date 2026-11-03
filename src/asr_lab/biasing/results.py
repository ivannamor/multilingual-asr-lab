"""Results of experiment E9: lists of entities for Whisper (a prompt) and for CTC (hotwords).

For each test set and model, with the normalizer fixed in the spec, this computes:

- the entity recall, the WER (with its errors by type) and the CER of the run without a list and of
  the run with each list, with the widest bootstrap interval;
- the recall and the WER of each list against no list, as paired bootstraps with Holm's correction
  over all the comparisons of the test set: both models, every list, both metrics;
- how often the entities of a list are written more times than the reference says them, with the
  list and, for comparison, without it;
- (post hoc) the transcripts shorter than half their reference, added after seeing that a list moved
  Whisper's WER on VoxPopuli far more than its entity recall: Whisper sometimes writes "Grazie a
  tutti" for a whole speech, and a prompt changes how often it does;
- a check that CTC decoding without hotwords gives the dev WER of E4, whose language model it uses.

Recall is counted like an error rate (a missed entity is a substitution), so its intervals and
comparisons come from the same bootstrap as the WER, and the signs are turned for the tables.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from asr_lab.biasing.experiment import (
    NO_LIST,
    conditions_for,
    ctc_lm_name,
    e4_settings,
    whisper_run_dir,
)
from asr_lab.biasing.scoring import entity_counts, invented_entities
from asr_lab.data.manifest import read_manifest
from asr_lab.evaluation.comparison import LEVEL, comparison_family, headline_interval
from asr_lab.evaluation.metrics import (
    ErrorCounts,
    align,
    char_tokens,
    count_errors,
    error_rate,
    utterance_counts,
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
from asr_lab.lm.corpus import load_spec
from asr_lab.provenance import git_commit, utc_now

WHISPER = "whisper"
CTC = "ctc"
MODELS = (WHISPER, CTC)
MODEL_LABELS = {WHISPER: "Whisper large-v3-turbo", CTC: "XLS-R CTC + Europarl 5-gram"}
METRICS = ("recall", "wer")
SHORT_MIN_WORDS = 5
_N_EXAMPLES = 5


def is_short(reference: Sequence[str], hypothesis: Sequence[str]) -> bool:
    """Whether a transcript has fewer than half the words of its reference, of five words or more.

    >>> is_short("signor presidente onorevoli colleghi oggi".split(), ["grazie"])
    True
    >>> is_short("grazie presidente".split(), [])
    False
    """
    return len(reference) >= SHORT_MIN_WORDS and len(hypothesis) < len(reference) / 2


def bias_results(
    spec_path: Path,
    entities_path: Path,
    selection_path: Path,
    ctc_grid_path: Path,
    runs_dir: Path,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Score the test runs of the biasing spec; see the module docstring."""
    spec = load_spec(spec_path)
    entities_report = json.loads(entities_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    options = {"n_resamples": n_resamples, "seed": seed}
    test_sets = [
        _test_set_results(spec, entities_report, selection, test_set, runs_dir, options)
        for test_set in spec["test_sets"]
    ]
    return {
        "spec": spec_path.as_posix(),
        "models": list(MODELS),
        "level": LEVEL,
        **options,
        "selection": _selection_rows(spec, selection),
        "check": _ctc_check(spec, ctc_grid_path),
        "test_sets": test_sets,
        "created": utc_now(),
        "git_commit": git_commit(),
    }


def bias_table(report: dict[str, Any], section: str) -> str:
    """One Markdown table of the results: ``selection``, ``check``, ``results``, ``comparisons``,
    ``invented`` or ``short``."""
    builders = {
        "selection": _selection_table,
        "check": _check_table,
        "results": _results_table,
        "comparisons": _comparisons_table,
        "invented": _invented_table,
        "short": _short_table,
    }
    if section not in builders:
        raise ValueError(f"unknown section {section!r}; choose from {sorted(builders)}")
    return builders[section](report)


def _test_set_results(
    spec: dict[str, Any],
    entities_report: dict[str, Any],
    selection: dict[str, Any],
    test_set: dict[str, Any],
    runs_dir: Path,
    options: dict[str, int],
) -> dict[str, Any]:
    normalize = get_normalizer_by_name(test_set["normalizer"])
    manifest = Path(test_set["test"])
    found = entities_report["test_sets"][test_set["name"]]["test"]
    entity_tokens = {
        key: [word_tokens(normalize(entity)) for entity in value] for key, value in found.items()
    }
    entries = read_manifest(manifest)
    units = resampling_units(entries)
    models, family, short_comparisons = {}, [], []
    for model in MODELS:
        lists = {NO_LIST: {}, **_test_lists(spec, entities_report, selection, test_set, model)}
        scored = {
            condition: _score_run(
                _run_dir(spec, test_set, model, condition, runs_dir),
                manifest,
                normalize,
                entity_tokens,
            )
            for condition in lists
        }
        baseline = scored[NO_LIST]
        systems = {}
        for condition, listed in lists.items():
            run = scored[condition]
            given = [
                [word_tokens(normalize(entity)) for entity in listed.get(entry.utterance_id, [])]
                for entry in run["entries"]
            ]
            invented = _invented(run, given)
            systems[condition] = {
                **_summary(run, units, options),
                "invented": sum(invented.values()),
                "invented_without_list": sum(_invented(baseline, given).values()),
                "most_invented": [[entity, n] for entity, n in _commonest(invented)],
            }
            if condition != NO_LIST:
                for metric in METRICS:
                    key = "entities" if metric == "recall" else "words"
                    family.append((f"{model}/{condition}/{metric}", baseline[key], run[key]))
                short_comparisons.append(
                    (model, condition, _without_short(baseline, run, units, options))
                )
        models[model] = {"runs": {c: s["info"] for c, s in scored.items()}, "systems": systems}
    comparisons = comparison_family(
        [(NO_LIST, name, a, b) for name, a, b in family], units, options
    )
    return {
        "name": test_set["name"],
        "lang": test_set["lang"],
        "normalizer": test_set["normalizer"],
        "n_utterances": len(entries),
        "n_words": sum(len(tokens) for tokens in baseline["references"]),
        "n_entities": sum(counts.ref_length for counts in baseline["entities"]),
        "n_speakers": len({entry.speaker_id for entry in entries}),
        "models": models,
        "comparisons": [_comparison_row(result) for result in comparisons],
        "without_short": [
            {"model": model, "list": condition, **result}
            for model, condition, result in short_comparisons
        ],
    }


def _test_lists(
    spec: dict[str, Any],
    entities_report: dict[str, Any],
    selection: dict[str, Any],
    test_set: dict[str, Any],
    model: str,
) -> dict[str, dict[str, list[str]]]:
    """The lists given to a model on a test set: none for CTC if no weight was chosen on dev."""
    if model == WHISPER:
        size = selection["whisper"]["domain_size"] if test_set.get("has_domain_list") else None
        return conditions_for(spec, entities_report, test_set, "test", size)
    chosen = selection["ctc"][test_set["name"]]
    size = chosen["domain"]["size"] if chosen["domain"] else None
    lists = conditions_for(spec, entities_report, test_set, "test", size)
    if chosen["oracle"] is None:
        lists = {name: value for name, value in lists.items() if name.startswith("domain")}
    return lists


def _run_dir(
    spec: dict[str, Any], test_set: dict[str, Any], model: str, condition: str, runs_dir: Path
) -> Path:
    if model == WHISPER:
        if condition == NO_LIST:
            return Path(test_set["whisper_baseline"])
        return whisper_run_dir(runs_dir, condition, test_set, "test")
    if condition == NO_LIST:
        return runs_dir / f"e4-{ctc_lm_name(spec)}-{test_set['name']}"
    return runs_dir / f"e9-ctc-{condition}-{test_set['name']}"


def _score_run(
    run_dir: Path,
    manifest: Path,
    normalize: Any,
    entity_tokens: Mapping[str, list[list[str]]],
) -> dict[str, Any]:
    """Per-utterance counts of one run: words, characters, entities, and its tokens."""
    run = load_run(run_dir, manifest)
    pairs = normalized_pairs(run.entries, run.hypotheses, normalize)
    references = [word_tokens(text) for text in pairs.references]
    hypotheses = [word_tokens(text) for text in pairs.hypotheses]
    return {
        "entries": run.entries,
        "references": references,
        "hypotheses": hypotheses,
        "words": [
            count_errors(align(ref, hyp)) for ref, hyp in zip(references, hypotheses, strict=True)
        ],
        "chars": utterance_counts(pairs.references, pairs.hypotheses, char_tokens),
        "entities": [
            entity_counts(ref, hyp, entity_tokens[entry.utterance_id])
            for entry, ref, hyp in zip(run.entries, references, hypotheses, strict=True)
        ],
        "short": [is_short(ref, hyp) for ref, hyp in zip(references, hypotheses, strict=True)],
        "info": {
            "run_dir": run_dir.as_posix(),
            "model": run.info["model"],
            "model_config": run.info["model_config"],
            "git_commit": run.info["git_commit"],
            "rtf": run.info["rtf"],
        },
    }


def _summary(
    run: dict[str, Any], units: dict[str, list[str] | None], options: dict[str, int]
) -> dict[str, Any]:
    counts = sum(run["words"], ErrorCounts())
    wer = headline_interval(run["words"], units, options)
    missed = headline_interval(run["entities"], units, options)
    short = [i for i, value in enumerate(run["short"]) if value]
    return {
        "recall": 1 - missed["wer"],
        "recall_low": 1 - missed["high"],
        "recall_high": 1 - missed["low"],
        "wer": wer["wer"],
        "low": wer["low"],
        "high": wer["high"],
        "unit": wer["unit"],
        "cer": error_rate(sum(run["chars"], ErrorCounts())),
        "substitutions": counts.substitutions,
        "deletions": counts.deletions,
        "insertions": counts.insertions,
        "errors": counts.errors,
        "short": len(short),
        "short_errors": sum(run["words"][i].errors for i in short),
        "missed_entities": sum(run["entities"], ErrorCounts()).substitutions,
        "short_missed_entities": sum(run["entities"][i].substitutions for i in short),
    }


def _invented(run: dict[str, Any], given: Sequence[Sequence[Sequence[str]]]) -> Counter[str]:
    """Entities of the lists written more times than the references say them, by entity."""
    invented: Counter[str] = Counter()
    for ref, hyp, listed in zip(run["references"], run["hypotheses"], given, strict=True):
        for entity in {tuple(e) for e in listed}:
            invented[" ".join(entity)] += invented_entities(ref, hyp, [entity])
    return +invented  # without the entities never invented


def _commonest(counter: Counter[str]) -> list[tuple[str, int]]:
    """The commonest items; in a tie, in a fixed order, so that the results are reproducible."""
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:_N_EXAMPLES]


def _without_short(
    baseline: dict[str, Any],
    run: dict[str, Any],
    units: dict[str, list[str] | None],
    options: dict[str, int],
) -> dict[str, Any]:
    """WER against no list without the utterances whose transcript is short in either run.

    Post hoc and descriptive: outside the family of Holm's correction.
    """
    keep = [
        i
        for i, (a, b) in enumerate(zip(baseline["short"], run["short"], strict=True))
        if not a and not b
    ]
    kept_units = {
        unit: None if groups is None else [groups[i] for i in keep]
        for unit, groups in units.items()
    }
    a = [baseline["words"][i] for i in keep]
    b = [run["words"][i] for i in keep]
    result = comparison_family([(NO_LIST, "list", a, b)], kept_units, options)[0]
    return {
        "n_utterances": len(keep),
        "wer_without_list": error_rate(sum(a, ErrorCounts())),
        "wer_with_list": error_rate(sum(b, ErrorCounts())),
        "delta": result["delta"],
        "low": result["low"],
        "high": result["high"],
    }


def _comparison_row(result: dict[str, Any]) -> dict[str, Any]:
    """A comparison against no list, with recall as a gain: positive when the list finds more."""
    model, condition, metric = result["b"].split("/")
    row = {
        "model": model,
        "list": condition,
        "metric": metric,
        "p_value": result["p_value"],
        "p_holm": result["p_holm"],
        "significant": result["significant"],
        "unit": result["unit"],
    }
    if metric == "recall":
        # Adding 0.0 turns -0.0, the gain of a list that changes nothing, into 0.0
        return {
            **row,
            "delta": -result["delta"] + 0.0,
            "low": -result["high"] + 0.0,
            "high": -result["low"] + 0.0,
        }
    return {**row, "delta": result["delta"], "low": result["low"], "high": result["high"]}


def _selection_rows(spec: dict[str, Any], selection: dict[str, Any]) -> list[dict[str, Any]]:
    """The settings chosen on the dev sets, one row per model, test set and kind of list."""
    with_domain = {t["name"] for t in spec["test_sets"] if t.get("has_domain_list")}
    whisper = selection["whisper"]
    chosen = next(
        (cell for cell in whisper["cells"] if cell["size"] == whisper["domain_size"]), None
    )
    rows = [
        {
            "model": WHISPER,
            "test_set": name,
            "list": "domain",
            "setting": f"{chosen['size']} entities" if chosen else "no list",
            "baseline": whisper["baseline"],
            "chosen": chosen,
        }
        for name in sorted(with_domain)
    ]
    for name, result in selection["ctc"].items():
        for kind in ("oracle", "domain"):
            setting = result[kind]
            if kind == "domain" and name not in with_domain:
                continue
            if setting is None:
                description = "no list"
            elif kind == "domain":
                description = f"{setting['size']} entities, weight {setting['weight']:g}"
            else:
                description = f"weight {setting['weight']:g}"
            rows.append(
                {
                    "model": CTC,
                    "test_set": name,
                    "list": "oracle and distractors" if kind == "oracle" else kind,
                    "setting": description,
                    "baseline": result["baseline"],
                    "chosen": setting,
                }
            )
    return rows


def _ctc_check(spec: dict[str, Any], ctc_grid_path: Path) -> list[dict[str, Any]]:
    """CTC without hotwords on the dev sets, against the dev WER of the weights chosen in E4."""
    _, e4_grid = e4_settings(spec)
    grid = json.loads(ctc_grid_path.read_text(encoding="utf-8"))["test_sets"]
    name = ctc_lm_name(spec)
    return [
        {
            "test_set": test_set,
            "e4_dev_wer": e4_grid[test_set][name]["selected"]["wer"],
            "e9_dev_wer": result["baseline"]["wer"],
        }
        for test_set, result in grid.items()
    ]


def _list_label(condition: str) -> str:
    """Name of a list in a table.

    >>> _list_label("domain-50"), _list_label("none")
    ('Domain (50)', 'No list')
    """
    if condition.startswith("domain-"):
        return f"Domain ({condition.removeprefix('domain-')})"
    return {"none": "No list", "oracle": "Oracle", "distractors": "Distractors"}[condition]


def _selection_table(report: dict[str, Any]) -> str:
    lines = [
        "| Model | Dev set | List | Chosen | Dev WER: no list → list "
        "| Dev entity recall: no list → list |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["selection"]:
        base, chosen = row["baseline"], row["chosen"]
        wer = f"{base['wer']:.2%} → {chosen['wer']:.2%}" if chosen else f"{base['wer']:.2%}"
        recall = (
            f"{base['recall']:.1%} → {chosen['recall']:.1%}" if chosen else f"{base['recall']:.1%}"
        )
        lines.append(
            f"| {MODEL_LABELS[row['model']]} | {row['test_set']} | {row['list']} "
            f"| {row['setting']} | {wer} | {recall} |"
        )
    return "\n".join(lines)


def _check_table(report: dict[str, Any]) -> str:
    lines = [
        "| Dev set | Dev WER in E4 | Dev WER here, without hotwords |",
        "|---|---|---|",
    ]
    for row in report["check"]:
        lines.append(f"| {row['test_set']} | {row['e4_dev_wer']:.2%} | {row['e9_dev_wer']:.2%} |")
    return "\n".join(lines)


def _rows(report: dict[str, Any]) -> Any:
    """Every (test set, model, list, system) of the report, in order."""
    for test_set in report["test_sets"]:
        for model in report["models"]:
            for condition, system in test_set["models"][model]["systems"].items():
                yield test_set, model, condition, system


def _results_table(report: dict[str, Any]) -> str:
    lines = [
        "| Test set | Model | List | Entity recall [95% CI] | WER [95% CI] | S / D / I | CER |",
        "|---|---|---|---|---|---|---|",
    ]
    for test_set, model, condition, s in _rows(report):
        lines.append(
            f"| {test_set['name']} | {MODEL_LABELS[model]} | {_list_label(condition)} "
            f"| {s['recall']:.1%} [{s['recall_low']:.1%}, {s['recall_high']:.1%}] "
            f"| {s['wer']:.2%} [{s['low']:.2%}, {s['high']:.2%}] "
            f"| {s['substitutions']:,} / {s['deletions']:,} / {s['insertions']:,} "
            f"| {s['cer']:.2%} |"
        )
    sizes = ", ".join(
        f"{t['name']} {t['n_utterances']:,} utterances, {t['n_words']:,} words, "
        f"{t['n_entities']:,} entities (`{t['normalizer']}`)"
        for t in report["test_sets"]
    )
    return "\n".join([*lines, "", f"*Sizes: {sizes}.*"])


def _comparisons_table(report: dict[str, Any]) -> str:
    lines = [
        "| Test set | Model | List | Metric | List - no list [95% CI] | p | p (Holm) "
        "| Significant |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for test_set in report["test_sets"]:
        family = test_set["comparisons"]
        for c in family:
            # A p-value of 0 means that no resample crossed zero: below 1 / n_resamples
            p = _p_value(c["p_value"], 1 / report["n_resamples"])
            p_holm = _p_value(c["p_holm"], len(family) / report["n_resamples"])
            metric = "Entity recall" if c["metric"] == "recall" else "WER"
            lines.append(
                f"| {test_set['name']} | {MODEL_LABELS[c['model']]} | {_list_label(c['list'])} "
                f"| {metric} | {c['delta']:+.2%} [{c['low']:+.2%}, {c['high']:+.2%}] "
                f"| {p} | {p_holm} | {'yes' if c['significant'] else 'no'} |"
            )
    return "\n".join(lines)


def _p_value(value: float, bound: float) -> str:
    return f"< {bound:.3f}" if value == 0 else f"{value:.3f}"


def _invented_table(report: dict[str, Any]) -> str:
    lines = [
        "| Test set | Model | List | Written more often than said: with the list "
        "| Same entities, without the list | Commonest with the list (times) |",
        "|---|---|---|---|---|---|",
    ]
    for test_set, model, condition, s in _rows(report):
        if condition != NO_LIST:
            lines.append(
                f"| {test_set['name']} | {MODEL_LABELS[model]} | {_list_label(condition)} "
                f"| {s['invented']:,} | {s['invented_without_list']:,} "
                f"| {', '.join(f'{entity} ({n})' for entity, n in s['most_invented']) or '-'} |"
            )
    return "\n".join(lines)


def _short_table(report: dict[str, Any]) -> str:
    """Only the test sets and models with a short transcript in some run."""
    lines = [
        "| Test set | Model | List | Short transcripts: no list → list "
        "| Their share of the errors: no list → list "
        "| Entities missed in them: no list → list | Utterances kept "
        "| WER of the rest: no list → list | Difference [95% CI] |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for test_set in report["test_sets"]:
        for row in test_set["without_short"]:
            systems = test_set["models"][row["model"]]["systems"]
            base, s = systems[NO_LIST], systems[row["list"]]
            if not base["short"] and not s["short"]:
                continue
            lines.append(
                f"| {test_set['name']} | {MODEL_LABELS[row['model']]} | {_list_label(row['list'])} "
                f"| {base['short']} → {s['short']} "
                f"| {_share(base)} → {_share(s)} "
                f"| {base['short_missed_entities']} of {base['missed_entities']} "
                f"→ {s['short_missed_entities']} of {s['missed_entities']} "
                f"| {row['n_utterances']:,} "
                f"| {row['wer_without_list']:.2%} → {row['wer_with_list']:.2%} "
                f"| {row['delta']:+.2%} [{row['low']:+.2%}, {row['high']:+.2%}] |"
            )
    return "\n".join(lines)


def _share(system: dict[str, Any]) -> str:
    return f"{system['short_errors'] / system['errors']:.1%}" if system["errors"] else "0.0%"
