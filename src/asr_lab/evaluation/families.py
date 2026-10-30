"""Results of experiment E1b: Whisper, CTC and Transducer models on the same test sets.

For each test set this computes, with the normalizer fixed in the spec:

- the WER of every model, with its widest bootstrap interval, its errors by type, its CER and its
  speed;
- the comparisons fixed in the spec (``[B, A]``: B - A, negative when B is better), as paired
  bootstraps with Holm's correction within the test set;
- the same on the utterances whose reference has no digit: the CTC vocabularies have no digits, so
  on the others they are penalized for a difference of notation that the normalizers do not always
  remove;
- (post hoc) the errors at the edges of the segments, the invented closing phrases and the empty
  transcripts of every model, and how the others do on the utterances where Parakeet writes nothing;
- (post hoc) the same results without the runaway transcripts, those written faster than anyone
  speaks: Transformers' TDT decoding has no limit on the tokens written at one audio frame, so a
  model that keeps predicting a duration of zero repeats itself until the length limit.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from asr_lab.evaluation.comparison import LEVEL, comparison_family, headline_interval
from asr_lab.evaluation.error_analysis import edge_errors
from asr_lab.evaluation.hallucination import invented_closing_phrase
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
from asr_lab.provenance import git_commit, utc_now

MODELS = ("whisper_small", "whisper_turbo", "ctc", "tdt")
MODEL_LABELS = {
    "whisper_small": "Whisper-small",
    "whisper_turbo": "Whisper-large-v3-turbo",
    "ctc": "XLS-R (CTC)",
    "tdt": "Parakeet TDT",
}
_DIGIT = re.compile(r"\d")
# Post hoc: well above the fastest normal transcript of every run (5.6 words per second); the two
# runaway transcripts of the experiment are at 63 and 125
RUNAWAY_WORDS_PER_SECOND = 20.0


def has_digit(text: str) -> bool:
    """True if a raw reference contains a digit.

    >>> has_digit("alle 8 di sera"), has_digit("alle otto di sera")
    (True, False)
    """
    return bool(_DIGIT.search(text))


def is_runaway(hypothesis: str, duration: float) -> bool:
    """True if a transcript has more words per second of audio than speech can have.

    >>> is_runaway("non " * 300, 5.0), is_runaway("non " * 20, 5.0)
    (True, False)
    """
    return len(word_tokens(hypothesis)) > RUNAWAY_WORDS_PER_SECOND * duration


def family_run_dir(test_set: dict[str, Any], model: str, runs_dir: Path) -> Path:
    """Run directory of a model on a test set: Whisper's come from the spec.

    >>> family_run_dir({"name": "fleurs-it"}, "ctc", Path("runs")).as_posix()
    'runs/e1b-ctc-fleurs-it'
    """
    if model in test_set:
        return Path(test_set[model])
    return runs_dir / f"e1b-{model}-{test_set['name']}"


def family_results(
    spec_path: Path,
    runs_dir: Path,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Score the runs of a model-family spec; see the module docstring.

    Raises:
        ValueError: if the runs of a test set do not cover the same utterances.
    """
    spec = tomllib.loads(spec_path.read_text(encoding="utf-8"))
    comparisons = [tuple(pair) for pair in spec["analysis"]["comparisons"]]
    options = {"n_resamples": n_resamples, "seed": seed}
    test_sets = []
    for test_set in spec["test_sets"]:
        scored = _score_test_set(test_set, runs_dir)
        entries = scored["entries"]
        no_digit = [i for i, entry in enumerate(entries) if not has_digit(entry.text)]
        runaway = [i for i in range(len(entries)) if any(scored["runaway"][m][i] for m in MODELS)]
        kept = [i for i in range(len(entries)) if i not in set(runaway)]
        test_sets.append(
            {
                "name": test_set["name"],
                "lang": test_set["lang"],
                "normalizer": test_set["normalizer"],
                "runs": scored["runs"],
                "n_with_digits": len(entries) - len(no_digit),
                "all": _subset(range(len(entries)), scored, comparisons, options),
                "no_digit": _subset(no_digit, scored, comparisons, options),
                "runaway": [
                    {
                        "utterance_id": entries[i].utterance_id,
                        "duration": entries[i].duration,
                        "reference_words": scored["words"]["tdt"][i].ref_length,
                        "words": {m: scored["n_hyp_words"][m][i] for m in MODELS},
                    }
                    for i in runaway
                ],
                "without_runaway": _subset(kept, scored, comparisons, options),
                "silent": _silent(scored),
            }
        )
    return {
        "spec": spec_path.as_posix(),
        "models": list(MODELS),
        "parameters": spec["analysis"]["parameters"],
        "comparisons": [list(pair) for pair in comparisons],
        "level": LEVEL,
        **options,
        "test_sets": test_sets,
        "created": utc_now(),
        "git_commit": git_commit(),
    }


def families_table(report: dict[str, Any], section: str) -> str:
    """One Markdown table of the results: ``wer``, ``comparisons``, ``no_digit``, ``speed``,
    ``errors``, ``silent`` or ``runaway``."""
    builders = {
        "wer": _wer_table,
        "comparisons": _comparisons_table,
        "no_digit": _no_digit_table,
        "speed": _speed_table,
        "errors": _errors_table,
        "silent": _silent_table,
        "runaway": _runaway_table,
    }
    if section not in builders:
        raise ValueError(f"unknown section {section!r}; choose from {sorted(builders)}")
    return builders[section](report)


def _score_test_set(test_set: dict[str, Any], runs_dir: Path) -> dict[str, Any]:
    """Per-utterance results of every model on one test set."""
    normalizer = get_normalizer_by_name(test_set["normalizer"])
    manifest = Path(test_set["manifest"])
    entries, runs = None, {}
    words, chars, edges, closing, empty, runaway, n_hyp_words = {}, {}, {}, {}, {}, {}, {}
    for model in MODELS:
        run = load_run(family_run_dir(test_set, model, runs_dir), manifest)
        if entries is None:
            entries = run.entries
        pairs = normalized_pairs(entries, run.hypotheses, normalizer)
        alignments = [
            align(word_tokens(ref), word_tokens(hyp))
            for ref, hyp in zip(pairs.references, pairs.hypotheses, strict=True)
        ]
        words[model] = [count_errors(alignment) for alignment in alignments]
        chars[model] = utterance_counts(pairs.references, pairs.hypotheses, char_tokens)
        edges[model] = [edge_errors(alignment) for alignment in alignments]
        closing[model] = [
            invented_closing_phrase(ref, hyp, test_set["lang"]) is not None
            for ref, hyp in zip(pairs.references, pairs.hypotheses, strict=True)
        ]
        empty[model] = [not hyp for hyp in pairs.hypotheses]
        runaway[model] = [
            is_runaway(hyp, entry.duration)
            for hyp, entry in zip(pairs.hypotheses, entries, strict=True)
        ]
        n_hyp_words[model] = [len(word_tokens(hyp)) for hyp in pairs.hypotheses]
        runs[model] = {
            "run_dir": family_run_dir(test_set, model, runs_dir).as_posix(),
            "model": run.info["model"],
            "model_config": run.info["model_config"],
            "git_commit": run.info["git_commit"],
            "rtf": run.info["rtf"],
            "processing_seconds": run.info["processing_seconds"],
            "audio_seconds": run.info["audio_seconds"],
        }
    return {
        "entries": entries,
        "words": words,
        "chars": chars,
        "edges": edges,
        "closing": closing,
        "empty": empty,
        "runaway": runaway,
        "n_hyp_words": n_hyp_words,
        "runs": runs,
    }


def _subset(
    keep: Sequence[int],
    scored: dict[str, Any],
    comparisons: Sequence[tuple[str, str]],
    options: dict[str, int],
) -> dict[str, Any]:
    entries = [scored["entries"][i] for i in keep]
    units = resampling_units(entries)
    words = {model: [scored["words"][model][i] for i in keep] for model in MODELS}

    def total(kind: str, model: str) -> ErrorCounts:
        return sum((scored[kind][model][i] for i in keep), ErrorCounts())

    systems = {}
    for model in MODELS:
        counts = total("words", model)
        systems[model] = {
            **headline_interval(words[model], units, options),
            "cer": error_rate(total("chars", model)),
            "substitutions": counts.substitutions,
            "deletions": counts.deletions,
            "insertions": counts.insertions,
            "errors": counts.errors,
            "edge_errors": total("edges", model).errors,
            "invented_closing_phrases": sum(scored["closing"][model][i] for i in keep),
            "empty": sum(scored["empty"][model][i] for i in keep),
        }
    return {
        "n_utterances": len(entries),
        "n_words": sum(counts.ref_length for counts in words[MODELS[0]]),
        "n_speakers": len({entry.speaker_id for entry in entries}),
        "systems": systems,
        "comparisons": comparison_family(
            [(a, b, words[a], words[b]) for b, a in comparisons], units, options
        ),
    }


def _silent(scored: dict[str, Any]) -> dict[str, Any]:
    """How every model does on the utterances where the Transducer writes nothing."""
    keep = [i for i, empty in enumerate(scored["empty"]["tdt"]) if empty]
    systems = {}
    for model in MODELS:
        counts = sum((scored["words"][model][i] for i in keep), ErrorCounts())
        systems[model] = {
            "errors": counts.errors,
            "hits": counts.hits,
            "insertions": counts.insertions,
            "closing": sum(scored["closing"][model][i] for i in keep),
        }
    durations = [scored["entries"][i].duration for i in keep]
    return {
        "n_utterances": len(keep),
        "n_words": sum(scored["words"]["tdt"][i].ref_length for i in keep),
        "mean_duration": sum(durations) / len(durations) if durations else 0.0,
        "systems": systems,
    }


def _label(model: str) -> str:
    return MODEL_LABELS[model]


def _wer_table(report: dict[str, Any]) -> str:
    lines = [
        "| Test set | Utterances / words | Model | WER [95% CI] | S / D / I | CER |",
        "|---|---|---|---|---|---|",
    ]
    for test_set in report["test_sets"]:
        subset = test_set["all"]
        size = f"{subset['n_utterances']:,} / {subset['n_words']:,}"
        for model in report["models"]:
            s = subset["systems"][model]
            lines.append(
                f"| {test_set['name']} (`{test_set['normalizer']}`) | {size} | {_label(model)} "
                f"| {s['wer']:.2%} [{s['low']:.2%}, {s['high']:.2%}] "
                f"| {s['substitutions']:,} / {s['deletions']:,} / {s['insertions']:,} "
                f"| {s['cer']:.2%} |"
            )
    return "\n".join(lines)


def _comparison_rows(
    report: dict[str, Any], key: str, test_sets: Sequence[dict[str, Any]] | None = None
) -> list[str]:
    lines = [
        "| Test set | B - A | WER difference [95% CI] | p | p (Holm) | Significant |",
        "|---|---|---|---|---|---|",
    ]
    for test_set in report["test_sets"] if test_sets is None else test_sets:
        family = test_set[key]["comparisons"]
        for c in family:
            # A p-value of 0 means that no resample crossed zero: below 1 / n_resamples
            p = _p_value(c["p_value"], 1 / report["n_resamples"])
            p_holm = _p_value(c["p_holm"], len(family) / report["n_resamples"])
            lines.append(
                f"| {test_set['name']} | {_label(c['b'])} - {_label(c['a'])} "
                f"| {c['delta']:+.2%} [{c['low']:+.2%}, {c['high']:+.2%}] | {p} | {p_holm} "
                f"| {'yes' if c['significant'] else 'no'} |"
            )
    return lines


def _p_value(value: float, bound: float) -> str:
    return f"< {bound:.3f}" if value == 0 else f"{value:.3f}"


def _comparisons_table(report: dict[str, Any]) -> str:
    return "\n".join(_comparison_rows(report, "all"))


def _no_digit_table(report: dict[str, Any]) -> str:
    """WER without the references with digits, then the comparisons repeated on them."""
    test_sets = report["test_sets"]
    lines = [
        "| Model | " + " | ".join(t["name"] for t in test_sets) + " |",
        "|---|" + "---|" * len(test_sets),
    ]
    for model in report["models"]:
        cells = [
            f"{t['no_digit']['systems'][model]['wer']:.2%} "
            f"[{t['no_digit']['systems'][model]['low']:.2%}, "
            f"{t['no_digit']['systems'][model]['high']:.2%}]"
            for t in test_sets
        ]
        lines.append(f"| {_label(model)} | " + " | ".join(cells) + " |")
    sizes = [
        f"{t['no_digit']['n_utterances']:,} of {t['all']['n_utterances']:,}" for t in test_sets
    ]
    lines.append("| *utterances kept* | " + " | ".join(sizes) + " |")
    return "\n".join([*lines, "", *_comparison_rows(report, "no_digit")])


def _speed_table(report: dict[str, Any]) -> str:
    test_sets = report["test_sets"]
    lines = [
        "| Model | Parameters | " + " | ".join(f"RTF {t['name']}" for t in test_sets) + " |",
        "|---|---|" + "---|" * len(test_sets),
    ]
    for model in report["models"]:
        cells = [f"{t['runs'][model]['rtf']:.4f}" for t in test_sets]
        lines.append(
            f"| {_label(model)} | {report['parameters'][model] / 1e6:.0f} M | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines)


def _errors_table(report: dict[str, Any]) -> str:
    lines = [
        "| Test set | Model | Errors | At the segment edges | Invented closing phrases "
        "| Empty transcripts |",
        "|---|---|---|---|---|---|",
    ]
    for test_set in report["test_sets"]:
        for model in report["models"]:
            s = test_set["all"]["systems"][model]
            lines.append(
                f"| {test_set['name']} | {_label(model)} | {s['errors']:,} "
                f"| {s['edge_errors']:,} ({s['edge_errors'] / max(s['errors'], 1):.0%}) "
                f"| {s['invented_closing_phrases']} | {s['empty']} |"
            )
    return "\n".join(lines)


def _silent_table(report: dict[str, Any]) -> str:
    lines = [
        "| Test set | Utterances / words / mean duration | Model | Words right | Errors "
        "| Inserted words | Invented closing phrases |",
        "|---|---|---|---|---|---|---|",
    ]
    for test_set in report["test_sets"]:
        silent = test_set["silent"]
        if not silent["n_utterances"]:
            continue
        size = f"{silent['n_utterances']} / {silent['n_words']} / {silent['mean_duration']:.1f} s"
        for model in report["models"]:
            s = silent["systems"][model]
            lines.append(
                f"| {test_set['name']} | {size} | {_label(model)} | {s['hits']} | {s['errors']} "
                f"| {s['insertions']} | {s['closing']} |"
            )
    return "\n".join(lines)


def _runaway_table(report: dict[str, Any]) -> str:
    """WER of the test sets with runaway transcripts, without those utterances, and comparisons."""
    test_sets = [t for t in report["test_sets"] if t["runaway"]]
    lines = [
        "| Test set | Runaway transcripts: words written / reference words / seconds | Model "
        "| WER with them | WER without them [95% CI] |",
        "|---|---|---|---|---|",
    ]
    for test_set in test_sets:
        found = "; ".join(
            f"{max(r['words'].values()):,} / {r['reference_words']} / {r['duration']:.1f}"
            for r in test_set["runaway"]
        )
        for model in report["models"]:
            before = test_set["all"]["systems"][model]
            after = test_set["without_runaway"]["systems"][model]
            lines.append(
                f"| {test_set['name']} | {found} | {_label(model)} | {before['wer']:.2%} "
                f"| {after['wer']:.2%} [{after['low']:.2%}, {after['high']:.2%}] |"
            )
    return "\n".join([*lines, "", *_comparison_rows(report, "without_runaway", test_sets)])
