"""Results of experiment E4: the CTC models of E1b, decoded with and without a language model.

For each test set this computes, with the normalizer fixed in the spec:

- the WER of greedy decoding and of every language model, with its widest bootstrap interval, its
  errors by type, its CER and its decoding speed;
- the comparisons fixed in the spec (``[B, A]``: B - A, negative when B is better), as paired
  bootstraps with Holm's correction within the test set;
- the same on the utterances whose reference has no digit, which these models cannot write;
- how many reference words each system gets right inside and outside the vocabulary of each
  language model, on those same utterances: an n-gram language model cannot propose a word it has
  never seen, so this shows whether it helps the words it knows at the expense of the others;
- (post hoc) which reference words a language model repairs and which it breaks, compared with
  greedy decoding, added after seeing how few of its effects show in the CER.

``greedy_check`` is a separate step because it reads the log probabilities, which are not committed:
it verifies that decoding them greedily gives back the transcripts of E1b.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from asr_lab.data.manifest import read_manifest
from asr_lab.evaluation.comparison import LEVEL, comparison_family, headline_interval
from asr_lab.evaluation.error_analysis import edge_errors
from asr_lab.evaluation.families import has_digit
from asr_lab.evaluation.metrics import (
    AlignedPair,
    EditOp,
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
from asr_lab.lm.decoder import greedy_decode
from asr_lab.lm.experiment import language_models
from asr_lab.lm.logprobs import iter_log_probs, read_metadata
from asr_lab.lm.tune import lm_paths
from asr_lab.provenance import git_commit, utc_now

GREEDY = "greedy"
CORPUS_LABELS = {"europarl": "Europarl", "opensubtitles": "OpenSubtitles"}
_SPECIAL_TOKENS = re.compile(r"<s>|</s>|<unk>|<pad>")
_N_EXAMPLES = 10


def recall_counts(alignment: Sequence[AlignedPair], vocabulary: set[str]) -> dict[str, ErrorCounts]:
    """Reference words right and wrong, inside and outside a vocabulary.

    A wrong word counts as a substitution, so that the error rate of these counts is 1 - recall
    and the bootstrap of error rates applies to it.

    >>> pairs = align("il gatto zibibbo".split(), "il cane zibibbo".split())
    >>> counts = recall_counts(pairs, {"il", "gatto"})
    >>> counts["known"]
    ErrorCounts(hits=1, substitutions=1, deletions=0, insertions=0)
    >>> counts["unknown"]
    ErrorCounts(hits=1, substitutions=0, deletions=0, insertions=0)
    """
    hits = {"known": 0, "unknown": 0}
    misses = {"known": 0, "unknown": 0}
    for pair in alignment:
        if pair.ref is None:  # an insertion has no reference word
            continue
        group = "known" if pair.ref in vocabulary else "unknown"
        if pair.op is EditOp.MATCH:
            hits[group] += 1
        else:
            misses[group] += 1
    return {group: ErrorCounts(hits=hits[group], substitutions=misses[group]) for group in hits}


def word_changes(
    greedy: Sequence[AlignedPair], with_lm: Sequence[AlignedPair]
) -> tuple[Counter[tuple[str | None, str]], Counter[tuple[str, str | None]]]:
    """The reference words that a language model repairs and breaks in one utterance.

    Repaired words are counted as (what greedy decoding wrote, the word), and broken ones as (the
    word, what the language model wrote); ``None`` means that nothing was written in its place.

    >>> greedy = align("il gatto dorme".split(), "il gato dorme".split())
    >>> with_lm = align("il gatto dorme".split(), "il gatto dorma".split())
    >>> repaired, broken = word_changes(greedy, with_lm)
    >>> repaired[("gato", "gatto")], broken[("dorme", "dorma")]
    (1, 1)
    """
    repaired: Counter[tuple[str | None, str]] = Counter()
    broken: Counter[tuple[str, str | None]] = Counter()
    # Both alignments hold every reference word once and in order, so they can be read in parallel
    pairs = zip(
        (pair for pair in greedy if pair.ref is not None),
        (pair for pair in with_lm if pair.ref is not None),
        strict=True,
    )
    for before, after in pairs:
        if before.op is not EditOp.MATCH and after.op is EditOp.MATCH:
            repaired[(before.hyp, after.hyp)] += 1
        elif before.op is EditOp.MATCH and after.op is not EditOp.MATCH:
            broken[(before.hyp, after.hyp)] += 1
    return repaired, broken


def read_vocabulary(text_path: Path) -> set[str]:
    """The words of the text of a language model."""
    with text_path.open(encoding="utf-8") as lines:
        return {word for line in lines for word in line.split()}


def lm_results(
    spec_path: Path,
    selection_path: Path,
    text_report_path: Path,
    models_dir: Path,
    runs_dir: Path,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Score the runs of the language-model spec; see the module docstring."""
    spec = load_spec(spec_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))["test_sets"]
    comparisons = [tuple(pair) for pair in spec["analysis"]["comparisons"]]
    names = language_models(spec)
    systems = [GREEDY, *(name for _, _, name in names)]
    options = {"n_resamples": n_resamples, "seed": seed}
    test_sets = []
    for test_set in spec["test_sets"]:
        scored = _score_test_set(test_set, systems, runs_dir)
        entries = scored["entries"]
        no_digit = [i for i, entry in enumerate(entries) if not has_digit(entry.text)]
        vocabularies = {
            name: read_vocabulary(lm_paths(models_dir, corpus, test_set["lang"], order)[0])
            for corpus, order, name in names
        }
        test_sets.append(
            {
                "name": test_set["name"],
                "lang": test_set["lang"],
                "normalizer": test_set["normalizer"],
                "runs": scored["runs"],
                "weights": {
                    name: selection[test_set["name"]][name]["selected"] for _, _, name in names
                },
                "n_with_digits": len(entries) - len(no_digit),
                "all": _subset(range(len(entries)), scored, systems, comparisons, options),
                "no_digit": _subset(no_digit, scored, systems, comparisons, options),
                "vocabulary": {
                    name: _vocabulary_recall(no_digit, scored, name, vocabularies[name], options)
                    for _, _, name in names
                },
                "changes": {name: _changes(scored, name) for _, _, name in names},
            }
        )
    return {
        "spec": spec_path.as_posix(),
        "systems": systems,
        "comparisons": [list(pair) for pair in comparisons],
        "level": LEVEL,
        **options,
        "text": json.loads(text_report_path.read_text(encoding="utf-8"))["text"],
        "test_sets": test_sets,
        "created": utc_now(),
        "git_commit": git_commit(),
    }


def greedy_check(spec_path: Path, logprobs_dir: Path) -> dict[str, Any]:
    """Compare greedy decoding of the saved log probabilities with the transcripts of E1b.

    The transcripts should differ only in spaces and in special tokens, which the decoder of this
    experiment never writes, while the checkpoint's tokenizer writes ``<unk>``.
    """
    spec = load_spec(spec_path)
    rows = []
    for test_set in spec["test_sets"]:
        path = logprobs_dir / f"{test_set['name']}-test.npz"
        tokens = read_metadata(path)["tokens"]
        decoded = {key: greedy_decode(values, tokens) for key, values in iter_log_probs(path)}
        run = load_run(Path(test_set["greedy_run"]), Path(test_set["test"]))
        original = {h.utterance_id: h.text for h in run.hypotheses}
        if decoded.keys() != original.keys():
            raise ValueError(f"{path} and {test_set['greedy_run']} cover different utterances")
        different = [key for key in decoded if decoded[key] != original[key]]
        in_spaces = [key for key in different if _words(decoded[key]) != _words(original[key])]
        in_tokens = [
            key
            for key in in_spaces
            if _words(decoded[key]) != _words(_SPECIAL_TOKENS.sub("", original[key]))
        ]
        rows.append(
            {
                "name": test_set["name"],
                "n_utterances": len(decoded),
                "different": len(different),
                "different_beyond_spaces": len(in_spaces),
                "different_beyond_special_tokens": len(in_tokens),
                "wer_e1b": _wer(run.entries, original, test_set["normalizer"]),
                "wer_log_probabilities": _wer(run.entries, decoded, test_set["normalizer"]),
                "dev_wer": _dev_greedy_wer(test_set, logprobs_dir),
            }
        )
    return {"test_sets": rows, "created": utc_now(), "git_commit": git_commit()}


def lm_table(report: dict[str, Any], section: str) -> str:
    """One Markdown table of the results: ``wer``, ``comparisons``, ``no_digit``, ``weights``,
    ``vocabulary``, ``changes``, ``text`` or (from ``greedy_check``) ``greedy_check``."""
    builders = {
        "wer": _wer_table,
        "comparisons": _comparisons_table,
        "no_digit": _no_digit_table,
        "weights": _weights_table,
        "vocabulary": _vocabulary_table,
        "changes": _changes_table,
        "text": _text_table,
        "greedy_check": _greedy_check_table,
    }
    if section not in builders:
        raise ValueError(f"unknown section {section!r}; choose from {sorted(builders)}")
    return builders[section](report)


def _words(text: str) -> list[str]:
    return text.split()


def _wer(entries: Sequence[Any], texts: dict[str, str], normalizer: str) -> float:
    normalize = get_normalizer_by_name(normalizer)
    references = [normalize(entry.text) for entry in entries]
    hypotheses = [normalize(texts[entry.utterance_id]) for entry in entries]
    return error_rate(sum(utterance_counts(references, hypotheses, word_tokens), ErrorCounts()))


def _dev_greedy_wer(test_set: dict[str, Any], logprobs_dir: Path) -> float:
    path = logprobs_dir / f"{test_set['name']}-dev.npz"
    tokens = read_metadata(path)["tokens"]
    texts = {key: greedy_decode(values, tokens) for key, values in iter_log_probs(path)}
    return _wer(read_manifest(Path(test_set["dev"])), texts, test_set["normalizer"])


def _score_test_set(
    test_set: dict[str, Any], systems: Sequence[str], runs_dir: Path
) -> dict[str, Any]:
    """Per-utterance results of every system on one test set."""
    normalizer = get_normalizer_by_name(test_set["normalizer"])
    manifest = Path(test_set["test"])
    entries, runs = None, {}
    words, chars, edges, alignments = {}, {}, {}, {}
    for system in systems:
        run_dir = (
            Path(test_set["greedy_run"])
            if system == GREEDY
            else runs_dir / f"e4-{system}-{test_set['name']}"
        )
        run = load_run(run_dir, manifest)
        if entries is None:
            entries = run.entries
        pairs = normalized_pairs(entries, run.hypotheses, normalizer)
        alignments[system] = [
            align(word_tokens(ref), word_tokens(hyp))
            for ref, hyp in zip(pairs.references, pairs.hypotheses, strict=True)
        ]
        words[system] = [count_errors(alignment) for alignment in alignments[system]]
        chars[system] = utterance_counts(pairs.references, pairs.hypotheses, char_tokens)
        edges[system] = [edge_errors(alignment) for alignment in alignments[system]]
        runs[system] = {
            "run_dir": run_dir.as_posix(),
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
        "alignments": alignments,
        "runs": runs,
    }


def _subset(
    keep: Sequence[int],
    scored: dict[str, Any],
    systems: Sequence[str],
    comparisons: Sequence[tuple[str, str]],
    options: dict[str, int],
) -> dict[str, Any]:
    entries = [scored["entries"][i] for i in keep]
    units = resampling_units(entries)
    words = {system: [scored["words"][system][i] for i in keep] for system in systems}
    results = {}
    for system in systems:
        counts = sum(words[system], ErrorCounts())
        chars = sum((scored["chars"][system][i] for i in keep), ErrorCounts())
        results[system] = {
            **headline_interval(words[system], units, options),
            "cer": error_rate(chars),
            "substitutions": counts.substitutions,
            "deletions": counts.deletions,
            "insertions": counts.insertions,
            "errors": counts.errors,
            "edge_errors": sum((scored["edges"][system][i] for i in keep), ErrorCounts()).errors,
        }
    return {
        "n_utterances": len(entries),
        "n_words": sum(counts.ref_length for counts in words[systems[0]]),
        "n_speakers": len({entry.speaker_id for entry in entries}),
        "systems": results,
        "comparisons": comparison_family(
            [(a, b, words[a], words[b]) for b, a in comparisons], units, options
        ),
    }


def _vocabulary_recall(
    keep: Sequence[int],
    scored: dict[str, Any],
    name: str,
    vocabulary: set[str],
    options: dict[str, int],
) -> dict[str, Any]:
    """Recall of greedy decoding and of one language model, for the words it knows and does not."""
    units = resampling_units([scored["entries"][i] for i in keep])
    counts = {
        system: [recall_counts(scored["alignments"][system][i], vocabulary) for i in keep]
        for system in (GREEDY, name)
    }
    result = {}
    for group in ("known", "unknown"):
        by_system = {
            system: [utterance[group] for utterance in counts[system]] for system in counts
        }
        greedy = headline_interval(by_system[GREEDY], units, options)
        with_lm = headline_interval(by_system[name], units, options)
        # Rates are of missed words: a negative difference means that the language model gets more
        # words right. The intervals are descriptive, outside the family of Holm's correction.
        difference = comparison_family(
            [(GREEDY, name, by_system[GREEDY], by_system[name])], units, options
        )[0]
        result[group] = {
            "n_words": sum(item.ref_length for item in by_system[GREEDY]),
            "greedy_recall": 1 - greedy["wer"],
            "greedy_low": 1 - greedy["high"],
            "greedy_high": 1 - greedy["low"],
            "lm_recall": 1 - with_lm["wer"],
            "lm_low": 1 - with_lm["high"],
            "lm_high": 1 - with_lm["low"],
            "difference": -difference["delta"],
            "difference_low": -difference["high"],
            "difference_high": -difference["low"],
        }
    return result


def _changes(scored: dict[str, Any], name: str) -> dict[str, Any]:
    """Words repaired and broken by a language model on a whole test set, with the commonest."""
    repaired: Counter[tuple[str | None, str]] = Counter()
    broken: Counter[tuple[str, str | None]] = Counter()
    for greedy, with_lm in zip(
        scored["alignments"][GREEDY], scored["alignments"][name], strict=True
    ):
        utterance_repaired, utterance_broken = word_changes(greedy, with_lm)
        repaired += utterance_repaired
        broken += utterance_broken
    return {
        "repaired": sum(repaired.values()),
        "broken": sum(broken.values()),
        "most_repaired": [[*pair, n] for pair, n in _commonest(repaired)],
        "most_broken": [[*pair, n] for pair, n in _commonest(broken)],
    }


def _commonest(counter: Counter[Any]) -> list[tuple[Any, int]]:
    """The commonest items; in a tie, in a fixed order, so that the results are reproducible."""
    return sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))[:_N_EXAMPLES]


def _label(system: str) -> str:
    """Name of a system in a table.

    >>> _label("europarl-5"), _label("greedy")
    ('Europarl 5-gram', 'Greedy')
    """
    if system == GREEDY:
        return "Greedy"
    corpus, order = system.rsplit("-", 1)
    return f"{CORPUS_LABELS[corpus]} {order}-gram"


def _wer_table(report: dict[str, Any]) -> str:
    lines = [
        "| Test set | Utterances / words | System | WER [95% CI] | S / D / I | CER |",
        "|---|---|---|---|---|---|",
    ]
    for test_set in report["test_sets"]:
        subset = test_set["all"]
        size = f"{subset['n_utterances']:,} / {subset['n_words']:,}"
        for system in report["systems"]:
            s = subset["systems"][system]
            lines.append(
                f"| {test_set['name']} (`{test_set['normalizer']}`) | {size} | {_label(system)} "
                f"| {s['wer']:.2%} [{s['low']:.2%}, {s['high']:.2%}] "
                f"| {s['substitutions']:,} / {s['deletions']:,} / {s['insertions']:,} "
                f"| {s['cer']:.2%} |"
            )
    return "\n".join(lines)


def _comparison_rows(report: dict[str, Any], key: str) -> list[str]:
    lines = [
        "| Test set | B - A | WER difference [95% CI] | p | p (Holm) | Significant |",
        "|---|---|---|---|---|---|",
    ]
    for test_set in report["test_sets"]:
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
        "| System | " + " | ".join(t["name"] for t in test_sets) + " |",
        "|---|" + "---|" * len(test_sets),
    ]
    for system in report["systems"]:
        cells = [
            f"{t['no_digit']['systems'][system]['wer']:.2%} "
            f"[{t['no_digit']['systems'][system]['low']:.2%}, "
            f"{t['no_digit']['systems'][system]['high']:.2%}]"
            for t in test_sets
        ]
        lines.append(f"| {_label(system)} | " + " | ".join(cells) + " |")
    sizes = [
        f"{t['no_digit']['n_utterances']:,} of {t['all']['n_utterances']:,}" for t in test_sets
    ]
    lines.append("| *utterances kept* | " + " | ".join(sizes) + " |")
    return "\n".join([*lines, "", *_comparison_rows(report, "no_digit")])


def _weights_table(report: dict[str, Any]) -> str:
    lines = [
        "| Test set | Language model | Alpha | Beta | Dev WER | RTF of the decoding |",
        "|---|---|---|---|---|---|",
    ]
    for test_set in report["test_sets"]:
        for name, weights in test_set["weights"].items():
            lines.append(
                f"| {test_set['name']} | {_label(name)} | {weights['alpha']:g} "
                f"| {weights['beta']:g} | {weights['wer']:.2%} "
                f"| {test_set['runs'][name]['rtf']:.4f} |"
            )
    return "\n".join(lines)


def _vocabulary_table(report: dict[str, Any]) -> str:
    lines = [
        "| Test set | Language model | Reference words outside its vocabulary "
        "| Words it knows: greedy, with LM, difference [95% CI] "
        "| Words it does not know: greedy, with LM, difference [95% CI] |",
        "|---|---|---|---|---|",
    ]
    for test_set in report["test_sets"]:
        for name, groups in test_set["vocabulary"].items():
            known, unknown = groups["known"], groups["unknown"]
            total = known["n_words"] + unknown["n_words"]
            lines.append(
                f"| {test_set['name']} | {_label(name)} "
                f"| {unknown['n_words']:,} of {total:,} ({unknown['n_words'] / total:.1%}) "
                f"| {_recall_cell(known)} | {_recall_cell(unknown)} |"
            )
    return "\n".join(lines)


def _changes_table(report: dict[str, Any]) -> str:
    """The 5-grams only: the 3-grams give nearly the same transcripts."""
    lines = [
        "| Test set | Language model | Reference words repaired | Reference words broken "
        "| Commonest repairs (greedy → with LM, times) "
        "| Commonest breaks (reference → with LM, times) |",
        "|---|---|---|---|---|---|",
    ]
    for test_set in report["test_sets"]:
        for name, changes in test_set["changes"].items():
            if name.endswith("-5"):
                lines.append(
                    f"| {test_set['name']} | {_label(name)} | {changes['repaired']:,} "
                    f"| {changes['broken']:,} | {_change_cell(changes['most_repaired'])} "
                    f"| {_change_cell(changes['most_broken'])} |"
                )
    return "\n".join(lines)


def _change_cell(changes: Sequence[Sequence[Any]]) -> str:
    return ", ".join(
        f"{before or '(nothing)'} → {after or '(nothing)'} ({n})"
        for before, after, n in changes[:5]
    )


def _recall_cell(group: dict[str, float]) -> str:
    return (
        f"{group['greedy_recall']:.1%}, {group['lm_recall']:.1%}, {group['difference']:+.1%} "
        f"[{group['difference_low']:+.1%}, {group['difference_high']:+.1%}]"
    )


def _text_table(report: dict[str, Any]) -> str:
    lines = [
        "| Source | Language | Lines read | Not writable | In a dev or test set (distinct) "
        "| Duplicates | Words available | Sentences used | Words used |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in report["text"]:
        lines.append(
            f"| {CORPUS_LABELS[row['corpus']]} | {row['lang']} | {row['read']:,} "
            f"| {row['not_writable']:,} ({row['not_writable'] / row['read']:.1%}) "
            f"| {row['in_dev_or_test']:,} ({row['distinct_in_dev_or_test']}) "
            f"| {row['duplicates']:,} | {row['available_words']:,} | {row['sentences']:,} "
            f"| {row['words']:,} |"
        )
    return "\n".join(lines)


def _greedy_check_table(report: dict[str, Any]) -> str:
    lines = [
        "| Test set | Utterances | Transcripts that differ | Beyond spaces "
        "| Beyond spaces and special tokens | WER of E1b | WER from the log probabilities "
        "| Greedy WER on the dev set |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in report["test_sets"]:
        lines.append(
            f"| {row['name']} | {row['n_utterances']:,} | {row['different']} "
            f"| {row['different_beyond_spaces']} | {row['different_beyond_special_tokens']} "
            f"| {row['wer_e1b']:.2%} | {row['wer_log_probabilities']:.2%} | {row['dev_wer']:.2%} |"
        )
    return "\n".join(lines)
