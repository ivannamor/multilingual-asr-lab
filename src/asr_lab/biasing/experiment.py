"""The steps of experiment E9, from the spec: entities, Whisper with prompts, CTC with hotwords."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from asr_lab.biasing.entities import distractors, domain_list, extract_entities, prompt_text
from asr_lab.biasing.scoring import entity_counts
from asr_lab.data.manifest import read_manifest
from asr_lab.evaluation.metrics import ErrorCounts, error_rate, utterance_counts, word_tokens
from asr_lab.evaluation.normalize import get_normalizer_by_name
from asr_lab.evaluation.scoring import load_run, normalized_pairs
from asr_lab.lm.corpus import clean_sentence, load_spec
from asr_lab.lm.decoder import KenLMModel, beam_search
from asr_lab.lm.logprobs import iter_log_probs, read_metadata
from asr_lab.lm.tune import decode_test_set, decoding_options, lm_name, lm_paths
from asr_lab.provenance import git_commit, utc_now

SPLITS = ("dev", "test")
NO_LIST = "none"
ORACLE = "oracle"
DISTRACTORS = "distractors"
_WORD_DELIMITER_AND_HYPHEN = "|-"
_worker: dict[str, Any] = {}


def run_entities(spec_path: Path, output: Path) -> Path:
    """Write the entities of every dev and test set, and the domain list, to a file."""
    spec = load_spec(spec_path)
    settings = spec["entities"]
    models, labels = settings["models"], settings["labels"]
    test_sets = {}
    for test_set in spec["test_sets"]:
        test_sets[test_set["name"]] = {}
        for split in SPLITS:
            entries = read_manifest(Path(test_set[split]))
            found = extract_entities(
                [entry.text for entry in entries], models[test_set["lang"]], labels
            )
            test_sets[test_set["name"]][split] = {
                entry.utterance_id: entities for entry, entities in zip(entries, found, strict=True)
            }
    domain = read_manifest(Path(settings["domain_manifest"]))
    domain_found = extract_entities(
        [entry.text for entry in domain], models[settings["domain_lang"]], labels
    )
    report = {
        "models": models,
        "labels": labels,
        "domain": domain_list(domain_found),
        "test_sets": test_sets,
        "created": utc_now(),
        "git_commit": git_commit(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return output


def oracle_lists(entities: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    """The distinct entities of each utterance.

    >>> oracle_lists({"a": ["Roma", "Roma"], "b": []})
    {'a': ['Roma'], 'b': []}
    """
    return {key: list(dict.fromkeys(value)) for key, value in entities.items()}


def domain_lists(
    domain: Sequence[dict[str, Any]], size: int, keys: Sequence[str]
) -> dict[str, list[str]]:
    """The same list for every utterance: the ``size`` commonest entities, the commonest last.

    >>> rows = [{"entity": "Europa", "count": 3}, {"entity": "UE", "count": 1}]
    >>> domain_lists(rows, 2, ["a"])
    {'a': ['UE', 'Europa']}
    """
    chosen = [row["entity"] for row in domain[:size]][::-1]
    return dict.fromkeys(keys, chosen) if chosen else {}


def conditions_for(
    spec: dict[str, Any],
    entities_report: dict[str, Any],
    test_set: dict[str, Any],
    split: str,
    domain_size: int | None,
) -> dict[str, dict[str, list[str]]]:
    """The lists of one test set and split, by condition."""
    found = entities_report["test_sets"][test_set["name"]][split]
    lists = {
        ORACLE: oracle_lists(found),
        DISTRACTORS: distractors(found, spec["entities"]["seed"]),
    }
    if test_set.get("has_domain_list") and domain_size:
        lists[f"domain-{domain_size}"] = domain_lists(
            entities_report["domain"], domain_size, list(found)
        )
    return lists


def whisper_run_dir(runs_dir: Path, condition: str, test_set: dict[str, Any], split: str) -> Path:
    """Run directory of Whisper with a list.

    >>> whisper_run_dir(Path("runs"), "oracle", {"name": "set"}, "dev").as_posix()
    'runs/e9-turbo-oracle-set-dev'
    >>> whisper_run_dir(Path("runs"), "oracle", {"name": "set"}, "test").as_posix()
    'runs/e9-turbo-oracle-set'
    """
    suffix = "-dev" if split == "dev" else ""
    return runs_dir / f"e9-turbo-{condition}-{test_set['name']}{suffix}"


def whisper_plan(
    spec: dict[str, Any],
    entities_report: dict[str, Any],
    split: str,
    selection: dict[str, Any] | None,
) -> list[tuple[dict[str, Any], str, dict[str, list[str]]]]:
    """Every Whisper run of a split, as (test set, condition, lists): empty lists for no list.

    On dev only the test set with a domain list runs, since it is the only setting to choose. On
    test the runs without a list are those of the earlier experiments.
    """
    plan = []
    for test_set in spec["test_sets"]:
        if split == "dev":
            if test_set.get("has_domain_list"):
                keys = list(entities_report["test_sets"][test_set["name"]]["dev"])
                plan.append((test_set, NO_LIST, {}))
                for size in spec["whisper"]["domain_sizes"]:
                    lists = domain_lists(entities_report["domain"], size, keys)
                    plan.append((test_set, f"domain-{size}", lists))
            continue
        size = None
        if test_set.get("has_domain_list"):
            size = (selection or {})["whisper"]["domain_size"]
        for condition, lists in conditions_for(
            spec, entities_report, test_set, "test", size
        ).items():
            plan.append((test_set, condition, lists))
    return plan


def run_whisper(
    spec_path: Path,
    entities_path: Path,
    split: str,
    runs_dir: Path,
    selection_path: Path | None,
    device: str,
) -> list[Path]:
    """Transcribe a split with Whisper under every list of the plan, loading the weights once."""
    from asr_lab.inference.faster_whisper import FasterWhisperModel  # needs ctranslate2's libraries
    from asr_lab.inference.run import transcribe_manifest

    spec = load_spec(spec_path)
    entities_report = json.loads(entities_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path else None
    base = FasterWhisperModel(spec["whisper"]["model_size"], device=device)
    written = []
    for test_set, condition, lists in whisper_plan(spec, entities_report, split, selection):
        prompts = {key: text for key, value in lists.items() if (text := prompt_text(value))}
        options = {}
        if "hallucination_silence_threshold" in test_set:
            options["hallucination_silence_threshold"] = test_set["hallucination_silence_threshold"]
        if lists:
            options["initial_prompts"] = prompts
        run_dir = whisper_run_dir(runs_dir, condition, test_set, split)
        print(f"=== {run_dir.name}", flush=True)
        transcribe_manifest(base.with_decoding(**options), Path(test_set[split]), run_dir)
        written.append(run_dir)
    return written


def score_lists(
    references: Sequence[str],
    hypotheses: Sequence[str],
    entities: Sequence[Sequence[str]],
    normalizer: Any,
) -> dict[str, float]:
    """WER and entity recall of hypotheses, normalized like the references."""
    counts = sum(utterance_counts(references, hypotheses, word_tokens), ErrorCounts())
    found = ErrorCounts()
    for reference, hypothesis, listed in zip(references, hypotheses, entities, strict=True):
        tokens = [word_tokens(normalizer(entity)) for entity in listed]
        found += entity_counts(word_tokens(reference), word_tokens(hypothesis), tokens)
    return {
        "wer": error_rate(counts),
        "recall": 1 - error_rate(found),
        "n_entities": found.ref_length,
    }


def select_setting(
    candidates: Sequence[dict[str, Any]], baseline_wer: float, margin: float
) -> dict[str, Any] | None:
    """The setting with the highest recall among those within the margin of WER; see the spec.

    >>> cells = [
    ...     {"size": 50, "weight": 2.0, "wer": 0.21, "recall": 0.9},
    ...     {"size": 50, "weight": 1.0, "wer": 0.20, "recall": 0.8},
    ... ]
    >>> select_setting(cells, 0.20, 0.002)["weight"]
    1.0
    >>> select_setting(cells, 0.10, 0.002) is None
    True
    """
    allowed = [c for c in candidates if c["wer"] <= baseline_wer + margin]
    if not allowed:
        return None
    return min(allowed, key=lambda c: (-c["recall"], c.get("size", 0), c.get("weight", 0.0)))


def hotword_bonuses(
    lists: Mapping[str, Sequence[str]], alphabet: frozenset[str], weight: float
) -> dict[str, dict[str, float]]:
    """The word bonus of each utterance: the words of its entities that the model can write.

    >>> hotword_bonuses(
    ...     {"a": ["Unione europea", "G7"]}, frozenset("abcdefghijklmnopqrstuvwxyz"), 2.0
    ... )
    {'a': {'unione': 2.0, 'europea': 2.0}}
    """
    bonuses = {}
    for key, entities in lists.items():
        words = [
            word
            for entity in entities
            if (cleaned := clean_sentence(entity, alphabet))
            for word in cleaned.split()
        ]
        bonuses[key] = dict.fromkeys(words, weight)
    return bonuses


def ctc_alphabet(tokens: Sequence[str]) -> frozenset[str]:
    """The characters that a CTC model can write, as for the text of its language models."""
    return frozenset(
        token for token in tokens if len(token) == 1 and token not in _WORD_DELIMITER_AND_HYPHEN
    )


def run_ctc_tune(
    spec_path: Path, entities_path: Path, models_dir: Path, logprobs_dir: Path, output: Path,
    workers: int,
) -> Path:  # fmt: skip
    """Every hotword setting of CTC on the dev sets, with its WER and entity recall."""
    spec = load_spec(spec_path)
    entities_report = json.loads(entities_path.read_text(encoding="utf-8"))
    results = {}
    for test_set in spec["test_sets"]:
        results[test_set["name"]] = _tune_test_set(
            spec, entities_report, test_set, models_dir, logprobs_dir, workers
        )
        print(f"{test_set['name']}: {results[test_set['name']]['baseline']}", flush=True)
    report = {"test_sets": results, "created": utc_now(), "git_commit": git_commit()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return output


def run_selection(
    spec_path: Path, entities_path: Path, ctc_grid_path: Path, runs_dir: Path, output: Path
) -> Path:
    """Choose the settings of the test runs from the dev results; see the spec."""
    spec = load_spec(spec_path)
    margin = spec["selection"]["wer_margin"]
    entities_report = json.loads(entities_path.read_text(encoding="utf-8"))
    grid = json.loads(ctc_grid_path.read_text(encoding="utf-8"))["test_sets"]
    ctc = {}
    for name, result in grid.items():
        baseline = result["baseline"]["wer"]
        ctc[name] = {
            "baseline": result["baseline"],
            "oracle": select_setting(result["oracle"], baseline, margin),
            "domain": select_setting(result.get("domain", []), baseline, margin),
        }
    whisper = _select_whisper(spec, entities_report, runs_dir, margin)
    report = {"whisper": whisper, "ctc": ctc, "created": utc_now(), "git_commit": git_commit()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return output


def run_ctc_decode(
    spec_path: Path,
    entities_path: Path,
    selection_path: Path,
    models_dir: Path,
    logprobs_dir: Path,
    runs_dir: Path,
) -> list[Path]:
    """Decode each test set once with its chosen lists; a setting not chosen gives no run."""
    spec = load_spec(spec_path)
    entities_report = json.loads(entities_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))["ctc"]
    lm_spec, weights = e4_settings(spec)
    written = []
    for test_set in spec["test_sets"]:
        chosen = selection[test_set["name"]]
        lm_test_set = _lm_test_set(lm_spec, test_set["name"])
        logprobs = logprobs_dir / f"{test_set['name']}-test.npz"
        alphabet = ctc_alphabet(read_metadata(logprobs)["tokens"])
        size = chosen["domain"]["size"] if chosen["domain"] else None
        lists = conditions_for(spec, entities_report, test_set, "test", size)
        for condition, listed in lists.items():
            setting = chosen["domain"] if condition.startswith("domain") else chosen["oracle"]
            if setting is None:
                continue
            bonuses = hotword_bonuses(listed, alphabet, setting["weight"])
            name = ctc_lm_name(spec)
            written.append(
                decode_test_set(
                    lm_spec,
                    lm_test_set,
                    name,
                    _ctc_lm_path(spec, models_dir, test_set),
                    logprobs,
                    weights[test_set["name"]][name]["selected"],
                    runs_dir / f"e9-ctc-{condition}-{test_set['name']}",
                    word_bonuses=bonuses,
                    extra_config={
                        "condition": condition,
                        "hotword_weight": setting["weight"],
                        "entities_sha256": _lists_hash(bonuses),
                    },
                )
            )
    return written


def _lists_hash(bonuses: Mapping[str, Mapping[str, float]]) -> str:
    text = json.dumps(bonuses, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _select_whisper(
    spec: dict[str, Any], entities_report: dict[str, Any], runs_dir: Path, margin: float
) -> dict[str, Any]:
    test_set = next(t for t in spec["test_sets"] if t.get("has_domain_list"))
    normalizer = get_normalizer_by_name(test_set["normalizer"])
    found = entities_report["test_sets"][test_set["name"]]["dev"]
    manifest = Path(test_set["dev"])

    def score(condition: str) -> dict[str, float]:
        run = load_run(whisper_run_dir(runs_dir, condition, test_set, "dev"), manifest)
        pairs = normalized_pairs(run.entries, run.hypotheses, normalizer)
        listed = [found[entry.utterance_id] for entry in run.entries]
        return score_lists(pairs.references, pairs.hypotheses, listed, normalizer)

    baseline = score(NO_LIST)
    cells = [{"size": size, **score(f"domain-{size}")} for size in spec["whisper"]["domain_sizes"]]
    chosen = select_setting(cells, baseline["wer"], margin)
    return {
        "baseline": baseline,
        "cells": cells,
        "domain_size": chosen["size"] if chosen else None,
    }


def e4_settings(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The spec of E4, for its decoding settings, and the weights chosen there."""
    lm_spec = load_spec(Path(spec["ctc"]["lm_spec"]))
    grid = json.loads(Path(spec["ctc"]["grid"]).read_text(encoding="utf-8"))["test_sets"]
    return lm_spec, grid


def _lm_test_set(lm_spec: dict[str, Any], name: str) -> dict[str, Any]:
    return next(t for t in lm_spec["test_sets"] if t["name"] == name)


def ctc_lm_name(spec: dict[str, Any]) -> str:
    """Name of the language model of E4 that the hotwords build on."""
    return lm_name(spec["ctc"]["language_model"], spec["ctc"]["order"])


def _ctc_lm_path(spec: dict[str, Any], models_dir: Path, test_set: dict[str, Any]) -> Path:
    settings = spec["ctc"]
    return lm_paths(models_dir, settings["language_model"], test_set["lang"], settings["order"])[1]


def _start_worker(logprobs: Path, lm_path: Path, normalizer: str, options: Any) -> None:
    _worker["utterances"] = dict(iter_log_probs(logprobs))
    _worker["tokens"] = read_metadata(logprobs)["tokens"]
    _worker["lm"] = KenLMModel(str(lm_path))
    _worker["normalizer"] = get_normalizer_by_name(normalizer)
    _worker["options"] = options


def _decode_cell(
    cell: tuple[dict[str, Any], list[str], list[str], list[list[str]], dict[str, dict[str, float]]],
) -> dict[str, Any]:
    description, ids, references, entities, bonuses = cell
    normalize = _worker["normalizer"]
    hypotheses = [
        normalize(
            beam_search(
                _worker["utterances"][key],
                _worker["tokens"],
                _worker["lm"],
                _worker["options"],
                bonuses.get(key),
            )
        )
        for key in ids
    ]
    return {**description, **score_lists(references, hypotheses, entities, normalize)}


def _tune_test_set(
    spec: dict[str, Any],
    entities_report: dict[str, Any],
    test_set: dict[str, Any],
    models_dir: Path,
    logprobs_dir: Path,
    workers: int,
) -> dict[str, Any]:
    lm_spec, grid = e4_settings(spec)
    name = ctc_lm_name(spec)
    weights = grid[test_set["name"]][name]["selected"]
    options = decoding_options(lm_spec, weights["alpha"], weights["beta"])
    logprobs = logprobs_dir / f"{test_set['name']}-dev.npz"
    alphabet = ctc_alphabet(read_metadata(logprobs)["tokens"])
    normalize = get_normalizer_by_name(test_set["normalizer"])
    entries = read_manifest(Path(test_set["dev"]))
    ids = [entry.utterance_id for entry in entries]
    references = [normalize(entry.text) for entry in entries]
    found = entities_report["test_sets"][test_set["name"]]["dev"]
    entities = [found[key] for key in ids]

    def cell(description: dict[str, Any], lists: Mapping[str, Sequence[str]], weight: float) -> Any:
        return description, ids, references, entities, hotword_bonuses(lists, alphabet, weight)

    cells = [cell({"list": NO_LIST}, {}, 0.0)]
    oracle = oracle_lists(found)
    for weight in spec["ctc"]["weights"]:
        cells.append(cell({"list": ORACLE, "weight": weight}, oracle, weight))
        if test_set.get("has_domain_list"):
            for size in spec["ctc"]["domain_sizes"]:
                lists = domain_lists(entities_report["domain"], size, ids)
                cells.append(
                    cell({"list": "domain", "size": size, "weight": weight}, lists, weight)
                )
    with ProcessPoolExecutor(
        workers,
        initializer=_start_worker,
        initargs=(
            logprobs, _ctc_lm_path(spec, models_dir, test_set), test_set["normalizer"], options,
        ),
    ) as pool:  # fmt: skip
        scored = list(pool.map(_decode_cell, cells))
    result: dict[str, Any] = {"baseline": scored[0], "oracle": [], "domain": []}
    for row in scored[1:]:
        result[row.pop("list")].append(row)
    return result
