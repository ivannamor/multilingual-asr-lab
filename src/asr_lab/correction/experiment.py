"""The steps of experiment E8, from the spec: N-best lists, the language model, the four methods.

1. ``run_nbest`` (CPU): the N-best lists of the CTC model with the language model of E4, on the dev
   and test sets, and the few-shot examples drawn from the dev lists.
2. ``run_llm`` (GPU): what the large language model says of each list. On dev only the probability
   of each transcript, the one thing the methods tune; on test, also its choice and its corrections.
3. ``run_methods`` (CPU): the rescoring weights chosen on dev, then one run per method on test.

The CTC transcripts that the methods start from are the first of each list, those of E4.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from asr_lab.biasing.experiment import ctc_alphabet
from asr_lab.correction.llm import LanguageModel
from asr_lab.correction.prompts import Example, Prompts, clean_answer, load_prompts
from asr_lab.data.manifest import read_manifest
from asr_lab.evaluation.metrics import ErrorCounts, error_rate, utterance_counts, word_tokens
from asr_lab.evaluation.normalize import get_normalizer_by_name
from asr_lab.inference.base import Hypothesis
from asr_lab.lm.corpus import clean_sentence, load_spec
from asr_lab.lm.decoder import KenLMModel, ScoredTranscript, beam_search_nbest
from asr_lab.lm.logprobs import iter_log_probs, read_metadata
from asr_lab.lm.tune import decoding_options, lm_name, lm_paths, write_run
from asr_lab.provenance import file_sha256, git_commit, utc_now

SPLITS = ("dev", "test")
CTC = "ctc"
RESCORE = "rescore"
CHOOSE = "choose"
ZERO_SHOT = "zero-shot"
FEW_SHOT = "few-shot"
SCORES_FILE = "nbest_scores.jsonl"
_worker: dict[str, Any] = {}


def nbest_run_dir(runs_dir: Path, test_set: dict[str, Any], split: str) -> Path:
    """Run directory of the N-best lists of a split; its predictions are the first transcripts.

    >>> nbest_run_dir(Path("runs"), {"name": "set"}, "dev").as_posix()
    'runs/e8-nbest-set-dev'
    >>> nbest_run_dir(Path("runs"), {"name": "set"}, "test").as_posix()
    'runs/e8-nbest-set'
    """
    suffix = "-dev" if split == "dev" else ""
    return runs_dir / f"e8-nbest-{test_set['name']}{suffix}"


def method_run_dir(runs_dir: Path, method: str, test_set: dict[str, Any]) -> Path:
    """Run directory of a method on a test set; ``ctc`` is the N-best run itself.

    >>> method_run_dir(Path("runs"), "few-shot", {"name": "set"}).as_posix()
    'runs/e8-few-shot-set'
    >>> method_run_dir(Path("runs"), "ctc", {"name": "set"}).as_posix()
    'runs/e8-nbest-set'
    """
    if method == CTC:
        return nbest_run_dir(runs_dir, test_set, "test")
    return runs_dir / f"e8-{method}-{test_set['name']}"


def llm_output_path(folder: Path, test_set: dict[str, Any], split: str) -> Path:
    """What the language model said of the lists of a split.

    >>> llm_output_path(Path("llm"), {"name": "set"}, "test").as_posix()
    'llm/set-test.jsonl'
    """
    return folder / f"{test_set['name']}-{split}.jsonl"


# 1. N-best lists


def e4_settings(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    """The spec of E4, its chosen weights and the name of the language model of the lists."""
    settings = spec["nbest"]
    lm_spec = load_spec(Path(settings["lm_spec"]))
    grid = json.loads(Path(settings["grid"]).read_text(encoding="utf-8"))["test_sets"]
    return lm_spec, grid, lm_name(settings["language_model"], settings["order"])


def run_nbest(
    spec_path: Path,
    models_dir: Path,
    logprobs_dir: Path,
    runs_dir: Path,
    examples_path: Path,
    workers: int = 8,
) -> list[Path]:
    """Write the N-best run of every dev and test set, then the few-shot examples."""
    spec = load_spec(spec_path)
    lm_spec, grid, name = e4_settings(spec)
    written = []
    examples = {}
    for test_set in spec["test_sets"]:
        weights = grid[test_set["name"]][name]["selected"]
        options = decoding_options(lm_spec, weights["alpha"], weights["beta"])
        settings = spec["nbest"]
        _, lm_path = lm_paths(
            models_dir, settings["language_model"], test_set["lang"], settings["order"]
        )
        for split in SPLITS:
            logprobs = logprobs_dir / f"{test_set['name']}-{split}.npz"
            output = nbest_run_dir(runs_dir, test_set, split)
            _decode_nbest(
                logprobs, lm_path, options, Path(test_set[split]), output, name,
                settings["n"], workers,
            )  # fmt: skip
            print(output, flush=True)
            written.append(output)
        dev_entries = read_manifest(Path(test_set["dev"]))
        dev_logprobs = logprobs_dir / f"{test_set['name']}-dev.npz"
        alphabet = ctc_alphabet(read_metadata(dev_logprobs)["tokens"])
        chosen = choose_examples(
            read_nbest(nbest_run_dir(runs_dir, test_set, "dev")),
            {entry.utterance_id: entry.text for entry in dev_entries},
            alphabet,
            get_normalizer_by_name(test_set["normalizer"]),
            spec["few_shot"],
        )
        examples[test_set["name"]] = [example.__dict__ for example in chosen]
    report = {"examples": examples, "created": utc_now(), "git_commit": git_commit()}
    _write_json(report, examples_path)
    return written


def _decode_nbest(
    logprobs: Path,
    lm_path: Path,
    options: Any,
    manifest: Path,
    output: Path,
    name: str,
    n: int,
    workers: int,
) -> None:
    metadata = read_metadata(logprobs)
    if metadata["manifest_sha256"] != file_sha256(manifest):
        raise ValueError(f"{logprobs} does not come from {manifest}")
    config = {
        "acoustic_model": metadata["model_config"],
        "language_model": name,
        "language_model_sha256": file_sha256(lm_path),
        "decoding": options.__dict__,
        "n": n,
    }
    run_path = output / "run.json"
    if (
        run_path.exists()
        and (output / SCORES_FILE).exists()
        and json.loads(run_path.read_text(encoding="utf-8")).get("model_config") == config
    ):
        return
    keys = [key for key, _ in iter_log_probs(logprobs)]
    started = time.perf_counter()
    with ProcessPoolExecutor(
        workers, initializer=_start_worker, initargs=(logprobs, lm_path, options, n)
    ) as pool:
        lists = list(pool.map(_nbest_of, keys, chunksize=8))
    seconds = time.perf_counter() - started
    hypotheses = [
        Hypothesis(key, best[0].text, tuple(t.text for t in best[1:]), best[0].score)
        for key, best in zip(keys, lists, strict=True)
    ]
    write_run(output, hypotheses, f"{metadata['model']}+{name}", config, manifest, seconds)
    lines = [
        json.dumps(
            {
                "utterance_id": key,
                "acoustic": [round(t.acoustic, 4) for t in best],
                "language": [round(t.language, 4) for t in best],
            }
        )
        for key, best in zip(keys, lists, strict=True)
    ]
    (output / SCORES_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _start_worker(logprobs: Path, lm_path: Path, options: Any, n: int) -> None:
    _worker["utterances"] = dict(iter_log_probs(logprobs))
    _worker["tokens"] = read_metadata(logprobs)["tokens"]
    _worker["lm"] = KenLMModel(str(lm_path))
    _worker["options"] = options
    _worker["n"] = n


def _nbest_of(key: str) -> list[ScoredTranscript]:
    return beam_search_nbest(
        _worker["utterances"][key], _worker["tokens"], _worker["lm"], _worker["options"],
        n=_worker["n"],
    )  # fmt: skip


def read_nbest(run_dir: Path) -> dict[str, list[ScoredTranscript]]:
    """The N-best list of each utterance of an N-best run, best first, with its scores."""
    texts = {}
    with (run_dir / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            texts[record["utterance_id"]] = [record["text"], *record["nbest"]]
    lists = {}
    with (run_dir / SCORES_FILE).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            key = record["utterance_id"]
            lists[key] = [
                ScoredTranscript(*row)
                for row in zip(texts[key], record["acoustic"], record["language"], strict=True)
            ]
    return lists


def choose_examples(
    nbest: Mapping[str, Sequence[ScoredTranscript]],
    references: Mapping[str, str],
    alphabet: frozenset[str],
    normalize: Any,
    settings: Mapping[str, Any],
) -> list[Example]:
    """Draw the few-shot examples of a test set from its dev lists; see the spec.

    The answer of an example is the reference as the CTC model would write it.
    """
    eligible = []
    for key in sorted(nbest):
        target = clean_sentence(references[key], alphabet)
        if target is None or len(target.split()) > settings["max_words"]:
            continue
        if normalize(nbest[key][0].text) != normalize(references[key]):
            eligible.append(Example(key, tuple(t.text for t in nbest[key]), target))
    return random.Random(settings["seed"]).sample(eligible, settings["n_examples"])


def load_examples(path: Path) -> dict[str, list[Example]]:
    """The few-shot examples of each test set."""
    report = json.loads(path.read_text(encoding="utf-8"))["examples"]
    return {
        name: [Example(e["utterance_id"], tuple(e["candidates"]), e["target"]) for e in rows]
        for name, rows in report.items()
    }


# 2. The language model


def load_llm(spec_path: Path, device: str = "cuda") -> LanguageModel:
    """The language model of the spec, at its revision (needs the extra ``hf``)."""
    from asr_lab.correction.llm import TransformersLLM

    settings = load_spec(spec_path)["llm"]
    return TransformersLLM(
        settings["id"],
        settings["revision"],
        dtype=settings["dtype"],
        batch_size=settings["batch_size"],
        device=device,
    )


def run_llm(
    spec_path: Path,
    runs_dir: Path,
    examples_path: Path,
    output_dir: Path,
    llm: LanguageModel,
    splits: Sequence[str] = SPLITS,
) -> list[Path]:
    """Ask the language model about every list of the splits; a complete output is kept."""
    spec = load_spec(spec_path)
    prompts = load_prompts(Path(spec["llm"]["prompts"]))
    examples = load_examples(examples_path)
    written = []
    for split in splits:
        for test_set in spec["test_sets"]:
            output = llm_output_path(output_dir, test_set, split)
            config = {
                "model": llm.name,
                **llm.config,
                "prompts": prompts.version,
                "prompts_sha256": prompts.sha256,
                "nbest_sha256": file_sha256(nbest_run_dir(runs_dir, test_set, split) / SCORES_FILE),
                "examples_sha256": file_sha256(examples_path),
            }
            if _complete(output, config):
                written.append(output)
                continue
            print(f"=== {test_set['name']} {split}", flush=True)
            nbest = read_nbest(nbest_run_dir(runs_dir, test_set, split))
            started = time.perf_counter()
            records = ask_llm(
                llm, prompts, nbest, test_set["lang"], examples[test_set["name"]], split == "test"
            )
            seconds = time.perf_counter() - started
            _write_llm_output(output, records, {**config, "seconds": seconds, "split": split})
            written.append(output)
    return written


def ask_llm(
    llm: LanguageModel,
    prompts: Prompts,
    nbest: Mapping[str, Sequence[ScoredTranscript]],
    lang: str,
    examples: Sequence[Example],
    everything: bool,
) -> list[dict[str, Any]]:
    """The language model's view of each list: the probability of each transcript and, with
    ``everything``, its choice and its zero-shot and few-shot corrections."""
    keys = sorted(nbest)
    candidates = [[t.text for t in nbest[key]] for key in keys]
    flat = [text for texts in candidates for text in texts]
    scores = iter(llm.text_logprobs(prompts.score_prefix(lang), flat))
    records = [
        {"utterance_id": key, "llm_logprob": [round(next(scores), 4) for _ in texts]}
        for key, texts in zip(keys, candidates, strict=True)
    ]
    if not everything:
        return records
    by_size: dict[int, list[int]] = {}
    for i, texts in enumerate(candidates):
        by_size.setdefault(len(texts), []).append(i)
    for size, indices in by_size.items():
        conversations = [prompts.choose(candidates[i], lang) for i in indices]
        for i, logprobs in zip(indices, llm.choice_logprobs(conversations, size), strict=True):
            records[i]["choice_logprob"] = [round(value, 4) for value in logprobs]
    budget = answer_budget(llm, flat)
    for method, shots in ((ZERO_SHOT, ()), (FEW_SHOT, examples)):
        conversations = [prompts.correct(texts, lang, shots) for texts in candidates]
        for record, answer in zip(records, llm.generate(conversations, budget), strict=True):
            record[method] = answer
    return records


def answer_budget(llm: LanguageModel, texts: Sequence[str]) -> int:
    """The most tokens an answer may take: half again the longest transcript, and some room."""
    return int(1.5 * max((llm.count_tokens(text) for text in texts), default=0)) + 16


def _complete(output: Path, config: Mapping[str, Any]) -> bool:
    meta = output.with_suffix(".json")
    if not (output.exists() and meta.exists()):
        return False
    saved = json.loads(meta.read_text(encoding="utf-8"))
    return all(saved.get(key) == value for key, value in config.items())


def _write_llm_output(output: Path, records: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    meta = {**meta, "created": utc_now(), "git_commit": git_commit()}
    _write_json(meta, output.with_suffix(".json"))


def read_llm_output(path: Path) -> dict[str, dict[str, Any]]:
    """The records of a language-model output, by utterance."""
    with path.open(encoding="utf-8") as handle:
        return {(r := json.loads(line))["utterance_id"]: r for line in handle if line.strip()}


# 3. The methods


def rescored(
    nbest: Sequence[ScoredTranscript], llm_logprob: Sequence[float], weight: float, bonus: float
) -> str:
    """The transcript with the best combined score; see the spec. In a tie, the earlier one.

    >>> lists = [ScoredTranscript("a b", -1.0, 0.0), ScoredTranscript("a c", -1.5, 0.0)]
    >>> rescored(lists, [-9.0, -4.0], 0.0, 0.0), rescored(lists, [-9.0, -4.0], 0.5, 0.0)
    ('a b', 'a c')
    """
    totals = [
        t.score + weight * logp + bonus * len(t.text.split())
        for t, logp in zip(nbest, llm_logprob, strict=True)
    ]
    return nbest[max(range(len(totals)), key=lambda i: (totals[i], -i))].text


def chosen(nbest: Sequence[ScoredTranscript], choice_logprob: Sequence[float]) -> str:
    """The candidate whose number the language model finds most likely; in a tie, the earlier.

    >>> chosen([ScoredTranscript("a", 0, 0), ScoredTranscript("b", 0, 0)], [-2.0, -0.1])
    'b'
    """
    return nbest[max(range(len(choice_logprob)), key=lambda i: (choice_logprob[i], -i))].text


def rescoring_grid(
    spec: dict[str, Any],
    nbest: Mapping[str, Sequence[ScoredTranscript]],
    llm_output: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, str],
    normalize: Any,
) -> list[dict[str, float]]:
    """The dev WER of every rescoring setting of the spec."""
    keys = sorted(references)
    normalized = [references[key] for key in keys]
    grid = []
    for weight in spec["rescoring"]["weights"]:
        for bonus in spec["rescoring"]["word_bonuses"]:
            hypotheses = [
                normalize(rescored(nbest[key], llm_output[key]["llm_logprob"], weight, bonus))
                for key in keys
            ]
            counts = sum(utterance_counts(normalized, hypotheses, word_tokens), ErrorCounts())
            grid.append({"weight": weight, "word_bonus": bonus, "wer": error_rate(counts)})
    return grid


def select_rescoring(grid: Sequence[dict[str, float]]) -> dict[str, float]:
    """The setting with the lowest WER; in a tie, the smaller weight, then the smaller bonus.

    >>> select_rescoring(
    ...     [
    ...         {"weight": 0.5, "word_bonus": 0.0, "wer": 0.1},
    ...         {"weight": 0.2, "word_bonus": 1.0, "wer": 0.1},
    ...     ]
    ... )
    {'weight': 0.2, 'word_bonus': 1.0, 'wer': 0.1}
    """
    return min(grid, key=lambda cell: (cell["wer"], cell["weight"], cell["word_bonus"]))


def run_methods(spec_path: Path, runs_dir: Path, llm_dir: Path, selection_path: Path) -> list[Path]:
    """Choose the rescoring setting of each test set on dev, then write a run per method on test."""
    spec = load_spec(spec_path)
    selection = {}
    written = []
    for test_set in spec["test_sets"]:
        normalize = get_normalizer_by_name(test_set["normalizer"])
        dev_references = {
            entry.utterance_id: normalize(entry.text)
            for entry in read_manifest(Path(test_set["dev"]))
        }
        grid = rescoring_grid(
            spec,
            read_nbest(nbest_run_dir(runs_dir, test_set, "dev")),
            read_llm_output(llm_output_path(llm_dir, test_set, "dev")),
            dev_references,
            normalize,
        )
        chosen_setting = select_rescoring(grid)
        selection[test_set["name"]] = {"grid": grid, "selected": chosen_setting}
        print(f"{test_set['name']}: {chosen_setting}", flush=True)
        written += _write_method_runs(spec, test_set, runs_dir, llm_dir, chosen_setting)
    _write_json(
        {"test_sets": selection, "created": utc_now(), "git_commit": git_commit()}, selection_path
    )
    return written


def _write_method_runs(
    spec: dict[str, Any],
    test_set: dict[str, Any],
    runs_dir: Path,
    llm_dir: Path,
    setting: Mapping[str, float],
) -> list[Path]:
    output = llm_output_path(llm_dir, test_set, "test")
    meta = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    nbest_dir = nbest_run_dir(runs_dir, test_set, "test")
    nbest = read_nbest(nbest_dir)
    llm_output = read_llm_output(output)
    keys = sorted(nbest)
    weight, bonus = setting["weight"], setting["word_bonus"]
    texts = {
        RESCORE: [rescored(nbest[k], llm_output[k]["llm_logprob"], weight, bonus) for k in keys],
        CHOOSE: [chosen(nbest[k], llm_output[k]["choice_logprob"]) for k in keys],
        ZERO_SHOT: [clean_answer(llm_output[k][ZERO_SHOT]) for k in keys],
        FEW_SHOT: [clean_answer(llm_output[k][FEW_SHOT]) for k in keys],
    }
    nbest_config = json.loads((nbest_dir / "run.json").read_text(encoding="utf-8"))["model_config"]
    llm_config = {key: value for key, value in meta.items() if key not in ("created", "seconds")}
    written = []
    for method in spec["analysis"]["methods"]:
        config: dict[str, Any] = {"method": method, "nbest": nbest_config, "llm": llm_config}
        if method == RESCORE:
            config["rescoring"] = {k: setting[k] for k in ("weight", "word_bonus")}
        hypotheses = [Hypothesis(k, text) for k, text in zip(keys, texts[method], strict=True)]
        run_dir = method_run_dir(runs_dir, method, test_set)
        # The language model's time on the whole test set, shared by its methods
        write_run(
            run_dir, hypotheses, f"{meta['model']}+e8-{method}", config, Path(test_set["test"]),
            meta["seconds"],
        )  # fmt: skip
        written.append(run_dir)
    return written


def _write_json(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, indent=1) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
