"""Experiment E8: N-best lists, prompts, the language model's answers and the four methods."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from asr_lab.correction.experiment import (
    CHOOSE,
    FEW_SHOT,
    RESCORE,
    ZERO_SHOT,
    answer_budget,
    ask_llm,
    choose_examples,
    load_examples,
    method_run_dir,
    nbest_run_dir,
    read_llm_output,
    read_nbest,
    rescoring_grid,
    run_llm,
    run_methods,
    select_rescoring,
)
from asr_lab.correction.prompts import Example, Message, clean_answer, load_prompts
from asr_lab.data.manifest import ManifestEntry, write_manifest
from asr_lab.evaluation.normalize import get_normalizer_by_name
from asr_lab.inference.base import Hypothesis
from asr_lab.inference.predictions import append_predictions
from asr_lab.lm.decoder import ScoredTranscript

PROMPTS = Path("experiments/E8/prompts.toml")
ALPHABET = frozenset("abcdefghijklmnopqrstuvwxyz'")


class FakeLLM:
    """Prefers texts with "roma", picks the last candidate, and corrects "rome" to "roma"."""

    name = "fake/llm"

    def __init__(self) -> None:
        self.config = {"revision": "abc", "batch_size": 2}
        self.conversations: list[list[Message]] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def text_logprobs(self, prefix: str, texts: Sequence[str]) -> list[float]:
        assert prefix.startswith("Italian")
        return [-1.0 if "roma" in text else -10.0 for text in texts]

    def choice_logprobs(self, conversations: Sequence[list[Message]], n: int) -> list[list[float]]:
        return [[-5.0] * (n - 1) + [-0.1] for _ in conversations]

    def generate(self, conversations: Sequence[list[Message]], max_new_tokens: int) -> list[str]:
        self.conversations += conversations
        answers = []
        for messages in conversations:
            first = messages[-1]["content"].split("\n1. ")[1].split("\n")[0]
            answers.append(f'"{first.replace("rome", "roma")}"')
        return answers


def scored(*texts: str) -> list[ScoredTranscript]:
    return [ScoredTranscript(text, -float(i), 0.0) for i, text in enumerate(texts)]


def test_prompts_list_the_candidates_and_put_examples_before_the_question() -> None:
    prompts = load_prompts(PROMPTS)
    assert prompts.version == "e8-v1"
    assert len(prompts.sha256) == 64
    messages = prompts.choose(["vado a rome", "vado a roma"], "it")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "2 most likely transcripts of one Italian utterance" in messages[1]["content"]
    assert "1. vado a rome\n2. vado a roma" in messages[1]["content"]
    example = Example("d0", ("la casa", "la cosa"), "la casa")
    few = prompts.correct(["vado a rome"], "it", [example])
    assert [m["role"] for m in few] == ["system", "user", "assistant", "user"]
    assert few[2]["content"] == "la casa"
    assert "1. vado a rome" in few[3]["content"]


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("vado a roma", "vado a roma"),
        ("  2) vado a roma  \n\nThe second one.", "vado a roma"),
        ("«vado a roma»", "vado a roma"),
        ("\n\n", ""),
    ],
)
def test_clean_answer_keeps_the_transcript_only(answer: str, expected: str) -> None:
    assert clean_answer(answer) == expected


def test_examples_have_an_error_and_a_reference_the_model_can_write() -> None:
    nbest = {
        "right": scored("la casa"),
        "wrong": scored("la cosa", "la casa"),
        "digits": scored("nel novantanove"),
        "long": scored("a " * 40),
    }
    references = {"right": "La casa.", "wrong": "La casa!", "digits": "Nel '99", "long": "b " * 40}
    settings = {"n_examples": 1, "max_words": 30, "seed": 0}
    normalize = get_normalizer_by_name("it-v2")
    examples = choose_examples(nbest, references, ALPHABET, normalize, settings)
    assert examples == [Example("wrong", ("la cosa", "la casa"), "la casa")]
    with pytest.raises(ValueError, match="larger than population"):
        choose_examples(nbest, references, ALPHABET, normalize, {**settings, "n_examples": 2})


def test_ask_llm_on_dev_only_scores_the_transcripts() -> None:
    llm = FakeLLM()
    nbest = {"u1": scored("vado a rome", "vado a roma")}
    records = ask_llm(llm, load_prompts(PROMPTS), nbest, "it", [], everything=False)
    assert records == [{"utterance_id": "u1", "llm_logprob": [-10.0, -1.0]}]
    assert llm.conversations == []


def test_ask_llm_on_test_groups_lists_by_size_for_the_choice() -> None:
    llm = FakeLLM()
    nbest = {"u1": scored("vado a rome", "vado a roma"), "u2": scored("a", "b", "c")}
    example = Example("d0", ("la cosa",), "la casa")
    records = ask_llm(llm, load_prompts(PROMPTS), nbest, "it", [example], everything=True)
    assert records[0]["choice_logprob"] == [-5.0, -0.1]
    assert len(records[1]["choice_logprob"]) == 3
    assert clean_answer(records[0][ZERO_SHOT]) == "vado a roma"
    # Zero-shot conversations have no examples, few-shot ones have one
    assert [len(c) for c in llm.conversations] == [2, 2, 4, 4]


def test_answer_budget_grows_with_the_longest_transcript() -> None:
    assert answer_budget(FakeLLM(), ["a b", "a b c d"]) == 22


# A whole experiment on two utterances per split


def write_experiment(root: Path) -> tuple[Path, Path, Path]:
    test_set = {"name": "set", "lang": "it", "normalizer": "it-v2"}
    references = {"dev": ["vado a roma", "parto oggi"], "test": ["vado a roma", "torno domani"]}
    lists = {
        "dev": [["vado a rome", "vado a roma"], ["parto oggi", "parto oggi a"]],
        "test": [["vado a rome", "vado a roma"], ["torno domani", "torno a domani"]],
    }
    runs = root / "runs"
    for split in ("dev", "test"):
        manifest = root / f"{split}.jsonl"
        write_manifest(
            manifest,
            [
                ManifestEntry(f"{split}{i}", f"{i}.wav", 2.0, text, "it", f"s{i}", "test")
                for i, text in enumerate(references[split])
            ],
        )
        test_set[split] = manifest.as_posix()
        run = nbest_run_dir(runs, test_set, split)
        run.mkdir(parents=True)
        append_predictions(
            run / "predictions.jsonl",
            [
                Hypothesis(f"{split}{i}", texts[0], tuple(texts[1:]))
                for i, texts in enumerate(lists[split])
            ],
        )
        scores = [
            {"utterance_id": f"{split}{i}", "acoustic": [-1.0, -1.2], "language": [0.0, 0.0]}
            for i in range(2)
        ]
        (run / "nbest_scores.jsonl").write_text(
            "\n".join(json.dumps(s) for s in scores) + "\n", encoding="utf-8"
        )
        (run / "run.json").write_text(json.dumps({"model_config": {"n": 2}}), encoding="utf-8")
    examples = root / "examples.json"
    rows = [{"utterance_id": "dev0", "candidates": ["vado a rome"], "target": "vado a roma"}]
    examples.write_text(json.dumps({"examples": {"set": rows}}), encoding="utf-8")
    spec = root / "correction.toml"
    sets = "\n".join(f'{key} = "{value}"' for key, value in test_set.items())
    spec.write_text(
        f"""
[llm]
prompts = "{PROMPTS.as_posix()}"

[rescoring]
weights = [0.0, 1.0]
word_bonuses = [0.0, 1.0]

[[test_sets]]
{sets}

[analysis]
methods = ["rescore", "choose", "zero-shot", "few-shot"]
""",
        encoding="utf-8",
    )
    return spec, examples, runs


@pytest.fixture
def experiment(tmp_path: Path) -> dict[str, Any]:
    spec, examples, runs = write_experiment(tmp_path)
    llm_dir = tmp_path / "llm"
    run_llm(spec, runs, examples, llm_dir, FakeLLM())
    return {"spec": spec, "examples": examples, "runs": runs, "llm": llm_dir, "root": tmp_path}


def test_run_llm_writes_each_split_and_keeps_a_complete_output(experiment: dict) -> None:
    dev = read_llm_output(experiment["llm"] / "set-dev.jsonl")
    test = read_llm_output(experiment["llm"] / "set-test.jsonl")
    assert set(dev["dev0"]) == {"utterance_id", "llm_logprob"}
    assert set(test["test0"]) >= {"choice_logprob", ZERO_SHOT, FEW_SHOT}
    meta = json.loads((experiment["llm"] / "set-test.json").read_text(encoding="utf-8"))
    assert meta["prompts"] == "e8-v1"
    assert meta["model"] == "fake/llm"

    class Refusing(FakeLLM):
        def text_logprobs(self, prefix: str, texts: Sequence[str]) -> list[float]:
            raise AssertionError("a complete output is asked again")

    run_llm(experiment["spec"], experiment["runs"], experiment["examples"], experiment["llm"],
            Refusing())  # fmt: skip


def test_rescoring_needs_the_llm_and_a_tie_keeps_the_smaller_setting(experiment: dict) -> None:
    from asr_lab.lm.corpus import load_spec

    spec = load_spec(experiment["spec"])
    nbest = read_nbest(experiment["runs"] / "e8-nbest-set-dev")
    llm_output = read_llm_output(experiment["llm"] / "set-dev.jsonl")
    references = {"dev0": "vado a roma", "dev1": "parto oggi"}
    grid = rescoring_grid(spec, nbest, llm_output, references, str)
    by_setting = {(c["weight"], c["word_bonus"]): c["wer"] for c in grid}
    assert by_setting[(0.0, 0.0)] == pytest.approx(1 / 5)
    assert by_setting[(1.0, 0.0)] == 0.0
    # A bonus per word makes "parto oggi a" win
    assert by_setting[(0.0, 1.0)] == pytest.approx(2 / 5)
    assert select_rescoring(grid) == {"weight": 1.0, "word_bonus": 0.0, "wer": 0.0}


def test_run_methods_writes_one_test_run_per_method(experiment: dict) -> None:
    selection = experiment["root"] / "selection.json"
    run_methods(experiment["spec"], experiment["runs"], experiment["llm"], selection)
    chosen = json.loads(selection.read_text(encoding="utf-8"))["test_sets"]["set"]["selected"]
    assert chosen["weight"] == 1.0
    test_set = {"name": "set"}
    texts = {}
    for method in (RESCORE, CHOOSE, ZERO_SHOT, FEW_SHOT):
        run = method_run_dir(experiment["runs"], method, test_set)
        records = [json.loads(line) for line in (run / "predictions.jsonl").open()]
        texts[method] = [r["text"] for r in records]
        config = json.loads((run / "run.json").read_text(encoding="utf-8"))["model_config"]
        assert config["method"] == method
        assert config["llm"]["prompts"] == "e8-v1"
    assert texts[RESCORE] == ["vado a roma", "torno domani"]
    assert texts[CHOOSE] == ["vado a roma", "torno a domani"]  # the fake always picks the last
    assert texts[ZERO_SHOT] == ["vado a roma", "torno domani"]


def test_examples_round_trip(experiment: dict) -> None:
    assert load_examples(experiment["examples"]) == {
        "set": [Example("dev0", ("vado a rome",), "vado a roma")]
    }
