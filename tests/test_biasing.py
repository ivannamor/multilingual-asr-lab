"""Telling a model which names to expect: entities, lists, scoring, selection, hotwords, prompts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from asr_lab.biasing.entities import clean_entity, distractors, domain_list, prompt_text
from asr_lab.biasing.experiment import (
    conditions_for,
    ctc_alphabet,
    domain_lists,
    hotword_bonuses,
    oracle_lists,
    run_selection,
    score_lists,
    select_setting,
    whisper_plan,
    whisper_run_dir,
)
from asr_lab.biasing.scoring import entity_counts, find_phrase, invented_entities
from asr_lab.data.manifest import ManifestEntry, write_manifest
from asr_lab.evaluation.metrics import ErrorCounts
from asr_lab.evaluation.normalize import get_normalizer_by_name
from asr_lab.inference.base import Hypothesis
from asr_lab.inference.predictions import append_predictions
from asr_lab.lm.decoder import beam_search

LETTERS = frozenset("abcdefghijklmnopqrstuvwxyzàèéìòù'")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("the Indian Ocean", "Indian Ocean"),
        ("Der Rhein", "Rhein"),
        ("North Korea's", "North Korea"),
        ("Nord\N{RIGHT SINGLE QUOTATION MARK}s", "Nord"),
        ("  Stati   Uniti ", "Stati Uniti"),
        ("truffe", None),
        ("Paesi dell'", None),
        ("the", None),
        ("", None),
    ],
)
def test_clean_entity(text: str, expected: str | None) -> None:
    assert clean_entity(text) == expected


def test_domain_list_joins_case_variants_and_sorts_by_count() -> None:
    rows = domain_list([["Europa", "UE"], ["EUROPA", "Europa", "Italia"], ["Italia"]])
    assert rows == [
        {"entity": "Europa", "count": 3},
        {"entity": "Italia", "count": 2},
        {"entity": "UE", "count": 1},
    ]


def test_distractors_share_no_entity_and_never_come_from_the_utterance() -> None:
    found = {
        "a": ["Roma", "Milano"],
        "b": ["Roma"],
        "c": ["Parigi"],
        "d": ["Berlino", "PARIGI"],
        "e": [],
    }
    lists = distractors(found, seed=0)
    assert lists.keys() == found.keys()
    for key, listed in lists.items():
        own = {entity.casefold() for entity in found[key]}
        assert listed
        assert not own & {entity.casefold() for entity in listed}


def test_distractors_are_reproducible_and_can_be_empty() -> None:
    found = {str(i): [f"Nome{i}"] for i in range(20)}
    assert distractors(found, seed=1) == distractors(found, seed=1)
    assert distractors(found, seed=1) != distractors(found, seed=2)
    assert distractors({"a": ["Roma"], "b": ["Roma"]}, seed=0) == {"a": [], "b": []}


def test_oracle_and_domain_lists() -> None:
    assert oracle_lists({"a": ["Roma", "Roma", "UE"]}) == {"a": ["Roma", "UE"]}
    rows = [{"entity": name, "count": 1} for name in ("Uno", "Due", "Tre")]
    assert domain_lists(rows, 2, ["a", "b"]) == {"a": ["Due", "Uno"], "b": ["Due", "Uno"]}
    assert domain_lists([], 2, ["a"]) == {}


def test_prompt_text() -> None:
    assert prompt_text(["Roma", "UE"]) == "Roma, UE."
    assert prompt_text([]) is None


def test_find_phrase_finds_every_position() -> None:
    assert find_phrase(["a", "b", "a"], ["a"]) == [0, 2]
    assert find_phrase(["a"], ["a", "b"]) == []


def test_an_entity_counts_only_if_all_its_words_are_right_in_place() -> None:
    reference = ["la", "unione", "europea"]
    assert entity_counts(reference, reference, [["unione", "europea"]]) == ErrorCounts(hits=1)
    wrong = ["la", "unione", "europa"]
    assert entity_counts(reference, wrong, [["unione", "europea"]]) == ErrorCounts(substitutions=1)
    elsewhere = entity_counts(["roma", "e", "milano"], ["milano", "e", "roma"], [["milano"]])
    assert elsewhere == ErrorCounts(substitutions=1)
    assert entity_counts(reference, reference, [["roma"]]) == ErrorCounts()


def test_a_shorter_entity_inside_a_longer_one_counts_once() -> None:
    reference = ["unione", "europea", "e", "unione"]
    counts = entity_counts(reference, reference, [["unione"], ["unione", "europea"]])
    assert counts == ErrorCounts(hits=2)


def test_repeated_entities_in_the_list_count_once_per_occurrence() -> None:
    reference = ["roma", "e", "roma"]
    counts = entity_counts(reference, ["roma", "e", "milano"], [["roma"], ["roma"]])
    assert counts == ErrorCounts(hits=1, substitutions=1)


def test_invented_entities_ignore_those_in_the_reference() -> None:
    assert invented_entities(["roma"], ["roma"], [["roma"]]) == 0
    assert invented_entities(["roma"], ["roma", "roma"], [["roma"]]) == 1
    assert invented_entities([], ["parigi", "roma"], [["parigi"], ["parigi"], ["berlino"]]) == 1


def test_score_lists_normalizes_the_entities() -> None:
    normalizer = get_normalizer_by_name("it-v1")
    scores = score_lists(["vado a roma"], ["vado a roma"], [["Roma", "Milano"]], normalizer)
    assert scores == {"wer": 0.0, "recall": 1.0, "n_entities": 1}


def test_selection_prefers_recall_within_the_margin_then_the_smaller_setting() -> None:
    cells = [
        {"size": 200, "weight": 1.0, "wer": 0.20, "recall": 0.9},
        {"size": 50, "weight": 2.0, "wer": 0.20, "recall": 0.9},
        {"size": 50, "weight": 1.0, "wer": 0.20, "recall": 0.9},
        {"size": 50, "weight": 3.0, "wer": 0.30, "recall": 1.0},
    ]
    assert select_setting(cells, 0.20, 0.01) == cells[2]
    assert select_setting(cells, 0.10, 0.01) is None
    assert select_setting([], 0.10, 0.01) is None


def test_hotwords_keep_the_words_the_model_can_write() -> None:
    bonuses = hotword_bonuses({"a": ["Unione europea", "G7", "Москва"], "b": []}, LETTERS, 2.0)
    assert bonuses == {"a": {"unione": 2.0, "europea": 2.0}, "b": {}}


def test_ctc_alphabet_leaves_out_delimiters_and_special_tokens() -> None:
    assert ctc_alphabet(["<pad>", "|", "-", "a", "'", "<unk>"]) == frozenset("a'")


def test_a_word_bonus_changes_the_word_chosen_by_the_search() -> None:
    tokens = ["<pad>", "|", "a", "b"]
    log_probs = np.log(np.array([[0.05, 0.05, 0.5, 0.4]]))
    assert beam_search(log_probs, tokens) == "a"
    assert beam_search(log_probs, tokens, word_bonus={"b": 1.0}) == "b"
    assert beam_search(log_probs, tokens, word_bonus={"c": 1.0}) == "a"


class _FakeWhisper:
    def __init__(self) -> None:
        self.prompts: list[str | None] = []

    def transcribe(self, waveform, **options):
        self.prompts.append(options.get("initial_prompt"))
        return iter(()), None


def test_whisper_gets_the_prompt_of_each_utterance_and_records_it() -> None:
    from asr_lab.inference.base import AudioSample
    from asr_lab.inference.faster_whisper import FasterWhisperModel

    fake = _FakeWhisper()
    model = FasterWhisperModel(model=fake, initial_prompts={"u1": "Roma, UE."})
    samples = [AudioSample(key, np.zeros(16_000, dtype=np.float32), "it") for key in ("u1", "u2")]
    model.transcribe(samples)
    assert fake.prompts == ["Roma, UE.", None]
    other = FasterWhisperModel(model=fake, initial_prompts={"u1": "Parigi."})
    assert "initial_prompts_sha256" in model.config
    assert model.config["initial_prompts_sha256"] != other.config["initial_prompts_sha256"]
    assert "initial_prompts_sha256" not in FasterWhisperModel(model=fake).config


SPEC = {
    "entities": {"seed": 0},
    "whisper": {"domain_sizes": [1, 2]},
    "test_sets": [{"name": "fleurs"}, {"name": "vox", "has_domain_list": True}],
}
REPORT = {
    "domain": [{"entity": "Europa", "count": 3}, {"entity": "UE", "count": 1}],
    "test_sets": {
        "fleurs": {"dev": {"f1": ["Roma"]}, "test": {"f2": ["Roma"], "f3": ["Parigi"]}},
        "vox": {"dev": {"v1": ["Europa"]}, "test": {"v2": ["Europa"], "v3": ["UE"]}},
    },
}


def test_whisper_runs_on_dev_only_where_there_is_a_setting_to_choose() -> None:
    plan = whisper_plan(SPEC, REPORT, "dev", None)
    assert [(test_set["name"], condition) for test_set, condition, _ in plan] == [
        ("vox", "none"),
        ("vox", "domain-1"),
        ("vox", "domain-2"),
    ]
    assert plan[0][2] == {}
    assert plan[2][2] == {"v1": ["UE", "Europa"]}


def test_whisper_test_plan_uses_the_chosen_size_and_never_reruns_no_list() -> None:
    plan = whisper_plan(SPEC, REPORT, "test", {"whisper": {"domain_size": 1}})
    names = [(test_set["name"], condition) for test_set, condition, _ in plan]
    assert names == [
        ("fleurs", "oracle"),
        ("fleurs", "distractors"),
        ("vox", "oracle"),
        ("vox", "distractors"),
        ("vox", "domain-1"),
    ]
    assert plan[4][2] == {"v2": ["Europa"], "v3": ["Europa"]}


def test_without_a_chosen_size_there_is_no_domain_run() -> None:
    plan = whisper_plan(SPEC, REPORT, "test", {"whisper": {"domain_size": None}})
    assert all(not condition.startswith("domain") for _, condition, _ in plan)


def test_conditions_never_use_another_split() -> None:
    lists = conditions_for(SPEC, REPORT, SPEC["test_sets"][1], "test", 2)
    assert set(lists) == {"oracle", "distractors", "domain-2"}
    assert set(lists["oracle"]) == {"v2", "v3"}


def test_selection_file_from_the_dev_results(tmp_path: Path) -> None:
    manifest = tmp_path / "dev.jsonl"
    write_manifest(
        manifest,
        [ManifestEntry("v1", "v1.wav", 2.0, "vado in Europa", "it", "s1", "test")],
    )
    test_set = {
        "name": "vox",
        "normalizer": "it-v1",
        "dev": manifest.as_posix(),
        "has_domain_list": True,
    }
    runs = tmp_path / "runs"
    for condition, text in (
        ("none", "vado in eurona"),
        ("domain-1", "vado in europa"),
        ("domain-2", "vado in europa"),
    ):
        run = whisper_run_dir(runs, condition, test_set, "dev")
        run.mkdir(parents=True)
        append_predictions(run / "predictions.jsonl", [Hypothesis("v1", text)])
    spec = tmp_path / "spec.toml"
    spec.write_text(
        f"""
[whisper]
domain_sizes = [1, 2]
[selection]
wer_margin = 0.002
[[test_sets]]
name = "vox"
normalizer = "it-v1"
dev = "{manifest.as_posix()}"
has_domain_list = true
""",
        encoding="utf-8",
    )
    entities = tmp_path / "entities.json"
    entities.write_text(
        json.dumps({"domain": [], "test_sets": {"vox": {"dev": {"v1": ["Europa"]}}}}),
        encoding="utf-8",
    )
    grid = tmp_path / "grid.json"
    baseline = {"list": "none", "wer": 0.2, "recall": 0.5, "n_entities": 2}
    good = {"weight": 1.0, "wer": 0.2, "recall": 0.9, "n_entities": 2}
    bad = {"weight": 8.0, "wer": 0.5, "recall": 1.0, "n_entities": 2}
    grid.write_text(
        json.dumps(
            {"test_sets": {"vox": {"baseline": baseline, "oracle": [good, bad], "domain": []}}}
        ),
        encoding="utf-8",
    )
    output = tmp_path / "selection.json"
    run_selection(spec, entities, grid, runs, output)
    selection = json.loads(output.read_text(encoding="utf-8"))
    assert selection["whisper"]["domain_size"] == 1
    assert selection["ctc"]["vox"]["oracle"]["weight"] == 1.0
    assert selection["ctc"]["vox"]["domain"] is None
