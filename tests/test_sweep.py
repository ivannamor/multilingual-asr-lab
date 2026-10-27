"""Tests for decoding sweeps and the selection rule.

The sweeps run with a fake Whisper model whose output depends on the decoding
options and on the utterance, recognised by the length of its audio, so each
test decides which configuration makes errors on which utterances.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from asr_lab import cli
from asr_lab.data.manifest import ManifestEntry, write_manifest
from asr_lab.inference.base import SAMPLE_RATE
from asr_lab.inference.faster_whisper import FasterWhisperModel
from asr_lab.sweep import load_spec, run_sweep, select_configuration

N_UTTERANCES = 40
SAMPLES_PER_STEP = 1600  # utterance i lasts (i + 1) * 0.1 seconds
TEXTS = {"it": ("il gatto dorme", "il cane dorme"), "de": ("die katze schläft", "der hund schläft")}
SPEC = """
model_size = "small"
languages = ["it", "de"]
manifest = "data/fleurs_{{lang}}/{{split}}.jsonl"
run_dir = "runs/{{config}}-{{lang}}-{{split}}"

[normalizers]
it = "it-v1"
de = "de-v1"
{configs}
"""
# Decides whether a configuration (its options) gets utterance `index` of `lang` wrong.
ErrorRule = Callable[[dict[str, Any], str, int], bool]


@dataclass
class FakeSegment:
    text: str
    avg_logprob: float = -0.1


@dataclass
class FakeWhisper:
    wrong: ErrorRule
    calls: list[dict[str, Any]] = field(default_factory=list)

    def transcribe(self, audio: np.ndarray, **options: Any) -> tuple[Any, None]:
        self.calls.append(options)
        index = len(audio) // SAMPLES_PER_STEP - 1
        right, wrong = TEXTS[options["language"]]
        text = wrong if self.wrong(options, options["language"], index) else right
        return iter([FakeSegment(text)]), None


@pytest.fixture(autouse=True)
def in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    for lang, (text, _) in TEXTS.items():
        folder = Path(f"data/fleurs_{lang}")
        (folder / "audio").mkdir(parents=True)
        entries = []
        for i in range(N_UTTERANCES):
            audio = f"audio/{lang}{i}.wav"
            waveform = np.zeros((i + 1) * SAMPLES_PER_STEP)
            sf.write(folder / audio, waveform, SAMPLE_RATE, subtype="PCM_16")
            entries.append(
                ManifestEntry(f"{lang}{i}", audio, (i + 1) / 10, text, lang, f"s{i}", "t")
            )
        write_manifest(folder / "dev.jsonl", entries)


def _spec(configs: str) -> Path:
    path = Path("sweep.toml")
    path.write_text(SPEC.format(configs=configs), encoding="utf-8")
    return path


def _sweep(configs: str, wrong: ErrorRule, speeds: dict[str, float] | None = None) -> FakeWhisper:
    """Run the sweep on the dev split; ``speeds`` sets each configuration's processing time."""
    spec = load_spec(_spec(configs))
    fake = FakeWhisper(wrong)
    run_sweep(spec, "dev", model=FasterWhisperModel(model=fake), progress=False)
    for name, seconds in (speeds or {}).items():
        for lang in spec.languages:
            _edit_run(spec.run_dir(name, lang, "dev"), processing_seconds=seconds)
    return fake


def _edit_run(run_dir: Path, **fields: Any) -> None:
    path = run_dir / "run.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    for key, value in fields.items():
        if key in run["model_config"]:
            run["model_config"][key] = value
        else:
            run[key] = value
    path.write_text(json.dumps(run), encoding="utf-8")


THREE = """
[[config]]
name = "baseline"
[[config]]
name = "beam1"
options = { beam_size = 1 }
[[config]]
name = "beam2"
options = { beam_size = 2 }
"""


# --- spec ------------------------------------------------------------------------------


def test_experiment_e2_spec_is_valid() -> None:
    spec = load_spec(Path(__file__).parents[1] / "experiments/E2/sweep.toml")
    assert spec.model_size == "large-v3-turbo"
    assert spec.languages == ("it", "de", "en")
    assert spec.configs[0].name == "baseline" and spec.configs[0].options == {}
    assert [c.name for c in spec.configs if not c.selectable] == ["baseline-repeat"]
    for config in spec.configs:  # every option value is accepted by the adapter
        FasterWhisperModel(spec.model_size, model=object(), **config.options)
    assert spec.run_dir("vad", "de", "dev") == Path("experiments/runs/e2-turbo-vad-fleurs-de-dev")


@pytest.mark.parametrize(
    ("configs", "message"),
    [
        ('[[config]]\nname = "a"\n[[config]]\nname = "a"', "unique"),
        ('[[config]]\nname = "a"\noptions = { beam = 2 }', r"unknown decoding options \['beam'\]"),
        ('[[config]]\nname = "a"\nselectable = false', "no selectable"),
    ],
)
def test_invalid_specs_are_rejected(configs: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_spec(_spec(configs))


def test_language_without_normalizer_is_rejected() -> None:
    text = SPEC.format(configs='[[config]]\nname = "a"').replace('de = "de-v1"', "")
    Path("sweep.toml").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=r"no normalizer for \['de'\]"):
        load_spec(Path("sweep.toml"))


# --- sweep -----------------------------------------------------------------------------


def test_sweep_transcribes_every_configuration_and_language_with_its_options() -> None:
    fake = _sweep(THREE, lambda options, lang, index: False)
    spec = load_spec(Path("sweep.toml"))
    assert len(fake.calls) == 1 + 3 * 2 * N_UTTERANCES  # one warm-up, then every utterance
    for name, beam in (("baseline", 5), ("beam1", 1), ("beam2", 2)):
        for lang in ("it", "de"):
            run = json.loads((spec.run_dir(name, lang, "dev") / "run.json").read_text("utf-8"))
            assert run["model_config"]["beam_size"] == beam
            assert run["finished_at"] is not None
    assert {call["beam_size"] for call in fake.calls[1:]} == {5, 1, 2}


def test_sweep_skips_complete_runs() -> None:
    _sweep(THREE, lambda options, lang, index: False)
    again = _sweep(THREE, lambda options, lang, index: False)
    assert len(again.calls) == 1  # only the warm-up


# --- selection -------------------------------------------------------------------------


def test_fastest_configuration_not_significantly_worse_is_selected() -> None:
    def wrong(options: dict[str, Any], lang: str, index: int) -> bool:
        if options["beam_size"] == 1:
            return index % 2 == 0  # many errors: significantly worse
        return options["beam_size"] == 2 and index == 0  # one error: not significant

    _sweep(THREE, wrong, speeds={"baseline": 10.0, "beam1": 2.0, "beam2": 5.0})
    selection = select_configuration(load_spec(Path("sweep.toml")), n_resamples=200)
    assert selection["best"] == {"it": "baseline", "de": "baseline"}
    configs = selection["configs"]
    assert configs["beam1"]["eligible"] is False
    assert configs["beam2"]["eligible"] is True
    assert configs["beam2"]["vs_best"]["it"]["delta"] > 0
    assert configs["beam2"]["processing_seconds"] == 10.0  # summed over the two languages
    assert selection["selected"] == "beam2"
    assert selection["none_eligible"] is False


def test_baseline_is_kept_when_no_configuration_is_safe_in_every_language() -> None:
    def wrong(options: dict[str, Any], lang: str, index: int) -> bool:
        if options["beam_size"] == 1:  # perfect in Italian only
            return lang == "de" and index % 2 == 0
        if options["beam_size"] == 2:  # perfect in German only
            return lang == "it" and index % 2 == 0
        return index % 2 == 0  # baseline: bad in both

    _sweep(THREE, wrong)
    selection = select_configuration(load_spec(Path("sweep.toml")), n_resamples=200)
    assert selection["best"] == {"it": "beam1", "de": "beam2"}
    assert not any(result["eligible"] for result in selection["configs"].values())
    assert selection["selected"] == "baseline"
    assert selection["none_eligible"] is True


def test_repeated_configuration_is_reported_but_never_selected() -> None:
    configs = THREE + '[[config]]\nname = "repeat"\nselectable = false\n'
    _sweep(configs, lambda options, lang, index: False, speeds={"repeat": 0.1})
    selection = select_configuration(load_spec(Path("sweep.toml")), n_resamples=200)
    assert "eligible" not in selection["configs"]["repeat"]
    assert selection["configs"]["repeat"]["wer"] == {"it": 0.0, "de": 0.0}
    assert selection["selected"] != "repeat"


def test_ties_go_to_the_configuration_listed_first() -> None:
    _sweep(
        THREE,
        lambda options, lang, index: False,
        speeds=dict.fromkeys(["baseline", "beam1", "beam2"], 1.0),
    )
    selection = select_configuration(load_spec(Path("sweep.toml")), n_resamples=200)
    assert selection["best"] == {"it": "baseline", "de": "baseline"}
    assert selection["selected"] == "baseline"


def test_run_with_other_settings_is_rejected() -> None:
    _sweep(THREE, lambda options, lang, index: False)
    spec = load_spec(Path("sweep.toml"))
    _edit_run(spec.run_dir("beam1", "de", "dev"), beam_size=5)
    with pytest.raises(ValueError, match=r"different \['beam_size'\]"):
        select_configuration(spec, n_resamples=200)


def test_incomplete_or_missing_runs_are_rejected() -> None:
    _sweep(THREE, lambda options, lang, index: False)
    spec = load_spec(Path("sweep.toml"))
    _edit_run(spec.run_dir("beam2", "it", "dev"), finished_at=None)
    with pytest.raises(ValueError, match="incomplete"):
        select_configuration(spec, n_resamples=200)
    with pytest.raises(ValueError, match="Missing run"):
        select_configuration(spec, "test", n_resamples=200)


def test_cli_select(capsys: pytest.CaptureFixture[str]) -> None:
    _sweep(
        THREE,
        lambda options, lang, index: False,
        speeds={"baseline": 3.0, "beam1": 1.0, "beam2": 2.0},
    )
    cli.main(["select", "--spec", "sweep.toml", "--n-resamples", "200",
              "--output", "results/selection.json"])  # fmt: skip
    out = capsys.readouterr().out
    assert "Selected: beam1" in out
    assert json.loads(Path("results/selection.json").read_text("utf-8"))["selected"] == "beam1"
