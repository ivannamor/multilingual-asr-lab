"""Tests for the faster-whisper adapter.

The unit tests inject a fake model, so no weights are downloaded. The test marked
``slow`` runs the real Whisper-small on one FLEURS utterance; it is skipped by
default and runs with ``uv run pytest -m slow``.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from asr_lab.data.manifest import read_manifest
from asr_lab.evaluation.metrics import wer
from asr_lab.evaluation.normalize import get_normalizer
from asr_lab.inference.base import SAMPLE_RATE, AudioSample, load_sample
from asr_lab.inference.faster_whisper import FasterWhisperModel

FLEURS_IT = Path("data/fleurs_it/test.jsonl")


@dataclass
class FakeSegment:
    text: str
    avg_logprob: float


@dataclass
class FakeWhisper:
    """Mimics ``faster_whisper.WhisperModel.transcribe``: returns a generator of segments."""

    segments: list[FakeSegment]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def transcribe(self, audio: np.ndarray, **options: Any) -> tuple[Any, None]:
        self.calls.append(options)
        return (segment for segment in self.segments), None


def _sample(utterance_id: str = "u1", lang: str | None = "it") -> AudioSample:
    return AudioSample(utterance_id, np.zeros(SAMPLE_RATE, dtype="float32"), lang)


def _adapter(segments: list[FakeSegment], **kwargs: Any) -> tuple[FasterWhisperModel, FakeWhisper]:
    fake = FakeWhisper(segments)
    return FasterWhisperModel(model=fake, **kwargs), fake


def test_segments_are_joined_into_one_hypothesis() -> None:
    model, _ = _adapter([FakeSegment(" Buongiorno a tutti.", -0.2), FakeSegment(" Come va?", -0.4)])
    (hypothesis,) = model.transcribe([_sample("u1")])
    assert hypothesis.utterance_id == "u1"
    assert hypothesis.text == "Buongiorno a tutti. Come va?"
    assert hypothesis.score == pytest.approx(-0.3)


def test_no_segments_gives_empty_text_and_no_score() -> None:
    model, _ = _adapter([])
    (hypothesis,) = model.transcribe([_sample()])
    assert hypothesis.text == ""
    assert hypothesis.score is None


def test_blank_segments_are_skipped() -> None:
    model, _ = _adapter([FakeSegment(" ciao", -0.1), FakeSegment("  ", -2.0)])
    assert model.transcribe([_sample()])[0].text == "ciao"


def test_decoding_options_are_passed_to_the_model() -> None:
    model, fake = _adapter([FakeSegment("x", -0.1)], beam_size=3)
    model.transcribe([_sample(lang="de")])
    assert fake.calls == [
        {"language": "de", "beam_size": 3, "condition_on_previous_text": False, "vad_filter": False}
    ]


def test_every_sample_is_transcribed_in_order() -> None:
    model, fake = _adapter([FakeSegment("x", -0.1)])
    hypotheses = model.transcribe([_sample("a"), _sample("b"), _sample("c")])
    assert [h.utterance_id for h in hypotheses] == ["a", "b", "c"]
    assert len(fake.calls) == 3


def test_name_and_config_describe_the_decoding_setup() -> None:
    model, _ = _adapter([], model_size="large-v3-turbo", compute_type="float32")
    assert model.name == "faster-whisper/large-v3-turbo-float32"
    config = model.config
    assert config["model_size"] == "large-v3-turbo"
    assert config["compute_type"] == "float32"
    assert config["beam_size"] == 5
    assert config["cpu_threads"] == 4
    assert config["condition_on_previous_text"] is False
    assert config["library_version"]


def test_default_precision_depends_on_the_device() -> None:
    cpu = FasterWhisperModel(model=FakeWhisper([]))
    gpu = FasterWhisperModel(model=FakeWhisper([]), device="cuda")
    assert (cpu.name, cpu.config["device"]) == ("faster-whisper/small-int8", "cpu")
    assert (gpu.name, gpu.config["device"]) == ("faster-whisper/small-float16", "cuda")
    explicit = FasterWhisperModel(model=FakeWhisper([]), device="cuda", compute_type="int8")
    assert explicit.compute_type == "int8"


def test_unknown_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="device must be one of"):
        FasterWhisperModel(model=FakeWhisper([]), device="tpu")


@pytest.mark.slow
@pytest.mark.skipif(not FLEURS_IT.exists(), reason="run the FLEURS ingest first")
def test_real_whisper_small_on_one_fleurs_utterance() -> None:
    entry = read_manifest(FLEURS_IT)[0]
    (hypothesis,) = FasterWhisperModel().transcribe([load_sample(entry, FLEURS_IT)])
    normalize = get_normalizer("it")
    assert wer([normalize(entry.text)], [normalize(hypothesis.text)]) < 0.3
