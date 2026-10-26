"""Tests for the common ASR model interface."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asr_lab.data.manifest import ManifestEntry
from asr_lab.inference.base import (
    SAMPLE_RATE,
    ASRModel,
    AudioSample,
    Hypothesis,
    load_sample,
)


class FakeModel(ASRModel):
    """Returns a fixed transcript; ``bug`` simulates broken adapters."""

    def __init__(self, bug: str | None = None) -> None:
        self.bug = bug
        self.calls = 0

    @property
    def name(self) -> str:
        return "fake-model"

    def _transcribe_batch(self, samples: Sequence[AudioSample]) -> list[Hypothesis]:
        self.calls += 1
        hypotheses = [Hypothesis(s.utterance_id, f"text of {s.utterance_id}") for s in samples]
        if self.bug == "drop":
            return hypotheses[:-1]
        if self.bug == "reverse":
            return hypotheses[::-1]
        return hypotheses


def _samples(*ids: str) -> list[AudioSample]:
    return [AudioSample(i, np.zeros(SAMPLE_RATE, dtype="float32"), "it") for i in ids]


def test_transcribe_returns_one_hypothesis_per_sample_in_order() -> None:
    hypotheses = FakeModel().transcribe(_samples("a", "b", "c"))
    assert [h.utterance_id for h in hypotheses] == ["a", "b", "c"]
    assert hypotheses[0] == Hypothesis("a", "text of a")


def test_empty_batch_does_not_call_the_model() -> None:
    model = FakeModel()
    assert model.transcribe([]) == []
    assert model.calls == 0


@pytest.mark.parametrize("bug", ["drop", "reverse"])
def test_misaligned_adapter_output_is_rejected(bug: str) -> None:
    with pytest.raises(ValueError, match="fake-model returned hypotheses"):
        FakeModel(bug=bug).transcribe(_samples("a", "b", "c"))


def test_adapter_must_implement_the_abstract_methods() -> None:
    class Incomplete(ASRModel):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()  # type: ignore[abstract]


def test_hypothesis_defaults() -> None:
    hypothesis = Hypothesis("a", "ciao")
    assert hypothesis.nbest == ()
    assert hypothesis.score is None


def _entry(tmp_path: Path, waveform: np.ndarray, sample_rate: int) -> tuple[ManifestEntry, Path]:
    sf.write(tmp_path / "audio" / "u1.wav", waveform, sample_rate, subtype="PCM_16")
    entry = ManifestEntry("u1", "audio/u1.wav", 1.0, "ciao", "it", "spk", "test")
    return entry, tmp_path / "test.jsonl"


def test_load_sample_reads_mono_16k_audio(tmp_path: Path) -> None:
    (tmp_path / "audio").mkdir()
    entry, manifest_path = _entry(tmp_path, np.full(SAMPLE_RATE, 0.5), SAMPLE_RATE)
    sample = load_sample(entry, manifest_path)
    assert sample.utterance_id == "u1"
    assert sample.lang == "it"
    assert sample.waveform.dtype == np.float32
    assert sample.waveform.shape == (SAMPLE_RATE,)
    assert sample.waveform[0] == pytest.approx(0.5, abs=1e-4)


@pytest.mark.parametrize(
    ("waveform", "sample_rate"),
    [(np.zeros(8_000), 8_000), (np.zeros((SAMPLE_RATE, 2)), SAMPLE_RATE)],
)
def test_load_sample_rejects_wrong_format(
    tmp_path: Path, waveform: np.ndarray, sample_rate: int
) -> None:
    (tmp_path / "audio").mkdir()
    entry, manifest_path = _entry(tmp_path, waveform, sample_rate)
    with pytest.raises(ValueError, match="expected mono 16000 Hz audio"):
        load_sample(entry, manifest_path)
