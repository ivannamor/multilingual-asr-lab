"""The common interface of every ASR model: audio samples in, one hypothesis per sample out.

Adapters (Whisper, wav2vec2, ...) subclass ``ASRModel`` and implement
``_transcribe_batch``. Audio loading happens once, outside the models, so every
adapter receives the same input: mono float32 waveforms at 16 kHz.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from asr_lab.data.manifest import ManifestEntry

SAMPLE_RATE = 16_000


@dataclass(frozen=True, eq=False)
class AudioSample:
    """One utterance ready for a model.

    ``eq=False``: comparing numpy arrays with ``==`` is element-wise, so the
    generated equality would be ambiguous; samples are compared by identity.
    """

    utterance_id: str
    waveform: np.ndarray  # mono, float32, SAMPLE_RATE Hz
    lang: str | None = None  # language hint, e.g. to force Whisper's language


@dataclass(frozen=True)
class Hypothesis:
    """The model output for one utterance, as raw text (not normalized)."""

    utterance_id: str
    text: str
    nbest: tuple[str, ...] = ()  # alternative transcripts, best first, if the decoder has them
    score: float | None = None  # model score of ``text`` (e.g. average log-probability)


class ASRModel(ABC):
    """Base class of all model adapters.

    ``transcribe`` is the public entry point: it calls the adapter's
    ``_transcribe_batch`` and checks that every sample got exactly one
    hypothesis, in the same order, so a buggy adapter cannot silently
    misalign predictions and references.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier reported with every result, e.g. ``"openai/whisper-small"``."""

    @property
    def config(self) -> dict[str, Any]:
        """Settings that can change the output, stored with every run; adapters override it."""
        return {}

    @abstractmethod
    def _transcribe_batch(self, samples: Sequence[AudioSample]) -> list[Hypothesis]:
        """Transcribe a batch of samples; return hypotheses in the same order."""

    def transcribe(self, samples: Sequence[AudioSample]) -> list[Hypothesis]:
        """Transcribe samples and validate the adapter's output.

        Raises:
            ValueError: if the adapter returns a different number of hypotheses
                or hypotheses in a different order than the samples.
        """
        if not samples:
            return []
        hypotheses = self._transcribe_batch(samples)
        expected = [sample.utterance_id for sample in samples]
        got = [hypothesis.utterance_id for hypothesis in hypotheses]
        if got != expected:
            raise ValueError(
                f"{self.name} returned hypotheses for {got[:3]}... "
                f"but was given {expected[:3]}... ({len(got)} vs {len(expected)} items)"
            )
        return hypotheses


def load_sample(entry: ManifestEntry, manifest_path: Path) -> AudioSample:
    """Read the audio of a manifest entry as a mono float32 waveform at 16 kHz.

    Raises:
        ValueError: if the file is not mono 16 kHz (run the ingest step first).
    """
    path = entry.audio_path(manifest_path)
    waveform, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != SAMPLE_RATE or waveform.ndim != 1:
        raise ValueError(
            f"{path}: expected mono {SAMPLE_RATE} Hz audio, got "
            f"{sample_rate} Hz with shape {waveform.shape}"
        )
    return AudioSample(entry.utterance_id, waveform, entry.lang)
