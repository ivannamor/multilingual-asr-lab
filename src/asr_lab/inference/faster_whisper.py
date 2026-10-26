"""Whisper adapter based on faster-whisper (CTranslate2), the fastest option on CPU.

On the development machine (ARM64 CPU, no GPU) Whisper-small runs at a real-time
factor of about 0.34 with beam size 5, int8 weights and 4 threads, about six times
faster than the PyTorch implementation; more threads were measured to be slower.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import version
from typing import Any

from faster_whisper import WhisperModel

from asr_lab.inference.base import ASRModel, AudioSample, Hypothesis


class FasterWhisperModel(ASRModel):
    """Whisper with fixed, reportable decoding settings.

    The language comes from each sample (the manifest), so Whisper never has to
    guess it. ``condition_on_previous_text`` is off: feeding back the previous
    window's text is a known source of repetition loops and hallucinations.

    Args:
        model_size: Whisper checkpoint, e.g. ``"small"`` or ``"large-v3-turbo"``.
        compute_type: weight precision; ``"int8"`` is fastest on CPU.
        beam_size: number of partial transcripts kept during decoding (1 = greedy).
        cpu_threads: threads used to transcribe one utterance.
        model: an already built model, used by tests to avoid downloading weights.
    """

    def __init__(
        self,
        model_size: str = "small",
        *,
        compute_type: str = "int8",
        beam_size: int = 5,
        cpu_threads: int = 4,
        model: Any | None = None,
    ) -> None:
        self.model_size = model_size
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.cpu_threads = cpu_threads
        if model is None:
            model = WhisperModel(
                model_size, device="cpu", compute_type=compute_type, cpu_threads=cpu_threads
            )
        self._model = model

    @property
    def name(self) -> str:
        return f"faster-whisper/{self.model_size}-{self.compute_type}"

    @property
    def config(self) -> dict[str, Any]:
        """Every setting that can change the output, to be stored with the results."""
        return {
            "library": "faster-whisper",
            "library_version": version("faster-whisper"),
            "model_size": self.model_size,
            "compute_type": self.compute_type,
            "device": "cpu",
            "cpu_threads": self.cpu_threads,
            "beam_size": self.beam_size,
            "condition_on_previous_text": False,
            "vad_filter": False,
        }

    def _transcribe_batch(self, samples: Sequence[AudioSample]) -> list[Hypothesis]:
        return [self._transcribe_one(sample) for sample in samples]

    def _transcribe_one(self, sample: AudioSample) -> Hypothesis:
        segments, _info = self._model.transcribe(
            sample.waveform,
            language=sample.lang,
            beam_size=self.beam_size,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        # ``segments`` is a generator: decoding actually runs while it is consumed.
        segments = list(segments)
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        score = (
            sum(segment.avg_logprob for segment in segments) / len(segments) if segments else None
        )
        return Hypothesis(sample.utterance_id, text, score=score)
