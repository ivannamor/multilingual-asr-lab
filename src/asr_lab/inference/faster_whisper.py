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

# Fastest weight precision measured or recommended for each device.
DEFAULT_COMPUTE_TYPES = {"cpu": "int8", "cuda": "float16"}
# faster-whisper's temperature fallback: decode at 0 (beam search), and if the output
# looks broken (too repetitive or too unlikely) retry by sampling at higher temperatures.
DEFAULT_TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
# Options that are never varied, passed explicitly so that the recorded config is complete.
_FIXED_OPTIONS = {
    "condition_on_previous_text": False,
    "best_of": 5,  # candidates sampled at each fallback temperature
    "patience": 1.0,
    "length_penalty": 1.0,
}
# Settings that were implicit (faster-whisper 1.2.1 defaults) before they were recorded:
# a run saved without them used these values.
_IMPLICIT_CONFIG = {
    "temperature": list(DEFAULT_TEMPERATURES),
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "hallucination_silence_threshold": None,
    "word_timestamps": False,
    "best_of": 5,
    "patience": 1.0,
    "length_penalty": 1.0,
}


class FasterWhisperModel(ASRModel):
    """Whisper with fixed, reportable decoding settings.

    The language comes from each sample (the manifest), so Whisper never has to
    guess it. ``condition_on_previous_text`` is off: feeding back the previous
    window's text is a known source of repetition loops and hallucinations. Every
    decoding option is passed explicitly and recorded in ``config``, including the
    ones left at faster-whisper's defaults.

    Args:
        model_size: Whisper checkpoint, e.g. ``"small"`` or ``"large-v3-turbo"``.
        device: ``"cpu"`` or ``"cuda"`` (an NVIDIA GPU, e.g. on Google Colab).
        compute_type: weight precision; by default ``"int8"`` on CPU (fastest there)
            and ``"float16"`` on a GPU.
        beam_size: number of partial transcripts kept during decoding (1 = greedy).
        cpu_threads: threads used to transcribe one utterance on CPU.
        temperature: fallback temperatures; ``(0.0,)`` disables the fallback, so
            decoding is deterministic.
        compression_ratio_threshold, log_prob_threshold: when an output exceeds the
            first or falls below the second, decoding is retried at the next
            temperature (``None`` disables the check).
        no_speech_threshold: output nothing for a window whose no-speech probability
            is above this value (and whose log-probability is low).
        vad_filter: remove non-speech audio with Silero VAD before decoding.
        hallucination_silence_threshold: skip silences longer than this many seconds
            around a likely hallucination; needs word timestamps, which it enables.
        model: an already built model, to share loaded weights (see ``with_decoding``)
            or, in tests, to avoid downloading them.
    """

    def __init__(
        self,
        model_size: str = "small",
        *,
        device: str = "cpu",
        compute_type: str | None = None,
        beam_size: int = 5,
        cpu_threads: int = 4,
        temperature: Sequence[float] = DEFAULT_TEMPERATURES,
        compression_ratio_threshold: float | None = 2.4,
        log_prob_threshold: float | None = -1.0,
        no_speech_threshold: float | None = 0.6,
        vad_filter: bool = False,
        hallucination_silence_threshold: float | None = None,
        model: Any | None = None,
    ) -> None:
        if device not in DEFAULT_COMPUTE_TYPES:
            raise ValueError(
                f"device must be one of {sorted(DEFAULT_COMPUTE_TYPES)}, got {device!r}"
            )
        if beam_size < 1:
            raise ValueError(f"beam_size must be at least 1, got {beam_size}")
        if not temperature or any(value < 0 for value in temperature):
            raise ValueError(f"temperature must be non-empty and non-negative, got {temperature}")
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type or DEFAULT_COMPUTE_TYPES[device]
        self.beam_size = beam_size
        self.cpu_threads = cpu_threads
        self.decoding = {
            "temperature": [float(value) for value in temperature],
            "compression_ratio_threshold": compression_ratio_threshold,
            "log_prob_threshold": log_prob_threshold,
            "no_speech_threshold": no_speech_threshold,
            "vad_filter": vad_filter,
            "hallucination_silence_threshold": hallucination_silence_threshold,
            "word_timestamps": hallucination_silence_threshold is not None,
            **_FIXED_OPTIONS,
        }
        if model is None:
            model = WhisperModel(
                model_size,
                device=device,
                compute_type=self.compute_type,
                cpu_threads=cpu_threads,
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
            "device": self.device,
            "cpu_threads": self.cpu_threads,
            "beam_size": self.beam_size,
            **self.decoding,
        }

    @property
    def implicit_config(self) -> dict[str, Any]:
        """Values used by runs saved before these settings were recorded (1.2.1 defaults)."""
        return dict(_IMPLICIT_CONFIG)

    def with_decoding(self, **options: Any) -> FasterWhisperModel:
        """The same loaded weights with other decoding options (the rest at their defaults).

        Loading a large checkpoint takes longer than transcribing a short test set,
        so a sweep of decoding options loads it once.
        """
        return FasterWhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
            model=self._model,
            **options,
        )

    def _transcribe_batch(self, samples: Sequence[AudioSample]) -> list[Hypothesis]:
        return [self._transcribe_one(sample) for sample in samples]

    def _transcribe_one(self, sample: AudioSample) -> Hypothesis:
        segments, _info = self._model.transcribe(
            sample.waveform,
            language=sample.lang,
            beam_size=self.beam_size,
            **self.decoding,
        )
        # ``segments`` is a generator: decoding actually runs while it is consumed.
        segments = list(segments)
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        score = (
            sum(segment.avg_logprob for segment in segments) / len(segments) if segments else None
        )
        return Hypothesis(sample.utterance_id, text, score=score)
