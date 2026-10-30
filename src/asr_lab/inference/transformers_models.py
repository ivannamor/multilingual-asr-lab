"""Adapters for models run with Hugging Face Transformers: a CTC and a Transducer family.

- ``Wav2Vec2CTCModel``: wav2vec2 with a CTC head. The encoder gives one distribution over the
  characters per audio frame; decoding here is greedy (the best character of each frame, then
  repeats and blanks removed), so no language model is involved.
- ``ParakeetTDTModel``: Parakeet, a Token-and-Duration Transducer. A small predictor network
  remembers the tokens written so far, and each step also predicts how many frames to skip.

Utterances are transcribed one at a time, as in the Whisper adapter: no padding can change the
output, and the real-time factors of the families compare the same way of working. PyTorch and
Transformers are imported only when a model is loaded: they are an optional extra (``hf``).
"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from collections.abc import Sequence
from importlib.metadata import version
from typing import Any

from asr_lab.inference.base import SAMPLE_RATE, ASRModel, AudioSample, Hypothesis

# The precision of each device, as for Whisper on a GPU; half precision is not useful on a CPU
DEFAULT_DTYPES = {"cpu": "float32", "cuda": "float16"}
DTYPES = ("float16", "float32")


class TransformersModel(ASRModel):
    """A checkpoint at a pinned revision, on one device and in one precision.

    Args:
        model_id: repository on the Hugging Face Hub.
        revision: commit of the repository, so that the weights can never change under a run.
        device: ``"cpu"`` or ``"cuda"``.
        dtype: ``"float16"`` or ``"float32"``; by default ``DEFAULT_DTYPES[device]``.
        model, processor: already built objects, to avoid downloading the weights in tests.
    """

    decoding = ""  # how the transcript is read from the model's output, stored with the results

    def __init__(
        self,
        model_id: str,
        *,
        revision: str,
        device: str = "cpu",
        dtype: str | None = None,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        if device not in DEFAULT_DTYPES:
            raise ValueError(f"device must be one of {sorted(DEFAULT_DTYPES)}, got {device!r}")
        dtype = dtype or DEFAULT_DTYPES[device]
        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}, got {dtype!r}")
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.dtype = dtype
        if model is None or processor is None:
            model, processor = self._load()
        self._model, self._processor = model, processor

    @property
    def name(self) -> str:
        return f"transformers/{self.model_id}-{self.dtype}"

    @property
    def config(self) -> dict[str, Any]:
        """Every setting that can change the output, to be stored with the results."""
        return {
            "library": "transformers",
            "library_version": version("transformers"),
            "model_id": self.model_id,
            "revision": self.revision,
            "dtype": self.dtype,
            "device": self.device,
            "decoding": self.decoding,
        }

    def _torch_dtype(self) -> Any:
        import torch

        return getattr(torch, self.dtype)

    def _transcribe_batch(self, samples: Sequence[AudioSample]) -> list[Hypothesis]:
        import torch

        with torch.inference_mode():
            return [Hypothesis(s.utterance_id, self._transcribe_one(s)) for s in samples]

    @abstractmethod
    def _load(self) -> tuple[Any, Any]:
        """Download (or read from the cache) the model and its processor."""

    @abstractmethod
    def _transcribe_one(self, sample: AudioSample) -> str:
        """The raw transcript of one utterance."""


class Wav2Vec2CTCModel(TransformersModel):
    """wav2vec2 with greedy CTC decoding. The language is the checkpoint's own."""

    decoding = "ctc-greedy"

    def _load(self) -> tuple[Any, Any]:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        processor = Wav2Vec2Processor.from_pretrained(self.model_id, revision=self.revision)
        model = Wav2Vec2ForCTC.from_pretrained(
            self.model_id, revision=self.revision, dtype=self._torch_dtype()
        )
        return model.to(self.device).eval(), processor

    def _transcribe_one(self, sample: AudioSample) -> str:
        inputs = self._processor(sample.waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = inputs.to(self.device, dtype=self._torch_dtype())
        logits = self._model(**inputs).logits
        return self._processor.batch_decode(logits.argmax(dim=-1))[0]


class ParakeetTDTModel(TransformersModel):
    """Parakeet TDT with greedy decoding. It has no language option: it finds the language."""

    decoding = "tdt-greedy"

    def _load(self) -> tuple[Any, Any]:
        from transformers import AutoModelForTDT, AutoProcessor

        processor = AutoProcessor.from_pretrained(self.model_id, revision=self.revision)
        model = AutoModelForTDT.from_pretrained(
            self.model_id, revision=self.revision, dtype=self._torch_dtype()
        )
        return model.to(self.device).eval(), processor

    def _transcribe_one(self, sample: AudioSample) -> str:
        inputs = self._processor([sample.waveform], sampling_rate=SAMPLE_RATE)
        inputs = inputs.to(self.device, dtype=self._torch_dtype())
        with warnings.catch_warnings():
            # The length limit comes from the audio's frames, so it cannot cut a transcript
            warnings.filterwarnings("ignore", message="Using the model-agnostic default")
            output = self._model.generate(**inputs, return_dict_in_generate=True)
        return self._processor.decode(output.sequences, skip_special_tokens=True)[0]


FAMILIES = {"ctc": Wav2Vec2CTCModel, "tdt": ParakeetTDTModel}
