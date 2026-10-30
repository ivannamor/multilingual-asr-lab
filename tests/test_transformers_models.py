"""Adapters of the CTC and Transducer families: settings, decoding, and a real forward pass."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from asr_lab.inference.base import AudioSample
from asr_lab.inference.transformers_models import ParakeetTDTModel, Wav2Vec2CTCModel

SAMPLE = AudioSample("u1", np.zeros(16_000, dtype=np.float32), "it")


def test_invalid_device_and_dtype_are_rejected() -> None:
    with pytest.raises(ValueError, match="device"):
        Wav2Vec2CTCModel("m", revision="r", device="tpu", model=object(), processor=object())
    with pytest.raises(ValueError, match="dtype"):
        Wav2Vec2CTCModel("m", revision="r", dtype="int8", model=object(), processor=object())


@pytest.mark.parametrize(
    ("family", "decoding"), [(Wav2Vec2CTCModel, "ctc-greedy"), (ParakeetTDTModel, "tdt-greedy")]
)
def test_name_and_config_identify_the_weights_and_the_decoding(family: Any, decoding: str) -> None:
    pytest.importorskip("transformers")
    model = family("org/m", revision="abc", device="cuda", model=object(), processor=object())
    assert model.name == "transformers/org/m-float16"  # half precision by default on a GPU
    assert model.config["revision"] == "abc"
    assert model.config["decoding"] == decoding
    assert family("org/m", revision="abc", model=object(), processor=object()).dtype == "float32"


def test_ctc_decodes_the_best_character_of_each_frame() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    class Processor:
        def __call__(self, waveform: np.ndarray, **_: Any) -> Any:
            return transformers.BatchFeature({"input_values": torch.zeros(1, len(waveform))})

        def batch_decode(self, ids: Any) -> list[str]:
            return ["".join("_abc"[i] for i in ids[0].tolist())]

    class Model:
        def __call__(self, input_values: Any) -> Any:
            logits = torch.tensor([[[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]])
            return transformers.modeling_outputs.CausalLMOutput(logits=logits)

    model = Wav2Vec2CTCModel("m", revision="r", model=Model(), processor=Processor())
    [hypothesis] = model.transcribe([SAMPLE])
    assert (hypothesis.utterance_id, hypothesis.text) == ("u1", "ac")


def test_tdt_returns_the_decoded_sequence() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    class Processor:
        def __call__(self, waveforms: list[np.ndarray], **_: Any) -> Any:
            assert len(waveforms) == 1  # one utterance at a time: no padding
            return transformers.BatchFeature({"input_features": torch.zeros(1, 4, 8)})

        def decode(self, sequences: Any, skip_special_tokens: bool) -> list[str]:
            assert skip_special_tokens
            return [f"token {sequences[0, 0].item()}"]

    class Output:
        sequences = torch.tensor([[7]])

    class Model:
        def generate(self, **kwargs: Any) -> Any:
            assert kwargs["return_dict_in_generate"]
            return Output()

    model = ParakeetTDTModel("m", revision="r", model=Model(), processor=Processor())
    assert model.transcribe([SAMPLE])[0].text == "token 7"


@pytest.mark.slow
def test_ctc_with_a_real_small_wav2vec2(tmp_path: Any) -> None:
    """A tiny random model built locally: the real forward pass and tokenizer, no download."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    vocab = tmp_path / "vocab.json"
    vocab.write_text('{"<pad>": 0, "|": 1, "<unk>": 2, "a": 3, "b": 4}', encoding="utf-8")
    processor = transformers.Wav2Vec2Processor(
        transformers.Wav2Vec2FeatureExtractor(return_attention_mask=True),
        transformers.Wav2Vec2CTCTokenizer(str(vocab)),
    )
    config = transformers.Wav2Vec2Config(
        vocab_size=5,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=8,
        conv_dim=(8, 8),
        conv_stride=(5, 5),
        conv_kernel=(10, 10),
        num_feat_extract_layers=2,
        do_stable_layer_norm=True,
        feat_extract_norm="layer",
        num_conv_pos_embedding_groups=1,
        mask_time_prob=0.0,
    )
    network = transformers.Wav2Vec2ForCTC(config).eval()
    model = Wav2Vec2CTCModel("tiny", revision="r", model=network, processor=processor)
    [hypothesis] = model.transcribe([SAMPLE])
    assert set(hypothesis.text) <= set("ab |<unk>")
