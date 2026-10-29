"""LoRA fine-tuning of E7: config, schedule, batches and labels; end to end on CPU (slow)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asr_lab.data.manifest import ManifestEntry
from asr_lab.training.lora import (
    TRAINING_FILE,
    batches,
    load_training_config,
    pad_labels,
    schedule,
)

SPEC = Path("experiments/E7/finetune.toml")


def test_the_e7_spec_loads() -> None:
    config = load_training_config(SPEC)
    assert config.base_model == "openai/whisper-small"
    assert config.target_modules == ("q_proj", "v_proj")
    assert config.epochs == 3


def test_schedule_warms_up_then_decays_to_zero() -> None:
    values = [schedule(step, total_steps=100, warmup_steps=10) for step in range(101)]
    assert values[0] == 0.0
    assert values[10] == 1.0
    assert values[100] == 0.0
    assert values[:11] == sorted(values[:11])
    assert values[10:] == sorted(values[10:], reverse=True)


def test_batches_cover_every_item_once_per_epoch() -> None:
    plan = batches(10, batch_size=4, epochs=3, seed=1)
    assert [len(batch) for batch in plan] == [4, 4, 2] * 3
    for epoch in range(3):
        seen = [i for batch in plan[epoch * 3 : epoch * 3 + 3] for i in batch]
        assert sorted(seen) == list(range(10))
    assert plan[:3] != plan[3:6]  # a new order every epoch
    assert batches(10, batch_size=4, epochs=3, seed=1) == plan


def test_pad_labels_keeps_sequences_without_the_start_token() -> None:
    labels = pad_labels([[5, 6], [1, 5]], start_token=1)
    assert labels.tolist() == [[5, 6], [5, -100]]
    assert labels.dtype == np.int64


@pytest.mark.slow
def test_fine_tune_and_transcribe_end_to_end(tmp_path: Path) -> None:
    """A few steps on CPU with whisper-tiny, then the converted model in faster-whisper."""
    pytest.importorskip("peft")
    from dataclasses import replace

    from asr_lab.inference.base import AudioSample
    from asr_lab.inference.faster_whisper import FasterWhisperModel
    from asr_lab.training.lora import fine_tune

    rng = np.random.default_rng(0)
    entries = []
    for i in range(3):
        sf.write(tmp_path / f"u{i}.wav", 0.1 * rng.standard_normal(16_000), 16_000)
        entries.append(ManifestEntry(f"u{i}", f"u{i}.wav", 1.0, "Buongiorno.", "it", "s", "t"))
    config = replace(load_training_config(SPEC), base_model="openai/whisper-tiny", batch_size=2)
    output = tmp_path / "model"

    ct2_dir = fine_tune(config, entries, tmp_path / "train.jsonl", output, seed=0, device="cpu")
    record = json.loads((output / TRAINING_FILE).read_text(encoding="utf-8"))
    assert record["steps"] == 6  # 2 batches x 3 epochs
    assert record["utterance_ids"] == ["u0", "u1", "u2"]
    assert not (output / "hf").exists()

    # Word timestamps (for the hallucination silence threshold) need the alignment heads
    model = FasterWhisperModel(str(ct2_dir), hallucination_silence_threshold=2.0)
    [hypothesis] = model.transcribe([AudioSample("u0", np.zeros(16_000, np.float32), "it")])
    assert isinstance(hypothesis.text, str)
    # Same config, data and seed: nothing to do
    modified = (ct2_dir / "model.bin").stat().st_mtime_ns
    fine_tune(config, entries, tmp_path / "train.jsonl", output, seed=0, device="cpu")
    assert (ct2_dir / "model.bin").stat().st_mtime_ns == modified

    # No entries: the base model through the same conversion, and different weights
    base_dir = fine_tune(config, [], Path(), tmp_path / "base", seed=0, device="cpu")
    assert json.loads((tmp_path / "base" / TRAINING_FILE).read_text(encoding="utf-8"))["steps"] == 0
    assert (base_dir / "model.bin").read_bytes() != (ct2_dir / "model.bin").read_bytes()
