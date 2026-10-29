"""Fine-tune Whisper with LoRA on a selection of utterances, then export it for faster-whisper.

LoRA (low-rank adaptation) freezes the model's weights and learns, for some of its matrices, a
small update ``B @ A`` of rank ``r`` (here about 1% of the parameters). It needs less memory than
a full fine-tuning and changes the model less, which limits forgetting. After training, the update
is merged into the weights, so the result is an ordinary Whisper checkpoint; it is converted to
CTranslate2 and transcribed with the same adapter and decoding options as every other model of
the project.

The training loop is plain PyTorch, so every choice is visible here: AdamW, linear warm-up and
decay, float16 autocast with loss scaling, gradient clipping, one fixed seed for the data order,
dropout and LoRA's initialization. The hyperparameters come from the ``[training]`` table of the
spec and are not tuned.

PyTorch, Transformers and PEFT are imported inside the functions: they are only needed where the
training runs (a GPU notebook), not by the rest of the package.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import time
import tomllib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from asr_lab.data.manifest import ManifestEntry
from asr_lab.inference.base import SAMPLE_RATE, load_sample
from asr_lab.provenance import environment, git_commit, utc_now

TRAINING_FILE = "training.json"
LOSS_EVERY = 10  # steps between two points of the recorded loss curve


@dataclass(frozen=True)
class TrainingConfig:
    """The ``[training]`` table of a fine-tuning spec (e.g. ``experiments/E7/finetune.toml``)."""

    base_model: str
    language: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]
    learning_rate: float
    warmup_ratio: float
    epochs: int
    batch_size: int
    max_grad_norm: float


def load_training_config(path: Path) -> TrainingConfig:
    """Read the ``[training]`` table of a fine-tuning spec."""
    table = tomllib.loads(path.read_text(encoding="utf-8"))["training"]
    return TrainingConfig(**{**table, "target_modules": tuple(table["target_modules"])})


def schedule(step: int, total_steps: int, warmup_steps: int) -> float:
    """Learning-rate multiplier: linear warm-up to 1, then linear decay to 0.

    >>> [schedule(s, total_steps=10, warmup_steps=2) for s in (0, 1, 2, 6, 10)]
    [0.0, 0.5, 1.0, 0.5, 0.0]
    """
    if step < warmup_steps:
        return step / warmup_steps
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


def batches(n_items: int, batch_size: int, epochs: int, seed: int) -> list[list[int]]:
    """Indices of each training batch: every epoch a new seeded shuffle, the last batch shorter.

    >>> batches(5, batch_size=2, epochs=1, seed=0)
    [[2, 4], [3, 0], [1]]
    """
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(epochs):
        order = rng.permutation(n_items).tolist()
        result += [order[i : i + batch_size] for i in range(0, n_items, batch_size)]
    return result


def pad_labels(sequences: Sequence[Sequence[int]], start_token: int) -> np.ndarray:
    """Token ids of the transcripts as one array, padded with -100 (ignored by the loss).

    The tokenizer starts every transcript with ``<|startoftranscript|>``; Whisper adds that token
    itself when it shifts the labels to make the decoder's input, so it is removed here.

    >>> pad_labels([[1, 7, 8], [1, 9]], start_token=1).tolist()
    [[7, 8], [9, -100]]
    """
    trimmed = [list(s[1:]) if s and s[0] == start_token else list(s) for s in sequences]
    width = max(len(s) for s in trimmed)
    return np.array([s + [-100] * (width - len(s)) for s in trimmed], dtype=np.int64)


def fine_tune(
    config: TrainingConfig,
    entries: Sequence[ManifestEntry],
    manifest_path: Path,
    output_dir: Path,
    *,
    seed: int,
    description: dict[str, Any] | None = None,
    device: str = "cuda",
) -> Path:
    """Train LoRA on ``entries``, merge it and export the model to ``output_dir/ct2``.

    With no entries, the base model is exported unchanged (the zero-shot baseline, through the
    same conversion). ``output_dir/training.json`` records the config, the data, the loss curve
    and the provenance; ``description`` (e.g. strategy and seed) is added to it. If it already
    exists for the same config, data and seed, nothing is done.

    Returns:
        The CTranslate2 model directory, to pass as ``model_size`` to ``FasterWhisperModel``.
    """
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    ids = [entry.utterance_id for entry in entries]
    # Through JSON, as it is read back from training.json (tuples become lists)
    identity = json.loads(
        json.dumps({"config": asdict(config), "seed": seed, "utterance_ids": ids})
    )
    record_path = output_dir / TRAINING_FILE
    ct2_dir = output_dir / "ct2"
    if record_path.exists() and (ct2_dir / "model.bin").exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if {key: record.get(key) for key in identity} == identity:
            return ct2_dir

    _seed_everything(seed)
    processor = WhisperProcessor.from_pretrained(config.base_model)
    processor.tokenizer.set_prefix_tokens(language=config.language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(config.base_model)
    start = time.perf_counter()
    losses: list[dict[str, float]] = []
    steps = 0
    if entries:
        model, losses, steps = _train(
            model, processor, config, entries, manifest_path, seed, device
        )
    train_seconds = time.perf_counter() - start

    hf_dir = output_dir / "hf"
    model.save_pretrained(hf_dir)
    processor.save_pretrained(hf_dir)
    # Transformers 5 folds the feature extractor's settings into processor_config.json;
    # faster-whisper and the converter read them from preprocessor_config.json
    processor.feature_extractor.save_pretrained(hf_dir)
    _convert(hf_dir, ct2_dir)
    shutil.rmtree(hf_dir)  # about 1 GB; the CTranslate2 copy is what gets evaluated

    import peft
    import transformers

    record = {
        **identity,
        **(description or {}),
        "utterances": len(entries),
        "hours": sum(entry.duration for entry in entries) / 3600,
        "steps": steps,
        "losses": losses,
        "train_seconds": train_seconds,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "git_commit": git_commit(),
        "environment": environment(),
        "created": utc_now(),
    }
    record_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return ct2_dir


def _train(
    model: Any,
    processor: Any,
    config: TrainingConfig,
    entries: Sequence[ManifestEntry],
    manifest_path: Path,
    seed: int,
    device: str,
) -> tuple[Any, list[dict[str, float]], int]:
    import torch
    from peft import LoraConfig, get_peft_model

    model.config.forced_decoder_ids = None  # the prefix tokens are in the labels
    model.generation_config.forced_decoder_ids = None
    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        bias="none",
    )
    model = get_peft_model(model, lora).to(device)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    plan = batches(len(entries), config.batch_size, config.epochs, seed)
    warmup = math.ceil(config.warmup_ratio * len(plan))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: schedule(step, len(plan), warmup)
    )
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
    start_token = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    losses, running = [], []
    for step, indices in enumerate(plan, start=1):
        batch = [entries[i] for i in indices]
        audio = [load_sample(entry, manifest_path).waveform for entry in batch]
        features = processor.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(device)
        tokens = processor.tokenizer([entry.text for entry in batch]).input_ids
        labels = torch.from_numpy(pad_labels(tokens, start_token)).to(device)
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            loss = model(input_features=features, labels=labels).loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        running.append(loss.item())
        if step % LOSS_EVERY == 0 or step == len(plan):
            losses.append({"step": step, "loss": sum(running) / len(running)})
            print(f"step {step}/{len(plan)}: loss {losses[-1]['loss']:.4f}", flush=True)
            running = []
    return model.merge_and_unload().cpu(), losses, len(plan)


def _convert(hf_dir: Path, ct2_dir: Path) -> None:
    """Convert a Transformers Whisper checkpoint to CTranslate2 (float16 weights)."""
    from ctranslate2.converters import TransformersConverter

    TransformersConverter(
        str(hf_dir), copy_files=["tokenizer.json", "preprocessor_config.json"]
    ).convert(str(ct2_dir), quantization="float16", force=True)


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 (libraries may use the global generator)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
