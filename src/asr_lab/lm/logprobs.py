"""The frame probabilities of a CTC model, saved once so that decoding never repeats inference.

Decoding a test set with many language models and settings would otherwise run the acoustic model
every time. Half precision keeps the files small (about 50 frames per second, one value per token)
and costs nothing that matters: at the scores that decide a decoding, its resolution is a few
hundredths of a log unit.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from asr_lab.data.manifest import read_manifest
from asr_lab.inference.base import load_sample
from asr_lab.provenance import file_sha256, utc_now

METADATA_KEY = "__metadata__"


def export_log_probs(model: Any, manifest: Path, output: Path) -> dict[str, Any]:
    """Save the log probabilities of every utterance of a manifest (``model``: see ``run``).

    A complete file for the same manifest is left as it is.
    """
    sha256 = file_sha256(manifest)
    if output.exists() and read_metadata(output)["manifest_sha256"] == sha256:
        return read_metadata(output)
    arrays = {
        entry.utterance_id: model.log_probabilities(load_sample(entry, manifest)).astype(np.float16)
        for entry in read_manifest(manifest)
    }
    metadata = {
        "model": model.name,
        "model_config": model.config,
        "tokens": model.tokens,
        "manifest": manifest.as_posix(),
        "manifest_sha256": sha256,
        "created": utc_now(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays, **{METADATA_KEY: np.array(json.dumps(metadata))})
    return metadata


def read_metadata(path: Path) -> dict[str, Any]:
    """The description of a log-probability file: model, tokens, manifest."""
    with np.load(path) as data:
        return json.loads(str(data[METADATA_KEY]))


def iter_log_probs(path: Path) -> Iterator[tuple[str, np.ndarray]]:
    """Utterance ids and log probabilities (frames x tokens) of a log-probability file."""
    with np.load(path) as data:
        for key in data.files:
            if key != METADATA_KEY:
                yield key, data[key].astype(np.float32)
