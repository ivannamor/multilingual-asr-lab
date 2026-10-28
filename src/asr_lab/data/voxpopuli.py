"""Convert part of VoxPopuli (Meta) into a manifest with 16 kHz mono WAV audio.

VoxPopuli is speech from the European Parliament with the parliament's own transcripts. The
Hugging Face Hub serves it as parquet files of a few gigabytes, each split into row groups of
about 1,000 utterances; only the row groups listed in the spec are read, with ranged requests, so
a sample of a few hours never downloads whole files. The dataset revision is pinned in the spec.
"""

from __future__ import annotations

import io
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import pyarrow.parquet as pq

from asr_lab.data.ingest import prepare_audio
from asr_lab.data.manifest import ManifestEntry, write_manifest

COLUMNS = ("audio_id", "audio", "raw_text", "normalized_text", "gender", "speaker_id")


@dataclass(frozen=True)
class DataSpec:
    """The ``[data]`` table of a quality spec: which part of VoxPopuli, and where it goes."""

    repo: str
    revision: str
    lang: str
    split: str
    row_groups: tuple[tuple[int, int], ...]  # (shard, row group) pairs
    output: Path
    normalizer: str

    @property
    def manifest(self) -> Path:
        return self.output / f"{self.split}.jsonl"


def load_data_spec(path: Path) -> DataSpec:
    """Read the ``[data]`` table of a quality spec (e.g. ``experiments/E6/quality.toml``)."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))["data"]
    return DataSpec(
        repo=data["repo"],
        revision=data["revision"],
        lang=data["lang"],
        split=data["split"],
        row_groups=tuple((int(shard), int(group)) for shard, group in data["row_groups"]),
        output=Path(data["output"]),
        normalizer=data["normalizer"],
    )


def read_row_group(file: Path | BinaryIO, index: int) -> list[dict[str, Any]]:
    """The rows of one row group of a VoxPopuli parquet file (only the columns used here)."""
    return pq.ParquetFile(file).read_row_group(index, columns=list(COLUMNS)).to_pylist()


def build_entries(
    rows: Iterable[dict[str, Any]], output_dir: Path, *, lang: str, origin: str
) -> list[ManifestEntry]:
    """Write the audio of each row as 16-bit WAV under ``output_dir/audio`` and return entries.

    Empty transcripts are kept: finding them is the job of the quality checks, not of ingestion.
    ``origin`` records where the rows come from (shard and row group).
    """
    entries = []
    for row in rows:
        # Audio ids contain colons (times of day), which Windows does not allow in file names.
        utterance_id = f"voxpopuli_{lang}_{row['audio_id'].replace(':', '-')}"
        audio_filepath = f"audio/{utterance_id}.wav"
        duration = prepare_audio(io.BytesIO(row["audio"]["bytes"]), output_dir / audio_filepath)
        entries.append(
            ManifestEntry(
                utterance_id=utterance_id,
                audio_filepath=audio_filepath,
                duration=duration,
                text=row["raw_text"] or "",
                lang=lang,
                speaker_id=f"voxpopuli_{row['speaker_id']}",
                source="voxpopuli",
                metadata={
                    "audio_id": row["audio_id"],
                    # The plenary session the speech belongs to, e.g. 20130522-0900-PLENARY-15
                    "session": row["audio_id"].split("_")[0].removesuffix(f"-{lang}"),
                    "gender": row["gender"] or "",
                    "voxpopuli_normalized_text": row["normalized_text"] or "",
                    "origin": origin,
                },
            )
        )
    return entries


def ingest_voxpopuli(spec: DataSpec) -> Path:
    """Download the row groups of the spec and write ``<output>/<split>.jsonl``.

    Raises:
        ValueError: if a shard of the spec does not exist at the pinned revision.
    """
    from huggingface_hub import HfFileSystem  # network access only when ingesting

    fs = HfFileSystem()
    entries = []
    for shard, group in spec.row_groups:
        pattern = f"datasets/{spec.repo}@{spec.revision}/{spec.lang}/{spec.split}-{shard:05d}-of-*"
        matches = fs.glob(pattern)
        if len(matches) != 1:
            raise ValueError(f"no single parquet file matches {pattern}: {matches}")
        with fs.open(matches[0]) as file:
            rows = read_row_group(file, group)
        origin = f"{Path(matches[0]).name}#{group}"
        entries += build_entries(rows, spec.output, lang=spec.lang, origin=origin)
        print(f"{origin}: {len(rows)} utterances")
    write_manifest(spec.manifest, sorted(entries, key=lambda entry: entry.utterance_id))
    return spec.manifest
