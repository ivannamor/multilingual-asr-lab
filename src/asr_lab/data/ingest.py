"""Convert public datasets into manifests with 16 kHz mono WAV audio.

Only FLEURS (Google) is supported so far. Run from the command line with::

    uv run python -m asr_lab.data.ingest --lang it --split test --output data/fleurs_it
"""

from __future__ import annotations

import argparse
import io
import random
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

import soundfile as sf
import soxr
from huggingface_hub import hf_hub_download

from asr_lab.data.manifest import ManifestEntry, write_manifest

TARGET_SAMPLE_RATE = 16_000
FLEURS_REPO = "google/fleurs"
FLEURS_LANGUAGES = {"it": "it_it", "de": "de_de", "en": "en_us"}
FLEURS_SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class FleursRow:
    """One line of a FLEURS ``<split>.tsv`` file (the file has no header)."""

    sentence_id: str  # the same sentence is read by several speakers
    file_name: str
    raw_transcription: str  # original casing and punctuation
    transcription: str  # lowercased and without punctuation by the FLEURS authors
    num_samples: int
    gender: str


def read_fleurs_tsv(path: Path) -> list[FleursRow]:
    """Parse a FLEURS TSV file with seven tab-separated columns per line.

    Raises:
        ValueError: with the file name and line number of the first malformed line.
    """
    rows = []
    with path.open(encoding="utf-8", newline="") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            columns = line.split("\t")
            if len(columns) != 7:
                raise ValueError(
                    f"{path}:{line_number}: expected 7 tab-separated columns, got {len(columns)}"
                )
            sentence_id, file_name, raw, normalized, _characters, num_samples, gender = columns
            if not num_samples.isdigit():
                raise ValueError(f"{path}:{line_number}: num_samples is not a number")
            rows.append(
                FleursRow(sentence_id, file_name, raw, normalized, int(num_samples), gender)
            )
    return rows


def prepare_audio(
    source: Path | BinaryIO, destination: Path, sample_rate: int = TARGET_SAMPLE_RATE
) -> float:
    """Convert any audio file soundfile can read to mono 16-bit WAV and return its duration.

    Channels are averaged and the signal is resampled if needed.
    """
    audio, source_rate = sf.read(source, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if source_rate != sample_rate:
        mono = soxr.resample(mono, source_rate, sample_rate)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, mono, sample_rate, subtype="PCM_16")
    return len(mono) / sample_rate


def build_fleurs_manifest(
    tsv_path: Path,
    archive_path: Path,
    output_dir: Path,
    *,
    lang: str,
    split: str,
    max_utterances: int | None = None,
    seed: int = 0,
) -> Path:
    """Extract the audio listed in a FLEURS TSV from its tar.gz archive and write a manifest.

    FLEURS has no speaker ids: ``speaker_id`` is a per-utterance placeholder and
    ``metadata["sentence_id"]`` identifies utterances that read the same text.

    Returns:
        The path of the manifest, ``<output_dir>/<split>.jsonl``.

    Raises:
        ValueError: if a file listed in the TSV is missing from the archive.
    """
    rows = _select(read_fleurs_tsv(tsv_path), max_utterances, seed)
    wanted = {row.file_name: row for row in rows}
    entries = []
    # A tar.gz is a single compressed stream: seeking backwards restarts decompression
    # from the start of the archive. Stream mode ("r|gz") forbids seeking, and each
    # file is read into memory before decoding, so the archive is decompressed once.
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            row = wanted.pop(PurePosixPath(member.name).name, None)
            if row is None:
                continue
            utterance_id = f"fleurs_{lang}_{PurePosixPath(row.file_name).stem}"
            audio_filepath = f"audio/{utterance_id}.wav"
            audio_bytes = io.BytesIO(archive.extractfile(member).read())
            duration = prepare_audio(audio_bytes, output_dir / audio_filepath)
            entries.append(_fleurs_entry(row, utterance_id, audio_filepath, duration, lang))
    if wanted:
        raise ValueError(
            f"{len(wanted)} files listed in {tsv_path} are missing from {archive_path}"
        )
    manifest_path = output_dir / f"{split}.jsonl"
    write_manifest(manifest_path, sorted(entries, key=lambda entry: entry.utterance_id))
    return manifest_path


def ingest_fleurs(
    lang: str,
    split: str,
    output_dir: Path,
    *,
    max_utterances: int | None = None,
    seed: int = 0,
) -> Path:
    """Download one FLEURS language/split from the Hugging Face Hub and build its manifest.

    Downloads are cached by ``huggingface_hub`` (by default in ``~/.cache/huggingface``).
    """
    if lang not in FLEURS_LANGUAGES:
        raise ValueError(f"Unsupported language {lang!r}; supported: {sorted(FLEURS_LANGUAGES)}")
    if split not in FLEURS_SPLITS:
        raise ValueError(f"Unsupported split {split!r}; supported: {list(FLEURS_SPLITS)}")
    folder = f"data/{FLEURS_LANGUAGES[lang]}"
    tsv_path = _download(f"{folder}/{split}.tsv")
    archive_path = _download(f"{folder}/audio/{split}.tar.gz")
    return build_fleurs_manifest(
        tsv_path,
        archive_path,
        output_dir,
        lang=lang,
        split=split,
        max_utterances=max_utterances,
        seed=seed,
    )


def _download(filename: str) -> Path:
    return Path(hf_hub_download(FLEURS_REPO, filename, repo_type="dataset"))


def _select(rows: list[FleursRow], max_utterances: int | None, seed: int) -> list[FleursRow]:
    """Return all rows, or a seeded random subset of at most ``max_utterances`` rows."""
    if max_utterances is None or max_utterances >= len(rows):
        return rows
    return random.Random(seed).sample(rows, max_utterances)


def _fleurs_entry(
    row: FleursRow, utterance_id: str, audio_filepath: str, duration: float, lang: str
) -> ManifestEntry:
    return ManifestEntry(
        utterance_id=utterance_id,
        audio_filepath=audio_filepath,
        duration=duration,
        text=row.raw_transcription,
        lang=lang,
        speaker_id=f"{utterance_id}_unknown_speaker",
        source="fleurs",
        metadata={
            "sentence_id": row.sentence_id,
            "gender": row.gender.lower(),
            "fleurs_transcription": row.transcription,
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download FLEURS and write a manifest.")
    parser.add_argument("--lang", required=True, choices=sorted(FLEURS_LANGUAGES))
    parser.add_argument("--split", default="test", choices=FLEURS_SPLITS)
    parser.add_argument("--output", required=True, type=Path, help="output directory")
    parser.add_argument("--max-utterances", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    manifest_path = ingest_fleurs(
        args.lang,
        args.split,
        args.output,
        max_utterances=args.max_utterances,
        seed=args.seed,
    )
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
