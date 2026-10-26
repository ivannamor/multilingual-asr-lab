"""JSONL manifests: the single data format shared by every module.

A manifest has one JSON object per line, one line per utterance. Audio paths are
stored relative to the manifest file with forward slashes, so a dataset folder
can be moved or copied to another machine (Linux or Windows) without rewriting it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

_REQUIRED_FIELDS = (
    "utterance_id",
    "audio_filepath",
    "duration",
    "text",
    "lang",
    "speaker_id",
    "source",
)
_OPTIONAL_FIELDS = ("metadata",)


@dataclass(frozen=True)
class ManifestEntry:
    """One utterance: where its audio is, what was said, and who said it.

    Example:
        >>> entry = ManifestEntry(
        ...     utterance_id="fleurs_it_0001",
        ...     audio_filepath="audio/fleurs_it_0001.wav",
        ...     duration=4.2,
        ...     text="Buongiorno a tutti.",
        ...     lang="it",
        ...     speaker_id="fleurs_it_spk12",
        ...     source="fleurs",
        ... )
        >>> entry.audio_path(Path("data/fleurs_it/test.jsonl")).as_posix()
        'data/fleurs_it/audio/fleurs_it_0001.wav'
    """

    utterance_id: str
    audio_filepath: str  # relative to the manifest file, forward slashes
    duration: float  # seconds
    text: str  # reference transcript, not normalized
    lang: str  # ISO 639-1 code: "it", "de", "en"
    speaker_id: str
    source: str  # dataset name, e.g. "fleurs"
    metadata: dict[str, str] = field(default_factory=dict)  # e.g. gender, accent

    def audio_path(self, manifest_path: Path) -> Path:
        """Path to the audio file, resolved from the location of the manifest file."""
        return manifest_path.parent / PurePosixPath(self.audio_filepath)


def write_manifest(path: Path, entries: Iterable[ManifestEntry]) -> int:
    """Write entries as JSONL (UTF-8, LF line endings) and return how many were written.

    Raises:
        ValueError: if an entry is invalid or an utterance id is repeated.
    """
    lines = []
    seen: set[str] = set()
    for number, entry in enumerate(entries, start=1):
        _validate(entry, seen, f"entry {number}")
        lines.append(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.writelines(lines)
    return len(lines)


def read_manifest(path: Path) -> list[ManifestEntry]:
    """Read and validate a JSONL manifest. Blank lines are ignored.

    Raises:
        ValueError: with the file name and line number of the first invalid line.
    """
    entries = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            where = f"{path}:{line_number}"
            entry = _parse_line(line, where)
            _validate(entry, seen, where)
            entries.append(entry)
    return entries


def _parse_line(line: str, where: str) -> ManifestEntry:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"{where}: invalid JSON ({error.msg})") from None
    if not isinstance(record, dict):
        raise ValueError(f"{where}: expected a JSON object")
    missing = [name for name in _REQUIRED_FIELDS if name not in record]
    if missing:
        raise ValueError(f"{where}: missing fields {missing}")
    unknown = sorted(set(record) - set(_REQUIRED_FIELDS) - set(_OPTIONAL_FIELDS))
    if unknown:
        raise ValueError(f"{where}: unknown fields {unknown} (put extra information in 'metadata')")
    return ManifestEntry(**record)


def _validate(entry: ManifestEntry, seen: set[str], where: str) -> None:
    """Check field values and id uniqueness; ``seen`` collects the ids found so far."""
    if not entry.utterance_id:
        raise ValueError(f"{where}: empty utterance_id")
    if entry.utterance_id in seen:
        raise ValueError(f"{where}: duplicate utterance_id {entry.utterance_id!r}")
    if isinstance(entry.duration, bool) or not isinstance(entry.duration, int | float):
        raise ValueError(f"{where}: duration must be a number of seconds")
    if not entry.duration > 0:
        raise ValueError(f"{where}: duration must be positive, got {entry.duration}")
    if PurePosixPath(entry.audio_filepath).is_absolute() or "\\" in entry.audio_filepath:
        raise ValueError(
            f"{where}: audio_filepath must be relative with forward slashes, "
            f"got {entry.audio_filepath!r}"
        )
    for name in ("text", "lang", "speaker_id", "source"):
        if not isinstance(getattr(entry, name), str):
            raise ValueError(f"{where}: {name} must be a string")
    seen.add(entry.utterance_id)
