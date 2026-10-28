"""Simulated usage sessions for experiment E10, built from real recordings.

A product sees sessions: someone speaks into a headset or a phone, in a quiet room or a noisy
one, in one of the supported languages. FLEURS has none of this, so E10 builds sessions from the
FLEURS dev sets, and keeps apart what is simulated and what is real:

- **simulated**: which sessions exist, i.e. the traffic mix of languages, channels and
  environments, drawn from a TOML spec with a fixed seed;
- **real**: the audio, degraded with real noise and a real telephone channel
  (``asr_lab.data.augment``), and, once it is transcribed, the model's errors and confidence.

``plan_sessions`` draws the sessions and returns them as manifest entries (text only, committed
with the results). ``render_sessions`` writes their audio, which is not committed: it is rebuilt
exactly from the manifest, the FLEURS dev sets and each session's noise seed.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from asr_lab.data.augment import SAMPLE_RATE, babble, mix_at_snr, pink_noise, telephone_channel
from asr_lab.data.manifest import ManifestEntry, read_manifest
from asr_lab.inference.base import load_sample

CHANNELS = ("headset", "phone")
ENVIRONMENTS = ("quiet", "stationary", "babble")
SOURCE = "fleurs-e10-simulated"
# Metadata kept from the FLEURS entry: sentence_id groups correlated sessions in the bootstrap.
_KEPT_METADATA = ("sentence_id", "gender")


@dataclass(frozen=True)
class SimulationSpec:
    """How many sessions to draw, from which recordings, and the traffic mix."""

    seed: int
    n_sessions: int
    source: str  # manifest of the recordings, with a {lang} placeholder
    languages: dict[str, float]  # share of the traffic
    channels: dict[str, float]
    environments: dict[str, float]
    snr_db: tuple[float, ...]  # drawn uniformly for the noisy environments
    babble_voices: int

    def source_manifest(self, lang: str) -> Path:
        return Path(self.source.format(lang=lang))


def load_simulation_spec(path: Path) -> SimulationSpec:
    """Read and check a TOML simulation spec.

    Raises:
        ValueError: if a mix has unknown names or shares that do not sum to 1.
    """
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    spec = SimulationSpec(
        seed=data["seed"],
        n_sessions=data["n_sessions"],
        source=data["source"],
        languages=dict(data["languages"]),
        channels=dict(data["channels"]),
        environments=dict(data["environments"]),
        snr_db=tuple(float(value) for value in data["snr_db"]),
        babble_voices=data["babble_voices"],
    )
    for name, mix, known in (
        ("languages", spec.languages, None),
        ("channels", spec.channels, CHANNELS),
        ("environments", spec.environments, ENVIRONMENTS),
    ):
        unknown = sorted(set(mix) - set(known)) if known else []
        if unknown:
            raise ValueError(f"{path}: unknown {name} {unknown}; expected some of {known}")
        if any(share <= 0 for share in mix.values()) or not math.isclose(sum(mix.values()), 1):
            raise ValueError(f"{path}: the {name} shares must be positive and sum to 1")
    if spec.n_sessions < 1 or not spec.snr_db or spec.babble_voices < 1:
        raise ValueError(f"{path}: n_sessions, snr_db and babble_voices must not be empty or zero")
    return spec


def plan_sessions(spec: SimulationSpec) -> list[ManifestEntry]:
    """Draw the sessions: language, recording, channel, environment, SNR and noise seed.

    Recordings are drawn with replacement, so one recording can appear in several sessions,
    under different conditions. Babble voices are other recordings of the same language that
    read a different sentence.
    """
    rng = np.random.default_rng(spec.seed)
    recordings = {lang: read_manifest(spec.source_manifest(lang)) for lang in spec.languages}
    sessions = []
    for index in range(spec.n_sessions):
        lang = _draw(rng, spec.languages)
        recording = recordings[lang][int(rng.integers(len(recordings[lang])))]
        metadata = {
            **{key: recording.metadata[key] for key in _KEPT_METADATA if key in recording.metadata},
            "recording": recording.utterance_id,
            "recording_manifest": spec.source_manifest(lang).as_posix(),
            "channel": _draw(rng, spec.channels),
            "environment": _draw(rng, spec.environments),
            "noise_seed": str(int(rng.integers(2**32))),
        }
        if metadata["environment"] != "quiet":
            metadata["snr_db"] = f"{spec.snr_db[int(rng.integers(len(spec.snr_db)))]:g}"
        if metadata["environment"] == "babble":
            others = [
                other.utterance_id
                for other in recordings[lang]
                if other.metadata.get("sentence_id") != recording.metadata.get("sentence_id")
            ]
            chosen = rng.choice(len(others), size=spec.babble_voices, replace=False)
            metadata["babble_voices"] = " ".join(others[i] for i in sorted(chosen))
        session_id = f"e10_{index:05d}"
        sessions.append(
            ManifestEntry(
                utterance_id=session_id,
                audio_filepath=f"audio/{session_id}.wav",
                duration=recording.duration,
                text=recording.text,
                lang=lang,
                speaker_id=recording.speaker_id,
                source=SOURCE,
                metadata=metadata,
            )
        )
    return sessions


def render_sessions(manifest_path: Path, *, progress: bool = True) -> int:
    """Write the audio of every session next to the manifest; return how many were written.

    Sessions whose audio already exists are skipped, so an interrupted run resumes. Each file is
    written under a temporary name and renamed at the end, so a file that exists is complete.
    """
    sessions = read_manifest(manifest_path)
    manifests = {session.metadata["recording_manifest"] for session in sessions}
    recordings = {
        entry.utterance_id: (entry, Path(manifest))
        for manifest in sorted(manifests)
        for entry in read_manifest(Path(manifest))
    }
    written = 0
    for session in tqdm(sessions, unit="session", disable=not progress):
        path = session.audio_path(manifest_path)
        if path.exists():
            continue
        waveform = render_session(session, recordings)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial.wav")
        sf.write(partial, waveform, SAMPLE_RATE, subtype="PCM_16")
        partial.replace(path)
        written += 1
    return written


def render_session(
    session: ManifestEntry, recordings: dict[str, tuple[ManifestEntry, Path]]
) -> np.ndarray:
    """The audio of one session: its recording, plus noise at its SNR, through its channel.

    Noise is added before the channel, as in reality: the phone carries the noisy room.
    ``recordings`` maps utterance ids to their manifest entry and manifest path.
    """
    metadata = session.metadata
    speech = _waveform(metadata["recording"], recordings)
    rng = np.random.default_rng(int(metadata["noise_seed"]))
    if metadata["environment"] == "stationary":
        noise = pink_noise(len(speech), rng)
    elif metadata["environment"] == "babble":
        voices = [_waveform(voice, recordings) for voice in metadata["babble_voices"].split()]
        noise = babble(voices, len(speech), rng)
    if metadata["environment"] != "quiet":
        speech = mix_at_snr(speech, noise, float(metadata["snr_db"]))
    if metadata["channel"] == "phone":
        speech = telephone_channel(speech)
    return speech


def _waveform(utterance_id: str, recordings: dict[str, tuple[ManifestEntry, Path]]) -> np.ndarray:
    entry, manifest = recordings[utterance_id]
    return load_sample(entry, manifest).waveform


def _draw(rng: np.random.Generator, shares: dict[str, float]) -> str:
    names: Sequence[str] = list(shares)
    probabilities = np.array([shares[name] for name in names])
    return names[int(rng.choice(len(names), p=probabilities / probabilities.sum()))]
