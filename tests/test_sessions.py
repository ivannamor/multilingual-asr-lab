"""Tests for the simulated usage sessions of E10, on a tiny synthetic dataset."""

from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asr_lab import cli
from asr_lab.analytics.sessions import (
    SOURCE,
    SimulationSpec,
    load_simulation_spec,
    plan_sessions,
    render_sessions,
)
from asr_lab.data.augment import SAMPLE_RATE, rms
from asr_lab.data.manifest import ManifestEntry, read_manifest, write_manifest

SPEC_TOML = """
seed = 0
n_sessions = 600
source = "{root}/{{lang}}/dev.jsonl"
snr_db = [10, 0]
babble_voices = 2

[languages]
it = 0.75
de = 0.25

[channels]
headset = 0.5
phone = 0.5

[environments]
quiet = 0.4
stationary = 0.3
babble = 0.3
"""


@pytest.fixture
def spec(tmp_path: Path) -> SimulationSpec:
    """Two languages, six recordings each: three sentences read twice, as pure tones."""
    time = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    for lang in ("it", "de"):
        (tmp_path / lang).mkdir()
        entries = []
        for i in range(6):
            utterance_id = f"{lang}_{i}"
            audio = 0.1 * np.sin(2 * np.pi * (200 + 50 * i) * time)
            sf.write(tmp_path / lang / f"{utterance_id}.wav", audio, SAMPLE_RATE, subtype="PCM_16")
            entries.append(
                ManifestEntry(utterance_id, f"{utterance_id}.wav", 1.0, f"frase {i // 2}", lang,
                              f"{utterance_id}_speaker", "fleurs",
                              {"sentence_id": str(i // 2), "gender": "female"})
            )  # fmt: skip
        write_manifest(tmp_path / lang / "dev.jsonl", entries)
    path = tmp_path / "simulation.toml"
    path.write_text(SPEC_TOML.format(root=tmp_path.as_posix()), encoding="utf-8")
    return load_simulation_spec(path)


def test_the_e10_spec_is_valid() -> None:
    spec = load_simulation_spec(Path("experiments/E10/simulation.toml"))
    assert spec.n_sessions == 3000
    assert spec.source_manifest("it") == Path("data/fleurs_it/dev.jsonl")


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("quiet = 0.4", "quiet = 0.5", "sum to 1"),
        ("quiet = 0.4", "quiet = 0.4\nrain = 0.0", "unknown environments"),
        ("phone = 0.5", "fax = 0.5", "unknown channels"),
        ("babble_voices = 2", "babble_voices = 0", "babble_voices"),
    ],
)
def test_invalid_specs_are_rejected(tmp_path: Path, old: str, new: str, message: str) -> None:
    path = tmp_path / "simulation.toml"
    path.write_text(SPEC_TOML.format(root=".").replace(old, new), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_simulation_spec(path)


def test_plan_is_reproducible_and_follows_the_mix(spec: SimulationSpec) -> None:
    sessions = plan_sessions(spec)
    assert sessions == plan_sessions(spec)
    assert sessions != plan_sessions(replace(spec, seed=1))
    assert len(sessions) == 600
    assert Counter(s.lang for s in sessions)["it"] / 600 == pytest.approx(0.75, abs=0.05)
    environments = Counter(s.metadata["environment"] for s in sessions)
    assert environments["quiet"] / 600 == pytest.approx(0.4, abs=0.05)


def test_sessions_keep_the_recording_and_record_their_conditions(spec: SimulationSpec) -> None:
    for session in plan_sessions(spec):
        metadata = session.metadata
        recording = metadata["recording"]
        assert session.lang == recording[:2] and session.source == SOURCE
        assert session.text == f"frase {int(recording[-1]) // 2}"
        assert metadata["sentence_id"] == str(int(recording[-1]) // 2)
        assert ("snr_db" in metadata) == (metadata["environment"] != "quiet")
        voices = metadata.get("babble_voices", "").split()
        assert len(voices) == (2 if metadata["environment"] == "babble" else 0)
        for voice in voices:  # same language, never the same sentence
            assert voice[:2] == session.lang
            assert int(voice[-1]) // 2 != int(recording[-1]) // 2


def test_render_writes_each_session_once(spec: SimulationSpec, tmp_path: Path) -> None:
    manifest = tmp_path / "sessions" / "sessions.jsonl"
    sessions = plan_sessions(replace(spec, n_sessions=40))
    write_manifest(manifest, sessions)
    assert render_sessions(manifest, progress=False) == 40
    assert render_sessions(manifest, progress=False) == 0  # resumable: nothing left to do
    assert not list(manifest.parent.rglob("*.partial.wav"))
    for session in sessions:
        audio, rate = sf.read(session.audio_path(manifest), dtype="float32")
        assert rate == SAMPLE_RATE and len(audio) == SAMPLE_RATE
        metadata = session.metadata
        clean, _ = sf.read(
            tmp_path / session.lang / f"{metadata['recording']}.wav", dtype="float32"
        )
        if metadata["channel"] == "headset" and metadata["environment"] == "quiet":
            assert np.allclose(audio, clean, atol=1e-4)  # unchanged, up to 16-bit rounding
        elif metadata["channel"] == "headset":
            measured = 20 * np.log10(rms(clean) / rms(audio - clean))
            assert measured == pytest.approx(float(metadata["snr_db"]), abs=0.1)


def test_cli_plan_and_render(spec: SimulationSpec, tmp_path: Path) -> None:
    spec_path = tmp_path / "simulation.toml"
    spec_path.write_text(
        spec_path.read_text(encoding="utf-8").replace("n_sessions = 600", "n_sessions = 5"),
        encoding="utf-8",
    )
    manifest = tmp_path / "out" / "sessions.jsonl"
    cli.main(["plan-sessions", "--spec", str(spec_path), "--output", str(manifest)])
    assert len(read_manifest(manifest)) == 5
    cli.main(["render-sessions", "--manifest", str(manifest)])
    assert len(list((manifest.parent / "audio").glob("*.wav"))) == 5
