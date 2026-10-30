"""Tuning on dev and decoding test once, on small log probabilities and a fake language model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from asr_lab.data.manifest import ManifestEntry, write_manifest
from asr_lab.evaluation.scoring import load_run
from asr_lab.lm import tune
from asr_lab.lm.decoder import DecodingOptions
from asr_lab.lm.experiment import language_models, logprobs_path
from asr_lab.lm.logprobs import export_log_probs, iter_log_probs, read_metadata
from asr_lab.lm.tune import decode_test_set, lm_paths, select_weights

TOKENS = ["<pad>", "<s>", "</s>", "<unk>", "|", "a", "b"]
SPEC = {"decoding": {"beam_width": 10, "token_min_logp": -100.0, "beam_prune_logp": -100.0}}


class FakeAcoustic:
    """Always the same frames: "a" a little more likely than "b"."""

    name = "fake"
    config = {"revision": "x"}  # noqa: RUF012
    tokens = TOKENS

    def log_probabilities(self, sample) -> np.ndarray:
        row = np.log([0.05, 1e-6, 1e-6, 1e-6, 0.05, 0.5, 0.4])
        return np.stack([row, row])


class FakeLM:
    """Prefers "b" whatever the weights' scale, so that alpha decides."""

    def __init__(self, path: str) -> None:
        self.path = path

    def begin(self) -> None:
        return None

    def score(self, state: None, word: str) -> tuple[float, None]:
        return (-0.1 if word == "b" else -5.0), state

    def end(self, state: None) -> float:
        return 0.0


def write_manifest_with_audio(path: Path, text: str) -> Path:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path.parent / "u.wav", np.zeros(1600, dtype=np.float32), 16_000)
    write_manifest(path, [ManifestEntry("u", "u.wav", 0.1, text, "it", "s", "t")])
    return path


def test_selection_breaks_ties_towards_smaller_weights() -> None:
    cells = [
        {"alpha": 0.5, "beta": 0.0, "wer": 0.2},
        {"alpha": 0.2, "beta": 1.0, "wer": 0.1},
        {"alpha": 0.2, "beta": 0.5, "wer": 0.1},
    ]
    assert select_weights(cells) == cells[2]


def test_names_and_paths() -> None:
    spec = {"corpora": [{"name": "c"}], "text": {"orders": [3, 5]}}
    assert language_models(spec) == [("c", 3, "c-3"), ("c", 5, "c-5")]
    assert lm_paths(Path("m"), "c", "it", 5)[1] == Path("m/c-it-5.bin")
    assert logprobs_path(Path("lp"), {"name": "s"}, "test") == Path("lp/s-test.npz")


def test_log_probabilities_round_trip_in_half_precision(tmp_path: Path) -> None:
    manifest = write_manifest_with_audio(tmp_path / "data" / "dev.jsonl", "a")
    output = tmp_path / "lp.npz"
    metadata = export_log_probs(FakeAcoustic(), manifest, output)
    assert metadata["tokens"] == TOKENS
    assert read_metadata(output)["model"] == "fake"
    [(key, values)] = list(iter_log_probs(output))
    assert key == "u"
    assert values.shape == (2, 7)
    assert values.dtype == np.float32
    assert np.allclose(values, FakeAcoustic().log_probabilities(None), atol=0.01)


def test_export_keeps_a_complete_file(tmp_path: Path) -> None:
    manifest = write_manifest_with_audio(tmp_path / "data" / "dev.jsonl", "a")
    output = tmp_path / "lp.npz"
    export_log_probs(FakeAcoustic(), manifest, output)
    before = output.stat().st_mtime_ns
    export_log_probs(FakeAcoustic(), manifest, output)
    assert output.stat().st_mtime_ns == before


def test_workers_score_a_cell_against_the_references(tmp_path: Path, monkeypatch) -> None:
    manifest = write_manifest_with_audio(tmp_path / "data" / "dev.jsonl", "b")
    logprobs = tmp_path / "lp.npz"
    export_log_probs(FakeAcoustic(), manifest, logprobs)
    monkeypatch.setattr(tune, "KenLMModel", FakeLM)
    tune._start_worker(logprobs, Path("x.bin"), {"u": "b"}, "it-v1")
    off = DecodingOptions(alpha=0.0, token_min_logp=-100, beam_prune_logp=-100)
    on = DecodingOptions(alpha=1.0, token_min_logp=-100, beam_prune_logp=-100)
    assert tune._score_cell(off)["wer"] == 1.0
    assert tune._score_cell(on)["wer"] == 0.0


def test_decode_writes_a_run_the_project_can_score(tmp_path: Path, monkeypatch) -> None:
    manifest = write_manifest_with_audio(tmp_path / "data" / "test.jsonl", "b")
    logprobs = tmp_path / "lp.npz"
    export_log_probs(FakeAcoustic(), manifest, logprobs)
    monkeypatch.setattr(tune, "KenLMModel", FakeLM)
    lm_path = tmp_path / "lm.bin"
    lm_path.write_bytes(b"model")
    test_set = {"test": manifest.as_posix()}
    output = tmp_path / "runs" / "e4"
    weights = {"alpha": 1.0, "beta": 0.0, "wer": 0.1}
    decode_test_set(SPEC, test_set, "c-5", lm_path, logprobs, weights, output)
    run = load_run(output)
    assert [h.text for h in run.hypotheses] == ["b"]
    assert run.info["model"] == "fake+c-5"
    assert run.info["model_config"]["decoding"]["alpha"] == 1.0
    # Decoding again with the same weights changes nothing
    before = (output / "run.json").read_text(encoding="utf-8")
    decode_test_set(SPEC, test_set, "c-5", lm_path, logprobs, weights, output)
    assert (output / "run.json").read_text(encoding="utf-8") == before
    assert json.loads(before)["rtf"] >= 0


def test_decode_refuses_log_probabilities_of_another_manifest(tmp_path: Path) -> None:
    dev = write_manifest_with_audio(tmp_path / "dev" / "m.jsonl", "a")
    test = write_manifest_with_audio(tmp_path / "test" / "m.jsonl", "b")
    logprobs = tmp_path / "lp.npz"
    export_log_probs(FakeAcoustic(), dev, logprobs)
    with pytest.raises(ValueError, match="does not come from"):
        decode_test_set(
            SPEC,
            {"test": test.as_posix()},
            "c-5",
            tmp_path / "lm.bin",
            logprobs,
            {"alpha": 1.0, "beta": 0.0},
            tmp_path / "out",
        )
