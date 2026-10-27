"""Tests for scoring saved predictions against a manifest."""

import json
from pathlib import Path

import pytest

from asr_lab import cli
from asr_lab.data.manifest import ManifestEntry, write_manifest
from asr_lab.evaluation.normalize import BASIC
from asr_lab.evaluation.scoring import format_summary, score_predictions, score_run
from asr_lab.inference.base import Hypothesis
from asr_lab.inference.predictions import append_predictions
from asr_lab.provenance import file_sha256


def _entry(i: int, text: str, sentence: str | None = None, speaker: str | None = None):
    return ManifestEntry(
        utterance_id=f"u{i}",
        audio_filepath=f"audio/u{i}.wav",
        duration=1.0,
        text=text,
        lang="it",
        speaker_id=speaker or f"spk{i}",
        source="test",
        metadata={"sentence_id": sentence} if sentence else {},
    )


ENTRIES = [
    _entry(0, "Il gatto dorme.", "s1"),
    _entry(1, "Il gatto dorme.", "s1"),
    _entry(2, "Oggi piove molto.", "s2"),
    _entry(3, "L'uomo è arrivato.", "s3"),
]
HYPOTHESES = [
    Hypothesis("u0", "il gatto dorme"),
    Hypothesis("u1", "il cane dorme"),  # 1 substitution
    Hypothesis("u2", "Oggi piove."),  # 1 deletion
    Hypothesis("u3", "l\u2019uomo è arrivato!"),  # correct after normalization
]


def test_corpus_wer_with_the_language_normalizer() -> None:
    scores = score_predictions(ENTRIES, HYPOTHESES, n_resamples=200)
    wer = scores["wer"]
    # References after it-v1: 3 + 3 + 3 + 4 words ("l uomo è arrivato") = 13
    assert wer["n_reference_tokens"] == 13
    assert (wer["substitutions"], wer["deletions"], wer["insertions"]) == (1, 1, 0)
    assert wer["estimate"] == pytest.approx(2 / 13)
    assert scores["normalizer"] == ["it-v1"]
    assert scores["n_utterances"] == 4
    assert scores["n_empty_hypotheses"] == 0


def test_intervals_for_every_meaningful_unit_and_widest_headline() -> None:
    wer = score_predictions(ENTRIES, HYPOTHESES, n_resamples=200)["wer"]
    intervals = wer["confidence_intervals"]
    assert set(intervals) == {"utterance", "sentence_id"}  # speakers are all different
    assert intervals["utterance"]["n_units"] == 4
    assert intervals["sentence_id"]["n_units"] == 3
    widths = {unit: ci["high"] - ci["low"] for unit, ci in intervals.items()}
    assert wer["headline_unit"] == max(widths, key=widths.get)
    for ci in intervals.values():
        assert ci["low"] <= wer["estimate"] <= ci["high"]


def test_speaker_unit_is_used_when_speakers_have_several_utterances() -> None:
    entries = [_entry(i, "ciao", speaker="anna" if i < 2 else "bruno") for i in range(4)]
    hypotheses = [Hypothesis(f"u{i}", "ciao") for i in range(4)]
    intervals = score_predictions(entries, hypotheses, n_resamples=50)["wer"][
        "confidence_intervals"
    ]
    assert set(intervals) == {"utterance", "speaker"}
    assert intervals["speaker"]["n_units"] == 2


def test_explicit_normalizer_and_empty_hypotheses() -> None:
    hypotheses = [*HYPOTHESES[:3], Hypothesis("u3", "")]
    scores = score_predictions(ENTRIES, hypotheses, normalizer=BASIC, n_resamples=50)
    assert scores["normalizer"] == ["basic-v1"]
    assert scores["n_empty_hypotheses"] == 1


@pytest.mark.parametrize(
    ("hypotheses", "message"),
    [
        (HYPOTHESES[:3], r"1 missing \(e\.g\. \['u3'\]\)"),
        ([*HYPOTHESES, Hypothesis("u9", "x")], r"1 not in the manifest \(e\.g\. \['u9'\]\)"),
    ],
)
def test_predictions_must_match_the_manifest(hypotheses: list, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        score_predictions(ENTRIES, hypotheses)


# --- run directories ------------------------------------------------------------------


def _run_dir(tmp_path: Path, limit: int | None = None) -> tuple[Path, Path]:
    manifest = tmp_path / "data" / "test.jsonl"
    write_manifest(manifest, ENTRIES)
    run_dir = tmp_path / "runs" / "r1"
    append_predictions(run_dir / "predictions.jsonl", HYPOTHESES[:limit])
    run = {
        "model": "fake-model",
        "model_config": {"beam_size": 5},
        "manifest": manifest.as_posix(),
        "manifest_sha256": file_sha256(manifest),
        "limit": limit,
    }
    (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
    return run_dir, manifest


def test_score_run_writes_a_scores_file(tmp_path: Path) -> None:
    run_dir, manifest = _run_dir(tmp_path)
    scores, path = score_run(run_dir, n_resamples=100)
    assert path == run_dir / "scores_it-v1_r100.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == scores
    assert scores["model"] == "fake-model"
    assert scores["manifest_sha256"] == file_sha256(manifest)
    assert scores["wer"]["estimate"] == pytest.approx(2 / 13)


def test_rescoring_unchanged_predictions_keeps_the_file(tmp_path: Path) -> None:
    run_dir, _ = _run_dir(tmp_path)
    _, path = score_run(run_dir, n_resamples=50)
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["scored_at"] = "2000-01-01T00:00:00+00:00"  # as if scored long ago
    path.write_text(json.dumps(saved), encoding="utf-8")
    text = path.read_text(encoding="utf-8")
    scores, _ = score_run(run_dir, n_resamples=50)
    assert path.read_text(encoding="utf-8") == text
    assert scores["scored_at"] == "2000-01-01T00:00:00+00:00"


def test_rescoring_changed_predictions_rewrites_the_file(tmp_path: Path) -> None:
    run_dir, _ = _run_dir(tmp_path)
    first, path = score_run(run_dir, n_resamples=50)
    predictions = run_dir / "predictions.jsonl"
    predictions.unlink()
    append_predictions(predictions, [Hypothesis(h.utterance_id, "") for h in HYPOTHESES])
    second, _ = score_run(run_dir, n_resamples=50)
    assert second["wer"]["estimate"] == 1.0 != first["wer"]["estimate"]
    assert json.loads(path.read_text(encoding="utf-8")) == second


def test_score_run_respects_the_run_limit(tmp_path: Path) -> None:
    run_dir, _ = _run_dir(tmp_path, limit=2)
    scores, _ = score_run(run_dir, n_resamples=50)
    assert scores["n_utterances"] == 2


def test_different_normalizers_are_saved_to_different_files(tmp_path: Path) -> None:
    run_dir, _ = _run_dir(tmp_path)
    _, default_path = score_run(run_dir)
    _, basic_path = score_run(run_dir, normalizer=BASIC)
    assert default_path.name == "scores_it-v1.json"
    assert basic_path.name == "scores_basic-v1.json"
    assert default_path.exists()  # not overwritten
    first = json.loads(default_path.read_text(encoding="utf-8"))
    _, again = score_run(run_dir)
    second = json.loads(again.read_text(encoding="utf-8"))
    assert first["wer"] == second["wer"]  # same settings, same result


def test_score_run_rejects_a_changed_manifest(tmp_path: Path) -> None:
    run_dir, manifest = _run_dir(tmp_path)
    write_manifest(manifest, ENTRIES[:3])
    with pytest.raises(ValueError, match="has changed since the run"):
        score_run(run_dir)


def test_score_run_needs_a_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    append_predictions(run_dir / "predictions.jsonl", HYPOTHESES)
    with pytest.raises(ValueError, match="pass the manifest"):
        score_run(run_dir)


def test_summary_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir, _ = _run_dir(tmp_path)
    cli.main(["score", "--run", str(run_dir), "--n-resamples", "50"])
    output = capsys.readouterr().out
    assert "fake-model" in output
    assert "4 utterances, 13 words" in output
    assert "WER 15.38%" in output
    assert "95% CI" in output
    path = run_dir / "scores_it-v1_r50.json"
    assert f"Scores: {path}" in output
    assert format_summary(json.loads(path.read_text(encoding="utf-8"))) in output


def test_cli_score_with_a_named_normalizer(tmp_path: Path) -> None:
    run_dir, _ = _run_dir(tmp_path)
    cli.main(
        ["score", "--run", str(run_dir), "--normalizer", "whisper-basic", "--n-resamples", "50"]
    )
    scores = json.loads((run_dir / "scores_whisper-basic_r50.json").read_text(encoding="utf-8"))
    assert scores["normalizer"] == ["whisper-basic"]
