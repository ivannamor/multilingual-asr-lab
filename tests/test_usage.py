"""Tests for the usage analytics of E10, on hand-made sessions with known errors."""

import json
from pathlib import Path

import numpy as np
import pytest

from asr_lab import cli
from asr_lab.analytics.usage import (
    Session,
    UserModel,
    analyze_usage,
    background_capture,
    captures_background,
    confidence_review,
    correction_rates,
    load_user_model,
    priorities,
    segment_rates,
    simulate_corrections,
    spearman,
    usage_table,
)
from asr_lab.data.manifest import ManifestEntry, write_manifest
from asr_lab.evaluation.metrics import ErrorCounts
from asr_lab.inference.base import Hypothesis
from asr_lab.inference.predictions import append_predictions
from asr_lab.provenance import file_sha256
from asr_lab.reporting import build_tables


def session(
    name: str,
    lang: str = "it",
    channel: str = "headset",
    environment: str = "quiet",
    snr_db: str | None = None,
    errors: int = 0,
    words: int = 10,
    confidence: float | None = -0.2,
) -> Session:
    return Session(
        name, lang, channel, environment, snr_db, sentence_id=name, confidence=confidence,
        counts=ErrorCounts(hits=words - errors, substitutions=errors),
    )  # fmt: skip


SESSIONS = [
    session("a", errors=1),
    session("b", errors=0),
    session("c", environment="babble", snr_db="0", errors=6),
    session("d", environment="babble", snr_db="10", errors=2),
    session("e", lang="de", errors=1),
    session("f", lang="de", channel="phone", environment="stationary", snr_db="5", errors=4),
]


def test_segment_rates_are_corpus_level_and_in_a_fixed_order() -> None:
    rows = segment_rates(SESSIONS, lambda s: s.condition, n_resamples=50)
    assert [row["segment"] for row in rows] == [
        "quiet", "stationary 5 dB", "babble 10 dB", "babble 0 dB",
    ]  # fmt: skip
    quiet = rows[0]
    assert quiet["sessions"] == 3 and quiet["words"] == 30
    assert quiet["wer"] == pytest.approx(2 / 30)
    assert quiet["session_share"] == pytest.approx(0.5)
    assert quiet["error_share"] == pytest.approx(2 / 14)
    assert sum(row["error_share"] for row in rows) == pytest.approx(1)


def test_priorities_count_errors_above_quiet_headset_sessions() -> None:
    rows = priorities(SESSIONS)
    top = rows[0]
    assert (top["lang"], top["channel"], top["environment"]) == ("it", "headset", "babble")
    assert top["reference_wer"] == pytest.approx(0.05)  # 1 error in 20 words
    assert top["avoidable_share"] == pytest.approx((8 - 0.05 * 20) / 14)
    german_phone = next(row for row in rows if row["channel"] == "phone")
    assert german_phone["avoidable_share"] == pytest.approx((4 - 0.1 * 10) / 14)
    quiet = [row for row in rows if row["environment"] == "quiet"]
    assert all(row["avoidable_share"] == 0 for row in quiet)
    assert [row["avoidable_share"] for row in rows] == sorted(
        (row["avoidable_share"] for row in rows), reverse=True
    )


def test_priorities_need_a_reference_for_every_language() -> None:
    with pytest.raises(ValueError, match="reference"):
        priorities([*SESSIONS, session("g", lang="en", environment="babble", snr_db="5")])


def test_a_perfect_confidence_reviews_like_the_oracle() -> None:
    perfect = [
        session(str(i), errors=e, confidence=-float(e)) for i, e in enumerate([0, 5, 1, 3, 0])
    ]
    review = confidence_review(perfect, budgets=(0.2, 0.4))
    assert review["spearman"] == pytest.approx(-1.0)
    first, second = review["review"]
    assert first["confidence"] == first["oracle"] == pytest.approx(5 / 9)
    assert second["confidence"] == second["oracle"] == pytest.approx(8 / 9)
    assert (first["random"], first["sessions"]) == (0.2, 1)


def test_sessions_without_confidence_are_left_out_of_the_review() -> None:
    review = confidence_review([*SESSIONS, session("x", confidence=None, errors=9)])
    assert review["sessions"] == len(SESSIONS)


def test_review_follows_sessions_that_captured_a_background_voice() -> None:
    sessions = [*SESSIONS, session("y", errors=9, confidence=-5.0)]
    sessions[-1] = Session(**{**sessions[-1].__dict__, "captured": True})
    review = confidence_review(sessions, budgets=(0.2,))
    assert review["captured"] == 1 and review["review"][0]["captured"] == 1


def test_a_transcript_closer_to_a_background_voice_is_captured() -> None:
    speaker = ErrorCounts(hits=1, substitutions=2)  # "a b c" heard as "a y z"
    assert captures_background("a y z", speaker, ["q r", "a y z w"])
    assert not captures_background("a y z", speaker, ["q r s"])
    assert not captures_background("a b c", ErrorCounts(hits=3), ["a b c"])  # ties are not


def test_background_capture_per_babble_condition() -> None:
    captured = Session(**{**session("z", environment="babble", snr_db="0", errors=6).__dict__,
                          "captured": True})  # fmt: skip
    rows = background_capture([*SESSIONS, captured])
    assert [row["segment"] for row in rows] == ["babble 10 dB", "babble 0 dB"]
    zero = rows[1]
    assert (zero["sessions"], zero["captured"]) == (2, 1)
    assert zero["error_share"] == pytest.approx(0.5)


def test_spearman_handles_ties() -> None:
    x = np.array([1.0, 1.0, 2.0, 3.0])
    assert spearman(x, x) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)
    assert np.isnan(spearman(x, np.ones(4)))  # undefined for a constant


def test_simulated_users_follow_their_model() -> None:
    always = simulate_corrections(SESSIONS, UserModel(p_notice=1.0, p_edit=0.0, seed=0))
    assert always == [s.counts.errors > 0 for s in SESSIONS]
    never = simulate_corrections(SESSIONS, UserModel(p_notice=0.0, p_edit=0.0, seed=0))
    assert not any(never)
    users = UserModel(p_notice=0.5, p_edit=0.1, seed=3)
    assert simulate_corrections(SESSIONS, users) == simulate_corrections(SESSIONS, users)


def test_correction_rates_per_segment() -> None:
    corrected = [True, False, True, False, False, True]
    rows = correction_rates(SESSIONS, corrected, lambda s: s.lang)
    italian = next(row for row in rows if row["segment"] == "it")
    assert italian["correction_rate"] == pytest.approx(2 / 4)
    assert italian["sessions_with_errors"] == pytest.approx(3 / 4)
    assert italian["wer"] == pytest.approx(9 / 40)


def test_the_e10_spec_has_a_user_model() -> None:
    users = load_user_model(Path("experiments/E10/simulation.toml"))
    assert 0 < users.p_notice <= 1 and 0 <= users.p_edit < 1


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A transcribed run of eight sessions: quiet and noisy, headset and phone, two languages.
    The German babble session transcribes its background voice."""
    source = tmp_path / "source.jsonl"
    write_manifest(source, [
        ManifestEntry("bg_it", "x.wav", 1.0, "sei sette otto", "it", "s", "test"),
        ManifestEntry("bg_de", "y.wav", 1.0, "uno tre", "de", "s", "test"),
    ])  # fmt: skip
    conditions = [
        ("headset", "quiet", None), ("phone", "quiet", None),
        ("headset", "babble", "5"), ("phone", "stationary", "0"),
    ]  # fmt: skip
    entries, hypotheses = [], []
    for lang in ("it", "de"):
        for i, (channel, environment, snr) in enumerate(conditions):
            session_id = f"{lang}_{i}"
            metadata = {"channel": channel, "environment": environment, "sentence_id": str(i)}
            if snr:
                metadata["snr_db"] = snr
            if environment == "babble":
                metadata |= {"babble_voices": f"bg_{lang}", "recording_manifest": str(source)}
            entries.append(
                ManifestEntry(session_id, f"audio/{session_id}.wav", 1.0, "uno due tre quattro",
                              lang, "speaker", "test", metadata)
            )  # fmt: skip
            text = "uno due tre quattro" if environment == "quiet" else "uno tre"
            hypotheses.append(Hypothesis(session_id, text, score=-0.1 if snr is None else -1.0))
    manifest = tmp_path / "sessions.jsonl"
    write_manifest(manifest, entries)
    run = tmp_path / "run"
    append_predictions(run / "predictions.jsonl", hypotheses)
    info = {
        "model": "fake",
        "manifest": manifest.as_posix(),
        "manifest_sha256": file_sha256(manifest),
    }
    (run / "run.json").write_text(json.dumps(info), encoding="utf-8")
    return run


def test_analyze_usage_and_its_tables(run_dir: Path) -> None:
    report = analyze_usage(run_dir, UserModel(1.0, 0.0, 0), n_resamples=50)
    assert report["sessions"] == 8
    assert report["overall"]["wer"] == pytest.approx(8 / 32)  # 2 deletions in each noisy one
    assert report["confidence"]["spearman"] < 0
    assert report["corrections"]["overall"]["correction_rate"] == pytest.approx(0.5)
    assert list(report["by_language_condition"]) == ["it", "de"]
    assert report["background_capture"][0]["captured"] == 1  # the German one
    assert report["confidence"]["captured"] == 1
    json.dumps(report)  # serializable
    for section in ("segments", "conditions", "priorities", "confidence", "capture", "corrections"):
        table = usage_table(report, section)
        assert table.startswith("| ") and len(table.splitlines()) > 2
    with pytest.raises(ValueError, match="unknown usage table"):
        usage_table(report, "nope")


def test_cli_usage_and_generated_table(
    run_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = tmp_path / "simulation.toml"
    spec.write_text("[users]\np_notice = 1.0\np_edit = 0.0\nseed = 0\n", encoding="utf-8")
    output = tmp_path / "usage.json"
    cli.main(["usage", "--run", str(run_dir), "--spec", str(spec), "--output", str(output),
              "--n-resamples", "50"])  # fmt: skip
    assert "8 sessions, WER 25.00%" in capsys.readouterr().out
    tables_spec = tmp_path / "tables.toml"
    tables_spec.write_text(
        f'[[table]]\nname = "x"\nkind = "usage"\nfile = "{output.as_posix()}"\n'
        'section = "confidence"\n',
        encoding="utf-8",
    )
    assert build_tables(tables_spec)["x"] == usage_table(
        json.loads(output.read_text()), "confidence"
    )


def test_cli_usage_figure(run_dir: Path, tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    spec = tmp_path / "simulation.toml"
    spec.write_text("[users]\np_notice = 1.0\np_edit = 0.0\nseed = 0\n", encoding="utf-8")
    figure = tmp_path / "usage.png"
    output = tmp_path / "usage.json"
    cli.main(["usage", "--run", str(run_dir), "--spec", str(spec), "--output", str(output),
              "--n-resamples", "50", "--figure", str(figure)])  # fmt: skip
    assert figure.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
