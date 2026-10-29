"""Tests for the quality features and filter rules, on synthetic utterances with known flaws."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asr_lab import cli
from asr_lab.data.manifest import ManifestEntry, read_manifest, write_manifest
from asr_lab.data.quality import (
    RULES,
    Filters,
    analyze_quality,
    clipping,
    compute_features,
    estimate_snr,
    export_collapsed,
    filter_summary,
    flag,
    high_wer_breakdown,
    quality_table,
    second_opinion,
    text_features,
)
from asr_lab.evaluation.normalize import get_normalizer_by_name
from asr_lab.inference.base import Hypothesis
from asr_lab.inference.predictions import append_predictions
from asr_lab.provenance import file_sha256
from asr_lab.reporting import build_tables

IT = get_normalizer_by_name("it-v2")
FILTERS = Filters(
    min_duration=1.0, max_duration=30.0, min_chars_per_second=5.0, max_chars_per_second=25.0,
    min_snr_db=10.0, max_clipping=0.001, max_wer=0.5,
)  # fmt: skip


def row(**changes: object) -> dict:
    """A feature row that breaks no rule, with some features changed."""
    return {
        "empty_text": False, "duration": 5.0, "chars_per_second": 12.0, "snr_db": 30.0,
        "clipping": 0.0, "wer": 0.1, "duplicate_text": False, "duplicate_audio": False,
    } | changes  # fmt: skip


def test_snr_estimate_tracks_the_noise_level() -> None:
    rng = np.random.default_rng(0)
    speech = np.concatenate([np.zeros(8_000), 0.3 * np.sin(np.arange(24_000) / 5)])
    quiet = estimate_snr(speech + 0.001 * rng.standard_normal(32_000), 16_000)
    noisy = estimate_snr(speech + 0.03 * rng.standard_normal(32_000), 16_000)
    assert quiet > noisy + 20
    assert estimate_snr(np.zeros(16_000), 16_000) == 0.0  # silence: no signal above the noise
    assert estimate_snr(np.concatenate([np.zeros(800), np.ones(800)]), 16_000) == 100.0  # floor
    assert np.isnan(estimate_snr(np.ones(10), 16_000))  # shorter than one frame


def test_clipping_counts_full_scale_samples() -> None:
    assert clipping(np.array([0.1, -1.0, 0.99999, 0.5])) == 0.5
    assert clipping(np.array([])) == 0.0


def test_text_features_use_the_normalized_text() -> None:
    features = text_features("Il 2 maggio.", 2.0, IT)
    assert features["normalized_text"] == "il due maggio"
    assert features["words"] == 3 and features["chars_per_second"] == 5.5
    assert text_features(" ... ", 1.0, IT)["empty_text"]


@pytest.mark.parametrize(
    ("changes", "rules"),
    [
        ({}, []),
        ({"empty_text": True, "chars_per_second": 0.0, "wer": None}, ["empty_text"]),
        ({"duration": 0.5}, ["too_short"]),
        ({"duration": 31.0}, ["too_long"]),
        ({"chars_per_second": 3.0}, ["too_slow"]),
        ({"chars_per_second": 30.0}, ["too_fast"]),
        ({"snr_db": 5.0}, ["low_snr"]),
        ({"snr_db": float("nan")}, []),  # unknown, not low
        ({"clipping": 0.01}, ["clipped"]),
        ({"wer": 0.8}, ["high_wer"]),
        ({"wer": None}, []),
        ({"duplicate_text": True, "duplicate_audio": True}, ["duplicate_text", "duplicate_audio"]),
    ],
)
def test_rules(changes: dict, rules: list[str]) -> None:
    assert flag(row(**changes), FILTERS) == rules


def test_summary_counts_each_rule_and_the_union() -> None:
    rows = [row(), row(wer=0.9, snr_db=2.0, duration=3600.0), row(snr_db=2.0)]
    summary = {item["rule"]: item for item in filter_summary(rows, FILTERS)}
    assert list(summary) == [*RULES, "any"]
    assert summary["low_snr"]["utterances"] == 2 and summary["high_wer"]["utterances"] == 1
    assert summary["any"]["utterances"] == 2 and summary["any"]["share"] == pytest.approx(2 / 3)
    assert summary["any"]["hours"] == pytest.approx((3600 + 5) / 3600)
    assert summary["too_long"]["hour_share"] == pytest.approx(3600 / 3610)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Eight utterances: clean, noisy, clipped, a repeated text and audio, an empty transcript;
    their transcription; and a quality spec pointing to them. Returns the spec."""
    rng = np.random.default_rng(1)
    speech = np.concatenate([np.zeros(8_000), 0.3 * np.sin(np.arange(32_000) / 7)])
    noise_level = {"clean": 0.001, "noisy": 0.2, "clipped": 0.001}
    texts = {
        "u1": ("clean", "il gatto dorme sul divano", "il gatto dorme sul divano"),
        "u2": ("clean", "oggi piove molto in città", "oggi piove molto in città"),
        "u3": ("noisy", "domani andiamo al mare", "domani"),
        "u4": ("clipped", "la musica è troppo alta", "la musica è troppo alta"),
        "u5": ("clean", "il gatto dorme sul divano", "il gatto dorme sul divano"),
        "u6": ("clean", "", "qualcosa"),
        "u7": ("clean", "parole che non sono state dette", "tutt altro discorso qui"),
        "u8": ("clean", "una frase qualunque per finire", "una frase qualunque per finire"),
    }
    data = tmp_path / "vp"
    entries, hypotheses, waveforms = [], [], {}
    for i, (name, (kind, reference, hypothesis)) in enumerate(texts.items()):
        path = data / "audio" / f"{name}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        waveform = (5 if kind == "clipped" else 1) * speech  # 1.5 at the peaks: clipped
        waveform = np.clip(waveform + noise_level[kind] * rng.standard_normal(40_000), -1, 1)
        waveforms[name] = waveforms["u1"] if name == "u5" else waveform  # u5 repeats u1's audio
        sf.write(path, waveforms[name].astype(np.float32), 16_000, subtype="PCM_16")
        entry = ManifestEntry(name, f"audio/{name}.wav", 2.5, reference, "it", f"spk{i % 3}",
                              "voxpopuli", {"session": f"s{i % 2}"})  # fmt: skip
        entries.append(entry)
        hypotheses.append(Hypothesis(name, hypothesis))
    manifest = data / "train.jsonl"
    write_manifest(manifest, entries)
    run = tmp_path / "run"
    append_predictions(run / "predictions.jsonl", hypotheses)
    (run / "run.json").write_text(json.dumps({
        "model": "fake", "manifest": manifest.as_posix(), "manifest_sha256": file_sha256(manifest),
    }), encoding="utf-8")  # fmt: skip
    spec = tmp_path / "quality.toml"
    spec.write_text(
        f'[data]\nrepo = "x"\nrevision = "{"0" * 40}"\nlang = "it"\nsplit = "train"\n'
        f'row_groups = [[0, 0]]\noutput = "{data.as_posix()}"\nnormalizer = "it-v2"\n'
        "[filters]\nmin_duration = 1.0\nmax_duration = 30.0\nmin_chars_per_second = 5.0\n"
        "max_chars_per_second = 25.0\nmin_snr_db = 10.0\nmax_clipping = 0.001\nmax_wer = 0.5\n"
        '[clusters]\nfeatures = ["log:wer", "snr_db", "length_ratio"]\nk = [2, 3]\nseed = 0\n'
        "examples = 1\n",
        encoding="utf-8",
    )
    return spec


def test_features_of_known_flaws(dataset: Path) -> None:
    manifest = dataset.parent / "vp" / "train.jsonl"
    rows = {r["utterance_id"]: r for r in compute_features(manifest, IT, dataset.parent / "run")}
    assert rows["u1"]["snr_db"] > rows["u3"]["snr_db"] + 20
    assert rows["u4"]["clipping"] > 0.1 and rows["u1"]["clipping"] == 0
    assert rows["u5"]["duplicate_text"] and rows["u5"]["duplicate_audio"]
    assert not rows["u1"]["duplicate_text"]  # the first occurrence is kept
    assert rows["u6"]["empty_text"] and rows["u6"]["wer"] is None
    assert rows["u3"]["wer"] == pytest.approx(3 / 4) and rows["u3"]["length_ratio"] == 0.25
    assert rows["u1"]["hypothesis_text"] == "il gatto dorme sul divano"
    without_run = compute_features(manifest, IT)
    assert without_run[0]["wer"] is None


def test_analysis_report_and_tables(dataset: Path) -> None:
    rows, report = analyze_quality(dataset, dataset.parent / "run")
    assert report["utterances"] == 8 and report["speakers"] == 3 and report["sessions"] == 2
    assert report["clusters"]["left_out"] == 1  # the empty transcript has no WER
    assert {r["utterance_id"]: r["cluster"] for r in rows}["u6"] is None
    by_id = {r["utterance_id"]: r["flags"] for r in rows}
    assert by_id["u7"] == ["high_wer"] and "clipped" in by_id["u4"]
    json.dumps(report)  # serializable
    kinds = {row["kind"]: row["utterances"] for row in report["high_wer"]}
    assert kinds == {"collapsed": 0, "similar_length": 1, "other": 1}  # u7: 4 of 6, u3: 1 of 4
    for section in ("flags", "clusters", "high_wer"):
        assert quality_table(report, section).startswith("| ")
    with pytest.raises(ValueError, match="unknown quality table"):
        quality_table(report, "nope")


def test_cli_quality_and_generated_table(dataset: Path, capsys: pytest.CaptureFixture) -> None:
    output = dataset.parent / "results"
    cli.main(["quality", "--spec", str(dataset), "--run", str(dataset.parent / "run"),
              "--output", str(output)])  # fmt: skip
    out = capsys.readouterr().out
    assert "8 utterances" in out and "cluster 1:" in out
    lines = (output / "features.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 8 and "normalized_text" not in json.loads(lines[0])
    tables = dataset.parent / "tables.toml"
    tables.write_text(
        f'[[table]]\nname = "q"\nkind = "quality"\nfile = "{(output / "quality.json").as_posix()}"'
        '\nsection = "flags"\n',
        encoding="utf-8",
    )
    report = json.loads((output / "quality.json").read_text(encoding="utf-8"))
    assert build_tables(tables)["q"] == quality_table(report, "flags")


def test_cli_quality_without_a_run_has_no_clusters(dataset: Path) -> None:
    output = dataset.parent / "results"
    cli.main(["quality", "--spec", str(dataset), "--output", str(output)])
    assert json.loads((output / "quality.json").read_text(encoding="utf-8"))["clusters"] is None


def test_high_wer_breakdown_and_table() -> None:
    rows = [
        {"flags": ["high_wer"], "length_ratio": 0.1, "duration": 3600.0, "speaker_id": "a",
         "session": "s1"},
        {"flags": ["high_wer"], "length_ratio": 1.0, "duration": 1800.0, "speaker_id": "b",
         "session": "s1"},
        {"flags": ["high_wer", "clipped"], "length_ratio": 0.15, "duration": 0.0,
         "speaker_id": "a", "session": "s2"},
        {"flags": [], "length_ratio": 0.0, "duration": 10.0, "speaker_id": "c", "session": "s3"},
    ]  # fmt: skip
    breakdown = {row["kind"]: row for row in high_wer_breakdown(rows)}
    assert list(breakdown) == ["collapsed", "similar_length", "other"]
    assert breakdown["collapsed"]["utterances"] == 2  # the unflagged row is not counted
    assert breakdown["collapsed"]["share"] == pytest.approx(2 / 3)
    assert breakdown["collapsed"]["speakers"] == 1 and breakdown["collapsed"]["sessions"] == 2
    assert breakdown["similar_length"]["hours"] == 0.5 and breakdown["other"]["utterances"] == 0
    table = quality_table({"high_wer": high_wer_breakdown(rows)}, "high_wer")
    assert "| collapsed (< 20% of the reference) | 2 (67%) | 1.00 | 1 | 2 |" in table


def test_second_opinion_on_the_exported_utterances(dataset: Path) -> None:
    rows = [
        {"utterance_id": name, "flags": ["high_wer"], "length_ratio": ratio}
        for name, ratio in (("u3", 0.1), ("u7", 0.67), ("u8", 0.05))
    ]
    exported = export_collapsed(rows, dataset)
    assert exported == dataset.parent / "vp" / "collapsed.jsonl"
    assert [e.utterance_id for e in read_manifest(exported)] == ["u3", "u8"]  # manifest order
    second = dataset.parent / "second"
    append_predictions(second / "predictions.jsonl", [
        Hypothesis("u3", "domani andiamo al mare"), Hypothesis("u8", "una frase"),
    ])  # fmt: skip
    (second / "run.json").write_text(json.dumps({
        "model": "small", "manifest": exported.as_posix(), "manifest_sha256": file_sha256(exported),
    }), encoding="utf-8")  # fmt: skip
    opinion = second_opinion(dataset, dataset.parent / "run", second)
    assert opinion["utterances"] == 2 and opinion["words"] == 4 + 5 and opinion["speakers"] == 2
    first, small = opinion["models"]
    assert first["model"] == "fake" and first["wer"] == pytest.approx(3 / 9)
    assert first["collapsed"] == 0 and first["within_max_wer"] == 1  # u8 was right the first time
    assert small["wer"] == pytest.approx(3 / 9) and small["collapsed"] == 0
    assert small["within_max_wer"] == 1  # u8: 2 words of 5 is 60% WER
    table = quality_table({"second_opinion": opinion}, "second_opinion")
    assert "| small | 33.33% [" in table and "| 0 / 2 | 1 / 2 |" in table


def test_second_opinion_needs_a_subset_of_the_first_run(dataset: Path) -> None:
    other = dataset.parent / "other.jsonl"
    write_manifest(other, [ManifestEntry("x1", "a.wav", 1.0, "ciao", "it", "s", "voxpopuli")])
    second = dataset.parent / "second"
    append_predictions(second / "predictions.jsonl", [Hypothesis("x1", "ciao")])
    (second / "run.json").write_text(json.dumps({"manifest": other.as_posix()}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not have"):
        second_opinion(dataset, dataset.parent / "run", second)


def test_cli_second_run_needs_a_first_run(dataset: Path) -> None:
    with pytest.raises(SystemExit):
        cli.main(["quality", "--spec", str(dataset), "--output", "x", "--export-collapsed"])
