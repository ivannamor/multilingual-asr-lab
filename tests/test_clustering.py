"""Tests for the clustering of quality features, on groups built to be separable."""

import numpy as np
import pytest

from asr_lab.data.clustering import ClusterSpec, choose_k, cluster_rows, describe, standardize


def rows_in_groups(seed: int = 0) -> list[dict]:
    """Three groups: clean (low WER, high SNR), noisy (low SNR), mislabelled (high WER)."""
    rng = np.random.default_rng(seed)
    centres = {"clean": (0.03, 30.0), "noisy": (0.15, 5.0), "mislabelled": (0.9, 28.0)}
    rows = []
    for name, (wer, snr) in centres.items():
        for i in range(20):
            rows.append({
                "utterance_id": f"{name}_{i}", "speaker_id": f"s{i % 4}", "duration": 10.0,
                "wer": max(0.0, wer + 0.02 * rng.standard_normal()),
                "snr_db": snr + rng.standard_normal(), "normalized_text": name,
            })  # fmt: skip
    return rows


SPEC = ClusterSpec(features=("log:wer", "snr_db"), k=(2, 3, 4, 5), seed=0, examples=2)


def test_standardize_handles_constant_columns() -> None:
    z = standardize(np.array([[1.0, 5.0], [3.0, 5.0]]))
    assert z.tolist() == [[-1.0, 0.0], [1.0, 0.0]]


def test_choose_k_finds_separated_groups() -> None:
    matrix = np.array([[row["wer"], row["snr_db"]] for row in rows_in_groups()])
    k, labels, scores = choose_k(standardize(matrix), SPEC.k, seed=0)
    assert k == 3 and len(set(labels)) == 3
    assert set(scores) == {2, 3, 4, 5} and scores[3] == max(scores.values())


def test_choose_k_needs_more_rows_than_clusters() -> None:
    with pytest.raises(ValueError, match="no number of clusters"):
        choose_k(np.zeros((2, 1)), (2, 3), seed=0)


def test_clusters_are_described_and_numbered_by_size() -> None:
    rows = rows_in_groups()
    rows.append({**rows[0], "utterance_id": "empty", "wer": None})  # left out, not clustered
    flags = [["high_wer"] if row["utterance_id"].startswith("mislabelled") else [] for row in rows]
    report = cluster_rows(rows, SPEC, flags)
    assert report["k"] == 3 and report["left_out"] == 1 and report["utterances"] == 60
    by_name = {
        c["examples"][0]["reference"]: c for c in report["clusters"]
    }  # every example of a cluster comes from the same group
    assert set(by_name) == {"clean", "noisy", "mislabelled"}
    assert by_name["mislabelled"]["description"].startswith("high WER")
    assert "low SNR" in by_name["noisy"]["description"]
    assert by_name["mislabelled"]["flagged"] == 1.0 and by_name["clean"]["flagged"] == 0.0
    assert by_name["mislabelled"]["rules"] == {"high_wer": 1.0}
    assert [c["utterances"] for c in report["clusters"]] == [20, 20, 20]
    assert set(report["labels"].values()) == {1, 2, 3} and "empty" not in report["labels"]
    assert by_name["clean"]["median"]["snr_db"] == pytest.approx(30, abs=1.5)


def test_describe_names_the_largest_deviations() -> None:
    assert describe({"log:duration": -0.6, "chars_per_second": 2.0}) == (
        "high speaking rate, low duration"
    )
    assert describe({"a": 1, "b": 2, "c": 3, "d": 4}, limit=2) == "high d, high c"
