"""Groups of utterances with similar quality features, described in words.

The features are standardized (each becomes "how many standard deviations from the average"), so
that seconds, decibels and error rates weigh the same, and k-means groups the utterances. The
number of groups is the one with the best silhouette score among those in the spec: how much
closer each utterance is to its own group than to the nearest other one (from -1 to 1).

Each cluster is described by the features where its average is far from the overall one, e.g.
"high WER, long model transcript": the description is a starting point for listening to its
examples, not a diagnosis.
"""

from __future__ import annotations

import math
import tomllib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_NAMES = {
    "duration": "duration",
    "chars_per_second": "speaking rate",
    "snr_db": "SNR",
    "clipping": "clipping",
    "wer": "WER",
    "length_ratio": "model transcript length",
}
DESCRIBE_Z = 0.5  # a cluster average this many standard deviations away is worth naming


@dataclass(frozen=True)
class ClusterSpec:
    """The ``[clusters]`` table of a quality spec."""

    features: tuple[str, ...]  # feature names, "log:" for log(1 + x)
    k: tuple[int, ...]  # candidate numbers of clusters
    seed: int
    examples: int  # utterances shown per cluster


def load_cluster_spec(path: Path) -> ClusterSpec:
    """Read the ``[clusters]`` table of a quality spec."""
    table = tomllib.loads(path.read_text(encoding="utf-8"))["clusters"]
    return ClusterSpec(
        tuple(table["features"]), tuple(table["k"]), table["seed"], table["examples"]
    )


def feature_matrix(
    rows: Sequence[dict[str, Any]], features: Sequence[str]
) -> tuple[np.ndarray, list[int]]:
    """The (transformed) features of the rows that have all of them, and those rows' indices.

    >>> rows = [{"wer": 0.0, "d": 2.0}, {"wer": None, "d": 1.0}, {"wer": 1.0, "d": 3.0}]
    >>> matrix, kept = feature_matrix(rows, ["log:wer", "d"])
    >>> kept, matrix.round(2).tolist()
    ([0, 2], [[0.0, 2.0], [0.69, 3.0]])
    """
    columns = [(name.removeprefix("log:"), name.startswith("log:")) for name in features]
    values, kept = [], []
    for index, row in enumerate(rows):
        raw = [row.get(name) for name, _ in columns]
        if any(value is None or math.isnan(value) for value in raw):
            continue
        values.append(
            [math.log1p(v) if log else v for v, (_, log) in zip(raw, columns, strict=True)]
        )
        kept.append(index)
    return np.array(values, dtype=float).reshape(len(values), len(columns)), kept


def standardize(matrix: np.ndarray) -> np.ndarray:
    """Each column to mean 0 and standard deviation 1 (a constant column becomes 0)."""
    std = matrix.std(axis=0)
    return (matrix - matrix.mean(axis=0)) / np.where(std > 0, std, 1.0)


def choose_k(
    standardized: np.ndarray, ks: Sequence[int], seed: int
) -> tuple[int, np.ndarray, dict[int, float]]:
    """The number of clusters with the best silhouette, its labels and every candidate's score.

    Raises:
        ValueError: if no candidate is smaller than the number of utterances.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    scores: dict[int, float] = {}
    labels: dict[int, np.ndarray] = {}
    for k in ks:
        if not 2 <= k < len(standardized):
            continue
        labels[k] = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(standardized)
        scores[k] = float(silhouette_score(standardized, labels[k]))
    if not scores:
        raise ValueError(f"no number of clusters in {list(ks)} fits {len(standardized)} rows")
    best = max(scores, key=lambda k: (scores[k], -k))  # ties: the fewer clusters
    return best, labels[best], scores


def cluster_rows(
    rows: Sequence[dict[str, Any]],
    spec: ClusterSpec,
    flags: Sequence[Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Cluster the rows on the spec's features and describe each cluster.

    Rows missing a feature (e.g. the WER of an empty transcript) are left out and counted.
    ``flags`` (the filter rules each row breaks) adds the share of each cluster flagged by
    each rule. Clusters are numbered from the largest.
    """
    matrix, kept = feature_matrix(rows, spec.features)
    standardized = standardize(matrix)
    k, labels, scores = choose_k(standardized, spec.k, spec.seed)
    order = [label for label, _ in Counter(labels.tolist()).most_common()]
    clusters = []
    for number, label in enumerate(order, start=1):
        members = np.flatnonzero(labels == label)
        z = standardized[members].mean(axis=0)
        centre_distance = np.linalg.norm(standardized[members] - z, axis=1)
        member_rows = [rows[kept[i]] for i in members]
        cluster: dict[str, Any] = {
            "cluster": number,
            "utterances": len(members),
            "share": len(members) / len(kept),
            "hours": sum(row["duration"] for row in member_rows) / 3600,
            "speakers": len({row["speaker_id"] for row in member_rows}),
            "median": _medians(member_rows),
            "z": dict(zip(spec.features, z.round(3).tolist(), strict=True)),
            "description": describe(dict(zip(spec.features, z.tolist(), strict=True))),
            "examples": [
                _example(rows[kept[members[i]]])
                for i in np.argsort(centre_distance, kind="stable")[: spec.examples]
            ],
        }
        if flags is not None:
            counts = Counter(rule for i in members for rule in flags[kept[i]])
            cluster["flagged"] = sum(bool(flags[kept[i]]) for i in members) / len(members)
            cluster["rules"] = {rule: n / len(members) for rule, n in counts.most_common()}
        clusters.append(cluster)
    return {
        "features": list(spec.features),
        "utterances": len(kept),
        "left_out": len(rows) - len(kept),
        "silhouette": {str(key): value for key, value in scores.items()},
        "k": k,
        "overall_median": _medians([rows[i] for i in kept]),
        "clusters": clusters,
        "labels": {
            rows[kept[i]]["utterance_id"]: order.index(label) + 1
            for i, label in enumerate(labels.tolist())
        },
    }


def describe(z: dict[str, float], limit: int = 3) -> str:
    """The features a cluster stands out on, largest deviation first.

    >>> describe({"log:wer": 1.8, "snr_db": -0.7, "duration": 0.1})
    'high WER, low SNR'
    >>> describe({"wer": 0.2})
    'typical'
    """
    notable = sorted(
        ((name, value) for name, value in z.items() if abs(value) >= DESCRIBE_Z),
        key=lambda item: -abs(item[1]),
    )[:limit]
    if not notable:
        return "typical"
    return ", ".join(
        f"{'high' if value > 0 else 'low'} "
        f"{FEATURE_NAMES.get(name.removeprefix('log:'), name.removeprefix('log:'))}"
        for name, value in notable
    )


def _medians(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Median of every feature of ``FEATURE_NAMES`` the rows have (clustered on or not)."""
    medians = {}
    for name in FEATURE_NAMES:
        values = [row[name] for row in rows if row.get(name) is not None]
        if values:
            medians[name] = float(np.median(values))
    return medians


def _example(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "utterance_id": row["utterance_id"],
        "duration": row["duration"],
        "wer": row.get("wer"),
        "reference": row.get("normalized_text"),
        "hypothesis": row.get("hypothesis_text"),
    }
