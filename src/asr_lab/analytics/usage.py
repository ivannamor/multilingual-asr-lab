"""What a product team would ask of its usage logs, answered on the simulated sessions of E10.

1. **Where do the errors come from?** Word error rate per language, channel and acoustic
   condition, with bootstrap intervals, and each segment's share of all word errors.
2. **Where to invest?** For each segment (language, channel, environment), the errors that would
   disappear if it were as accurate as quiet headset sessions of the same language: its share of
   all errors is the priority. It combines how much traffic the segment has (simulated) with how
   much worse it is (measured).
3. **Can the model tell when it is wrong?** Sessions sorted by the model's confidence (average
   log-probability): how many of the word errors a review of the least confident sessions
   would catch, against a random review and the best possible one. In babble, the model can
   transcribe a background voice instead of the speaker: such sessions are flagged, counted
   per condition and followed through the review.
4. **What does the product see?** Users correct a transcript with a probability that grows with
   its errors (a simulated behaviour, set in the spec). The correction rate is what telemetry
   measures; comparing it with the true WER shows what it hides.

Everything is computed on normalized text, with each session's language normalizer, from the
saved predictions; ``analyze_usage`` returns a JSON-serializable report.
"""

from __future__ import annotations

import json
import tomllib
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from asr_lab.data.manifest import ManifestEntry, read_manifest
from asr_lab.evaluation.error_analysis import analyze
from asr_lab.evaluation.metrics import ErrorCounts, align, count_errors, error_rate, word_tokens
from asr_lab.evaluation.normalize import get_normalizer
from asr_lab.evaluation.scoring import DEFAULT_N_RESAMPLES, DEFAULT_SEED, load_run
from asr_lab.evaluation.stats import bootstrap_ci

REVIEW_BUDGETS = (0.05, 0.1, 0.2, 0.3)  # shares of the sessions sent to review
LANGUAGE_NAMES = {"it": "Italian", "de": "German", "en": "English"}
# Order of segments in every table: languages, channels, environments (cleanest first).
_ORDER = (*LANGUAGE_NAMES.values(), "headset", "phone", "quiet", "stationary", "babble")


@dataclass(frozen=True)
class Session:
    """One transcribed session: its conditions, its errors and the model's confidence."""

    session_id: str
    lang: str
    channel: str
    environment: str
    snr_db: str | None  # None in quiet sessions
    sentence_id: str
    confidence: float | None  # average log-probability of the transcript
    counts: ErrorCounts
    captured: bool = False  # the transcript follows a background voice (see captures_background)

    @property
    def condition(self) -> str:
        """The acoustic condition, e.g. ``"quiet"`` or ``"babble 5 dB"``."""
        return self.environment if self.snr_db is None else f"{self.environment} {self.snr_db} dB"


@dataclass(frozen=True)
class UserModel:
    """Simulated users: each word error is noticed with ``p_notice``, and a transcript with at
    least one noticed error is corrected; ``p_edit`` is the chance of editing a correct one."""

    p_notice: float
    p_edit: float
    seed: int


def load_sessions(run_dir: Path) -> list[Session]:
    """The sessions of a run, aligned after normalization with each language's normalizer.

    Babble sessions are compared with the transcripts of their background voices, read from
    the manifest their recording comes from (``recording_manifest``).
    """
    run = load_run(run_dir)
    utterances = analyze(run.entries, run.hypotheses)
    backgrounds = _background_texts(run.entries)
    return [
        Session(
            session_id=entry.utterance_id,
            lang=entry.lang,
            channel=entry.metadata["channel"],
            environment=entry.metadata["environment"],
            snr_db=entry.metadata.get("snr_db"),
            sentence_id=entry.metadata["sentence_id"],
            confidence=hypothesis.score,
            counts=utterance.counts,
            captured=captures_background(
                utterance.hypothesis, utterance.counts, backgrounds.get(entry.utterance_id, [])
            ),
        )
        for entry, hypothesis, utterance in zip(
            run.entries, run.hypotheses, utterances, strict=True
        )
    ]


def captures_background(hypothesis: str, counts: ErrorCounts, backgrounds: Sequence[str]) -> bool:
    """Whether a transcript is closer, in word error rate, to one of the background voices than
    to the speaker (``counts`` are its errors against the speaker's reference). All texts are
    normalized.

    >>> counts = ErrorCounts(substitutions=3)  # "a b c" heard as "x y z"
    >>> captures_background("x y z", counts, ["x y z w"]), captures_background("x y z", counts, [])
    (True, False)
    """
    words = word_tokens(hypothesis)
    return any(
        error_rate(count_errors(align(word_tokens(text), words))) < error_rate(counts)
        for text in backgrounds
    )


def load_user_model(path: Path) -> UserModel:
    """The ``[users]`` table of the E10 simulation spec."""
    users = tomllib.loads(path.read_text(encoding="utf-8"))["users"]
    return UserModel(users["p_notice"], users["p_edit"], users["seed"])


def segment_rates(
    sessions: Sequence[Session],
    key: Callable[[Session], str],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """WER of each segment with a bootstrap interval resampling sentences, its share of the
    sessions and its share of all word errors; segments in a fixed order (see ``_sort_key``)."""
    groups = _group(sessions, key)
    total_errors = sum(s.counts.errors for s in sessions)
    rows = []
    for name, members in sorted(groups.items(), key=lambda item: _sort_key(item[0])):
        counts = [s.counts for s in members]
        total = sum(counts, ErrorCounts())
        ci = bootstrap_ci(
            counts, groups=[s.sentence_id for s in members], n_resamples=n_resamples, seed=seed
        )
        rows.append(
            {
                "segment": name,
                "sessions": len(members),
                "session_share": len(members) / len(sessions),
                "words": total.ref_length,
                "wer": error_rate(total),
                "ci_low": ci.low,
                "ci_high": ci.high,
                "error_share": total.errors / total_errors if total_errors else 0.0,
            }
        )
    return rows


def priorities(sessions: Sequence[Session]) -> list[dict[str, Any]]:
    """Errors each segment (language, channel, environment) would lose at the WER of quiet
    headset sessions of its language, as a share of all errors; largest first.

    Quiet headset segments are the reference and have no avoidable errors by definition.

    Raises:
        ValueError: if a language has no quiet headset session to serve as reference.
    """
    reference = {
        lang: error_rate(sum((s.counts for s in members), ErrorCounts()))
        for lang, members in _group(
            [s for s in sessions if s.environment == "quiet" and s.channel == "headset"],
            lambda s: s.lang,
        ).items()
    }
    missing = sorted({s.lang for s in sessions} - set(reference))
    if missing:
        raise ValueError(f"no quiet headset session to use as reference for {missing}")
    total_errors = sum(s.counts.errors for s in sessions)
    rows = []
    for (lang, channel, environment), members in _group(
        sessions, lambda s: (s.lang, s.channel, s.environment)
    ).items():
        total = sum((s.counts for s in members), ErrorCounts())
        avoidable = max(total.errors - reference[lang] * total.ref_length, 0.0)
        rows.append(
            {
                "lang": lang,
                "channel": channel,
                "environment": environment,
                "sessions": len(members),
                "session_share": len(members) / len(sessions),
                "wer": error_rate(total),
                "reference_wer": reference[lang],
                "error_share": total.errors / total_errors,
                "avoidable_share": avoidable / total_errors,
            }
        )
    return sorted(rows, key=lambda row: -row["avoidable_share"])


def confidence_review(
    sessions: Sequence[Session], budgets: Sequence[float] = REVIEW_BUDGETS
) -> dict[str, Any]:
    """How many word errors a review of the least confident sessions would catch.

    For each budget (share of the sessions reviewed), the share of all word errors in the
    reviewed sessions when they are chosen by lowest confidence, at random (in expectation, the
    budget itself) and by an oracle that knows the errors, and how many sessions that captured
    a background voice the confidence review reaches. Also the Spearman rank correlation
    between confidence and session WER (negative: the lower the confidence, the higher the WER).
    """
    scored = [s for s in sessions if s.confidence is not None]
    errors = np.array([s.counts.errors for s in scored], dtype=float)
    confidence = np.array([s.confidence for s in scored], dtype=float)
    wers = np.array([error_rate(s.counts) for s in scored], dtype=float)
    order = np.argsort(confidence, kind="stable")
    by_confidence = errors[order]
    captured = np.array([s.captured for s in scored], dtype=bool)[order]
    by_errors = np.sort(errors)[::-1]
    total = errors.sum()
    rows = []
    for budget in budgets:
        k = round(budget * len(scored))
        rows.append(
            {
                "budget": budget,
                "sessions": k,
                "confidence": by_confidence[:k].sum() / total,
                "random": k / len(scored),
                "oracle": by_errors[:k].sum() / total,
                "captured": int(captured[:k].sum()),
            }
        )
    return {
        "sessions": len(scored),
        "captured": int(captured.sum()),
        "spearman": spearman(confidence, wers),
        "review": rows,
    }


def background_capture(sessions: Sequence[Session]) -> list[dict[str, Any]]:
    """Per babble condition: the sessions whose transcript follows a background voice, and their
    share of the condition's word errors."""
    rows = []
    babble = [s for s in sessions if s.environment == "babble"]
    for name, members in sorted(
        _group(babble, lambda s: s.condition).items(), key=lambda i: _sort_key(i[0])
    ):
        captured = [s for s in members if s.captured]
        errors = sum(s.counts.errors for s in members)
        rows.append(
            {
                "segment": name,
                "sessions": len(members),
                "captured": len(captured),
                "captured_share": len(captured) / len(members),
                "error_share": sum(s.counts.errors for s in captured) / errors if errors else 0.0,
            }
        )
    return rows


def simulate_corrections(sessions: Sequence[Session], users: UserModel) -> list[bool]:
    """Whether the (simulated) user corrected each session."""
    rng = np.random.default_rng(users.seed)
    corrected = []
    for session in sessions:
        noticed = rng.binomial(session.counts.errors, users.p_notice) > 0
        corrected.append(bool(noticed or rng.random() < users.p_edit))
    return corrected


def correction_rates(
    sessions: Sequence[Session], corrected: Sequence[bool], key: Callable[[Session], str]
) -> list[dict[str, Any]]:
    """Per segment: share of sessions corrected (what telemetry sees), share of sessions with
    at least one error, and the true WER."""
    flags = dict(zip((s.session_id for s in sessions), corrected, strict=True))
    rows = []
    for name, members in sorted(_group(sessions, key).items(), key=lambda i: _sort_key(i[0])):
        rows.append(
            {
                "segment": name,
                "sessions": len(members),
                "correction_rate": sum(flags[s.session_id] for s in members) / len(members),
                "sessions_with_errors": sum(s.counts.errors > 0 for s in members) / len(members),
                "wer": error_rate(sum((s.counts for s in members), ErrorCounts())),
            }
        )
    return rows


def analyze_usage(
    run_dir: Path,
    users: UserModel,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Every analysis of the module on one run, as a JSON-serializable report."""
    sessions = load_sessions(run_dir)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    def rates(key: Callable[[Session], str]) -> list[dict[str, Any]]:
        return segment_rates(sessions, key, n_resamples=n_resamples, seed=seed)

    corrected = simulate_corrections(sessions, users)
    return {
        "run": run_dir.as_posix(),
        "model": run.get("model"),
        "model_config": run.get("model_config"),
        "manifest": run.get("manifest"),
        "sessions": len(sessions),
        "n_resamples": n_resamples,
        "seed": seed,
        "simulated": [
            "which sessions exist: languages, channels, environments and SNRs (traffic mix)",
            "whether a user corrects a transcript (user model)",
        ],
        "overall": rates(lambda s: "all sessions")[0],
        "by_language": rates(lambda s: LANGUAGE_NAMES.get(s.lang, s.lang)),
        "by_channel": rates(lambda s: s.channel),
        "by_environment": rates(lambda s: s.environment),
        "by_condition": {
            channel: _rates_where(
                sessions, lambda s, channel=channel: s.channel == channel, n_resamples, seed
            )
            for channel in ("headset", "phone")
        },
        "by_language_condition": {
            lang: {
                channel: _rates_where(
                    sessions,
                    lambda s, lang=lang, channel=channel: s.lang == lang and s.channel == channel,
                    n_resamples,
                    seed,
                )
                for channel in ("headset", "phone")
            }
            for lang in LANGUAGE_NAMES
            if any(s.lang == lang for s in sessions)
        },
        "priorities": priorities(sessions),
        "confidence": confidence_review(sessions),
        "background_capture": background_capture(sessions),
        "user_model": {"p_notice": users.p_notice, "p_edit": users.p_edit, "seed": users.seed},
        "corrections": {
            "overall": correction_rates(sessions, corrected, lambda s: "all sessions")[0],
            "by_condition": correction_rates(sessions, corrected, lambda s: s.condition),
        },
    }


def usage_table(report: dict[str, Any], section: str) -> str:
    """One Markdown table of a usage report: ``segments``, ``conditions``, ``priorities``,
    ``confidence``, ``capture`` or ``corrections``."""
    tables = {
        "segments": _segments_table,
        "conditions": _conditions_table,
        "priorities": _priorities_table,
        "confidence": _confidence_table,
        "capture": _capture_table,
        "corrections": _corrections_table,
    }
    if section not in tables:
        raise ValueError(f"unknown usage table {section!r}; expected one of {sorted(tables)}")
    return tables[section](report)


def save_usage(report: dict[str, Any], path: Path) -> None:
    """Write a usage report as JSON (UTF-8, LF)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _segments_table(report: dict[str, Any]) -> str:
    lines = [
        "| Segment | Sessions (share) | Words | WER [95% CI] | Share of all word errors |",
        "|---|---|---|---|---|",
    ]
    for group in ("overall", "by_language", "by_channel", "by_environment"):
        rows = report[group] if isinstance(report[group], list) else [report[group]]
        for row in rows:
            lines.append(
                f"| {row['segment']} | {row['sessions']:,} ({row['session_share']:.0%}) "
                f"| {row['words']:,} | {_wer_ci(row)} | {row['error_share']:.0%} |"
            )
    return "\n".join(lines)


def _conditions_table(report: dict[str, Any]) -> str:
    channels = list(report["by_condition"])
    lines = [
        "| Condition | " + " | ".join(f"{c}: WER [95% CI] (sessions)" for c in channels) + " |",
        "|---|" + "---|" * len(channels),
    ]
    by_channel = {c: {row["segment"]: row for row in report["by_condition"][c]} for c in channels}
    conditions = list(dict.fromkeys(name for c in channels for name in by_channel[c]))
    for condition in sorted(conditions, key=_sort_key):
        cells = [
            f"{_wer_ci(by_channel[c][condition])} ({by_channel[c][condition]['sessions']})"
            if condition in by_channel[c]
            else "-"
            for c in channels
        ]
        lines.append(f"| {condition} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _priorities_table(report: dict[str, Any]) -> str:
    lines = [
        "| Segment | Sessions (share) | WER | WER in quiet, headset | Share of all word errors "
        "| Avoidable share of all word errors |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["priorities"]:
        segment = f"{LANGUAGE_NAMES.get(row['lang'], row['lang'])}, {row['channel']}, "
        segment += row["environment"]
        lines.append(
            f"| {segment} | {row['sessions']:,} ({row['session_share']:.0%}) | {row['wer']:.2%} "
            f"| {row['reference_wer']:.2%} | {row['error_share']:.1%} "
            f"| {row['avoidable_share']:.1%} |"
        )
    return "\n".join(lines)


def _confidence_table(report: dict[str, Any]) -> str:
    review = report["confidence"]
    lines = [
        "| Sessions reviewed | Word errors caught: lowest confidence first | At random "
        "| Best possible | Background-voice transcripts reached |",
        "|---|---|---|---|---|",
    ]
    for row in review["review"]:
        lines.append(
            f"| {row['budget']:.0%} ({row['sessions']:,}) | {row['confidence']:.0%} "
            f"| {row['random']:.0%} | {row['oracle']:.0%} "
            f"| {row['captured']} of {review['captured']} |"
        )
    return "\n".join(lines)


def _capture_table(report: dict[str, Any]) -> str:
    lines = [
        "| Condition | Sessions | Transcript follows a background voice "
        "| Their share of the condition's word errors |",
        "|---|---|---|---|",
    ]
    for row in report["background_capture"]:
        lines.append(
            f"| {row['segment']} | {row['sessions']:,} | {row['captured']} "
            f"({row['captured_share']:.1%}) | {row['error_share']:.1%} |"
        )
    return "\n".join(lines)


def _corrections_table(report: dict[str, Any]) -> str:
    lines = [
        "| Condition | Sessions | WER | Sessions with at least one error "
        "| Sessions corrected (simulated users) |",
        "|---|---|---|---|---|",
    ]
    corrections = report["corrections"]
    for row in [corrections["overall"], *corrections["by_condition"]]:
        lines.append(
            f"| {row['segment']} | {row['sessions']:,} | {row['wer']:.2%} "
            f"| {row['sessions_with_errors']:.0%} | {row['correction_rate']:.0%} |"
        )
    return "\n".join(lines)


def _wer_ci(row: dict[str, Any]) -> str:
    return f"{row['wer']:.2%} [{row['ci_low']:.2%}, {row['ci_high']:.2%}]"


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (Pearson correlation of the ranks, ties get their mean rank);
    NaN if either variable is constant, since the correlation is then undefined.

    >>> round(spearman(np.array([1.0, 2.0, 3.0]), np.array([10.0, 30.0, 20.0])), 2)
    0.5
    """
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(_ranks(x), _ranks(y))[0, 1])


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    for value in np.unique(values):  # ties share the mean of their ranks
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def _background_texts(entries: Sequence[ManifestEntry]) -> dict[str, list[str]]:
    """Normalized transcripts of the background voices of each babble session, by session id."""
    manifests: dict[str, dict[str, str]] = {}
    texts = {}
    for entry in entries:
        voices = entry.metadata.get("babble_voices")
        if not voices:
            continue
        source = entry.metadata["recording_manifest"]
        if source not in manifests:
            manifests[source] = {e.utterance_id: e.text for e in read_manifest(Path(source))}
        normalizer = get_normalizer(entry.lang)
        texts[entry.utterance_id] = [normalizer(manifests[source][v]) for v in voices.split()]
    return texts


def _rates_where(
    sessions: Sequence[Session],
    keep: Callable[[Session], bool],
    n_resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    kept = [s for s in sessions if keep(s)]
    return segment_rates(kept, lambda s: s.condition, n_resamples=n_resamples, seed=seed)


def _sort_key(name: str) -> tuple[int, float, str]:
    """Known names in ``_ORDER``, then conditions from the highest SNR to the lowest."""
    first, *rest = name.split()
    rank = _ORDER.index(first) if first in _ORDER else len(_ORDER)
    snr = -float(rest[0]) if rest and rest[-1] == "dB" else float("-inf")
    return rank, snr, name


def _group(sessions: Sequence[Session], key: Callable[[Session], Any]) -> dict[Any, list[Session]]:
    groups: dict[Any, list[Session]] = defaultdict(list)
    for session in sessions:
        groups[key(session)].append(session)
    return dict(groups)
