"""Figures of the results document, drawn from saved JSON results.

Needs the optional ``plots`` extra (``uv sync --extra plots``), imported only
when a figure is drawn. The figures are static PNGs for GitHub, so they use the
light theme only; every number they show is also in a generated table.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# Chart chrome (light theme) and the one highlight colour.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
HIGHLIGHT = "#2a78d6"
LANGUAGE_NAMES = {"it": "Italian", "de": "German", "en": "English"}


def pareto_front(points: Sequence[tuple[float, float]]) -> list[int]:
    """Indices of the points that no other point beats on both coordinates (lower is
    better), sorted by the first coordinate.

    >>> pareto_front([(1, 3), (2, 2), (3, 3), (2, 1)])
    [0, 3]
    """
    front = [
        i
        for i, (x, y) in enumerate(points)
        if not any(
            other_x <= x and other_y <= y and (other_x, other_y) != (x, y)
            for other_x, other_y in points
        )
    ]
    return sorted(front, key=lambda i: points[i])


def plot_selection(selection: dict[str, Any], path: Path) -> Path:
    """WER against processing time for each candidate of a decoding sweep, one panel per
    language, with the Pareto front and the selected configuration highlighted.

    The time is the total over all languages relative to the baseline (the first
    configuration), as in the selection rule.
    """
    plt = _import_pyplot()
    configs = {name: r for name, r in selection["configs"].items() if r["selectable"]}
    names = list(configs)
    baseline_seconds = next(iter(selection["configs"].values()))["processing_seconds"]
    times = [100 * (configs[n]["processing_seconds"] / baseline_seconds - 1) for n in names]
    languages = selection["languages"]

    wers = {lang: [100 * configs[n]["wer"][lang] for n in names] for lang in languages}
    # The same WER span in every panel, so that equal gaps look equal across languages.
    span = 1.3 * max(max(values) - min(values) for values in wers.values())

    figure, axes = plt.subplots(
        1, len(languages), figsize=(3.6 * len(languages), 4.2), sharex=True, facecolor=SURFACE
    )
    for axis, lang in zip(axes, languages, strict=True):
        front = pareto_front(list(zip(times, wers[lang], strict=True)))
        axis.plot(
            [times[i] for i in front], [wers[lang][i] for i in front],
            color=MUTED, linewidth=1, linestyle="--", zorder=1,
        )  # fmt: skip
        for name, x, y in zip(names, times, wers[lang], strict=True):
            selected = name == selection["selected"]
            axis.scatter(
                x, y, s=60, color=HIGHLIGHT if selected else MUTED,
                edgecolors=SURFACE, linewidths=2, zorder=3,
            )  # fmt: skip
        middle = (max(wers[lang]) + min(wers[lang])) / 2
        axis.set_ylim(middle - span / 2, middle + span / 2)
        _style(axis)
        axis.set_title(LANGUAGE_NAMES.get(lang, lang), loc="left", fontsize=10, color=INK)
    axes[0].xaxis.set_major_formatter(
        lambda x, _: f"{x:+.0f}".replace("-", "\u2212") if round(x) else "0"
    )
    axes[0].set_ylabel("WER (%)", color=SECONDARY, fontsize=9)
    figure.supxlabel(
        "Processing time on the three dev sets, relative to the baseline (%)",
        color=SECONDARY, fontsize=9,
    )  # fmt: skip
    figure.tight_layout()  # before the labels, whose placement depends on the final layout
    for axis, lang in zip(axes, languages, strict=True):
        _label_points(axis, _label_groups(names, times, wers[lang], selection["selected"]),
                      selection["selected"])  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, facecolor=SURFACE, metadata={"Software": None})
    plt.close(figure)
    return path


def plot_usage(report: dict[str, Any], path: Path) -> Path:
    """WER against the signal-to-noise ratio of the E10 sessions, one panel per language: one
    line per noise type (colour) and channel (solid for headset, dashed for phone), each
    starting from the quiet sessions of its channel.
    """
    plt = _import_pyplot()
    languages = list(report["by_language_condition"])
    positions = {"quiet": 0}
    figure, axes = plt.subplots(
        1, len(languages), figsize=(3.6 * len(languages), 4.2), sharey=True, facecolor=SURFACE
    )
    axes = np.atleast_1d(axes)
    snrs: set[float] = set()
    for axis, lang in zip(axes, languages, strict=True):
        for channel, rows in report["by_language_condition"][lang].items():
            by_condition = {row["segment"]: row for row in rows}
            for environment, colour in NOISE_COLOURS.items():
                points = sorted(
                    (float(name.split()[1]), row["wer"])
                    for name, row in by_condition.items()
                    if name.startswith(environment)
                )[::-1]  # from the highest SNR to the lowest
                snrs.update(snr for snr, _ in points)
                xs = [positions["quiet"]] + [_SNR_OFFSET - snr for snr, _ in points]
                ys = [by_condition["quiet"]["wer"]] + [wer for _, wer in points]
                axis.plot(
                    xs, [100 * y for y in ys], color=colour, linewidth=2, marker="o",
                    markersize=5, markeredgecolor=SURFACE, markeredgewidth=1,
                    linestyle="-" if channel == "headset" else "--",
                    label=f"{environment}, {channel}",
                )  # fmt: skip
        _style(axis)
        axis.margins(x=0.08)
        axis.set_title(LANGUAGE_NAMES.get(lang, lang), loc="left", fontsize=10, color=INK)
    ticks = [positions["quiet"], *(_SNR_OFFSET - snr for snr in sorted(snrs, reverse=True))]
    labels = ["quiet", *(f"{snr:g}" for snr in sorted(snrs, reverse=True))]
    for axis in axes:
        axis.set_xticks(ticks, labels)
    axes[0].set_ylabel("WER (%)", color=SECONDARY, fontsize=9)
    figure.supxlabel("Signal-to-noise ratio (dB)", color=SECONDARY, fontsize=9)
    handles, names = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, names, loc="upper center", ncol=len(names), frameon=False, fontsize=8,
        labelcolor=SECONDARY, bbox_to_anchor=(0.5, 1.0),
    )  # fmt: skip
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, facecolor=SURFACE, metadata={"Software": None})
    plt.close(figure)
    return path


# Noise types of E10 (categorical slots 1 and 2 of the palette) and the x position of an SNR:
# quiet at 0, then 15 dB at 5 ... 0 dB at 20, so that the noise grows to the right.
NOISE_COLOURS = {"stationary": HIGHLIGHT, "babble": "#eb6834"}
_SNR_OFFSET = 20.0


# Label positions (offset in points, alignment): single labels go up right of their
# point, the selected one down right; a group label takes the first position that
# does not overlap a label already placed.
_SINGLE = ((6, 5, "left"),)
_SELECTED = ((6, -13, "left"),)
_GROUP = ((0, 9, "center"), (-6, -14, "right"), (0, 20, "center"), (0, -16, "center"))


def _label_points(axis: Any, labels: list[tuple[str, float, float]], selected: str) -> None:
    renderer = axis.figure.canvas.get_renderer()
    placed: list[Any] = []
    # Groups last, so that they move around the single labels.
    for label, x, y in sorted(labels, key=lambda item: ", " in item[0]):
        positions = _SELECTED if label == selected else _GROUP if ", " in label else _SINGLE
        for dx, dy, align in positions:
            text = axis.annotate(
                label, (x, y), xytext=(dx, dy), textcoords="offset points", ha=align,
                fontsize=8, color=INK if label == selected else SECONDARY,
                fontweight="bold" if label == selected else None,
            )  # fmt: skip
            box = text.get_window_extent(renderer).expanded(1.05, 1.15)
            if not any(box.overlaps(other) for other in placed) or (dx, dy, align) == positions[-1]:
                placed.append(box)
                break
            text.remove()


def _label_groups(
    names: Sequence[str], xs: Sequence[float], ys: Sequence[float], selected: str
) -> list[tuple[str, float, float]]:
    """One label per group of nearly coincident points (within 3 points of time and
    0.01 points of WER), so that labels do not overlap; the selected point keeps its own."""
    groups: list[tuple[list[str], float, float]] = []
    for name, x, y in sorted(zip(names, xs, ys, strict=True), key=lambda item: item[1]):
        group = next(
            (g for g in groups if name != selected and selected not in g[0]
             and abs(g[1] - x) < 3 and abs(g[2] - y) < 0.01),
            None,
        )  # fmt: skip
        if group is None:
            groups.append(([name], x, y))
        else:
            group[0].append(name)
    return [(", ".join(members), x, y) for members, x, y in groups]


def _style(axis: Any) -> None:
    """Recessive chrome: hairline grid, no top or right spine, muted ticks."""
    axis.set_facecolor(SURFACE)
    axis.grid(color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(AXIS)
    axis.tick_params(colors=MUTED, labelsize=8, labelcolor=SECONDARY)
    axis.margins(x=0.25)  # room for the labels at the edges


def _import_pyplot() -> Any:
    try:
        import matplotlib
    except ImportError as error:
        raise RuntimeError(
            "matplotlib is not installed: run `uv sync --extra plots` to draw figures"
        ) from error
    matplotlib.use("Agg")  # files only, no window
    import matplotlib.pyplot as plt

    return plt
