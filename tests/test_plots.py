"""Tests for the figures. Drawing needs the optional ``plots`` extra (matplotlib)."""

import json
import sys
from pathlib import Path

import pytest

from asr_lab import cli
from asr_lab.plots import _label_groups, pareto_front, plot_selection

SELECTION = {
    "languages": ["it", "de"],
    "selected": "beam1",
    "configs": {
        "baseline": {"selectable": True, "wer": {"it": 0.024, "de": 0.052},
                     "processing_seconds": 500.0},
        "beam1": {"selectable": True, "wer": {"it": 0.024, "de": 0.054},
                  "processing_seconds": 450.0},
        "no-fallback": {"selectable": True, "wer": {"it": 0.024, "de": 0.052},
                        "processing_seconds": 505.0},
        "repeat": {"selectable": False, "wer": {"it": 0.024, "de": 0.052},
                   "processing_seconds": 502.0},
    },
}  # fmt: skip


@pytest.mark.parametrize(
    ("points", "front"),
    [
        ([], []),
        ([(1, 1)], [0]),
        ([(2, 1), (1, 2), (3, 3)], [1, 0]),  # sorted by the first coordinate
        ([(1, 1), (1, 1)], [0, 1]),  # identical points do not dominate each other
        ([(1, 2), (1, 1)], [1]),  # same time, lower WER wins
    ],
)
def test_pareto_front(points: list[tuple[float, float]], front: list[int]) -> None:
    assert pareto_front(points) == front


def test_coincident_points_share_a_label_except_the_selected_one() -> None:
    labels = _label_groups(
        ["baseline", "no-fallback", "hst2", "beam8"], [0.0, 0.7, 2.3, 13.0],
        [2.37, 2.37, 2.37, 2.38], selected="hst2",
    )  # fmt: skip
    assert [label for label, _, _ in labels] == ["baseline, no-fallback", "hst2", "beam8"]


def test_plot_selection_writes_a_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    path = plot_selection(SELECTION, tmp_path / "figures" / "pareto.png")
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_cli_plot_selection(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("matplotlib")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(SELECTION), encoding="utf-8")
    output = tmp_path / "pareto.png"
    cli.main(["plot-selection", "--selection", str(selection), "--output", str(output)])
    assert output.exists()
    assert f"Figure: {output}" in capsys.readouterr().out


def test_clear_error_when_matplotlib_is_not_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)  # makes `import matplotlib` fail
    with pytest.raises(RuntimeError, match="uv sync --extra plots"):
        plot_selection(SELECTION, tmp_path / "pareto.png")
