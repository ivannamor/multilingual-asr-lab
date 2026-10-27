"""The notebooks must stay valid without being executed in CI.

Running it needs a model and a dataset download, so the tests check what can
break silently: outputs committed by mistake, syntax errors, and imports of
``asr_lab`` names that have been renamed or removed.
"""

import ast
import importlib
import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted(Path("notebooks").glob("*.ipynb"))
CELLS = {path: json.loads(path.read_text(encoding="utf-8"))["cells"] for path in NOTEBOOKS}
CODE = {
    path: ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]
    for path, cells in CELLS.items()
}


def test_notebooks_are_found() -> None:
    assert Path("notebooks/quickstart.ipynb") in NOTEBOOKS


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=str)
def test_notebook_has_no_outputs(notebook: Path) -> None:
    for cell in CELLS[notebook]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


@pytest.mark.parametrize("source", [source for sources in CODE.values() for source in sources])
def test_code_cells_are_valid_python(source: str) -> None:
    ast.parse(source)


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=str)
def test_asr_lab_imports_exist(notebook: Path) -> None:
    imported = [
        (node.module, alias.name)
        for source in CODE[notebook]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("asr_lab")
        for alias in node.names
    ]
    assert imported, "the notebook should use the asr_lab package"
    for module, name in imported:
        assert hasattr(importlib.import_module(module), name), f"{module}.{name} no longer exists"
