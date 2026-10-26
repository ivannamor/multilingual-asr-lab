"""The quickstart notebook must stay valid without being executed in CI.

Running it needs a model and a dataset download, so the tests check what can
break silently: outputs committed by mistake, syntax errors, and imports of
``asr_lab`` names that have been renamed or removed.
"""

import ast
import importlib
import json
from pathlib import Path

import pytest

NOTEBOOK = Path("notebooks/quickstart.ipynb")
CELLS = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
CODE = ["".join(cell["source"]) for cell in CELLS if cell["cell_type"] == "code"]


def test_notebook_has_no_outputs() -> None:
    for cell in CELLS:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


@pytest.mark.parametrize("source", CODE)
def test_code_cells_are_valid_python(source: str) -> None:
    ast.parse(source)


def test_asr_lab_imports_exist() -> None:
    imported = [
        (node.module, alias.name)
        for source in CODE
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("asr_lab")
        for alias in node.names
    ]
    assert imported, "the notebook should use the asr_lab package"
    for module, name in imported:
        assert hasattr(importlib.import_module(module), name), f"{module}.{name} no longer exists"
