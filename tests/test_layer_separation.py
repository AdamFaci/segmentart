# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet

"""The business layer must not depend on the user interface.

No module under `segmentart/backend/`, and not `segmentart/config.py`, may import
Streamlit or touch `st.session_state`. Another frontend has to be able to reuse the
backend, so this rule is enforced rather than merely documented.
"""

import ast
import pathlib

import pytest

BACKEND = pathlib.Path(__file__).resolve().parent.parent / "segmentart" / "backend"
CONFIG = pathlib.Path(__file__).resolve().parent.parent / "segmentart" / "config.py"

UI_MODULES = {"streamlit", "streamlit_drawable_canvas"}

MODULES = sorted(BACKEND.glob("*.py")) + [CONFIG]


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_streamlit_import(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    offending = imported & UI_MODULES
    assert not offending, f"{path.name} imports the interface layer: {sorted(offending)}"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_session_state(path):
    assert "session_state" not in path.read_text(encoding="utf-8"), (
        f"{path.name} reads Streamlit session state; pass the data in explicitly"
    )


def test_backend_imports_without_optional_dependencies():
    """Every backend module must import with only NumPy and Pillow present."""
    import importlib

    for path in sorted(BACKEND.glob("*.py")):
        if path.name == "__init__.py":
            continue
        importlib.import_module(f"segmentart.backend.{path.stem}")
