"""The version is written ONCE, and everything that states one states that one.

``cogno_anima.__version__`` used to be ``importlib.metadata.version("cogno-anima")`` — the
version of whatever distribution is INSTALLED, not of the code that is running. Imported from a
source tree (an editable install older than a bump, a checkout on ``PYTHONPATH``) the two part
ways and the code reports the old number. Measured on a development box, on the tree before
this change: ``pyproject.toml`` declared 0.1.5 and ``cogno_anima.__version__`` answered 0.1.0.

The number now lives in ``cogno_anima/_version.py`` and nowhere else:

* ``cogno_anima.__version__`` is that constant;
* ``pyproject.toml`` declares no number — ``dynamic = ["version"]``, read from the same line;
* the built wheel carries it (asserted in ``test_py_typed_ships.py``, which already builds one);
* ``CHANGELOG.md`` opens with it — the one other place a number is typed by hand.

Each test below fails when one of them says something else.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import cogno_anima
from cogno_anima import _version

ROOT = Path(__file__).resolve().parent.parent.parent
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
ATTR = "cogno_anima._version.__version__"


def _section(name: str) -> str:
    """The body of one ``[name]`` table. Regex, not tomllib: the floor is 3.10, where it does
    not exist (the choice ``test_packaging.py`` already made)."""
    parts = PYPROJECT.split(f"[{name}]\n", 1)
    assert len(parts) == 2, f"[{name}] is not in pyproject.toml"
    return parts[1].split("\n[", 1)[0]


def read_statically() -> str:
    """The version as the BUILD reads it: the one string literal assigned to ``__version__`` in
    ``_version.py``, taken off the AST without importing anything."""
    tree = ast.parse((ROOT / "cogno_anima" / "_version.py").read_text(encoding="utf-8"))
    found = [node.value.value for node in tree.body
             if isinstance(node, ast.Assign) and len(node.targets) == 1
             and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "__version__"
             and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)]
    assert len(found) == 1, (
        "`_version.py` must assign ONE plain string literal to `__version__`: setuptools reads "
        f"it from the source without importing the package. Found: {found}")
    return found[0]


def test_the_package_reports_the_line_the_build_reads():
    assert cogno_anima.__version__ == _version.__version__ == read_statically()
    assert re.fullmatch(r"\d+\.\d+\.\d+", cogno_anima.__version__), cogno_anima.__version__


def test_pyproject_declares_no_number_of_its_own():
    """A static `version = "…"` beside the dynamic one would be a second source — and
    setuptools refuses the pair, so the build would say so too. This says it first."""
    project = _section("project")
    assert not re.search(r"(?m)^version\s*=", project), "a static version is back in [project]"
    dynamic = re.search(r"(?m)^dynamic\s*=\s*\[([^\]]*)\]", project)
    assert dynamic and '"version"' in dynamic.group(1)
    source = re.search(r'(?m)^version\s*=\s*\{\s*attr\s*=\s*"([^"]+)"\s*\}',
                       _section("tool.setuptools.dynamic"))
    assert source and source.group(1) == ATTR, (
        f"the build must read the version from {ATTR}, the line `cogno_anima.__version__` is")


def test_the_package_does_not_ask_the_installed_distribution():
    """The measured defect, as a shape: `importlib.metadata` answers for the INSTALL. Nothing
    in `cogno_anima/__init__.py` may read a version from it again."""
    tree = ast.parse((ROOT / "cogno_anima" / "__init__.py").read_text(encoding="utf-8"))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names}
    imported |= {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not {m for m in imported if m.startswith("importlib.metadata")}, imported
    assigned = [node for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)]
    assert not assigned, "`__init__.py` assigns `__version__` itself — a second source"


def test_the_changelog_opens_with_this_version():
    """The other place a number is typed by hand. The FIRST version heading is the release in
    the tree; a bump without an entry, or an entry without a bump, fails here."""
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = re.findall(r"(?m)^## (\d+\.\d+\.\d+)\b", changelog)
    assert headings, "no version heading in CHANGELOG.md"
    assert headings[0] == cogno_anima.__version__, (
        f"CHANGELOG.md opens with {headings[0]} and the package is {cogno_anima.__version__}")


def test_the_checks_above_can_fail():
    """The readers produce the failure they look for, on text built to have it."""
    assert re.search(r"(?m)^version\s*=", 'name = "x"\nversion = "9.9.9"\n')
    assert not re.search(r"(?m)^version\s*=", 'name = "x"\ndynamic = ["version"]\n')
    computed = ast.parse('from importlib.metadata import version\n__version__ = version("x")')
    assert not [n for n in computed.body if isinstance(n, ast.Assign)
                and isinstance(n.value, ast.Constant)]
    assert re.findall(r"(?m)^## (\d+\.\d+\.\d+)\b", "## Unreleased\n\n## 0.0.1 — x\n") == ["0.0.1"]
