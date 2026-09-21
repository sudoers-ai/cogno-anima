"""A bare ``pytest`` must never collect ``tests/integration``.

The integration suites default to the LOCAL Ollama (``tests/integration/backends.py``), and
their skip fires only when that server does not answer. On a box whose Ollama serves live
traffic it does answer, so the most obvious command in this repo — ``python3 -m pytest`` with
no path — used to queue every real-model test in ``tests/integration`` ahead of real turns.
``testpaths`` in ``pyproject.toml`` now scopes a bare run to ``tests/unit``; integration is
opt-in BY PATH (``pytest tests/integration``), which is exactly how CI already invokes it
(``test_ci_runs_every_test_file.py`` holds the CI side of that).

Three collections, every one of them ``--collect-only``: nothing in this module RUNS a test,
so nothing in it can reach a model. Each assertion is about node ids, not a bare count:

* the TWIN — a bare collection holds no id under ``tests/integration``;
* its CONTROL — ``pytest tests/integration`` does collect ids there, more than zero and from
  every test module on disk. Without it an empty collection would satisfy the twin for free,
  the ``'x' in []`` shape;
* a second CONTROL — the bare collection is still the WHOLE unit suite, id for id, so the fix
  cannot pass by quietly collecting less than ``pytest tests/unit`` does.
"""

from __future__ import annotations

import functools
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]
INTEGRATION = "tests/integration"
UNIT = "tests/unit"

_SUMMARY = re.compile(r"^(\d+)(?:/\d+)? tests? collected\b")


class Collection(NamedTuple):
    returncode: int
    ids: "tuple[str, ...]"
    reported: int          # the "N tests collected" figure pytest printed (0 when none)
    output: str


@functools.lru_cache(maxsize=None)
def _collect(*paths: str) -> Collection:
    """``pytest --collect-only -q [paths]``, run from the repo root in a subprocess.

    ``--collect-only`` is the whole safety argument of this module: it imports the test
    modules and runs no test. ``PYTEST_ADDOPTS`` is dropped so a stray value in someone's
    shell cannot hand the child a path this module did not ask for.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "no:cacheprovider", "-p", "no:importnb", *paths],
        cwd=REPO_ROOT, capture_output=True, text=True, env=env, timeout=600,
    )
    lines = proc.stdout.splitlines()
    ids = tuple(line for line in lines if "::" in line and not line[:1].isspace())
    reported = next((int(m.group(1)) for line in lines if (m := _SUMMARY.match(line))), 0)
    return Collection(proc.returncode, ids, reported, proc.stdout[-3000:] + proc.stderr[-3000:])


def _files(ids: "tuple[str, ...]") -> "set[str]":
    return {node.split("::", 1)[0] for node in ids}


def _modules_on_disk(directory: str) -> "set[str]":
    return {p.relative_to(REPO_ROOT).as_posix()
            for p in (REPO_ROOT / directory).rglob("test_*.py")}


# ── the twin ───────────────────────────────────────────────────────────────────────────
def test_a_bare_pytest_collects_not_one_integration_test():
    bare = _collect()
    assert bare.returncode == 0, bare.output
    assert bare.ids, f"a bare collection returned no ids at all:\n{bare.output}"
    assert len(set(bare.ids)) == bare.reported, "the id parser disagrees with pytest's count"
    leaked = sorted(i for i in bare.ids if i.startswith(INTEGRATION + "/"))
    assert not leaked, (
        f"a bare `pytest` collects {len(leaked)} integration tests, which default to the "
        f"LOCAL model server. Scope `testpaths` to {UNIT!r}; integration is asked for by "
        f"path.\n  first: {leaked[:5]}")
    assert not set(bare.ids) & set(_collect(INTEGRATION).ids)


# ── control: the integration suite is still there, and reachable by path ───────────────
def test_the_integration_suite_is_collected_when_asked_for_by_path():
    integ = _collect(INTEGRATION)
    assert integ.ids, (
        f"`pytest {INTEGRATION}` collected nothing — the twin above would pass over an empty "
        f"set:\n{integ.output}")
    assert integ.returncode == 0, integ.output
    assert len(set(integ.ids)) == integ.reported, "the id parser disagrees with pytest's count"
    assert all(i.startswith(INTEGRATION + "/") for i in integ.ids)
    assert _files(integ.ids) == _modules_on_disk(INTEGRATION)


# ── control: a bare run is still the whole unit suite ──────────────────────────────────
def test_a_bare_pytest_still_collects_the_whole_unit_suite():
    bare, unit = _collect(), _collect(UNIT)
    assert unit.returncode == 0, unit.output
    assert unit.ids, unit.output
    assert unit.reported == bare.reported == len(bare.ids) == len(unit.ids)
    assert set(bare.ids) == set(unit.ids)
