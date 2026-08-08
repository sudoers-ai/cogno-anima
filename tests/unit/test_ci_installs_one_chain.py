"""Every CI job must test against the SAME siblings.

``cogno-anima`` depends on ``cogno-synapse`` -> ``cogno-homeo``. Both ARE on PyPI (0.1.2 and
0.1.1) and git main is ahead of both under those same version numbers — so ``pip install -e .``
alone does not fail, it quietly installs the old release, and pip cannot tell you because the
version string is what it always was.

The ``test`` job installs the chain from git explicitly. The nightly ``ollama-integration`` job
did not. ``cogno-soma`` had the identical gap and spent four nights red on
``module 'cogno_anima.metakeys' has no attribute 'JUDGE_VERDICT'`` — one job testing main, the
other a release from before the key existed (soma #27). Here it still passed, which is worse:
nothing marks the day the two stop agreeing. And the integration job only runs on ``schedule``,
so a PR is green either way.

This asserts the invariant instead of the wording: any job that installs this package must also
install every sibling from git. It reads the SETS, so reordering or reformatting the workflow is
free and only a real divergence fails.
"""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"

_JOB = re.compile(r"^  ([a-z0-9-]+):\s*$")
_GIT_DEP = re.compile(r"pip install \"(cogno-[a-z]+) @ git\+")
_SELF = re.compile(r"pip install -e \"?\.")


def _jobs() -> "dict[str, str]":
    """Job name → its raw block. Crude on purpose: a YAML parser would need a dependency, and
    the two-space job indent is stable in this file."""
    text = WORKFLOW.read_text(encoding="utf-8")
    names = [(m.group(1), m.start()) for m in _JOB.finditer(text, re.MULTILINE)
             ] or [(m.group(1), m.start()) for m in
                   (m for m in re.finditer(r"(?m)^  ([a-z0-9-]+):\s*$", text))]
    out = {}
    for i, (name, start) in enumerate(names):
        end = names[i + 1][1] if i + 1 < len(names) else len(text)
        out[name] = text[start:end]
    return out


def test_the_workflow_is_where_we_think_it_is():
    """Guards the guard: a moved or renamed workflow would make every check below pass over
    an empty string."""
    assert WORKFLOW.is_file()
    jobs = _jobs()
    assert "test" in jobs and "ollama-integration" in jobs, sorted(jobs)


def test_every_job_that_installs_soma_installs_the_same_git_chain():
    jobs = {n: b for n, b in _jobs().items() if _SELF.search(b)}
    assert jobs, "no job installs this package — the check would be vacuous"
    chains = {name: frozenset(_GIT_DEP.findall(block)) for name, block in jobs.items()}
    reference = max(chains.values(), key=len)
    assert reference, "no job installs the siblings from git — all of them resolve PyPI"
    drift = {name: sorted(reference - deps) for name, deps in chains.items()
             if deps != reference}
    assert not drift, (
        "these jobs install this package WITHOUT the git siblings, so pip resolves the stale "
        f"PyPI wheels instead — same version number, older code: {drift}")
