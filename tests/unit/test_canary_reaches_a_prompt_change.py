"""The canary must be reachable from the event that introduces the defects it catches.

## What was measured

The `ollama-integration` job ran on `schedule` and `workflow_dispatch` only. Between
2026-09-09 and 2026-09-17 it failed on EIGHT consecutive nightlies, always the same three
tests, while every push and every pull request in that window went green — including the PRs
that caused two of the three. The worst of them was the SUPEREGO judge approving a draft that
invented a class listing over an empty read: the fail-CLOSED guarantee, gone, merged by a PR
whose own CI could not run the test that says so.

So the trigger is the fix, and it is the kind of fix that fails SILENTLY: a `needs.` reference
that does not resolve is not an error, it is simply never true, and the canary quietly goes
back to nightly-only. Nobody would see it until the next eight-day-old red. These assertions
are cheap and they read the workflow, not a memory of it — the same reason
`test_ci_runs_every_test_file.py` and `test_ci_installs_one_chain.py` exist.

It stays OFF for a PR that touches neither the stage logic nor the prompt text, which is most
of them: this is a real model on a CPU runner, and the nightly comment's original reasoning
("a merge gate here would make main is always green depend on sampling") is why the scope is
narrow rather than absent.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"

#: The directories whose content the canary is the only automated reader of. A prompt is code
#: here — two of the three defects were prose inside `stages/`.
WATCHED = ("cogno_anima/stages/", "cogno_anima/prompt_templates/")


@pytest.fixture(scope="module")
def ci() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def _canary(ci: dict) -> dict:
    return ci["jobs"]["ollama-integration"]


def test_the_canary_still_runs_on_schedule_and_dispatch(ci):
    """The nightly is the baseline and nothing here replaces it."""
    condition = " ".join(_canary(ci)["if"].split())
    assert "github.event_name == 'schedule'" in condition
    assert "github.event_name == 'workflow_dispatch'" in condition


def test_the_canary_is_reachable_from_a_pull_request(ci):
    """The eight invisible reds. Without this the job cannot run on the event that breaks it."""
    condition = " ".join(_canary(ci)["if"].split())
    assert "github.event_name == 'pull_request'" in condition, condition


def test_the_pull_request_arm_is_gated_on_the_watched_paths(ci):
    """Scope, not absence: the job costs a real model run, so it is spent on the PRs that can
    break it. Both directories are named in the filter that computes the gate."""
    scope = ci["jobs"]["canary_scope"]
    script = "\n".join(s.get("run", "") for s in scope["steps"] if isinstance(s, dict))
    for path in WATCHED:
        # the filter is a regex over `git diff --name-only`, so the path appears with its
        # separator escaped or not — assert on the directory, not on the regex spelling
        assert path.rstrip("/").replace("cogno_anima/", "") in script, (path, script)
    assert "cogno_anima" in script


def test_the_scope_job_only_runs_on_a_pull_request(ci):
    """On a schedule it is skipped, which is why the canary's own `if` carries `!cancelled()`
    — a skipped dependency would otherwise take the nightly down with it."""
    assert ci["jobs"]["canary_scope"]["if"] == "github.event_name == 'pull_request'"
    assert "!cancelled()" in _canary(ci)["if"]


def test_every_needs_reference_names_a_job_that_exists(ci):
    """THE SILENT FAILURE, pinned mechanically.

    `needs.canary-scope.outputs.hit` and `needs.canary_scope.outputs.hit` differ by one
    character, both parse, and the wrong one does not raise — it evaluates to nothing, the
    pull_request arm is never true, and the canary is back to nightly-only with a green check
    beside it. This reads the identifiers out of every `if` and insists each one is a real job,
    so the mismatch is a red unit test instead of a silence.
    """
    jobs = set(ci["jobs"])
    for name, job in ci["jobs"].items():
        for referenced in re.findall(r"needs\.([A-Za-z0-9_-]+)\.", str(job.get("if", ""))):
            assert referenced in jobs, (
                f"job {name!r} tests `needs.{referenced}.…` and there is no such job "
                f"(jobs: {sorted(jobs)}) — this does not error, it is merely never true")
            assert referenced in (job.get("needs") or []), (
                f"job {name!r} reads `needs.{referenced}` without declaring it in `needs:`")


# ── the per-test cap on the shard that overran ───────────────────────────────────────

def test_the_semantics_shard_has_a_per_test_timeout(ci):
    """1h49m03s MEASURED (run 35207468672) against the ~30 min the shard was split for."""
    shards = {s["shard"]: s for s in _canary(ci)["strategy"]["matrix"]["include"]}
    assert int(shards["semantics"]["timeout"]) > 0
    # Above the transport ceiling on purpose: at most two sequential generations per test,
    # each capped by COGNO_OLLAMA_TIMEOUT, so this catches a HANG and not a slow model. A cap
    # below that would start failing tests that work — the instrument defect this PR also fixes.
    step = [s for s in _canary(ci)["steps"] if "pytest" in str(s.get("run", ""))][0]
    ceiling = int(step["env"]["COGNO_OLLAMA_TIMEOUT"])
    assert int(shards["semantics"]["timeout"]) > 2 * ceiling


def test_the_timeout_reaches_pytest_without_touching_the_pytest_command(ci):
    """`PYTEST_TIMEOUT` is read natively by pytest-timeout. The command line stays exactly as
    `test_ci_runs_every_test_file.py` lexes it — that guard proves the path handed to pytest is
    a DIRECTORY, and a new flag there is one more thing for it to have to understand."""
    step = [s for s in _canary(ci)["steps"] if "pytest" in str(s.get("run", ""))][0]
    assert step["env"]["PYTEST_TIMEOUT"] == "${{ matrix.timeout }}"
    assert "--timeout" not in step["run"]


def test_every_shard_declares_a_timeout(ci):
    """`0` is pytest-timeout's "no timeout", and it is written out rather than omitted: an
    absent matrix key renders as an empty string, which is not a documented value."""
    for shard in _canary(ci)["strategy"]["matrix"]["include"]:
        assert "timeout" in shard, shard["shard"]
        assert int(shard["timeout"]) >= 0


def test_pytest_timeout_is_installed_where_the_canary_installs_from():
    """The env var is inert without the plugin, and an inert cap is worse than none — it reads
    as a bound that is not there."""
    pyproject = (WORKFLOW.parents[2] / "pyproject.toml").read_text()
    assert "pytest-timeout" in pyproject
