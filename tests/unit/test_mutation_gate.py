"""Mutation gate, stub-only half (plan 0.8, M1/M3).

A case whose checks stay green while its stage is SABOTAGED does not observe
that stage — it is dead weight wearing a guard's uniform. These tests run the
harness with one stage broken on purpose and assert the target dimension's
score DROPS versus the clean stub baseline (and, for M3 garbage, that the
failure surfaces as scored model-fault checks — never a silent pass).

The model-dependent half of the gate (M2 swapped labels, M4 weak model) needs a
real model and lives in the sweep protocol, not in CI.
"""

import asyncio

import pytest

from cognobench.runner import run_bench


def _run(mutate: str | None = None, only: list[str] | None = None):
    return asyncio.run(run_bench(
        model="stub", embed_model="stub", base_url="http://localhost:11434",
        only=only or [], stub=True, limit=None, calibrate=False,
        mutate=mutate,
    ))


def _correct(report, dim_name: str) -> tuple[int, int]:
    d = next(x for x in report.dimensions if x.name == dim_name)
    return d.correct_count, d.total


@pytest.mark.parametrize("mutate,dim", [
    ("noumeno_echo", "noumeno"),
    ("ner_unknown", "ner"),
    ("voice_empty", "superego_voice"),
])
def test_sabotage_drops_the_target_dimension(mutate, dim):
    only = [dim.split("_")[0] if dim.startswith("superego") else dim]
    baseline = _correct(_run(only=only), dim)
    sabotaged = _correct(_run(mutate=mutate, only=only), dim)
    assert sabotaged[1] > 0, f"{dim}: sabotage produced no checks"
    assert sabotaged[0] < baseline[0], (
        f"{dim}: score did not drop under {mutate} "
        f"({baseline[0]}/{baseline[1]} → {sabotaged[0]}/{sabotaged[1]}) — the "
        f"dimension does not observe its own stage"
    )


def test_judge_approve_flips_the_must_reject_cases():
    """The judge sabotage approves everything with an empty critique — the
    exact mistral failure mode (3/3 false approvals). The clean-stub baseline
    already rejects everything (fail-closed default on unparseable judge JSON),
    so the AGGREGATE need not drop — the assertion is directional: every
    must-REJECT case flips red under always-approve."""
    report = _run(mutate="judge_approve", only=["superego"])
    judge = next(d for d in report.dimensions if d.name == "superego_judge")
    must_reject = [c for c in judge.checks
                   if c.field == "judge(soft)" and c.expected == "False"]
    assert must_reject, "expected must-reject judge cases"
    assert all(not c.correct for c in must_reject), (
        "an always-approve judge still passed a must-reject case — that check "
        "does not observe the judge"
    )


def test_scope_allow_flips_the_must_block_cases():
    """The scope sabotage waves everything through; every expect_blocked=True
    case must flip. (The clean stub may fail some scope cases already, so the
    assertion is on the must-block subset, not the aggregate.)"""
    report = _run(mutate="scope_allow", only=["superego"])
    scope = next(d for d in report.dimensions if d.name == "superego_scope")
    must_block = [c for c in scope.checks
                  if c.field == "scope(soft)" and c.expected == "True"]
    assert must_block, "expected must-block scope cases"
    assert all(not c.correct for c in must_block), (
        "an always-ALLOW scope guard still passed a must-block case — that "
        "check does not observe the guard"
    )


def test_garbage_surfaces_as_scored_model_fault_never_silent():
    """M3: non-JSON output must surface as failed checks in the full
    denominator (StageParseError = the model's fault) — the run stays VALID and
    nothing silently passes."""
    report = _run(mutate="garbage", only=["ner"])
    ner = next(d for d in report.dimensions if d.name == "ner")
    assert ner.valid, "garbage output must not invalidate the run (it is a result)"
    errs = [c for c in ner.checks if c.field == "case_error"]
    assert errs and all(not c.correct for c in errs)
    assert ner.correct_count < ner.total


def test_safety_deterministic_checks_survive_ner_sabotage():
    """The asymmetry the mutation report exists to show: the deterministic PII
    checks are decided by the regex detector on the ORIGINAL text, so an
    all-UNKNOWN NER must NOT move them."""
    report = _run(mutate="ner_unknown", only=["safety"])
    saf = next(d for d in report.dimensions if d.name == "safety")
    hard = [c for c in saf.checks if c.field.endswith("_deterministic")]
    assert hard, "expected deterministic safety checks"
    assert all(c.correct for c in hard)
