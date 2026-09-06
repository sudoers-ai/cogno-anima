"""
Integration tests for the SUPEREGO stage (Stage 5) against a real model.

Scope guard + judge consume JSON → use a json-constrained backend; voice writes
free text → plain backend. Auto-skipped when the configured model is unreachable.
temperature=0.0.
"""

import pytest

from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.types import (
    PipelineContext, NoumenoResult, IntentResult, StageMetrics,
    EgoResult, EgoStep, ToolExecution,
)
from tests.integration import backends

# The judge is a reasoning role, and the model has to be able to do it. Measured on the
# goal↔execution case below (user asked to record an EXPENSE, the EGO recorded INCOME),
# running the real stage n=3 per model: mistral:latest APPROVED the mismatch 3/3 with an
# empty critique; qwen3:8b rejected 3/3 with the correct critique, as did gpt-4o-mini.
# The judge is fail-closed by design — never approve unverified — so on mistral it was the
# exact inverse, a false-pass machine. The prompt was never at fault.
# The model is whatever COGNO_TEST_MODEL names — see tests/integration/backends.py.


def _json_backend():
    return backends.json_backend()


def _text_backend():
    return backends.text_backend()


def _m(s):
    return StageMetrics(stage=s, elapsed_ms=1.0, tokens_in=1, tokens_out=1, model="t")


def _ctx(user, intent_class="ACTION_REQUEST", goal="", tool=None, args=None, result="",
         side_effect=False, mutating=None, draft="done"):
    """``side_effect``/``mutating`` were omitted here until the READ-ONLY judge branch existed,
    because nothing read them — and a `record_expense` that answers "Recorded expense of 50 BRL"
    was being described to the judge as a call that WROTE NOTHING. That is a defect in the
    double, not an accommodation of the new branch: the two tests below are about a turn that
    acted, and a double that says otherwise sends them down the wrong criteria."""
    noumeno = NoumenoResult(
        original=user, rewritten=user, context_turn="", language="pt",
        canonical_language="en", drift_score=0.0, drift_tag="PASS_THROUGH", changed=False,
        confidence=1.0, change_subject=False, subject_similarity=1.0, context_used=False,
        preserved_terms=[], rewrite_warnings=[], metrics=_m("noumeno"),
    )
    intent = IntentResult(
        intent_class=intent_class, sentiment="NEUTRAL", confidence=1.0,
        temporal_class="TIMELESS", triad_signal="EGO", goal=goal or user, domains=["FINANCE"],
        metrics=_m("ner"),
    )
    ctx = PipelineContext(user_input=user, noumeno=noumeno, intent=intent)
    if tool:
        ctx.ego_result = EgoResult(steps=[EgoStep(
            index=0, path="native", assistant_text=draft,
            tool_calls=[ToolExecution(tool=tool, arguments=args or {}, result=result, ok=True,
                                      side_effect=side_effect, tool_mutating=mutating)],
        )], metrics=_m("ego"))
    return ctx


SCOPE = "You are a personal finance assistant. You only help with money, expenses, income, budgets."


@pytest.mark.asyncio
async def test_scope_blocks_off_topic():
    await backends.skip_unless_available()
    r = await SuperegoStage().check_input_scope(
        _ctx("Como faço um bolo de chocolate?", intent_class="INFORMATION_REQUEST"),
        _json_backend(), scope_prompt=SCOPE)
    assert r.blocked is True, f"expected BLOCK, got allow; msg={r.refusal_message!r}"
    assert r.refusal_message


@pytest.mark.asyncio
async def test_scope_allows_in_scope():
    await backends.skip_unless_available()
    r = await SuperegoStage().check_input_scope(
        _ctx("Quanto gastei esse mês?", intent_class="INFORMATION_REQUEST"),
        _json_backend(), scope_prompt=SCOPE)
    assert r.blocked is False


@pytest.mark.asyncio
async def test_judge_approves_correct_execution():
    await backends.skip_unless_available()
    ctx = _ctx("registra uma despesa de 50 do almoço", goal="record an expense of 50 for lunch",
               tool="record_expense", args={"amount": 50, "description": "lunch"},
               result="Recorded expense of 50 BRL", side_effect=True, mutating=True)
    r = await SuperegoStage().evaluate(ctx, _json_backend(), limits_prompt="")
    assert r.approved is True, f"expected approve, got reject: {r.critique!r}"


@pytest.mark.asyncio
async def test_judge_rejects_goal_execution_mismatch():
    await backends.skip_unless_available()
    # asked to record an EXPENSE, but the EGO recorded INCOME → goal↔execution miss
    ctx = _ctx("registra uma despesa de 50 do almoço", goal="record an expense of 50 for lunch",
               tool="record_income", args={"amount": 50, "description": "lunch"},
               result="Recorded income of 50 BRL", side_effect=True, mutating=True)
    r = await SuperegoStage().evaluate(ctx, _json_backend(), limits_prompt="")
    assert r.approved is False, "judge should catch income-instead-of-expense"
    assert r.critique


@pytest.mark.asyncio
async def test_voice_writes_grounded_response():
    await backends.skip_unless_available()
    ctx = _ctx("qual meu saldo?", intent_class="INFORMATION_REQUEST", goal="get balance",
               tool="get_balance", args={}, result="Current balance: 1000 BRL")
    r = await SuperegoStage().voice(ctx, _text_backend(), voice_prompt="You are a friendly finance assistant.")
    assert r.response and "1000" in r.response, f"expected grounded figure; got {r.response!r}"
    assert r.metrics.stage == "superego_voice"
    assert r.metrics.tokens_in > 0 and r.metrics.tokens_out > 0


# ── the READ-ONLY judge branch (model half) ──────────────────────────────────────────
# The deterministic half — which branch is chosen and which criteria render — is pinned in
# `tests/unit/test_judge_readonly_branch.py` and needs no model. These two are the half only a
# model can answer, and they are a PAIR on purpose: the first says the branch stops rejecting a
# correct read, the second says it did not become a free pass. Shipping only the first would be
# shipping a rubber stamp.


@pytest.mark.asyncio
async def test_judge_approves_a_truthful_empty_read():
    """The measured defect: the read succeeded, returned nothing, and the draft said so.

    Under the execution criteria this is judged as "the goal was not met" — 730 production
    traces put a turn that only READ at 65.9% rejected against 14.1% for one that WROTE, and
    one critique conceded "the execution CORRECTLY stated ... but it did not fully meet".
    """
    ctx = _ctx("e as aulas do mês que vem?", intent_class="INFORMATION_REQUEST",
               goal="list the classes scheduled for next month",
               tool="get_schedule", args={"month": "2026-10"}, result="No classes found.",
               draft="I found no classes scheduled for next month (October 2026).")
    await backends.skip_unless_available()
    r = await SuperegoStage().evaluate(ctx, _json_backend(), limits_prompt="")
    assert r.approved is True, f"expected approve, got reject: {r.critique!r}"


@pytest.mark.asyncio
async def test_judge_still_rejects_a_read_whose_draft_invents():
    """The twin that keeps the branch honest — and the one that must die if it goes lax.

    Same clean read, same empty result, but the draft fills the emptiness with classes nobody
    returned. GROUNDING is criterion #1 in this branch precisely so this stays a rejection:
    an empty read grounds a NEGATIVE answer and nothing else.
    """
    ctx = _ctx("e as aulas do mês que vem?", intent_class="INFORMATION_REQUEST",
               goal="list the classes scheduled for next month",
               tool="get_schedule", args={"month": "2026-10"}, result="No classes found.",
               draft="You have Algebra on 5 October at 09:00 and Physics on 12 October at 14:00.")
    await backends.skip_unless_available()
    r = await SuperegoStage().evaluate(ctx, _json_backend(), limits_prompt="")
    assert r.approved is False, "an empty read grounds a negative answer, never a listing"
    assert r.critique
