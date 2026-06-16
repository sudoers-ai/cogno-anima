"""
Integration tests for the SUPEREGO stage (Stage 5) against a real Ollama model.

Scope guard + judge consume JSON → use a json-constrained backend; voice writes
free text → plain backend. Auto-skipped if Ollama is unreachable. temperature=0.0.
"""

import httpx
import pytest

from cogno_anima.llm import OllamaBackend
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.types import (
    PipelineContext, NoumenoResult, IntentResult, StageMetrics,
    EgoResult, EgoStep, ToolExecution,
)

MODEL = "mistral:latest"


async def is_ollama_available() -> bool:
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            return (await client.get("http://localhost:11434/")).status_code == 200
    except Exception:
        return False


def _json_backend():
    return OllamaBackend(model=MODEL, temperature=0.0, format="json")


def _text_backend():
    return OllamaBackend(model=MODEL, temperature=0.0)


def _m(s):
    return StageMetrics(stage=s, elapsed_ms=1.0, tokens_in=1, tokens_out=1, model="t")


def _ctx(user, intent_class="ACTION_REQUEST", goal="", tool=None, args=None, result=""):
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
            index=0, path="native", assistant_text="done",
            tool_calls=[ToolExecution(tool=tool, arguments=args or {}, result=result, ok=True)],
        )], metrics=_m("ego"))
    return ctx


SCOPE = "You are a personal finance assistant. You only help with money, expenses, income, budgets."


@pytest.mark.asyncio
async def test_scope_blocks_off_topic():
    if not await is_ollama_available():
        pytest.skip("Ollama not running")
    r = await SuperegoStage().check_input_scope(
        _ctx("Como faço um bolo de chocolate?", intent_class="INFORMATION_REQUEST"),
        _json_backend(), scope_prompt=SCOPE)
    assert r.blocked is True, f"expected BLOCK, got allow; msg={r.refusal_message!r}"
    assert r.refusal_message


@pytest.mark.asyncio
async def test_scope_allows_in_scope():
    if not await is_ollama_available():
        pytest.skip("Ollama not running")
    r = await SuperegoStage().check_input_scope(
        _ctx("Quanto gastei esse mês?", intent_class="INFORMATION_REQUEST"),
        _json_backend(), scope_prompt=SCOPE)
    assert r.blocked is False


@pytest.mark.asyncio
async def test_judge_approves_correct_execution():
    if not await is_ollama_available():
        pytest.skip("Ollama not running")
    ctx = _ctx("registra uma despesa de 50 do almoço", goal="record an expense of 50 for lunch",
               tool="record_expense", args={"amount": 50, "description": "lunch"},
               result="Recorded expense of 50 BRL")
    r = await SuperegoStage().evaluate(ctx, _json_backend(), limits_prompt="")
    assert r.approved is True, f"expected approve, got reject: {r.critique!r}"


@pytest.mark.asyncio
async def test_judge_rejects_goal_execution_mismatch():
    if not await is_ollama_available():
        pytest.skip("Ollama not running")
    # asked to record an EXPENSE, but the EGO recorded INCOME → goal↔execution miss
    ctx = _ctx("registra uma despesa de 50 do almoço", goal="record an expense of 50 for lunch",
               tool="record_income", args={"amount": 50, "description": "lunch"},
               result="Recorded income of 50 BRL")
    r = await SuperegoStage().evaluate(ctx, _json_backend(), limits_prompt="")
    assert r.approved is False, "judge should catch income-instead-of-expense"
    assert r.critique


@pytest.mark.asyncio
async def test_voice_writes_grounded_response():
    if not await is_ollama_available():
        pytest.skip("Ollama not running")
    ctx = _ctx("qual meu saldo?", intent_class="INFORMATION_REQUEST", goal="get balance",
               tool="get_balance", args={}, result="Current balance: 1000 BRL")
    r = await SuperegoStage().voice(ctx, _text_backend(), voice_prompt="You are a friendly finance assistant.")
    assert r.response and "1000" in r.response, f"expected grounded figure; got {r.response!r}"
    assert r.metrics.stage == "superego_voice"
    assert r.metrics.tokens_in > 0 and r.metrics.tokens_out > 0
