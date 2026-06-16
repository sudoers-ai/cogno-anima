"""Unit tests for SuperegoStage (Stage 5) — guard, judge, voicer."""

import pytest

from cogno_core.stages.superego import SuperegoStage
from cogno_core.stages.drift import DriftCalculator
from cogno_core.types import (
    StageMetrics, NoumenoResult, IntentResult, PipelineContext,
    EgoResult, EgoStep, ToolExecution, SuperegoResult,
)


# ── test doubles ─────────────────────────────────────────────────────

class ScriptedBackend:
    def __init__(self, responses, model="stub-se", ti=5, to=3):
        self.responses = list(responses)
        self.model = model
        self.ti = ti
        self.to = to
        self.calls = []

    async def generate(self, system, prompt):
        self.calls.append({"system": system, "prompt": prompt})
        r = self.responses.pop(0) if self.responses else ""
        return r, self.ti, self.to


class RaisingBackend:
    model = "boom"

    async def generate(self, system, prompt):
        raise ConnectionError("backend down")


def _m(stage="x"):
    return StageMetrics(stage=stage, elapsed_ms=0.0, tokens_in=0, tokens_out=0, model="t")


def _ctx(user="record 50", intent_class="ACTION_REQUEST", sentiment="NEUTRAL",
         goal="record expense", with_ego=True, pii_risk="NONE", emotional=None):
    noumeno = NoumenoResult(
        original=user, rewritten=user, context_turn="", language="pt",
        canonical_language="en", drift_score=0.0, drift_tag="PASS_THROUGH", changed=False,
        confidence=1.0, change_subject=False, subject_similarity=1.0, context_used=False,
        preserved_terms=[], rewrite_warnings=[], metrics=_m("noumeno"),
    )
    intent = IntentResult(
        intent_class=intent_class, sentiment=sentiment, confidence=1.0,
        temporal_class="TIMELESS", triad_signal="EGO", goal=goal, domains=["FINANCE"],
        pii_risk=pii_risk, metrics=_m("ner"),
    )
    ctx = PipelineContext(user_input=user, noumeno=noumeno, intent=intent)
    if with_ego:
        ctx.ego_result = EgoResult(steps=[EgoStep(
            index=0, path="native", assistant_text="recorded",
            tool_calls=[ToolExecution(tool="record_expense", arguments={"amount": 50},
                                      result="Recorded 50", ok=True, side_effect=True)],
        )], metrics=_m("ego"))
    if emotional and ctx.id_result is None:
        from cogno_core.types import IdResult
        ctx.id_result = IdResult(triad_route="SUPEREGO", emotional_override=emotional, metrics=_m("id"))
    return ctx


# ── strip_cot ────────────────────────────────────────────────────────

def test_strip_cot_variants():
    assert SuperegoStage.strip_cot("<think>x</think>Hi") == ("Hi", True)
    assert SuperegoStage.strip_cot("<thinking>y</thinking> Yo ") == ("Yo", True)
    assert SuperegoStage.strip_cot("plain") == ("plain", False)
    assert SuperegoStage.strip_cot("") == ("", False)


def test_detect_adjustments():
    adj = SuperegoStage.detect_adjustments(_ctx(sentiment="FRUSTRATED"))
    assert "tone:empathetic" in adj
    adj2 = SuperegoStage.detect_adjustments(_ctx(intent_class="SOCIAL", sentiment="PLAYFUL"))
    assert "style:warm" in adj2 and "tone:playful" in adj2
    adj3 = SuperegoStage.detect_adjustments(_ctx(pii_risk="HIGH"))
    assert any(a.startswith("pii:risk_") for a in adj3)


# ── scope guard ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scope_no_rules_allows_without_llm():
    b = ScriptedBackend([])
    r = await SuperegoStage().check_input_scope(_ctx(), b, scope_prompt="")
    assert r.blocked is False and b.calls == []   # no LLM call


@pytest.mark.asyncio
async def test_scope_ner_bypass_for_social():
    b = ScriptedBackend([])
    r = await SuperegoStage().check_input_scope(
        _ctx(intent_class="SOCIAL"), b, scope_prompt="finance only")
    assert r.blocked is False and b.calls == []   # bypassed, no LLM call


@pytest.mark.asyncio
async def test_scope_blocks_off_topic():
    b = ScriptedBackend(['{"blocked": true, "refusal_message": "Sou financeiro, não ajudo com bolo."}'])
    r = await SuperegoStage().check_input_scope(
        _ctx(user="como faço bolo?", intent_class="INFORMATION_REQUEST"),
        b, scope_prompt="finance only")
    assert r.blocked is True and "bolo" in r.refusal_message
    assert r.metrics.stage == "superego_scope" and r.metrics.tokens_in == 5


@pytest.mark.asyncio
async def test_scope_allows_in_scope():
    b = ScriptedBackend(['{"blocked": false, "refusal_message": ""}'])
    r = await SuperegoStage().check_input_scope(
        _ctx(user="quanto custa o plano?", intent_class="INFORMATION_REQUEST"),
        b, scope_prompt="finance")
    assert r.blocked is False


@pytest.mark.asyncio
async def test_scope_fails_open_on_error():
    r = await SuperegoStage().check_input_scope(
        _ctx(intent_class="INFORMATION_REQUEST"), RaisingBackend(), scope_prompt="finance")
    assert r.blocked is False   # fail-open: never refuse on error


@pytest.mark.asyncio
async def test_scope_fails_open_on_garbage():
    b = ScriptedBackend(["not json at all"])
    r = await SuperegoStage().check_input_scope(
        _ctx(intent_class="INFORMATION_REQUEST"), b, scope_prompt="finance")
    assert r.blocked is False


# ── judge (evaluate) ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_judge_no_ego_approves():
    r = await SuperegoStage().evaluate(_ctx(with_ego=False), ScriptedBackend([]), limits_prompt="")
    assert r.approved is True and r.critique is None


@pytest.mark.asyncio
async def test_judge_approves():
    b = ScriptedBackend(['{"approved": true, "critique": ""}'])
    r = await SuperegoStage().evaluate(_ctx(), b, limits_prompt="must confirm before write")
    assert r.approved is True and r.critique is None
    assert r.metrics.stage == "superego_judge"


@pytest.mark.asyncio
async def test_judge_rejects_with_critique():
    b = ScriptedBackend(['{"approved": false, "critique": "recorded income instead of expense"}'])
    r = await SuperegoStage().evaluate(_ctx(), b, limits_prompt="")
    assert r.approved is False
    assert "income instead of expense" in r.critique   # goal↔execution catch


@pytest.mark.asyncio
async def test_judge_fails_closed_on_error():
    r = await SuperegoStage().evaluate(_ctx(), RaisingBackend(), limits_prompt="")
    assert r.approved is False and r.critique   # fail-closed: don't pass unverified


@pytest.mark.asyncio
async def test_judge_prompt_includes_goal_and_execution():
    b = ScriptedBackend(['{"approved": true}'])
    await SuperegoStage().evaluate(_ctx(goal="record an expense of 50"), b, limits_prompt="LIM")
    p = b.calls[0]["prompt"]
    assert "record an expense of 50" in p          # goal
    assert "record_expense" in p                   # execution
    assert "LIM" in p                              # limits


# ── voice ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_voice_writes_and_strips_cot():
    b = ScriptedBackend(["<think>plan</think>Prontinho, registrei R$50 de almoço ✅"])
    r = await SuperegoStage().voice(_ctx(), b, voice_prompt="warm assistant")
    assert r.response == "Prontinho, registrei R$50 de almoço ✅"
    assert r.cot_stripped is True
    assert r.approved is True
    assert r.metrics.stage == "superego_voice"


@pytest.mark.asyncio
async def test_voice_applies_tone_adjustments():
    b = ScriptedBackend(["resposta"])
    r = await SuperegoStage().voice(_ctx(sentiment="FRUSTRATED"), b, voice_prompt="x")
    assert "tone:empathetic" in r.adjustments


@pytest.mark.asyncio
async def test_voice_pii_backstop_flags_output():
    b = ScriptedBackend(["Seu email cadastrado é joao.silva@example.com"])
    r = await SuperegoStage().voice(_ctx(), b, voice_prompt="x")
    assert "pii:flagged_in_output" in r.adjustments


@pytest.mark.asyncio
async def test_voice_feeds_synthesis_drift():
    ctx = _ctx()
    ctx.drift = DriftCalculator().compute(ctx.noumeno, ctx.intent)
    ctx.drift.synthesis_drift = -1.0   # sentinel
    b = ScriptedBackend(["Recorded 50 for lunch"])
    await SuperegoStage().voice(ctx, b, voice_prompt="x")
    assert ctx.drift.synthesis_drift >= 0.0   # voice computed it


@pytest.mark.asyncio
async def test_voice_propagates_backend_error():
    with pytest.raises(ConnectionError):
        await SuperegoStage().voice(_ctx(), RaisingBackend(), voice_prompt="x")


# ── blocked + wiring ─────────────────────────────────────────────────

def test_blocked_response_uses_host_message():
    r = SuperegoStage()._blocked_response(_ctx(), block_message="Dados sensíveis detectados.")
    assert r.blocked is True and r.response == "Dados sensíveis detectados."


def test_blocked_response_fallback():
    r = SuperegoStage()._blocked_response(_ctx())
    assert r.blocked is True and r.response   # non-empty fallback


def test_pipeline_context_superego_wiring():
    ctx = PipelineContext(user_input="hi")
    assert ctx.superego_result is None and ctx.superego_metrics is None
    assert ctx.needs_handoff is False and ctx.stop_reason == "completed"

    ctx.superego_result = SuperegoResult(response="ok", metrics=_m("superego_voice"))
    ctx.superego_result.metrics.tokens_in = 8
    ctx.superego_result.metrics.tokens_out = 4
    ctx.superego_result.metrics.tokens_total = 12
    assert ctx.superego_metrics is not None
    assert ctx.superego_metrics in ctx.stage_metrics
    assert ctx.total_tokens == 12


def test_retry_metrics_accumulate_judge_calls():
    # host appends scope + judge attempts into retry_metrics; they fold into totals
    ctx = PipelineContext(user_input="hi")
    ctx.retry_metrics.append(StageMetrics(stage="superego_judge", elapsed_ms=1.0,
                                          tokens_in=3, tokens_out=2, model="t"))
    ctx.retry_metrics.append(StageMetrics(stage="superego_scope", elapsed_ms=1.0,
                                          tokens_in=1, tokens_out=1, model="t"))
    assert ctx.total_tokens == 7
