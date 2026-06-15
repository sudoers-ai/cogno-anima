"""Unit tests for EgoStage (Stage 4) — the executor agent loop."""

import json
import pytest

from cogno_core.stages.ego import EgoStage
from cogno_core.llm.base import ToolCallingBackend
from cogno_core.types import (
    StageMetrics, NoumenoResult, IntentResult, PipelineContext, EgoResult,
    EgoStep, ToolExecution, ToolResult,
)
from cogno_core.errors import MCPDispatchError, ToolExecutionError


# ── test doubles (self-contained; the import-from-conftest path is brittle
#     with this package layout, so they live here) ─────────────────────

class ScriptedToolCallingBackend:
    """Native-FC test double replaying scripted chat_with_tools turns.

    Each turn is a message_dict: ``{"content": str, "tool_calls": [...]}`` or a
    plain ``{"content": "final"}`` to end the loop. With ``native=False`` the
    same script drives the text-fallback path via ``generate`` (tool_calls are
    rendered as ``<TOOL_CALL>`` tags).
    """

    def __init__(self, turns, model="stub-fc", tokens_in=7, tokens_out=3, native=True):
        self.turns = list(turns)
        self.model = model
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self._native = native
        self.calls = []

    async def generate(self, system, prompt):
        turn = self.turns.pop(0)
        text = turn.get("content", "") or ""
        for tc in turn.get("tool_calls", []):
            fn = tc["function"]
            args = fn["arguments"] if isinstance(fn["arguments"], str) else json.dumps(fn["arguments"])
            text += f'\n<TOOL_CALL>{{"tool": "{fn["name"]}", "args": {args}}}</TOOL_CALL>'
        return text, self.tokens_in, self.tokens_out

    async def chat_with_tools(self, messages, tools, tool_choice=None):
        self.calls.append({"messages": list(messages), "tools": tools, "tool_choice": tool_choice})
        return self.turns.pop(0), self.tokens_in, self.tokens_out

    def supports_native_tools(self):
        return self._native


class PlainBackend:
    """A text-only backend (only generate + model) → never satisfies ToolCallingBackend."""
    model = "plain"

    async def generate(self, system, prompt):
        return "ok", 1, 1


class StubDispatcher:
    """Host ToolDispatcher test double. handlers: name -> callable(args) -> str|ToolResult."""

    def __init__(self, schema=None, handlers=None, side_effects=None):
        self._schema = schema or []
        self._handlers = handlers or {}
        self._side_effects = side_effects or {}
        self.executed = []

    @classmethod
    def with_tools(cls, *names, **kwargs):
        schema = [{"type": "function", "function": {"name": n, "description": n, "parameters": {}}}
                  for n in names]
        return cls(schema=schema, **kwargs)

    def tools_schema(self):
        return self._schema

    async def execute(self, name, arguments):
        self.executed.append((name, dict(arguments)))
        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult(output=f"{name} ok", side_effect=self._side_effects.get(name, False))
        res = handler(arguments)
        if isinstance(res, ToolResult):
            return res
        return ToolResult(output=str(res), side_effect=self._side_effects.get(name, False))


# ── helpers ──────────────────────────────────────────────────────────

def _m(stage):
    return StageMetrics(stage=stage, elapsed_ms=0.0, tokens_in=0, tokens_out=0, model="stub")


def _ctx(user="record 40", rewritten=None, intent_class="ACTION_REQUEST", **meta):
    noumeno = NoumenoResult(
        original=user, rewritten=(rewritten if rewritten is not None else user),
        context_turn="", language="en", drift_score=0.0, drift_tag="PASS_THROUGH",
        changed=False, confidence=0.9, change_subject=False, subject_similarity=1.0,
        context_used=False, preserved_terms=[], rewrite_warnings=[], metrics=_m("noumeno"),
    )
    intent = IntentResult(
        intent_class=intent_class, sentiment="NEUTRAL", confidence=0.9,
        temporal_class="TIMELESS", triad_signal="EGO", goal="record income",
        domains=["FINANCE"], entities_objects=["income"], metrics=_m("ner"),
    )
    ctx = PipelineContext(user_input=user, noumeno=noumeno, intent=intent)
    ctx.metadata.update(meta)
    return ctx


def _tool_turn(name, args):
    return {"content": "", "tool_calls": [{
        "id": f"c_{name}", "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)}}]}


SYS = "You are an executor. Use tools to record finances."


# ── path selection ───────────────────────────────────────────────────

def test_isinstance_gate():
    assert isinstance(ScriptedToolCallingBackend([]), ToolCallingBackend)
    assert not isinstance(PlainBackend(), ToolCallingBackend)


@pytest.mark.asyncio
async def test_native_single_tool_then_final():
    backend = ScriptedToolCallingBackend([
        _tool_turn("add_income", {"amount": 40}),
        {"content": "Recorded 40."},
    ])
    disp = StubDispatcher.with_tools("add_income")
    ctx = await EgoStage().process(_ctx(), backend, disp, system_prompt=SYS)
    res = ctx.ego_result
    assert [t.tool for t in res.tools_executed] == ["add_income"]
    assert res.tools_executed[0].arguments == {"amount": 40}
    assert res.draft == "Recorded 40."
    assert res.steps[0].path == "native"
    assert res.interrupted is False
    assert len(res.steps) == 2
    assert disp.executed == [("add_income", {"amount": 40})]
    # tokens summed across the 2 chat_with_tools calls
    assert res.metrics.tokens_in == 14 and res.metrics.tokens_out == 6
    assert res.metrics.model == "stub-fc"


@pytest.mark.asyncio
async def test_fallback_path_when_native_disabled():
    backend = ScriptedToolCallingBackend([
        _tool_turn("add_income", {"amount": 40}),
        {"content": "Recorded."},
    ], native=False)
    disp = StubDispatcher.with_tools("add_income")
    ctx = await EgoStage().process(_ctx(), backend, disp, system_prompt=SYS)
    res = ctx.ego_result
    assert res.steps[0].path == "fallback"
    assert [t.tool for t in res.tools_executed] == ["add_income"]
    assert res.draft == "Recorded."
    assert disp.executed == [("add_income", {"amount": 40})]


@pytest.mark.asyncio
async def test_conversational_no_tool():
    backend = ScriptedToolCallingBackend([{"content": "Hi, how can I help?"}])
    disp = StubDispatcher.with_tools("add_income")
    ctx = await EgoStage().process(_ctx(intent_class="SOCIAL"), backend, disp, system_prompt=SYS)
    res = ctx.ego_result
    assert res.tools_executed == []
    assert res.draft == "Hi, how can I help?"
    assert len(res.steps) == 1
    assert disp.executed == []


# ── budget / convergence signals ─────────────────────────────────────

@pytest.mark.asyncio
async def test_max_steps_interrupted():
    # always asks for a (distinct) tool, never stops
    turns = [_tool_turn("add_income", {"amount": i}) for i in range(10)]
    backend = ScriptedToolCallingBackend(turns)
    disp = StubDispatcher.with_tools("add_income")
    ctx = await EgoStage().process(_ctx(ego_max_steps=2), backend, disp, system_prompt=SYS)
    res = ctx.ego_result
    assert res.interrupted is True
    assert res.interrupt_reason == "max_steps"
    assert len(res.steps) == 2


@pytest.mark.asyncio
async def test_duplicate_calls_abort():
    # same tool + same args every turn → blocked after MAX_DUPLICATE_CALLS, then abort
    turns = [_tool_turn("add_income", {"amount": 40}) for _ in range(6)]
    backend = ScriptedToolCallingBackend(turns)
    disp = StubDispatcher.with_tools("add_income")
    ctx = await EgoStage().process(_ctx(), backend, disp, system_prompt=SYS)
    res = ctx.ego_result
    assert res.interrupted is True
    assert res.interrupt_reason == "duplicate_calls"
    # executed only twice; subsequent identical calls were blocked, not dispatched
    assert disp.executed == [("add_income", {"amount": 40}), ("add_income", {"amount": 40})]
    blocked = [t for t in res.tools_executed if t.error == "duplicate"]
    assert len(blocked) >= 2


# ── tool name / error handling ───────────────────────────────────────

@pytest.mark.asyncio
async def test_unknown_tool_fed_back():
    backend = ScriptedToolCallingBackend([
        _tool_turn("drop_db", {}),
        {"content": "Sorry, I can't do that."},
    ])
    disp = StubDispatcher.with_tools("add_income")
    ctx = await EgoStage().process(_ctx(), backend, disp, system_prompt=SYS)
    res = ctx.ego_result
    first = res.tools_executed[0]
    assert first.ok is False and "unknown tool" in first.error
    assert disp.executed == []                     # never dispatched
    assert res.draft == "Sorry, I can't do that."


@pytest.mark.asyncio
async def test_recoverable_error_fed_back_and_loop_continues():
    disp = StubDispatcher.with_tools(
        "add_income",
        handlers={"add_income": lambda a: ToolResult(output="", ok=False, error="amount must be > 0")},
    )
    backend = ScriptedToolCallingBackend([
        _tool_turn("add_income", {"amount": -1}),
        {"content": "Let me fix that."},
    ])
    ctx = await EgoStage().process(_ctx(), backend, disp, system_prompt=SYS)
    res = ctx.ego_result
    assert res.tools_executed[0].ok is False
    assert res.tools_executed[0].error == "amount must be > 0"
    assert res.draft == "Let me fix that."          # loop continued past the error


@pytest.mark.asyncio
async def test_fatal_error_propagates():
    def boom(_):
        raise MCPDispatchError("add_income", {}, ConnectionError("server down"))
    disp = StubDispatcher.with_tools("add_income", handlers={"add_income": boom})
    backend = ScriptedToolCallingBackend([_tool_turn("add_income", {"amount": 40})])
    with pytest.raises(MCPDispatchError):
        await EgoStage().process(_ctx(), backend, disp, system_prompt=SYS)


@pytest.mark.asyncio
async def test_stray_exception_wrapped():
    def boom(_):
        raise ValueError("unexpected bug")
    disp = StubDispatcher.with_tools("add_income", handlers={"add_income": boom})
    backend = ScriptedToolCallingBackend([_tool_turn("add_income", {"amount": 40})])
    with pytest.raises(ToolExecutionError) as ei:
        await EgoStage().process(_ctx(), backend, disp, system_prompt=SYS)
    assert ei.value.tool == "add_income"
    assert isinstance(ei.value.__cause__, ValueError)


# ── tool_choice / correction / side effects ──────────────────────────

@pytest.mark.asyncio
async def test_action_request_forces_tool_choice_first():
    backend = ScriptedToolCallingBackend([_tool_turn("add_income", {"amount": 1}), {"content": "ok"}])
    disp = StubDispatcher.with_tools("add_income")
    await EgoStage().process(_ctx(intent_class="ACTION_REQUEST"), backend, disp, system_prompt=SYS)
    assert backend.calls[0]["tool_choice"] == "required"
    assert backend.calls[1]["tool_choice"] is None


@pytest.mark.asyncio
async def test_information_request_no_force():
    backend = ScriptedToolCallingBackend([{"content": "here is info"}])
    disp = StubDispatcher.with_tools("get_summary")
    await EgoStage().process(_ctx(intent_class="INFORMATION_REQUEST"), backend, disp, system_prompt=SYS)
    assert backend.calls[0]["tool_choice"] is None


@pytest.mark.asyncio
async def test_correction_injects_actions_block_and_attempt():
    prior = EgoResult(steps=[EgoStep(
        index=0, path="native", tool_calls=[
            ToolExecution(tool="add_income", arguments={"amount": 40}, result="ok",
                          ok=True, side_effect=True)])], metrics=_m("ego"))
    ctx = _ctx()
    ctx.ego_result = prior
    ctx.metadata["ego_correction"] = {"reason": "valor errado, era 50", "attempt": 2}
    backend = ScriptedToolCallingBackend([{"content": "redone"}])
    disp = StubDispatcher.with_tools("add_income")
    ctx = await EgoStage().process(ctx, backend, disp, system_prompt=SYS)
    sys_msg = backend.calls[0]["messages"][0]["content"]
    assert "ACTIONS ALREADY EXECUTED" in sys_msg
    assert "add_income" in sys_msg
    assert "valor errado" in sys_msg
    assert ctx.ego_result.attempt == 2


@pytest.mark.asyncio
async def test_side_effect_recorded():
    disp = StubDispatcher.with_tools("add_income", side_effects={"add_income": True})
    backend = ScriptedToolCallingBackend([_tool_turn("add_income", {"amount": 40}), {"content": "ok"}])
    ctx = await EgoStage().process(_ctx(), backend, disp, system_prompt=SYS)
    assert ctx.ego_result.has_side_effects is True
    assert ctx.ego_result.tools_executed[0].side_effect is True


@pytest.mark.asyncio
async def test_persona_label_echoed():
    backend = ScriptedToolCallingBackend([{"content": "ok"}])
    disp = StubDispatcher.with_tools("add_income")
    ctx = await EgoStage().process(_ctx(ego_persona="ANALYST"), backend, disp, system_prompt=SYS)
    assert ctx.ego_result.persona == "ANALYST"


@pytest.mark.asyncio
async def test_requires_noumeno_and_intent():
    backend = ScriptedToolCallingBackend([{"content": "ok"}])
    disp = StubDispatcher.with_tools("add_income")
    bad = PipelineContext(user_input="hi")          # no noumeno/intent
    with pytest.raises(ValueError):
        await EgoStage().process(bad, backend, disp, system_prompt=SYS)
