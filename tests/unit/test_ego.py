"""Unit tests for EgoStage (Stage 4) — the executor agent loop."""

import json
import pytest

from cogno_anima import metakeys as mk
from cogno_anima.stages.ego import EgoStage
from cogno_synapse.base import ToolCallingBackend
from cogno_anima.types import (
    StageMetrics, NoumenoResult, IntentResult, PipelineContext, EgoResult,
    EgoStep, ToolExecution, ToolResult,
)
from cogno_anima.errors import MCPDispatchError, ToolExecutionError


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
        self.calls.append({"system": system, "prompt": prompt})
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


class PolicyDispatcher(StubDispatcher):
    """ToolDispatcher that also satisfies ToolPolicyDispatcher (read/write +
    destructive classification), for read-only mask and confirmation-gate tests."""

    def __init__(self, *a, mutating=(), destructive=(), **kw):
        super().__init__(*a, **kw)
        self._mutating = set(mutating)
        self._destructive = set(destructive)

    @classmethod
    def with_tools(cls, *names, mutating=(), destructive=(), **kwargs):
        schema = [{"type": "function", "function": {"name": n, "description": n, "parameters": {}}}
                  for n in names]
        return cls(schema=schema, mutating=mutating, destructive=destructive, **kwargs)

    def is_mutating(self, name):
        return name in self._mutating

    def requires_confirmation(self, name):
        return name in self._destructive


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
async def test_fallback_prompt_lists_tools_and_mechanics():
    backend = ScriptedToolCallingBackend([{"content": "done"}], native=False)
    disp = StubDispatcher.with_tools("add_income", "get_summary")
    await EgoStage().process(_ctx(), backend, disp, system_prompt=SYS)
    system = backend.calls[0]["system"]
    assert "# Available tools" in system
    assert "add_income" in system and "get_summary" in system
    assert "<TOOL_CALL>" in system           # mechanics block present on fallback


@pytest.mark.asyncio
async def test_native_prompt_omits_tool_mechanics():
    backend = ScriptedToolCallingBackend([{"content": "done"}])   # native
    disp = StubDispatcher.with_tools("add_income")
    await EgoStage().process(_ctx(), backend, disp, system_prompt=SYS)
    system = backend.calls[0]["messages"][0]["content"]
    assert "<TOOL_CALL>" not in system       # API carries tool format on native FC


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


@pytest.mark.asyncio
async def test_duplicate_and_interrupt_log_warnings(caplog):
    import logging
    turns = [_tool_turn("add_income", {"amount": 40}) for _ in range(6)]
    backend = ScriptedToolCallingBackend(turns)
    disp = StubDispatcher.with_tools("add_income")
    with caplog.at_level(logging.WARNING, logger="cogno_anima.ego"):
        await EgoStage().process(_ctx(), backend, disp, system_prompt=SYS)
    msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("event=duplicate_call" in m for m in msgs)
    assert any("event=done" in m and "interrupted=true" in m and "reason=duplicate_calls" in m
               for m in msgs)


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
async def test_unknown_tool_logs_warning(caplog):
    import logging
    backend = ScriptedToolCallingBackend([
        _tool_turn("drop_db", {}),
        {"content": "Sorry, I can't do that."},
    ])
    disp = StubDispatcher.with_tools("add_income")
    with caplog.at_level(logging.WARNING, logger="cogno_anima.ego"):
        await EgoStage().process(_ctx(), backend, disp, system_prompt=SYS)
    assert any("event=unknown_tool" in r.message and "tool=drop_db" in r.message
               for r in caplog.records)


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
async def test_failed_call_retryable_after_new_information():
    """A host guard may refuse a write until the SAME turn reads first (id provenance),
    then expect the IDENTICAL call again — the retry must execute, not hit blocked_retry.
    A success clears the failed-sig memory (new information arrived)."""
    state = {"read": False}

    def guarded_write(_args):
        if not state["read"]:
            return ToolResult(output="", ok=False,
                              error="id not read this turn — call get_summary first")
        return ToolResult(output="income recorded", ok=True)

    def read(_args):
        state["read"] = True
        return ToolResult(output="pending: amount 40", ok=True)

    disp = StubDispatcher.with_tools(
        "add_income", "get_summary",
        handlers={"add_income": guarded_write, "get_summary": read})
    backend = ScriptedToolCallingBackend([
        _tool_turn("add_income", {"amount": 40}),    # refused by the guard
        _tool_turn("get_summary", {}),               # the requested read succeeds
        _tool_turn("add_income", {"amount": 40}),    # identical retry → must RUN
        {"content": "done"},
    ])
    ctx = await EgoStage().process(_ctx(), backend, disp, system_prompt=SYS)
    res = ctx.ego_result
    writes = [t for t in res.tools_executed if t.tool == "add_income"]
    assert writes[0].ok is False
    assert writes[1].ok is True                      # dispatched again, not blocked
    assert not [t for t in res.tools_executed if t.error == "blocked_retry"]
    assert res.interrupted is False


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
async def test_host_force_tool_forces_first_tool_without_rewriting_intent():
    """metadata[EGO_FORCE_TOOL]: the host routed a SOCIAL/short turn ("sim",
    "confirmar") to the executor. tool_choice is forced on step 1, but the NER's
    intent_class stays untouched — the perception record must remain honest."""
    backend = ScriptedToolCallingBackend([_tool_turn("add_income", {"amount": 1}),
                                          {"content": "ok"}])
    disp = StubDispatcher.with_tools("add_income")
    ctx = _ctx(intent_class="SOCIAL", **{mk.EGO_FORCE_TOOL: True})
    ctx = await EgoStage().process(ctx, backend, disp, system_prompt=SYS)
    assert backend.calls[0]["tool_choice"] == "required"
    assert ctx.intent.intent_class == "SOCIAL"          # NER record untouched


def test_force_tool_adds_directive_to_task_context():
    """Fallback-path parity: the flag renders a host directive into the task
    context (the pressure the old intent_class rewrite used to give)."""
    ctx = _ctx(intent_class="SOCIAL", **{mk.EGO_FORCE_TOOL: True})
    assert "REQUIRES tool execution" in EgoStage()._task_context(ctx)
    assert "REQUIRES tool execution" not in EgoStage()._task_context(_ctx(intent_class="SOCIAL"))


@pytest.mark.asyncio
async def test_readonly_wins_over_force_tool():
    """A tentative user (read-only mask) beats the force flag: propose, don't force."""
    backend = ScriptedToolCallingBackend([{"content": "Want me to record it?"}])
    disp = StubDispatcher.with_tools("record_expense")
    ctx = _ctx(intent_class="SOCIAL", ego_readonly=True, **{mk.EGO_FORCE_TOOL: True})
    await EgoStage().process(ctx, backend, disp, system_prompt=SYS)
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


# ── NER signal enrichment: constraints/negation (Block 1) ────────────

def test_task_context_includes_constraints_and_negation():
    ctx = _ctx()
    ctx.intent.constraints = ["only this month"]
    ctx.intent.negation = ["do not delete records"]
    block = EgoStage()._task_context(ctx)
    assert "Constraints (must respect): only this month" in block
    assert "Must NOT: do not delete records" in block


def test_task_context_surfaces_aristotelian_slots():
    ctx = _ctx()
    ctx.intent.aristotelian = {
        "ACTION": "CREATE_MEETING_AND_SEARCH | Schedule event + search rate",
        "TIME": "TOMORROW | Relative temporal marker",
        "QUANTITY": "DOLLAR_VALUE | Monetary quantity to retrieve",
    }
    block = EgoStage()._task_context(ctx)
    assert "Request breakdown (map these to tool arguments):" in block
    # tag + description surfaced per category, so the loop can fill args from the user's words
    assert "TIME=TOMORROW (Relative temporal marker)" in block
    assert "QUANTITY=DOLLAR_VALUE (Monetary quantity to retrieve)" in block
    # a bare tag (no " | ") still renders, description omitted
    ctx.intent.aristotelian = {"ACTION": "BOOK"}
    assert "Request breakdown (map these to tool arguments): ACTION=BOOK" in EgoStage()._task_context(ctx)
    # empty → no line
    ctx.intent.aristotelian = {}
    assert "Request breakdown" not in EgoStage()._task_context(ctx)


# ── Read-only mask (Fonte A) ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_readonly_masks_mutating_tools():
    """ego_readonly → only non-mutating tools are offered; force_first off."""
    backend = ScriptedToolCallingBackend([{"content": "Both 13:00 and 15:00 are open — which?"}])
    disp = PolicyDispatcher.with_tools("get_balance", "record_expense", mutating=["record_expense"])
    ctx = _ctx(intent_class="ACTION_REQUEST", ego_readonly=True)
    ctx = await EgoStage().process(ctx, backend, disp, system_prompt=SYS)
    # native call must NOT force a tool, and only the read tool is visible
    assert backend.calls[0]["tool_choice"] is None
    offered = {t["function"]["name"] for t in backend.calls[0]["tools"]}
    assert offered == {"get_balance"}                       # record_expense masked
    assert ctx.ego_result.tools_executed == []
    assert "PROPOSE mode" in EgoStage()._task_context(ctx)


@pytest.mark.asyncio
async def test_readonly_without_policy_masks_everything():
    """Fail-safe: a plain dispatcher (no policy) in read-only mode offers no tools."""
    backend = ScriptedToolCallingBackend([{"content": "Want me to record 50?"}])
    disp = StubDispatcher.with_tools("record_expense")      # no is_mutating
    ctx = _ctx(intent_class="ACTION_REQUEST", ego_readonly=True)
    ctx = await EgoStage().process(ctx, backend, disp, system_prompt=SYS)
    assert backend.calls[0]["tools"] == []
    assert ctx.ego_result.tools_executed == []


@pytest.mark.asyncio
async def test_firm_action_still_forces_first_tool():
    backend = ScriptedToolCallingBackend([_tool_turn("record_expense", {"amount": 50}),
                                          {"content": "done"}])
    disp = StubDispatcher.with_tools("record_expense")
    ctx = _ctx(intent_class="ACTION_REQUEST")   # no readonly
    ctx = await EgoStage().process(ctx, backend, disp, system_prompt=SYS)
    assert backend.calls[0]["tool_choice"] == "required"


# ── Confirmation gate (Fonte B) ──────────────────────────────────────

@pytest.mark.asyncio
async def test_destructive_tool_held_without_confirmation():
    """A requires_confirmation tool is NOT executed; it surfaces as pending."""
    backend = ScriptedToolCallingBackend([_tool_turn("delete_all", {}), {"content": "x"}])
    disp = PolicyDispatcher.with_tools("delete_all", mutating=["delete_all"],
                                       destructive=["delete_all"])
    ctx = _ctx(intent_class="ACTION_REQUEST")
    ctx = await EgoStage().process(ctx, backend, disp, system_prompt=SYS)
    res = ctx.ego_result
    assert disp.executed == []                              # never ran the host tool
    assert [t.tool for t in res.pending_confirmation] == ["delete_all"]
    assert res.has_side_effects is False


@pytest.mark.asyncio
async def test_destructive_tool_runs_once_confirmed():
    """With ego_confirmed set, the gate opens and the tool executes."""
    backend = ScriptedToolCallingBackend([_tool_turn("delete_all", {}), {"content": "done"}])
    disp = PolicyDispatcher.with_tools("delete_all", mutating=["delete_all"],
                                       destructive=["delete_all"])
    ctx = _ctx(intent_class="ACTION_REQUEST", ego_confirmed=True)
    ctx = await EgoStage().process(ctx, backend, disp, system_prompt=SYS)
    assert disp.executed == [("delete_all", {})]
    assert ctx.ego_result.pending_confirmation == []


@pytest.mark.asyncio
async def test_confirmed_calls_execute_deterministically_without_model_reissue():
    """Gate-B completion: the approved calls run even if the model re-issues NOTHING (a small
    model often just replies 'done' on the confirm turn) — the side effect must not be skipped."""
    backend = PlainBackend()   # emits no tool call, ever
    disp = StubDispatcher.with_tools("book_appointment", side_effects={"book_appointment": True})
    ctx = _ctx(ego_confirmed=True,
               ego_confirmed_calls=[{"tool": "book_appointment",
                                     "arguments": {"host_id": "dr_x", "date": "2026-07-02",
                                                   "time": "11:00"}}])
    ctx = await EgoStage().process(ctx, backend, disp, system_prompt=SYS)
    # executed exactly once, recorded in the trace, and flagged a side effect
    assert disp.executed == [("book_appointment", {"host_id": "dr_x", "date": "2026-07-02",
                                                   "time": "11:00"})]
    booked = [tc for s in ctx.ego_result.steps for tc in s.tool_calls
              if tc.tool == "book_appointment"]
    assert len(booked) == 1 and booked[0].ok and booked[0].side_effect
    assert ctx.ego_result.has_side_effects


@pytest.mark.asyncio
async def test_confirmed_call_blocks_a_redundant_model_reissue():
    """If the model DOES re-issue the same confirmed call, the dedup guard blocks it — never
    execute the destructive action twice."""
    backend = ScriptedToolCallingBackend([_tool_turn("book_appointment", {"host_id": "dr_x"}),
                                          {"content": "done"}])
    disp = StubDispatcher.with_tools("book_appointment")
    ctx = _ctx(ego_confirmed=True,
               ego_confirmed_calls=[{"tool": "book_appointment", "arguments": {"host_id": "dr_x"}}])
    ctx = await EgoStage().process(ctx, backend, disp, system_prompt=SYS)
    assert disp.executed == [("book_appointment", {"host_id": "dr_x"})]   # exactly once


@pytest.mark.asyncio
async def test_confirmed_call_failure_feeds_the_error_not_already_executed():
    """A confirmed call can still fail execute-time business validation (slot taken, limit
    reached). The model must receive the ERROR — '[ALREADY EXECUTED] → (empty)' makes it
    hallucinate success — and the trace must record the failed execution."""
    class RecordingBackend:
        model = "plain"
        def __init__(self):
            self.prompts = []
        async def generate(self, system, prompt):
            self.prompts.append(prompt)
            return "That slot is taken; 10:00 is free.", 1, 1

    backend = RecordingBackend()
    disp = StubDispatcher.with_tools("book_appointment", handlers={
        "book_appointment": lambda a: ToolResult(
            output="", ok=False, error="09:00 on 2026-07-02 is already booked. Free: 10:00",
            side_effect=True),
    })
    ctx = _ctx(ego_confirmed=True,
               ego_confirmed_calls=[{"tool": "book_appointment",
                                     "arguments": {"date": "2026-07-02", "time": "09:00"}}])
    ctx = await EgoStage().process(ctx, backend, disp, system_prompt=SYS)
    # the model saw the failure (not a success note) with the business error verbatim
    assert "[EXECUTION FAILED]" in backend.prompts[0]
    assert "already booked" in backend.prompts[0]
    assert "[ALREADY EXECUTED] book_appointment → \n" not in backend.prompts[0]
    # trace records the failed execution (the judge sees ERROR, not silence)
    call = ctx.ego_result.steps[0].tool_calls[0]
    assert call.ok is False and "already booked" in call.error
    # loop converged to an honest draft
    assert "taken" in ctx.ego_result.draft


# ── 2R-B: composite budget + sequential ordering ─────────────────────

@pytest.mark.asyncio
async def test_composite_raises_default_max_steps():
    """A multi-task (is_composite) request gets more loop budget by default."""
    turns = [_tool_turn("add_income", {"amount": i}) for i in range(12)]
    backend = ScriptedToolCallingBackend(turns)
    disp = StubDispatcher.with_tools("add_income")
    ctx = _ctx()                       # no ego_max_steps override
    ctx.intent.is_composite = True
    ctx = await EgoStage().process(ctx, backend, disp, system_prompt=SYS)
    assert ctx.ego_result.interrupt_reason == "max_steps"
    assert len(ctx.ego_result.steps) == EgoStage.MAX_STEPS_COMPOSITE  # 8, not 5


@pytest.mark.asyncio
async def test_host_max_steps_overrides_composite():
    """The host's explicit ego_max_steps always wins over the composite default."""
    turns = [_tool_turn("add_income", {"amount": i}) for i in range(12)]
    backend = ScriptedToolCallingBackend(turns)
    disp = StubDispatcher.with_tools("add_income")
    ctx = _ctx(ego_max_steps=2)
    ctx.intent.is_composite = True
    ctx = await EgoStage().process(ctx, backend, disp, system_prompt=SYS)
    assert len(ctx.ego_result.steps) == 2


def test_sequential_adds_order_hint_to_task_context():
    """is_sequential renders an ordering instruction + the causal chain as a plan."""
    ctx = _ctx()
    ctx.intent.is_sequential = True
    ctx.intent.causal_chain = ["convert to USD", "record the expense"]
    task_ctx = EgoStage()._task_context(ctx)
    assert "Execution order" in task_ctx
    assert "1) convert to USD" in task_ctx and "2) record the expense" in task_ctx


def test_non_sequential_has_no_order_hint():
    ctx = _ctx()
    assert "Execution order" not in EgoStage()._task_context(ctx)
