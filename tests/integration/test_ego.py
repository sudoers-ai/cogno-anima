"""
Integration tests for the EGO stage (Stage 4) against a real model.

The default (Ollama) backend does text generation only (no ``chat_with_tools``),
so the EGO runs the **text-fallback path** — the model emits ``<TOOL_CALL>`` tags
which ``parse_tool_calls_from_text`` reads. That is the path the distilled student
will use, so it is the one most worth exercising end-to-end; point
``COGNO_TEST_MODEL`` at a cloud model and the same test walks the native path
instead, which is what the assertion below reads off the backend rather than
assuming.

Tool execution is delegated to an in-process ``InMemoryDispatcher`` test host (no
DB/MCP). Auto-skipped when the configured model is unreachable (``backends``
decides what that means). temperature=0.0 for determinism.
"""

import pytest

from cogno_synapse.base import ToolCallingBackend
from cogno_anima.stages.ego import EgoStage
from cogno_anima.types import (
    PipelineContext, IntentResult, NoumenoResult, StageMetrics, ToolResult,
)
from tests.integration import backends


# ── in-process host dispatcher (the "hands") ─────────────────────────

TOOLS = [
    {"type": "function", "function": {
        "name": "record_expense",
        "description": "Record an expense (money the user spent).",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number", "description": "value in BRL"},
            "description": {"type": "string", "description": "what it was spent on"},
        }, "required": ["amount", "description"]}}},
    {"type": "function", "function": {
        "name": "get_balance",
        "description": "Get the current account balance.",
        "parameters": {"type": "object", "properties": {}}}},
]


class InMemoryDispatcher:
    def __init__(self):
        self.executed = []

    def tools_schema(self):
        return TOOLS

    async def execute(self, name, arguments):
        self.executed.append((name, dict(arguments)))
        if name == "record_expense":
            amt = arguments.get("amount")
            return ToolResult(output=f"Recorded expense of {amt} BRL.", side_effect=True)
        if name == "get_balance":
            return ToolResult(output="Current balance: 1000 BRL.")
        return ToolResult(output="", ok=False, error=f"unknown tool {name!r}")


SYSTEM = (
    "You are a finance assistant's execution engine. For ANY data operation you "
    "MUST call the appropriate tool — never invent or compute the data yourself. "
    "When the task is done, reply with a short confirmation."
)


def _m(stage):
    return StageMetrics(stage=stage, elapsed_ms=1.0, tokens_in=1, tokens_out=1, model="test")


def _ctx(task: str, intent_class: str = "ACTION_REQUEST") -> PipelineContext:
    noumeno = NoumenoResult(
        original=task, rewritten=task, context_turn="", language="en",
        canonical_language="en", drift_score=0.0, drift_tag="PASS_THROUGH",
        changed=False, confidence=1.0, change_subject=False, subject_similarity=1.0,
        context_used=False, preserved_terms=[], rewrite_warnings=[], metrics=_m("noumeno"),
    )
    intent = IntentResult(
        intent_class=intent_class, sentiment="NEUTRAL", confidence=1.0,
        temporal_class="TIMELESS", triad_signal="EGO", goal=task, domains=["FINANCE"],
        metrics=_m("ner"),
    )
    ctx = PipelineContext(user_input=task, noumeno=noumeno, intent=intent)
    return ctx


@pytest.mark.asyncio
async def test_ego_executes_tool():
    await backends.skip_unless_available()
    backend = backends.text_backend()
    # A text-only backend (the local default) has no native FC → the EGO takes the
    # text-fallback path; a cloud backend satisfies ToolCallingBackend and takes the
    # native one. Which one it is follows the CONFIGURED backend, so read it off the
    # object instead of asserting the local case as if it were the only one.
    expected_path = "native" if isinstance(backend, ToolCallingBackend) else "fallback"

    disp = InMemoryDispatcher()
    ctx = await EgoStage().process(
        _ctx("Record an expense of 50 reais for lunch."), backend, disp, system_prompt=SYSTEM)
    res = ctx.ego_result

    assert res is not None
    assert res.steps and res.steps[0].path == expected_path
    names = [t.tool for t in res.tools_executed]
    assert "record_expense" in names, f"expected record_expense, got {names}; draft={res.draft!r}"
    assert ("record_expense", ) in [(n,) for n, _ in disp.executed]
    assert res.has_side_effects is True


@pytest.mark.asyncio
async def test_ego_metrics_are_real():
    await backends.skip_unless_available()
    backend = backends.text_backend()
    disp = InMemoryDispatcher()
    ctx = await EgoStage().process(
        _ctx("What is my current balance?"), backend, disp, system_prompt=SYSTEM)
    res = ctx.ego_result
    assert res.metrics.model == backend.model
    assert res.metrics.tokens_in > 0 and res.metrics.tokens_out > 0
    assert res.metrics.tokens_total == res.metrics.tokens_in + res.metrics.tokens_out
    # folds into the pipeline totals
    assert ctx.total_tokens >= res.metrics.tokens_total


@pytest.mark.asyncio
async def test_ego_produces_draft():
    await backends.skip_unless_available()
    backend = backends.text_backend()
    disp = InMemoryDispatcher()
    ctx = await EgoStage().process(
        _ctx("Thanks, that's all for now.", intent_class="SOCIAL"),
        backend, disp, system_prompt=SYSTEM)
    res = ctx.ego_result
    # a valid result with a draft for the SUPEREGO to voice; no crash
    assert res is not None
    assert isinstance(res.draft, str)


@pytest.mark.asyncio
async def test_a_real_model_stops_repeating_once_it_SEES_the_frustration():
    """The unit tests pin the prompt TEXT; only a real model shows whether it acts on it.

    Reproduces the shape of the August WhatsApp failure: the canonical rewrite, the intent and
    the goal are byte-identical across the turns (the NOUMENO normalises "Sugere horário ai",
    "Sugere horario" and "Sugere horário porra" to the same sentence), so the ONLY thing that
    can tell them apart is the sentiment — which the executor did not receive. It answered with
    the same sentence twice and the contact wrote "IA burra".

    Asserts the DRAFTS DIFFER, not any particular wording: what the model says when it finally
    sees the frustration is its business; what the pipeline owes it is a task that is no longer
    identical. Pinning a phrase here would be pinning this model's taste.
    """
    await backends.skip_unless_available()
    backend = backends.text_backend()
    disp = InMemoryDispatcher()

    async def draft(sentiment: str) -> str:
        ctx = _ctx("Please suggest a time.", intent_class="ACTION_REQUEST")
        ctx.intent.sentiment = sentiment
        ctx.intent.goal = "suggest a time"
        out = await EgoStage().process(ctx, backend, disp,
                                       system_prompt="You are a sales consultant. Be brief.")
        return (out.ego_result.draft or "").strip()

    calm = await draft("NEUTRAL")
    angry = await draft("FRUSTRATED")
    assert calm and angry, "both turns must produce a draft"
    assert calm != angry, (
        "same rewrite, same intent, same goal — the sentiment is the only thing left to make "
        "the executor say something different, and it must be enough")
