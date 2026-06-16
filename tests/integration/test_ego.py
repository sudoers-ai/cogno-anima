"""
Integration tests for the EGO stage (Stage 4) against a real Ollama model.

The default ``OllamaBackend`` does text generation only (no ``chat_with_tools``),
so the EGO runs the **text-fallback path** — the model emits ``<TOOL_CALL>`` tags
which ``parse_tool_calls_from_text`` reads. This is exactly the path the distilled
student will use, so it is the one most worth exercising end-to-end.

Tool execution is delegated to an in-process ``InMemoryDispatcher`` test host (no
DB/MCP). Auto-skipped if Ollama is unreachable. temperature=0.0 for determinism.
"""

import httpx
import pytest

from cogno_core.llm import OllamaBackend
from cogno_core.llm.base import ToolCallingBackend
from cogno_core.stages.ego import EgoStage
from cogno_core.types import (
    PipelineContext, IntentResult, NoumenoResult, StageMetrics, ToolResult,
)

MODEL = "mistral:latest"


async def is_ollama_available() -> bool:
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.get("http://localhost:11434/")
            return resp.status_code == 200
    except Exception:
        return False


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
async def test_ego_executes_tool_via_fallback():
    if not await is_ollama_available():
        pytest.skip("Local Ollama server (http://localhost:11434) is not running.")
    backend = OllamaBackend(model=MODEL, temperature=0.0)
    # default OllamaBackend has no native FC → EGO uses the text-fallback path
    assert not isinstance(backend, ToolCallingBackend)

    disp = InMemoryDispatcher()
    ctx = await EgoStage().process(
        _ctx("Record an expense of 50 reais for lunch."), backend, disp, system_prompt=SYSTEM)
    res = ctx.ego_result

    assert res is not None
    assert res.steps and res.steps[0].path == "fallback"
    names = [t.tool for t in res.tools_executed]
    assert "record_expense" in names, f"expected record_expense, got {names}; draft={res.draft!r}"
    assert ("record_expense", ) in [(n,) for n, _ in disp.executed]
    assert res.has_side_effects is True


@pytest.mark.asyncio
async def test_ego_metrics_are_real():
    if not await is_ollama_available():
        pytest.skip("Local Ollama server is not running.")
    backend = OllamaBackend(model=MODEL, temperature=0.0)
    disp = InMemoryDispatcher()
    ctx = await EgoStage().process(
        _ctx("What is my current balance?"), backend, disp, system_prompt=SYSTEM)
    res = ctx.ego_result
    assert res.metrics.model == MODEL
    assert res.metrics.tokens_in > 0 and res.metrics.tokens_out > 0
    assert res.metrics.tokens_total == res.metrics.tokens_in + res.metrics.tokens_out
    # folds into the pipeline totals
    assert ctx.total_tokens >= res.metrics.tokens_total


@pytest.mark.asyncio
async def test_ego_produces_draft():
    if not await is_ollama_available():
        pytest.skip("Local Ollama server is not running.")
    backend = OllamaBackend(model=MODEL, temperature=0.0)
    disp = InMemoryDispatcher()
    ctx = await EgoStage().process(
        _ctx("Thanks, that's all for now.", intent_class="SOCIAL"),
        backend, disp, system_prompt=SYSTEM)
    res = ctx.ego_result
    # a valid result with a draft for the SUPEREGO to voice; no crash
    assert res is not None
    assert isinstance(res.draft, str)
