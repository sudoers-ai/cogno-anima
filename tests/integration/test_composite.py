"""
Integration test for CompositeDispatcher against a real EGO + Ollama model.

Proves the merge works end-to-end: two tools live in *separate* source
dispatchers (as a finance module and a scheduling module would), are merged into
one CompositeDispatcher, and the real EGO — seeing a single flat tool set — picks
the right tool, which routes to the source that owns it. Auto-skipped without
Ollama. temperature=0.0 for determinism.
"""

import httpx
import pytest

from cogno_synapse import OllamaBackend
from cogno_anima.stages.ego import EgoStage
from cogno_anima.tools import CompositeDispatcher
from cogno_anima.types import (
    IntentResult,
    NoumenoResult,
    PipelineContext,
    StageMetrics,
    ToolResult,
)

MODEL = "mistral:latest"


async def is_ollama_available() -> bool:
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            return (await client.get("http://localhost:11434/")).status_code == 200
    except Exception:
        return False


class FinanceDispatcher:
    """Source A — owns get_balance."""

    def __init__(self):
        self.executed = []

    def tools_schema(self):
        return [{"type": "function", "function": {
            "name": "get_balance", "description": "Get the current account balance.",
            "parameters": {"type": "object", "properties": {}}}}]

    async def execute(self, name, arguments):
        self.executed.append(name)
        return ToolResult(output="Current balance: 1000 BRL.")


class SchedulingDispatcher:
    """Source B — owns check_availability."""

    def __init__(self):
        self.executed = []

    def tools_schema(self):
        return [{"type": "function", "function": {
            "name": "check_availability",
            "description": "Check free appointment slots for a date.",
            "parameters": {"type": "object", "properties": {"date": {"type": "string"}},
                           "required": ["date"]}}}]

    async def execute(self, name, arguments):
        self.executed.append(name)
        return ToolResult(output="Free at 14:00 and 16:00.")


SYSTEM = (
    "You are an assistant's execution engine for finance and scheduling. For ANY "
    "data operation you MUST call the appropriate tool — never invent the data. "
    "When the task is done, reply with a short confirmation."
)


def _m(stage):
    return StageMetrics(stage=stage, elapsed_ms=1.0, tokens_in=1, tokens_out=1, model="test")


def _ctx(task: str) -> PipelineContext:
    noumeno = NoumenoResult(
        original=task, rewritten=task, context_turn="", language="en",
        canonical_language="en", drift_score=0.0, drift_tag="PASS_THROUGH",
        changed=False, confidence=1.0, change_subject=False, subject_similarity=1.0,
        context_used=False, preserved_terms=[], rewrite_warnings=[], metrics=_m("noumeno"))
    intent = IntentResult(
        intent_class="ACTION_REQUEST", sentiment="NEUTRAL", confidence=1.0,
        temporal_class="TIMELESS", triad_signal="EGO", goal=task, domains=["FINANCE"],
        metrics=_m("ner"))
    return PipelineContext(user_input=task, noumeno=noumeno, intent=intent)


@pytest.mark.asyncio
async def test_ego_executes_tool_from_merged_source():
    if not await is_ollama_available():
        pytest.skip("Local Ollama server (http://localhost:11434) is not running.")
    backend = OllamaBackend(model=MODEL, temperature=0.0)

    finance, scheduling = FinanceDispatcher(), SchedulingDispatcher()
    composite = CompositeDispatcher([finance, scheduling])
    # the EGO sees both tools as one flat set
    names = {s["function"]["name"] for s in composite.tools_schema()}
    assert names == {"get_balance", "check_availability"}

    ctx = await EgoStage().process(
        _ctx("What is my current balance?"), backend, composite, system_prompt=SYSTEM)
    res = ctx.ego_result

    assert res is not None
    executed = [t.tool for t in res.tools_executed]
    assert "get_balance" in executed, f"expected get_balance, got {executed}; draft={res.draft!r}"
    # the call routed to the finance source, not scheduling
    assert finance.executed == ["get_balance"]
    assert scheduling.executed == []
