"""
End-to-end integration test: the FULL pipeline through ReferencePipeline against
a real Ollama model (NOUMENO → NER → ID → EGO ⇄ judge → voice).

This closes the gap where integration tests only exercised one stage at a time —
here a real turn flows through every seam. Auto-skipped if Ollama is unreachable.
temperature=0.0 for determinism. Assertions are INVARIANTS (valid route, terminal
reached, no crash, no hallucinated dispatch) — never exact model wording.
"""

import httpx
import pytest

from cogno_anima.llm import OllamaBackend, OllamaEmbedder
from cogno_anima.llm.cache import CachingEmbedder
from cogno_anima.types import PipelineContext, ToolResult
from cogno_anima.vocab import VALID_TRIAD, VALID_STOP_REASONS
from cognobench.pipeline import ReferencePipeline

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent.parent.parent / "cogno_anima" / "prompt_templates"
MODEL = "mistral:latest"
BASE_URL = "http://localhost:11434"


async def is_ollama_available() -> bool:
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.get(f"{BASE_URL}/")
            return resp.status_code == 200
    except Exception:
        return False


TOOLS = [
    {"type": "function", "function": {
        "name": "record_expense", "description": "Record an expense (money spent).",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number"}, "description": {"type": "string"}},
            "required": ["amount", "description"]}}},
    {"type": "function", "function": {
        "name": "get_balance", "description": "Get the current account balance.",
        "parameters": {"type": "object", "properties": {}}}},
]
VALID_TOOLS = {t["function"]["name"] for t in TOOLS}


class InMemoryDispatcher:
    def __init__(self):
        self.executed = []

    def tools_schema(self):
        return TOOLS

    async def execute(self, name, arguments):
        self.executed.append((name, dict(arguments)))
        if name == "record_expense":
            return ToolResult(output=f"Recorded expense of {arguments.get('amount')} BRL.",
                              side_effect=True)
        if name == "get_balance":
            return ToolResult(output="Current balance: 1000 BRL.")
        return ToolResult(output="", ok=False, error=f"unknown tool {name!r}")


KW = dict(
    ego_prompt=("You are the execution engine of a finance assistant. For ANY data "
                "operation you MUST call the appropriate tool — never invent data. "
                "If the user is only chatting, do not call a tool."),
    limits_prompt="Only act on the user's stated request; confirm what was done.",
    voice_prompt="You are a warm, concise finance assistant. Reply in the user's language.",
)


def _backends():
    """gen=JSON-constrained (NOUMENO/NER/scope/judge); ego/voice=free text (TOOL_CALL)."""
    gen = OllamaBackend(model=MODEL, base_url=BASE_URL, temperature=0.0, format="json")
    text = OllamaBackend(model=MODEL, base_url=BASE_URL, temperature=0.0)
    embedder = CachingEmbedder(OllamaEmbedder(model="nomic-embed-text", base_url=BASE_URL))
    return gen, text, embedder


def _assert_pipeline_invariants(ctx, dispatcher):
    # Perception + routing always populated and valid.
    assert ctx.noumeno is not None and ctx.intent is not None and ctx.id_result is not None
    assert ctx.id_result.triad_route in VALID_TRIAD
    assert ctx.stop_reason in VALID_STOP_REASONS
    # A terminal is always reached: either a response was written, or it was a
    # blocking/handoff terminal.
    terminal = (
        (ctx.superego_result is not None and ctx.superego_result.response)
        or ctx.needs_handoff
        or ctx.stop_reason in ("pii_blocked", "scope_blocked")
    )
    assert terminal, "pipeline must reach a terminal state"
    # No hallucinated dispatch — only exposed tools were ever executed.
    for name, _ in dispatcher.executed:
        assert name in VALID_TOOLS


@pytest.mark.asyncio
async def test_e2e_action_turn_full_pipeline():
    if not await is_ollama_available():
        pytest.skip("Local Ollama server (http://localhost:11434) is not running.")
    gen, text, embedder = _backends()
    pipe = ReferencePipeline(prompts_dir=PROMPTS_DIR, embedder=embedder)
    disp = InMemoryDispatcher()

    ctx = await pipe.run_turn(
        PipelineContext(user_input="record an expense of 50 BRL for lunch"),
        gen_backend=gen, ego_backend=text, dispatcher=disp, **KW)

    _assert_pipeline_invariants(ctx, disp)
    # An action request routes to the EGO (the tool gateway).
    assert ctx.id_result.triad_route == "EGO"
    assert ctx.total_tokens > 0


@pytest.mark.asyncio
async def test_e2e_social_turn_full_pipeline():
    if not await is_ollama_available():
        pytest.skip("Local Ollama server (http://localhost:11434) is not running.")
    gen, text, embedder = _backends()
    pipe = ReferencePipeline(prompts_dir=PROMPTS_DIR, embedder=embedder)
    disp = InMemoryDispatcher()

    ctx = await pipe.run_turn(
        PipelineContext(user_input="thank you so much, that's all for now!"),
        gen_backend=gen, ego_backend=text, dispatcher=disp, **KW)

    _assert_pipeline_invariants(ctx, disp)
    # Pure social chat is voiced directly — the EGO must not run, no tool fires.
    assert ctx.ego_result is None
    assert disp.executed == []
    assert ctx.superego_result.response
