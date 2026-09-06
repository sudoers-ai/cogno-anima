"""The provider's prompt cache reaches ``StageMetrics`` — the number that was worth money
and that nothing carried.

Measured live 2026-09-03: a second call with the same prefix reported **2432 cached of 2625**
prompt tokens (92.6%). ``cogno-synapse`` now reports it; these pin that every stage that runs a
model records it, that it is a SUBSET (never added to ``tokens_total``), and — the twin that
matters most — that a backend which reports nothing produces exactly the numbers it produced
before.
"""

import json

import pytest

from cogno_anima.stages.ego import EgoStage
from cogno_anima.stages.ner import IntentAnalyzer
from cogno_anima.stages.noumeno import Noumeno
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.types import (
    EgoResult, EgoStep, IntentResult, NoumenoResult, PipelineContext, StageMetrics,
    ToolExecution, ToolResult,
)

_NOUMENO_JSON = ('{"rewritten": "Hello.", "context_turn": "", "confidence": 0.95, '
                 '"changed": false, "preserved_terms": [], "rewrite_warnings": []}')
_NER_JSON = ('{"intent_class": "SOCIAL", "sentiment": "NEUTRAL", "confidence": 0.9, '
             '"temporal_class": "TIMELESS", "triad_signal": "SUPEREGO", "goal": "greet", '
             '"domains": [], "mandatory_tags": [], "pii": []}')


class CachingBackend:
    """A backend that reports a cached-prompt count, the way ``OpenAIBackend`` does.

    ``last_cached_tokens`` is per CALL and is what ``cogno_synapse.cached_tokens_of`` reads.
    ``cached=None`` models a backend that reports nothing at all (Ollama, a stub, the
    distilled student) — the attribute is simply absent.
    """

    def __init__(self, responses, *, cached=None, model="stub-cache", ti=100, to=10):
        self.responses = list(responses)
        self.model = model
        self.ti = ti
        self.to = to
        self._cached = list(cached) if isinstance(cached, list) else cached
        self.calls = 0

    def _next_cached(self):
        if self._cached is None:
            return
        value = self._cached.pop(0) if isinstance(self._cached, list) else self._cached
        self.last_cached_tokens = value

    async def generate(self, system, prompt):
        self.calls += 1
        self._next_cached()
        return (self.responses.pop(0) if self.responses else ""), self.ti, self.to

    async def chat_with_tools(self, messages, tools, tool_choice=None):
        self.calls += 1
        self._next_cached()
        turn = self.responses.pop(0)
        return turn, self.ti, self.to

    def supports_native_tools(self):
        return True


class _Embedder:
    async def embed(self, text):
        return [1.0, 0.0, 0.0]

    async def similarity(self, a, b):
        return 1.0 if a == b else 0.9


def _m(stage="x"):
    return StageMetrics(stage=stage, elapsed_ms=0.0, tokens_in=0, tokens_out=0, model="t")


def _ctx(user="hello"):
    noumeno = NoumenoResult(
        original=user, rewritten=user, context_turn="", language="en", drift_score=0.0,
        drift_tag="PASS_THROUGH", changed=False, confidence=0.9, change_subject=False,
        subject_similarity=1.0, context_used=False, preserved_terms=[], rewrite_warnings=[],
        metrics=_m("noumeno"))
    intent = IntentResult(
        intent_class="ACTION_REQUEST", sentiment="NEUTRAL", confidence=0.9,
        temporal_class="TIMELESS", triad_signal="EGO", goal="record income",
        domains=["FINANCE"], entities_objects=["income"], metrics=_m("ner"))
    return PipelineContext(user_input=user, noumeno=noumeno, intent=intent)


class _Dispatcher:
    def __init__(self, *names):
        self._schema = [{"type": "function",
                         "function": {"name": n, "description": n, "parameters": {}}}
                        for n in names]
        self.executed = []

    def tools_schema(self):
        return self._schema

    async def execute(self, name, arguments):
        self.executed.append(name)
        return ToolResult(output=f"{name} ok", side_effect=False)


def _tool_turn(name):
    return {"content": "", "tool_calls": [{
        "id": f"c_{name}", "type": "function",
        "function": {"name": name, "arguments": json.dumps({})}}]}


# ── the shape of the field itself ─────────────────────────────────────────────────────────

def test_cached_tokens_is_a_subset_and_never_swells_the_total():
    """It is already inside ``tokens_in``. Adding it to ``tokens_total`` would double-count the
    same tokens in the allowance, in the rollup and on every dashboard that reads them."""
    plain = StageMetrics(stage="ego", elapsed_ms=1.0, tokens_in=2625, tokens_out=100,
                         model="gpt-4o-mini")
    cached = StageMetrics(stage="ego", elapsed_ms=1.0, tokens_in=2625, tokens_out=100,
                          cached_tokens=2432, model="gpt-4o-mini")
    assert plain.tokens_total == cached.tokens_total == 2725
    assert cached.cached_tokens == 2432 and plain.cached_tokens == 0


# ── every stage that runs a model records it ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_noumeno_records_what_the_backend_reported():
    ctx = PipelineContext(user_input="hello")
    await Noumeno(_Embedder(), default_language="en").process(
        ctx, CachingBackend([_NOUMENO_JSON], cached=1800))
    assert ctx.noumeno.metrics.cached_tokens == 1800


@pytest.mark.asyncio
async def test_ner_records_what_the_backend_reported():
    ctx = PipelineContext(user_input="hello")
    await Noumeno(_Embedder(), default_language="en").process(
        ctx, CachingBackend([_NOUMENO_JSON]))
    await IntentAnalyzer().process(ctx, CachingBackend([_NER_JSON], cached=900))
    assert ctx.intent.metrics.cached_tokens == 900


@pytest.mark.asyncio
async def test_the_ego_sums_the_cache_across_every_step_of_the_loop():
    """This loop is where the money is: every step after the first re-sends the same system
    prompt and tool schemas. One read at the end would describe the last call while the tokens
    describe all of them."""
    backend = CachingBackend([_tool_turn("book"), {"content": "done"}], cached=[500, 2432])
    ctx = _ctx()
    await EgoStage().process(ctx, backend, _Dispatcher("book"), system_prompt="do it")
    assert backend.calls == 2
    assert ctx.ego_result.metrics.tokens_in == 200        # 2 steps x 100
    assert ctx.ego_result.metrics.cached_tokens == 2932   # 500 + 2432, summed per step


@pytest.mark.asyncio
async def test_the_ego_sums_it_on_the_TEXT_fallback_path_too():
    """A plain ``LLMBackend`` takes the ``<TOOL_CALL>`` path. Recording it on only one of the
    two branches would meter half the deployments."""

    class _TextOnly(CachingBackend):
        def supports_native_tools(self):
            return False

        async def generate(self, system, prompt):
            self.calls += 1
            self._next_cached()
            turn = self.responses.pop(0)
            text = turn.get("content", "") or ""
            for tc in turn.get("tool_calls", []):
                text += ('\n<TOOL_CALL>{"tool": "%s", "args": {}}</TOOL_CALL>'
                         % tc["function"]["name"])
            return text, self.ti, self.to

    backend = _TextOnly([_tool_turn("book"), {"content": "done"}], cached=[300, 700])
    ctx = _ctx()
    await EgoStage().process(ctx, backend, _Dispatcher("book"), system_prompt="do it")
    assert ctx.ego_result.metrics.cached_tokens == 1000


@pytest.mark.asyncio
async def test_the_superego_records_it_on_all_three_of_its_calls():
    stage = SuperegoStage()
    ctx = _ctx()
    ctx.ego_result = EgoResult(steps=[EgoStep(
        index=0, path="native", assistant_text="booked",
        tool_calls=[ToolExecution(tool="book", arguments={}, result="ok",
                                  ok=True, side_effect=True)])], metrics=_m("ego"))

    scope = await stage.check_input_scope(
        ctx, CachingBackend(['{"decision": "ALLOW"}'], cached=120), scope_prompt="only money")
    assert scope.metrics.cached_tokens == 120

    verdict = await stage.evaluate(
        ctx, CachingBackend(['{"approved": true}'], cached=340), limits_prompt="be safe")
    assert verdict.metrics.cached_tokens == 340

    voiced = await stage.voice(
        ctx, CachingBackend(["Feito."], cached=560), voice_prompt="speak")
    assert voiced.metrics.cached_tokens == 560


@pytest.mark.asyncio
async def test_a_scope_check_that_never_called_the_model_reports_no_cache():
    """The backend is SHARED between stages and turns, so its last count belongs to somebody
    else's call. An early exit that ran no model must report 0, not inherit it."""
    stage = SuperegoStage()
    ctx = _ctx()
    backend = CachingBackend([], cached=9999)
    backend.last_cached_tokens = 9999           # a previous call's number, already on the object
    result = await stage.check_input_scope(ctx, backend, scope_prompt="")   # no rules → no call
    assert backend.calls == 0
    assert result.metrics.cached_tokens == 0


# ── the twin that must not move ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_backend_that_reports_nothing_produces_the_numbers_it_always_did():
    """Ollama, a stub, the distilled student: no attribute at all. Every count is what it was
    before this field existed, and ``cached_tokens`` is 0 — which downstream means FULL price."""
    ctx = PipelineContext(user_input="hello")
    backend = CachingBackend([_NOUMENO_JSON])
    assert not hasattr(backend, "last_cached_tokens")
    await Noumeno(_Embedder(), default_language="en").process(ctx, backend)
    m = ctx.noumeno.metrics
    assert m.cached_tokens == 0
    assert (m.tokens_in, m.tokens_out) == (100, 10)
    assert m.tokens_total == 110 + m.embedding_tokens

    ego_ctx = _ctx()
    await EgoStage().process(ego_ctx, CachingBackend([{"content": "done"}]),
                             _Dispatcher("book"), system_prompt="do it")
    assert ego_ctx.ego_result.metrics.cached_tokens == 0
    assert ego_ctx.ego_result.metrics.tokens_in == 100


# ── the resilient JSON helper sums across attempts, like the tokens beside it ──────────────

@pytest.mark.asyncio
async def test_the_truncation_retry_sums_the_cache_it_spent_on_both_attempts():
    """A retry re-sends the same prefix, so it is precisely the attempt most likely to be
    cached. Reading the number once at the end would describe one call while the tokens
    describe two."""
    from cogno_anima.utils import generate_json_resilient

    backend = CachingBackend(['{"a": 1', '{"a": 1}'], cached=[400, 2000])
    data, tin, tout, cached = await generate_json_resilient(
        backend, "sys", "prompt", json.loads, stage="noumeno")
    assert data == {"a": 1} and backend.calls == 2
    assert (tin, tout) == (200, 20)
    assert cached == 2400
