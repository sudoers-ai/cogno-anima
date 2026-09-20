"""WHO answered this call reaches ``StageMetrics`` — the question a trace could not answer.

Measured on a downstream host: the same scope-guard input classified ALLOW 4/4 and, forty-five
minutes later, BLOCK 7/7 at ``temperature=0`` — stable inside each period, across process
restarts of the same build, with the host-written prompt slot proven byte-identical by digest
and this library at one revision throughout. Nothing persisted said whether the same backend
had answered, so the question could only be argued.

``cogno_synapse.system_fingerprint_of`` / ``served_model_of`` report it per call; these pin
that every stage that runs a model records it, that a backend which reports nothing produces
``None`` and never an empty string, and — the twin that matters most — that a row never keeps
an EARLIER call's identity. A stale fingerprint is not a small error: it is a false statement
about who answered, and comparison is the only thing these values are ever used for.
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


class StampingBackend:
    """A backend that stamps the provider's per-call identifiers, like ``OpenAIBackend`` does.

    ``fingerprints``/``models`` are per CALL and are consumed in order; a ``None`` entry models
    a provider that said nothing on that call. The attributes are reset BEFORE every call for
    the same reason the real backends reset them — a request that says nothing must not leave
    the previous one's identifier behind for the next read to find.

    ``fingerprints=None`` models a backend with no such notion at all (Ollama, a stub, the
    distilled student): the attributes are never set, so ``getattr`` finds nothing.
    """

    def __init__(self, responses, *, fingerprints=None, models=None, model="stub-fp",
                 ti=100, to=10):
        self.responses = list(responses)
        self.model = model
        self.ti = ti
        self.to = to
        self._fingerprints = list(fingerprints) if fingerprints is not None else None
        self._models = list(models) if models is not None else None
        self.calls = 0
        self.seen = []                     # (system, prompt) of every call, in order

    def _stamp(self):
        if self._fingerprints is not None:
            self.last_system_fingerprint = (
                self._fingerprints.pop(0) if self._fingerprints else None)
        if self._models is not None:
            self.last_served_model = self._models.pop(0) if self._models else None

    async def generate(self, system, prompt):
        self.calls += 1
        self.seen.append((system, prompt))
        self._stamp()
        return (self.responses.pop(0) if self.responses else ""), self.ti, self.to

    async def chat_with_tools(self, messages, tools, tool_choice=None):
        self.calls += 1
        self.seen.append((messages, tools))
        self._stamp()
        return self.responses.pop(0), self.ti, self.to

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

    def tools_schema(self):
        return self._schema

    async def execute(self, name, arguments):
        return ToolResult(output=f"{name} ok", side_effect=False)


def _tool_turn(name):
    return {"content": "", "tool_calls": [{
        "id": f"c_{name}", "type": "function",
        "function": {"name": name, "arguments": json.dumps({})}}]}


def _executed_ctx():
    ctx = _ctx()
    ctx.ego_result = EgoResult(steps=[EgoStep(
        index=0, path="native", assistant_text="booked",
        tool_calls=[ToolExecution(tool="book", arguments={}, result="ok",
                                  ok=True, side_effect=True)])], metrics=_m("ego"))
    return ctx


# ── the shape of the fields themselves ────────────────────────────────────────────────────

def test_who_answered_is_not_a_count_and_never_swells_the_total():
    """They are strings, and the only operation ever performed on them is a COMPARISON."""
    plain = StageMetrics(stage="superego_scope", elapsed_ms=1.0, tokens_in=300, tokens_out=20,
                         model="gpt-4o-mini")
    stamped = StageMetrics(stage="superego_scope", elapsed_ms=1.0, tokens_in=300,
                           tokens_out=20, model="gpt-4o-mini",
                           system_fingerprint="fp_a", served_model="gpt-4o-mini-2024-07-18")
    assert plain.tokens_total == stamped.tokens_total == 320
    assert stamped.system_fingerprint == "fp_a"
    assert stamped.served_model == "gpt-4o-mini-2024-07-18"


def test_unstamped_is_None_and_not_an_empty_string():
    """``None`` is "the provider did not say"; ``""`` would compare EQUAL to another ``""``
    and assert that two calls were served by the same configuration when neither said
    anything at all."""
    m = _m()
    assert m.system_fingerprint is None
    assert m.served_model is None


# ── every stage that runs a model records it ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_noumeno_records_who_answered():
    ctx = PipelineContext(user_input="hello")
    await Noumeno(_Embedder(), default_language="en").process(
        ctx, StampingBackend([_NOUMENO_JSON], fingerprints=["fp_a"], models=["snap-1"]))
    assert ctx.noumeno.metrics.system_fingerprint == "fp_a"
    assert ctx.noumeno.metrics.served_model == "snap-1"


@pytest.mark.asyncio
async def test_ner_records_who_answered():
    ctx = PipelineContext(user_input="hello")
    await Noumeno(_Embedder(), default_language="en").process(
        ctx, StampingBackend([_NOUMENO_JSON]))
    await IntentAnalyzer().process(
        ctx, StampingBackend([_NER_JSON], fingerprints=["fp_b"], models=["snap-2"]))
    assert ctx.intent.metrics.system_fingerprint == "fp_b"
    assert ctx.intent.metrics.served_model == "snap-2"


@pytest.mark.asyncio
async def test_the_superego_records_it_on_all_three_of_its_calls():
    stage = SuperegoStage()
    ctx = _executed_ctx()

    scope = await stage.check_input_scope(
        ctx, StampingBackend(['{"blocked": false}'], fingerprints=["fp_scope"],
                             models=["snap-scope"]), scope_prompt="only money")
    assert scope.metrics.system_fingerprint == "fp_scope"
    assert scope.metrics.served_model == "snap-scope"

    verdict = await stage.evaluate(
        ctx, StampingBackend(['{"approved": true}'], fingerprints=["fp_judge"],
                             models=["snap-judge"]), limits_prompt="be safe")
    assert verdict.metrics.system_fingerprint == "fp_judge"
    assert verdict.metrics.served_model == "snap-judge"

    voiced = await stage.voice(
        ctx, StampingBackend(["Done."], fingerprints=["fp_voice"], models=["snap-voice"]),
        voice_prompt="speak")
    assert voiced.metrics.system_fingerprint == "fp_voice"
    assert voiced.metrics.served_model == "snap-voice"


@pytest.mark.asyncio
async def test_the_ego_records_the_LAST_step_of_the_loop():
    """Not a sum — there is no sum of two fingerprints. The row names the last call, and the
    correction loop already gives one row per attempt."""
    backend = StampingBackend([_tool_turn("book"), {"content": "done"}],
                              fingerprints=["fp_first", "fp_last"],
                              models=["snap-first", "snap-last"])
    ctx = _ctx()
    await EgoStage().process(ctx, backend, _Dispatcher("book"), system_prompt="do it")
    assert backend.calls == 2
    assert ctx.ego_result.metrics.system_fingerprint == "fp_last"
    assert ctx.ego_result.metrics.served_model == "snap-last"


@pytest.mark.asyncio
async def test_the_ego_records_it_on_the_TEXT_fallback_path_too():
    """A plain ``LLMBackend`` takes the ``<TOOL_CALL>`` path. Recording it on only one of the
    two branches would leave half the deployments unable to answer the question."""

    class _TextOnly(StampingBackend):
        def supports_native_tools(self):
            return False

        async def generate(self, system, prompt):
            self.calls += 1
            self.seen.append((system, prompt))
            self._stamp()
            turn = self.responses.pop(0)
            text = turn.get("content", "") or ""
            for tc in turn.get("tool_calls", []):
                text += ('\n<TOOL_CALL>{"tool": "%s", "args": {}}</TOOL_CALL>'
                         % tc["function"]["name"])
            return text, self.ti, self.to

    backend = _TextOnly([_tool_turn("book"), {"content": "done"}],
                        fingerprints=["fp_first", "fp_last"], models=["snap-1", "snap-2"])
    ctx = _ctx()
    await EgoStage().process(ctx, backend, _Dispatcher("book"), system_prompt="do it")
    assert ctx.ego_result.metrics.system_fingerprint == "fp_last"
    assert ctx.ego_result.metrics.served_model == "snap-2"


# ── the twin that matters: a row never keeps an EARLIER call's identity ────────────────────

@pytest.mark.asyncio
async def test_the_second_call_on_the_same_backend_does_not_inherit_the_first():
    """The same instance, two guard calls: the first is answered with a fingerprint, the
    second with none. The second row says ``None`` — the whole point of the field is to
    compare two calls, and a stale value makes them compare EQUAL when one said nothing."""
    stage = SuperegoStage()
    backend = StampingBackend(['{"blocked": false}', '{"blocked": false}'],
                              fingerprints=["fp_a", None], models=["snap-a", None])

    first = await stage.check_input_scope(_ctx(), backend, scope_prompt="only money")
    second = await stage.check_input_scope(_ctx(), backend, scope_prompt="only money")

    assert backend.calls == 2
    assert first.metrics.system_fingerprint == "fp_a"
    assert second.metrics.system_fingerprint is None
    assert second.metrics.served_model is None


@pytest.mark.asyncio
async def test_the_truncation_retry_lets_the_LAST_attempt_answer():
    """``generate_json_resilient`` SUMS the counts across attempts and REPLACES the identity:
    the retry is the call that produced the data, including when it reports nothing."""
    from cogno_anima.utils import generate_json_resilient

    backend = StampingBackend(['{"a": 1', '{"a": 1}'],
                              fingerprints=["fp_first", "fp_retry"],
                              models=["snap-1", "snap-2"])
    data, tin, tout, cached, fingerprint, served = await generate_json_resilient(
        backend, "sys", "prompt", json.loads, stage="noumeno")
    assert data == {"a": 1} and backend.calls == 2
    assert (tin, tout) == (200, 20)                 # counts sum...
    assert fingerprint == "fp_retry"                # ...identity does not
    assert served == "snap-2"

    silent_retry = StampingBackend(['{"a": 1', '{"a": 1}'],
                                   fingerprints=["fp_first", None], models=["snap-1", None])
    *_, fingerprint, served = await generate_json_resilient(
        silent_retry, "sys", "prompt", json.loads, stage="noumeno")
    assert fingerprint is None and served is None


@pytest.mark.asyncio
async def test_the_ego_loop_reports_None_when_the_LAST_step_said_nothing():
    backend = StampingBackend([_tool_turn("book"), {"content": "done"}],
                              fingerprints=["fp_first", None], models=["snap-1", None])
    ctx = _ctx()
    await EgoStage().process(ctx, backend, _Dispatcher("book"), system_prompt="do it")
    assert backend.calls == 2
    assert ctx.ego_result.metrics.system_fingerprint is None
    assert ctx.ego_result.metrics.served_model is None


@pytest.mark.asyncio
async def test_a_blank_echo_reads_as_None_and_not_as_a_value():
    """``cogno_synapse`` normalises a blank to ``None``, and this layer must not undo it: two
    calls both answering ``""`` told us nothing, and must not compare equal."""
    stage = SuperegoStage()
    backend = StampingBackend(['{"blocked": false}'], fingerprints=["   "], models=[""])
    result = await stage.check_input_scope(_ctx(), backend, scope_prompt="only money")
    assert backend.calls == 1
    assert result.metrics.system_fingerprint is None
    assert result.metrics.served_model is None


# ── a backend with no such notion, and a path that ran no call ────────────────────────────

@pytest.mark.asyncio
async def test_a_backend_that_reports_nothing_records_None_and_never_raises():
    """Ollama, a stub, the distilled student: the attributes do not exist at all."""
    backend = StampingBackend([_NOUMENO_JSON])
    assert not hasattr(backend, "last_system_fingerprint")
    ctx = PipelineContext(user_input="hello")
    await Noumeno(_Embedder(), default_language="en").process(ctx, backend)
    assert ctx.noumeno.metrics.system_fingerprint is None
    assert ctx.noumeno.metrics.served_model is None

    ego_ctx = _ctx()
    await EgoStage().process(ego_ctx, StampingBackend([{"content": "done"}]),
                             _Dispatcher("book"), system_prompt="do it")
    assert ego_ctx.ego_result.metrics.system_fingerprint is None


@pytest.mark.asyncio
async def test_a_scope_bypass_does_not_inherit_the_backends_last_fingerprint():
    """The backend is SHARED between stages and turns, so its last identifier belongs to
    somebody else's call. A path that awaited nothing must say ``None``, which here means
    "no call ran" — reading the attribute on that path would name a backend for a call that
    never happened."""
    stage = SuperegoStage()
    backend = StampingBackend(['{"blocked": false}'], fingerprints=["fp_a"], models=["snap-a"])

    served = await stage.check_input_scope(_ctx(), backend, scope_prompt="only money")
    assert served.metrics.system_fingerprint == "fp_a"
    assert backend.last_system_fingerprint == "fp_a"        # still on the object

    ctx = _ctx()
    ctx.intent.intent_class = "SOCIAL"                      # NER-assisted bypass
    bypassed = await stage.check_input_scope(ctx, backend, scope_prompt="only money")
    assert backend.calls == 1                               # no second call ran
    assert bypassed.metrics.system_fingerprint is None
    assert bypassed.metrics.served_model is None


@pytest.mark.asyncio
async def test_the_judge_reports_None_when_it_could_not_ask():
    """Fail-CLOSED and nothing-to-judge both ran no completed call."""
    stage = SuperegoStage()
    backend = StampingBackend(['{"approved": true}'], fingerprints=["fp_a"], models=["snap-a"])
    await stage.evaluate(_executed_ctx(), backend, limits_prompt="be safe")

    nothing = await stage.evaluate(_ctx(), backend, limits_prompt="be safe")   # no ego_result
    assert nothing.metrics.system_fingerprint is None

    class _Boom(StampingBackend):
        async def generate(self, system, prompt):
            self.calls += 1
            self._stamp()
            raise RuntimeError("provider down")

    closed = await stage.evaluate(
        _executed_ctx(), _Boom([], fingerprints=["fp_x"]), limits_prompt="be safe")
    assert closed.approved is False
    assert closed.metrics.system_fingerprint is None


@pytest.mark.asyncio
async def test_a_re_voiced_reply_is_named_by_the_SECOND_call():
    """The voice can call twice (the divergence backstop). The counts are the turn's and
    accumulate; WHO answered is the last call's and is replaced — including when the last call
    reports nothing, and whichever of the two replies the backstop then keeps."""
    from cogno_anima import metakeys as mk

    draft = "The configured rate is R$ 120,00 per hour."

    def _diverging_turn():
        ctx = _ctx(user="what is the hourly rate?")
        ctx.ego_result = EgoResult(
            steps=[EgoStep(index=0, path="native", assistant_text=draft,
                           tool_calls=[ToolExecution(tool="lookup_rules", arguments={},
                                                     result="No records found.", ok=True,
                                                     side_effect=False)])],
            metrics=_m("ego"))
        ctx.metadata[mk.JUDGE_VERDICT] = {"approved": True, "attempts": 1}
        return ctx

    replies = ["Infelizmente, não consegui encontrar informações.", draft]

    backend = StampingBackend(list(replies), fingerprints=["fp_first", "fp_second"],
                              models=["snap-1", "snap-2"], ti=50, to=5)
    out = await SuperegoStage().voice(_diverging_turn(), backend, voice_prompt="persona")
    assert backend.calls == 2, "the divergence backstop did not re-voice"
    assert out.metrics.tokens_in == 100                      # counts sum across both...
    assert out.metrics.system_fingerprint == "fp_second"     # ...identity is the last call's
    assert out.metrics.served_model == "snap-2"

    silent = StampingBackend(list(replies), fingerprints=["fp_first", None],
                             models=["snap-1", None])
    out = await SuperegoStage().voice(_diverging_turn(), silent, voice_prompt="persona")
    assert silent.calls == 2
    assert out.metrics.system_fingerprint is None
    assert out.metrics.served_model is None
