"""The per-turn switch that lets a JSON stage ask its backend to carry a schema — OFF by default.

Four things are held here, over the REAL stages and scripted replies (no model, no network):

* **OFF is byte for byte.** With ``mk.STAGE_SCHEMA`` absent — or holding anything that is not
  exactly ``"open"`` / ``"strict"`` under the stage's own label — no schema is derived
  (``schema_for`` is never called), no backend is probed, every call is
  ``backend.generate(system, prompt)`` with two positional strings, and ``schema_use`` is ``None``.
* **ON never changes a byte of a prompt.** The (system, prompt) a stage sends is the same with
  the switch off, ``open`` and ``strict``.
* **ON, the schema reaches the backend** — the stage's own, in the form asked for, under the
  stage's label — or the record says it did not: ``applied`` is the serving backend's receipt,
  ``None`` when it has no dialect for the form (never another form's dialect), ``"refused"``
  when it was asked for and did not travel.
* **The record is closed.** ``{"requested", "applied"}``, two closed alphabets, nothing else.

The last section runs the stages over the REAL ``cogno_synapse`` backends with the SDK client
replaced by a double: the request a provider would receive carries the converted schema, and its
``messages`` are the ones the switched-off call sends.
"""

from __future__ import annotations

import json

import pytest
from cogno_synapse import (SCHEMA_FORMS, SCHEMA_RECEIPTS, FallbackBackend, OpenAIBackend,
                           schema_digest, to_dialect)

from cogno_anima import metakeys as mk
from cogno_anima import schemas, structured
from cogno_anima.stages import scope_options
from cogno_anima.stages.ner import IntentAnalyzer
from cogno_anima.stages.noumeno import Noumeno
from cogno_anima.stages.proposal_judge import ProposalJudge
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.structured import (SchemaAsk, closed_schema_record, requested_form,
                                    schema_ask)
from cogno_anima.tools.pre_judge import Proposal
from cogno_anima.types import PipelineContext, StageMetrics
from cogno_anima.utils import call_json_stage, generate_json_resilient
from tests.conftest import StubEmbedder
from tests.unit.test_repairs_are_counted import _ALLOW, _APPROVE, _BLOCK, _INTENT, _REWRITE
from tests.unit.test_superego import _ctx

JSON_STAGES = ("noumeno", "ner", "superego_scope", "superego_judge")
_REPLY = {"noumeno": json.dumps(_REWRITE), "ner": json.dumps(_INTENT),
          "superego_scope": _ALLOW, "superego_judge": _APPROVE}


# ── doubles ───────────────────────────────────────────────────────────────────────────────────

class Plain:
    """A backend with NO schema support and a ``generate`` of exactly two arguments."""
    model = "plain"

    def __init__(self, reply):
        self.replies = [reply] if isinstance(reply, str) else list(reply)
        self.calls: list = []
        self.probes = 0

    def _next(self):
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]

    async def generate(self, system, prompt):
        self.calls.append(("generate", system, prompt, None))
        return self._next(), 5, 3


class Structured(Plain):
    """A leaf that announces ``dialects`` (form → dialect) and records what it was handed.
    ``receipt`` overrides what it leaves as its receipt (default: the dialect of the form)."""
    model = "structured"

    def __init__(self, reply, dialects=None, receipt="same"):
        super().__init__(reply)
        self.dialects = ({"open": "openai_json_schema", "strict": "openai_strict"}
                         if dialects is None else dialects)
        self.receipt = receipt
        self.last_schema_dialect = None

    def schema_dialect(self, form):
        self.probes += 1
        return self.dialects.get(form)

    async def generate(self, system, prompt):
        self.last_schema_dialect = None
        return await super().generate(system, prompt)

    async def generate_structured(self, system, prompt, *, schema, name, form):
        self.calls.append(("generate_structured", system, prompt,
                           {"schema": schema, "name": name, "form": form}))
        self.last_schema_dialect = self.dialects[form] if self.receipt == "same" else self.receipt
        return self._next(), 5, 3


async def run(stage: str, backend, switch=None) -> StageMetrics:
    """Run ONE real stage over ``backend`` with ``switch`` as ``mk.STAGE_SCHEMA`` (``None`` =
    the key is absent). Returns the ``StageMetrics`` of the call."""
    def stamp(ctx):
        if switch is not None:
            ctx.metadata[mk.STAGE_SCHEMA] = switch
        return ctx

    if stage == "noumeno":
        ctx = stamp(PipelineContext(user_input="Manda a fatura.", force_language="pt"))
        await Noumeno(embedder=StubEmbedder()).process(ctx, backend)
        return ctx.noumeno.metrics
    if stage == "ner":
        ctx = stamp(_ctx())
        await IntentAnalyzer().process(ctx, backend)
        return ctx.intent.metrics
    if stage == "superego_scope":
        ctx = stamp(_ctx(user="what can I study?", intent_class="INFORMATION_REQUEST"))
        result = await SuperegoStage().check_input_scope(ctx, backend,
                                                         scope_prompt="Only school things.")
        return result.metrics
    assert stage == "superego_judge"
    result = await SuperegoStage().evaluate(stamp(_ctx()), backend, limits_prompt="")
    return result.metrics


def sent(backend) -> "list[tuple[str, str]]":
    return [(system, prompt) for _, system, prompt, _ in backend.calls]


# ══ the switch ════════════════════════════════════════════════════════════════════════════════

def test_the_key_and_the_alphabets():
    assert mk.STAGE_SCHEMA == "stage_schema"
    assert set(JSON_STAGES) == set(schemas.STAGE_SCHEMAS)
    assert SCHEMA_FORMS == ("open", "strict")
    for stage in JSON_STAGES:
        for form in SCHEMA_FORMS:
            assert requested_form({mk.STAGE_SCHEMA: {stage: form}}, stage) == form


@pytest.mark.parametrize("value", [
    None, True, 1, "strict", ["ner"], ("ner", "strict"), {"ner": True}, {"ner": 1},
    {"ner": "STRICT"}, {"ner": "Open"}, {"ner": "on"}, {"ner": "yes"}, {"ner": ""},
    {"ner": " strict"}, {"ner": "closed"}, {"ner": "chain"}, {"ner": "openai_strict"},
    {"ner": ["strict"]}, {"ner": {"form": "strict"}}, {"NER": "strict"}, {"noumeno": "strict"},
    {"*": "strict"}, {"all": "strict"},
])
def test_anything_that_is_not_exactly_a_form_under_the_stages_label_is_off(value):
    """A switch that a typo could turn ON is not a switch."""
    assert requested_form({mk.STAGE_SCHEMA: value}, "ner") is None
    assert not schema_ask({mk.STAGE_SCHEMA: value}, "ner").on
    assert requested_form({mk.STAGE_SCHEMA: {"ner": "strict"}}, "ner") == "strict"   # control


@pytest.mark.parametrize("metadata", [None, {}, [], "x", {"other": {"ner": "strict"}}])
def test_no_key_is_off_and_nothing_raises(metadata):
    assert requested_form(metadata, "ner") is None
    ask = schema_ask(metadata, "ner")
    assert not ask.on and ask.kwargs() == {"schema": None, "name": "ner", "form": None}
    assert ask.record() is None


def test_a_stage_with_no_schema_is_off_whatever_the_switch_says():
    for stage in ("superego_select", "judge_pre", "superego_voice", "ego", "", "unknown"):
        ask = schema_ask({mk.STAGE_SCHEMA: {stage: "strict"}}, stage)
        assert not ask.on and ask.schema is None and ask.form is None, stage
    assert schema_ask({mk.STAGE_SCHEMA: {"ner": "strict"}}, "ner").on            # control


def test_an_ask_that_is_on_holds_the_stages_schema_in_the_form_asked():
    for stage in JSON_STAGES:
        for form in SCHEMA_FORMS:
            ask = schema_ask({mk.STAGE_SCHEMA: {stage: form}}, stage)
            assert ask.on and ask.kwargs() == {"schema": schemas.schema_for(stage, form),
                                               "name": stage, "form": form}
    # half an ask is no ask: a schema with no form, a form with no schema
    assert not SchemaAsk("ner", {"type": "object"}, None).on
    assert not SchemaAsk("ner", None, "strict").on
    assert SchemaAsk("ner", None, "strict").kwargs() == {"schema": None, "name": "ner", "form": None}


def test_off_derives_nothing(monkeypatch):
    """With the key absent ``schema_for`` is never called."""
    calls = []
    real = schemas.schema_for
    monkeypatch.setattr(schemas, "schema_for", lambda *a, **k: calls.append(a) or real(*a, **k))
    for stage in JSON_STAGES:
        schema_ask({}, stage)
        schema_ask({mk.STAGE_SCHEMA: {stage: "STRICT"}}, stage)
    assert calls == []
    schema_ask({mk.STAGE_SCHEMA: {"ner": "strict"}}, "ner")                       # control
    assert calls == [("ner", "strict")]


# ══ OFF is byte for byte ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("stage", JSON_STAGES)
@pytest.mark.parametrize("switch", [None, {}, {"other": "strict"}, "strict", {"X": "open"}])
async def test_off_the_call_is_generate_and_nothing_is_probed_or_derived(stage, switch, monkeypatch):
    derived = []
    real = schemas.schema_for
    monkeypatch.setattr(schemas, "schema_for", lambda *a, **k: derived.append(a) or real(*a, **k))
    if isinstance(switch, dict) and "X" in switch:      # a row — of ANOTHER stage
        switch = {s: "open" for s in JSON_STAGES if s != stage}
    backend = Structured(_REPLY[stage])
    metrics = await run(stage, backend, switch)
    assert backend.calls and all(kind == "generate" for kind, *_ in backend.calls)
    assert backend.probes == 0, "the backend was probed with the switch off"
    assert derived == [], "a schema was derived with the switch off"
    assert metrics.schema_use is None
    assert metrics.stage == stage


@pytest.mark.parametrize("stage", JSON_STAGES)
async def test_off_a_backend_of_two_arguments_is_called_with_two_arguments(stage):
    """The dead turn this design exists to make impossible: a keyword into a two-argument
    ``generate``. ``Plain.generate`` takes exactly ``(system, prompt)``."""
    for switch in (None, {stage: "open"}, {stage: "strict"}):
        backend = Plain(_REPLY[stage])
        await run(stage, backend, switch)
        assert backend.calls and all(kind == "generate" for kind, *_ in backend.calls), switch


# ══ ON never changes a byte of a prompt ═══════════════════════════════════════════════════════

@pytest.mark.parametrize("stage", JSON_STAGES)
async def test_the_prompt_is_the_same_bytes_off_open_and_strict(stage):
    off = Structured(_REPLY[stage])
    await run(stage, off, None)
    for form in SCHEMA_FORMS:
        on = Structured(_REPLY[stage])
        await run(stage, on, {stage: form})
        assert [kind for kind, *_ in on.calls] == ["generate_structured"] * len(on.calls), form
        assert sent(on) == sent(off), f"{stage}: the prompt moved with the switch {form}"
    assert sent(off) and all(isinstance(s, str) and isinstance(p, str) for s, p in sent(off))


async def test_the_guards_prompt_digest_does_not_move_with_the_switch():
    """The digest a trace keeps of the guard's WHOLE prompt is over the two strings sent — which
    do not change — so it is the same label with the switch off and on."""
    shas = []
    for switch in (None, {"superego_scope": "open"}, {"superego_scope": "strict"}):
        ctx = _ctx(user="what can I study?", intent_class="INFORMATION_REQUEST")
        if switch:
            ctx.metadata[mk.STAGE_SCHEMA] = switch
        result = await SuperegoStage().check_input_scope(ctx, Structured(_ALLOW),
                                                         scope_prompt="Only school things.")
        shas.append((result.prompt_sha,
                     json.dumps(result.prompt_blocks, sort_keys=True, default=str)))
    assert shas[0][0] and len(set(shas)) == 1


# ══ ON: the schema reaches the backend, or the record says it did not ═════════════════════════

@pytest.mark.parametrize("stage", JSON_STAGES)
@pytest.mark.parametrize("form", SCHEMA_FORMS)
async def test_on_the_stage_hands_over_its_own_schema_in_the_form_asked(stage, form):
    backend = Structured(_REPLY[stage])
    metrics = await run(stage, backend, {stage: form})
    handed = [extra for kind, _, _, extra in backend.calls if kind == "generate_structured"]
    assert handed, "nothing was handed over"
    for extra in handed:
        assert extra == {"schema": schemas.schema_for(stage, form), "name": stage, "form": form}
    assert metrics.schema_use == {"requested": form, "applied": backend.dialects[form]}


@pytest.mark.parametrize("stage", JSON_STAGES)
async def test_a_backend_with_no_schema_support_records_applied_none(stage):
    """``not offered``: the stage was asked, the backend has nothing to carry it with."""
    metrics = await run(stage, Plain(_REPLY[stage]), {stage: "strict"})
    assert metrics.schema_use == {"requested": "strict", "applied": None}
    # control: the same turn with the switch off records NOTHING — a different fact
    assert (await run(stage, Plain(_REPLY[stage]), None)).schema_use is None


@pytest.mark.parametrize("stage", JSON_STAGES)
async def test_strict_asked_of_a_backend_that_only_has_open_is_never_recorded_as_open(stage):
    """The twin the design names: applied ``None`` — never the open dialect."""
    open_only = Structured(_REPLY[stage], dialects={"open": "ollama_format"})
    metrics = await run(stage, open_only, {stage: "strict"})
    assert metrics.schema_use == {"requested": "strict", "applied": None}
    assert all(kind == "generate" for kind, *_ in open_only.calls)
    # control: asked for open, the same backend carries it
    again = Structured(_REPLY[stage], dialects={"open": "ollama_format"})
    assert (await run(stage, again, {stage: "open"})).schema_use == {
        "requested": "open", "applied": "ollama_format"}


@pytest.mark.parametrize("stage", JSON_STAGES)
async def test_a_refused_schema_is_recorded_as_refused(stage):
    metrics = await run(stage, Structured(_REPLY[stage], receipt="refused"), {stage: "strict"})
    assert metrics.schema_use == {"requested": "strict", "applied": "refused"}


async def test_a_receipt_outside_the_alphabet_never_reaches_the_record():
    for junk in ("OPENAI_STRICT", "chain", "yes", "", 7):
        metrics = await run("ner", Structured(_REPLY["ner"], receipt=junk), {"ner": "open"})
        assert metrics.schema_use == {"requested": "open", "applied": None}, junk


async def test_one_stage_switched_on_leaves_the_others_as_they_were():
    for stage in JSON_STAGES:
        other = next(s for s in JSON_STAGES if s != stage)
        backend = Structured(_REPLY[stage])
        metrics = await run(stage, backend, {other: "strict"})
        assert metrics.schema_use is None and backend.probes == 0, (stage, other)
        assert all(kind == "generate" for kind, *_ in backend.calls)


# ── the paths of a stage that make no call, or whose call raised ──────────────────────────────

async def test_a_guard_bypass_and_a_guard_whose_call_raised_record_nothing():
    switch = {"superego_scope": "strict"}
    social = _ctx(user="oi", intent_class="SOCIAL")
    social.metadata[mk.STAGE_SCHEMA] = switch
    backend = Structured(_ALLOW)
    result = await SuperegoStage().check_input_scope(social, backend, scope_prompt="Only school.")
    assert backend.calls == [] and result.metrics.schema_use is None          # the bypass

    class Raising(Structured):
        async def generate_structured(self, *a, **k):
            raise RuntimeError("boom")

    ctx = _ctx(user="what can I study?", intent_class="INFORMATION_REQUEST")
    ctx.metadata[mk.STAGE_SCHEMA] = switch
    result = await SuperegoStage().check_input_scope(ctx, Raising(_ALLOW), scope_prompt="Only school.")
    assert result.blocked is False and result.metrics.schema_use is None      # fail-OPEN, no claim
    # control: the same turn with a call that completes does record
    ctx = _ctx(user="what can I study?", intent_class="INFORMATION_REQUEST")
    ctx.metadata[mk.STAGE_SCHEMA] = switch
    result = await SuperegoStage().check_input_scope(ctx, Structured(_BLOCK), scope_prompt="Only school.")
    assert result.blocked is True
    assert result.metrics.schema_use == {"requested": "strict", "applied": "openai_strict"}


async def test_a_judge_with_nothing_to_judge_and_a_judge_whose_call_raised_record_nothing():
    switch = {"superego_judge": "open"}
    idle = _ctx(with_ego=False)
    idle.metadata[mk.STAGE_SCHEMA] = switch
    backend = Structured(_APPROVE)
    result = await SuperegoStage().evaluate(idle, backend, limits_prompt="")
    assert backend.calls == [] and result.metrics.schema_use is None

    class Raising(Structured):
        async def generate_structured(self, *a, **k):
            raise RuntimeError("boom")

    ctx = _ctx()
    ctx.metadata[mk.STAGE_SCHEMA] = switch
    result = await SuperegoStage().evaluate(ctx, Raising(_APPROVE), limits_prompt="")
    assert result.approved is False and result.metrics.schema_use is None     # fail-CLOSED, no claim


# ── several calls, one row: the last call wins ────────────────────────────────────────────────

async def test_a_truncation_retry_keeps_the_receipt_of_the_last_attempt():
    """The first attempt arrives cut and its schema travelled; the second is served without one
    (receipt ``None``). The row says what the LAST call says, the honest ``None`` included."""
    class Flaky(Structured):
        async def generate_structured(self, system, prompt, *, schema, name, form):
            text, tin, tout = await super().generate_structured(
                system, prompt, schema=schema, name=name, form=form)
            if len(self.calls) > 1:
                self.last_schema_dialect = None
            return text, tin, tout

    ask = schema_ask({mk.STAGE_SCHEMA: {"ner": "strict"}}, "ner")
    backend = Flaky(['{"intent_class": "ACTION_REQ', json.dumps(_INTENT)])
    data, tin, tout, *_ = await generate_json_resilient(
        backend, "s", "p", json.loads, stage="ner", ask=ask)
    assert data == _INTENT and (tin, tout) == (10, 6) and len(backend.calls) == 2
    assert ask.record() == {"requested": "strict", "applied": None}
    # control: one attempt, its own receipt
    ask = schema_ask({mk.STAGE_SCHEMA: {"ner": "strict"}}, "ner")
    await generate_json_resilient(Structured(json.dumps(_INTENT)), "s", "p", json.loads,
                                  stage="ner", ask=ask)
    assert ask.record() == {"requested": "strict", "applied": "openai_strict"}


async def test_the_noumenos_second_reading_owns_the_row_only_when_it_is_adopted():
    """The NOUMENO may read twice (a rewrite that echoes one of the prompt's own examples is read
    again without them). When the second reading is adopted its tokens and its fingerprint are
    the row's, and so is its schema receipt. When it FAILS nothing of it is kept — the row goes
    on describing the first call, its receipt included."""
    from tests.unit.test_noumeno import TestFewShotEchoBackstop, _resp, make_noumeno
    echo = TestFewShotEchoBackstop.ECHO

    class TwoReceipts(Structured):
        """The first call's schema travels strict; the second call's is refused."""
        async def generate_structured(self, system, prompt, *, schema, name, form):
            result = await super().generate_structured(system, prompt, schema=schema, name=name, form=form)
            if len(self.calls) > 1:
                self.last_schema_dialect = "refused"
            return result

    async def run_twice(second_reply):
        backend = TwoReceipts([_resp(echo), second_reply, second_reply])
        ctx = PipelineContext(user_input="Sim")
        ctx.metadata[mk.STAGE_SCHEMA] = {"noumeno": "strict"}
        await make_noumeno().process(ctx, backend)
        assert len(backend.calls) == 2, "the premise: the second reading ran"
        return ctx.noumeno

    adopted = await run_twice(_resp("Yes, I have lost customers due to slow replies."))
    assert adopted.rewritten == "Yes, I have lost customers due to slow replies."
    assert adopted.metrics.schema_use == {"requested": "strict", "applied": "refused"}

    failed = await run_twice("not json at all")
    assert failed.rewritten == echo and "FEW_SHOT_ECHO" in failed.rewrite_warnings
    assert failed.metrics.schema_use == {"requested": "strict", "applied": "openai_strict"}


async def test_the_resilient_helper_without_an_ask_is_the_call_it_always_was():
    backend = Structured(json.dumps(_INTENT))
    data, tin, tout, cached, fingerprint, served = await generate_json_resilient(
        backend, "s", "p", json.loads, stage="ner")
    assert data == _INTENT and backend.calls == [("generate", "s", "p", None)]
    assert backend.probes == 0


# ══ the funnel ════════════════════════════════════════════════════════════════════════════════

async def test_the_funnel_has_no_default_for_any_of_its_three_keywords():
    import inspect
    params = inspect.signature(call_json_stage).parameters
    for name in ("schema", "name", "form"):
        assert params[name].default is inspect.Parameter.empty, name
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name
    with pytest.raises(TypeError):
        await call_json_stage(Plain("{}"), "s", "p", schema={"type": "object"}, name="ner")  # type: ignore[call-arg]
    assert inspect.signature(schemas.schema_for).parameters["form"].default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        schemas.schema_for("ner")                                              # type: ignore[call-arg]


async def test_the_funnel_off_and_on():
    schema = schemas.schema_for("ner", "open")
    plain = Structured("{}")
    assert await call_json_stage(plain, "s", "p", schema=None, name="ner", form=None) == ("{}", 5, 3, None)
    assert await call_json_stage(plain, "s", "p", schema=schema, name="ner", form=None) == ("{}", 5, 3, None)
    assert await call_json_stage(plain, "s", "p", schema=None, name="ner", form="open") == ("{}", 5, 3, None)
    assert plain.probes == 0 and [kind for kind, *_ in plain.calls] == ["generate"] * 3
    on = Structured("{}")
    assert await call_json_stage(on, "s", "p", schema=schema, name="ner", form="open") == (
        "{}", 5, 3, "openai_json_schema")
    assert on.calls == [("generate_structured", "s", "p",
                         {"schema": schema, "name": "ner", "form": "open"})]
    # a form outside the alphabet is served by nothing: today's call, no receipt
    odd = Structured("{}", dialects={"closed": "openai_strict"})
    assert await call_json_stage(odd, "s", "p", schema=schema, name="ner", form="closed") == ("{}", 5, 3, None)
    assert [kind for kind, *_ in odd.calls] == ["generate"]


async def test_the_stages_with_no_schema_go_through_the_funnel_asking_for_none():
    """The selector and the pre-judge: switched on by name or not, their call is ``generate``."""
    backend = Structured('{"answers": [], "maybe": [], "asked": ""}')
    await scope_options.select_options("is there a gym?", ["Timetable"], backend)
    judge_backend = Structured('{"approved": true}')
    verdict = await ProposalJudge(judge_backend, request="send her a note")(
        Proposal(tool="send_note", arguments={"text": "hello"}))
    assert verdict.verdict == "approved"
    for b in (backend, judge_backend):
        assert [kind for kind, *_ in b.calls] == ["generate"] and b.probes == 0


# ══ the record, on the far side of a carrier ══════════════════════════════════════════════════

def test_the_record_is_two_closed_alphabets():
    for form in SCHEMA_FORMS:
        for applied in sorted(SCHEMA_RECEIPTS) + [None]:
            assert closed_schema_record({"requested": form, "applied": applied}) == {
                "requested": form, "applied": applied}


@pytest.mark.parametrize("raw", [
    None, "strict", ["strict", None], {}, {"applied": "openai_strict"},
    {"requested": "STRICT", "applied": "openai_strict"}, {"requested": True, "applied": None},
    {"requested": None, "applied": "refused"}, {"requested": ["open"], "applied": None},
])
def test_a_record_with_no_requested_form_is_no_record(raw):
    assert closed_schema_record(raw) is None


@pytest.mark.parametrize("applied", ["chain", "OPENAI_STRICT", "free text a model wrote", "", 3,
                                     True, ["refused"], {"x": 1}])
def test_an_applied_outside_the_alphabet_closes_to_none(applied):
    assert closed_schema_record({"requested": "open", "applied": applied}) == {
        "requested": "open", "applied": None}
    assert closed_schema_record({"requested": "open", "applied": "refused", "extra": "leak",
                                 "schema": {"type": "object"}}) == {
        "requested": "open", "applied": "refused"}


def test_the_field_defaults_to_none_and_is_not_a_count():
    row = StageMetrics(stage="ner", elapsed_ms=1.0, tokens_in=3, tokens_out=2, model="m")
    assert row.schema_use is None and row.tokens_total == 5
    row = StageMetrics(stage="ner", elapsed_ms=1.0, tokens_in=3, tokens_out=2, model="m",
                       schema_use={"requested": "strict", "applied": None})
    assert row.tokens_total == 5
    assert row.model_dump()["schema_use"] == {"requested": "strict", "applied": None}


def test_the_ask_records_only_a_call_that_completed():
    ask = schema_ask({mk.STAGE_SCHEMA: {"ner": "open"}}, "ner")
    assert ask.record() is None                              # asked, and no call completed
    ask.note(None)
    assert ask.record() == {"requested": "open", "applied": None}
    ask.note("ollama_format")
    assert ask.record() == {"requested": "open", "applied": "ollama_format"}
    off = schema_ask({}, "ner")
    off.note("openai_strict")                                # an OFF ask never claims a schema
    assert off.record() is None
    assert structured.REQUESTED == "requested" and structured.APPLIED == "applied"


# ══ over the REAL transport, with the SDK client replaced by a double ═════════════════════════

def _openai(reply: str, model: str = "gpt-4o-mini"):
    seen: list = []

    class _Completions:
        async def create(self, **kw):
            seen.append(kw)
            message = type("M", (), {"content": reply, "tool_calls": None, "refusal": None})()
            return type("R", (), {
                "choices": [type("C", (), {"finish_reason": "stop", "message": message})()],
                "usage": type("U", (), {"prompt_tokens": 9, "completion_tokens": 4})(),
                "system_fingerprint": "fp_x", "model": "served-x"})()

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

        async def close(self):
            return None

    backend = OpenAIBackend(model=model, api_key="sk-x", temperature=0.0)
    backend._client = _Client  # type: ignore[method-assign]
    return backend, seen


@pytest.mark.parametrize("stage", JSON_STAGES)
@pytest.mark.parametrize("form,dialect", [("open", "openai_json_schema"), ("strict", "openai_strict")])
async def test_through_a_chain_in_a_chain_the_provider_receives_the_converted_schema(stage, form, dialect):
    """The host's shape: a chain wrapping a chain wrapping the leaf. The request the provider
    would receive carries the stage's schema converted to the leaf's dialect — once — and its
    ``messages`` are byte for byte the ones the switched-off call sends."""
    off_leaf, off_seen = _openai(_REPLY[stage])
    await run(stage, FallbackBackend([FallbackBackend([off_leaf])]), None)
    leaf, seen = _openai(_REPLY[stage])
    metrics = await run(stage, FallbackBackend([FallbackBackend([leaf])]), {stage: form})
    assert metrics.schema_use == {"requested": form, "applied": dialect}
    assert len(seen) == len(off_seen) >= 1
    for with_schema, without in zip(seen, off_seen):
        assert with_schema["messages"] == without["messages"]
        assert {k: v for k, v in with_schema.items() if k != "response_format"} == \
               {k: v for k, v in without.items() if k != "response_format"}
        sent_schema = with_schema["response_format"]["json_schema"]
        assert sent_schema == {"name": stage, "strict": form == "strict",
                               "schema": to_dialect(schemas.schema_for(stage, form), dialect)}
    assert off_seen[0].get("response_format") == {"type": "json_object"}
    assert metrics.system_fingerprint == "fp_x" and metrics.served_model == "served-x"


async def test_off_through_the_real_transport_is_the_request_it_always_was():
    for stage in JSON_STAGES:
        leaf, seen = _openai(_REPLY[stage])
        metrics = await run(stage, FallbackBackend([leaf]), None)
        assert metrics.schema_use is None
        assert all(kw.get("response_format") == {"type": "json_object"} for kw in seen), stage
        assert all(set(kw) == {"model", "messages", "max_tokens", "temperature", "response_format"}
                   for kw in seen), stage


def test_the_schema_a_stage_sends_is_named_by_one_digest_per_form():
    """What a log line carries of a schema is its digest; two forms of one stage that differ
    are told apart by it, and two that do not differ are not."""
    digests = {(stage, form): schema_digest(schemas.schema_for(stage, form))
               for stage in JSON_STAGES for form in SCHEMA_FORMS}
    assert digests[("ner", "open")] != digests[("ner", "strict")]
    for stage in ("noumeno", "superego_scope", "superego_judge"):
        assert digests[(stage, "open")] == digests[(stage, "strict")], stage
