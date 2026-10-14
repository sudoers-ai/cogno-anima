"""The fifteen log lines that carried somebody else's text — what they say now, at run time.

Read by AST on ``32803f5``, 15 of the package's 76 log calls formatted text somebody else
wrote: the judge's critique, the NOUMENO's rewrite, the tail of a reply that arrived cut, an
exception's message, an id the model put in a call, a tool name it invented, trait values the
tenant typed. ``test_logs_carry_no_third_party_text.py`` holds the CLASS, by AST. This file
drives each of those lines with a sentinel in that text:

* the line still comes out — one per occurrence, at its level, with its new fields (compared
  whole, so a field that goes missing fails too);
* no record of the package, at any level, holds a byte of the sentinel.

It imports nothing the change added, so it runs on the tree before it. There 15 of its tests
were red because the sentinel was in the log, and a sixteenth — the control that a tool of
the catalogue is still named — because the line it reads had no event name yet.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

import cogno_anima
from cogno_anima import metakeys as mk
from cogno_anima.stages import superego as superego_module
from cogno_anima.stages.ego import EgoStage, _as_streak
from cogno_anima.stages.ner import IntentAnalyzer
from cogno_anima.stages.noumeno import Noumeno
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.tools.id_provenance import IdProvenanceDispatcher
from cogno_anima.types import PipelineContext, ToolResult
from cogno_anima.utils import generate_json_resilient
from tests.conftest import StubBackend, StubEmbedder
from tests.unit.test_ego import (SYS, ScriptedToolCallingBackend, StubDispatcher, _tool_turn)
from tests.unit.test_ego import _ctx as ego_ctx
from tests.unit.test_ner import PERFECT_JSON, make_noumeno_result
from tests.unit.test_superego import ScriptedBackend
from tests.unit.test_superego import _ctx as superego_ctx

PROMPTS_DIR = Path(cogno_anima.__file__).parent / "prompt_templates"

SENTINEL = "ZQX7"                      # in every third-party text below, and in nothing else
WHO = "Odile Varnes"
MAIL = "odile.varnes@example.com"


class Captured:
    """Every record the package emitted, at every level, formatted as a handler would."""

    def __init__(self, caplog):
        self._caplog = caplog

    @property
    def lines(self) -> "list[str]":
        return [r.getMessage() + (f"\n{r.exc_text or ''}" if r.exc_info else "")
                for r in self._caplog.records if r.name.startswith("cogno_anima")]

    def of(self, event: str) -> "list[str]":
        return [ln for ln in self.lines if f"event={event} " in f"{ln} "]

    def clean(self, *needles: str) -> None:
        for needle in needles:
            leaked = [ln for ln in self.lines if needle.casefold() in ln.casefold()]
            assert not leaked, f"{needle!r} reached the log: {leaked}"


@pytest.fixture
def log(caplog):
    for gate in (superego_module._WARNED_TRAITS, superego_module._WARNED_CONTACT_STATE):
        gate.reset()
    with caplog.at_level(logging.DEBUG, logger="cogno_anima"):
        yield Captured(caplog)


def _rewrite(rewritten: str) -> str:
    return json.dumps({"rewritten": rewritten, "context_turn": "", "confidence": 0.95,
                       "changed": False, "preserved_terms": [], "rewrite_warnings": []})


class Replies(StubBackend):
    """One scripted reply per call; an exception in the script is raised."""

    def __init__(self, replies):
        super().__init__()
        self.replies = list(replies)

    async def generate(self, system, prompt):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply, self.tokens_in, self.tokens_out


def _noumeno(embedder=None) -> Noumeno:
    return Noumeno(embedder=embedder or StubEmbedder(), prompts_dir=PROMPTS_DIR)


#: An example output of the shipped NOUMENO prompt — the echo backstop's own material.
ECHO = "Yes, I would like to know how to repot the fern."


async def test_the_judges_rejection_logs_how_it_rejected_and_not_what_it_wrote(log):
    """The line of the brief. One line per rejection, with the branch, the read, the attempt,
    the critique's length and the reply's repair marks — and not a byte of the critique."""
    critique = f"{WHO} <{MAIL}> was charged 1.250,00 twice {SENTINEL}"
    ctx = superego_ctx()
    ctx.ego_result.attempt = 2
    result = await SuperegoStage().evaluate(
        ctx, ScriptedBackend([json.dumps({"approved": False, "critique": critique})]),
        limits_prompt="")
    assert result.approved is False and result.critique == critique     # the words survive
    log.clean(SENTINEL, WHO, "Odile", "Varnes", MAIL, "1.250,00", "charged")
    assert log.of("judge") == [
        f"stage=superego event=judge approved=false branch=execution read=boolean attempt=2 "
        f"critique_len={len(critique)} repairs=-"]
    assert [r.levelno for r in log._caplog.records if "event=judge " in r.getMessage()] == [
        logging.WARNING]


async def test_the_rejection_line_is_one_per_rejection_and_carries_the_marks(log):
    stage, ctx = SuperegoStage(), superego_ctx()
    await stage.evaluate(ctx, ScriptedBackend(['{"approved": false}']), limits_prompt="")
    await stage.evaluate(ctx, ScriptedBackend(
        ['```json\n{"approved": false, "critique": "' + SENTINEL + '"}\n```']), limits_prompt="")
    await stage.evaluate(ctx, ScriptedBackend(['{"approved": true}']), limits_prompt="")
    log.clean(SENTINEL)
    assert log.of("judge") == [
        "stage=superego event=judge approved=false branch=execution read=boolean attempt=1 "
        "critique_len=18 repairs=default:critique",            # len("execution rejected")
        f"stage=superego event=judge approved=false branch=execution read=boolean attempt=1 "
        f"critique_len={len(SENTINEL)} repairs=fence_or_think_stripped",
        "stage=superego event=judge approved=true branch=execution"]


async def test_a_judge_call_that_raises_logs_the_class_of_the_error(log):
    result = await SuperegoStage().evaluate(
        superego_ctx(), Replies([RuntimeError(f"400: prompt quoted back {SENTINEL} {MAIL}")]),
        limits_prompt="")
    assert result.approved is False
    log.clean(SENTINEL, MAIL)
    assert log.of("judge_failed") == [
        "stage=superego event=judge_failed error=RuntimeError — not approving (fail-closed)"]


async def test_a_scope_call_that_raises_logs_the_class_of_the_error(log):
    result = await SuperegoStage().check_input_scope(
        superego_ctx(intent_class="INFORMATION_REQUEST"),
        Replies([RuntimeError(f"400: prompt quoted back {SENTINEL} {MAIL}")]),
        scope_prompt="finance")
    assert result.blocked is False
    log.clean(SENTINEL, MAIL)
    assert log.of("scope_failed") == [
        "stage=superego event=scope_failed error=RuntimeError — allowing by default (fail-open)"]


async def test_the_trait_values_the_vocabulary_does_not_have_are_counted(log):
    ctx = superego_ctx()
    ctx.metadata[mk.VOICE_TRAITS] = ["warm", f"{SENTINEL} {WHO}", "sassy"]
    ctx.metadata[mk.ACTIVE_PERSONA_ID] = "front_desk"
    assert SuperegoStage.persona_traits(ctx) == ["warm"]
    log.clean(SENTINEL, WHO, "sassy")
    assert log.of("voice_traits_dropped") == [
        "stage=superego event=voice_traits_dropped persona=front_desk dropped=2 kept=1"]


async def test_the_rewrite_that_echoes_an_example_is_measured_not_printed(log):
    """``few_shot_echo`` and, when the second call fails to parse, ``…_retry_failed``: the
    first printed the rewrite, the second printed a ``StageParseError`` — which quotes the
    reply."""
    ctx = PipelineContext(user_input="Sim")
    await _noumeno().process(ctx, Replies([_rewrite(ECHO), f"not json {SENTINEL} {MAIL}"]))
    assert ctx.noumeno.rewritten == ECHO
    log.clean(SENTINEL, MAIL, "repot the fern")
    assert log.of("few_shot_echo") == [
        f"stage=noumeno event=few_shot_echo rewritten_len={len(ECHO)} — retrying without the "
        "examples block"]
    assert log.of("few_shot_echo_retry_failed") == [
        "stage=noumeno event=few_shot_echo_retry_failed error=StageParseError — keeping the "
        "first answer, flagged"]


async def test_the_degenerate_second_rewrite_is_measured_not_printed(log):
    ctx = PipelineContext(user_input="Sim")
    await _noumeno().process(ctx, Replies([_rewrite(ECHO), _rewrite(f"{SENTINEL}.")]))
    assert ctx.noumeno.rewritten == f"{SENTINEL}."
    log.clean(SENTINEL)
    assert log.of("few_shot_echo_retry_degenerate") == [
        f"stage=noumeno event=few_shot_echo_retry_degenerate retried_len={len(SENTINEL) + 1} — "
        "adopted (honest beats fabricated), flagged"]


async def test_the_rewrite_that_copies_a_rule_illustration_is_measured_not_printed(log):
    copied = "Book with Dr. Heitor Lacerda."            # an illustration in the prompt's rules
    ctx = PipelineContext(user_input="com a Ana")
    await _noumeno().process(ctx, Replies([_rewrite(copied)]))
    log.clean("Heitor", "Lacerda")
    assert log.of("few_shot_echo_rule_illustration") == [
        f"stage=noumeno event=few_shot_echo_rule_illustration rewritten_len={len(copied)} — "
        "unsupported by input/conversation, flagged"]


async def test_an_embedder_that_raises_logs_the_class_of_the_error(log):
    class Down(StubEmbedder):
        async def similarity(self, a, b):
            raise RuntimeError(f"POST /embed body={a!r} {SENTINEL}")

    ctx = PipelineContext(user_input=f"o meu e-mail é {MAIL}", force_language="pt")
    ctx.metadata[mk.LAST_REWRITTEN] = "an earlier turn"
    await _noumeno(Down()).process(ctx, Replies([_rewrite("My e-mail is given.")]))
    assert ctx.noumeno.drift_score is None
    log.clean(SENTINEL, MAIL)
    assert log.of("embed_unavailable") and all(
        ln == "stage=noumeno event=embed_unavailable error=RuntimeError — drift is UNKNOWN for "
              "this turn (not 0.0, not 1.0); the turn continues."
        for ln in log.of("embed_unavailable"))


async def test_a_language_detector_that_raises_logs_the_class_of_the_error(log, monkeypatch):
    langdetect = pytest.importorskip("langdetect")

    def detect(text):
        raise ValueError(f"no features in {text!r} {SENTINEL}")

    monkeypatch.setattr(langdetect, "detect", detect)
    ctx = PipelineContext(user_input=f"{MAIL}")
    await _noumeno().process(ctx, Replies([_rewrite("An address.")]))
    assert ctx.noumeno.language == "und"
    log.clean(SENTINEL, MAIL)
    assert log.of("language_undetected") == [
        "stage=noumeno event=language_undetected error=ValueError — defaulting to 'und'"]


async def test_a_reply_that_arrived_cut_logs_its_length_and_not_its_tail(log):
    cut = '{"rewritten": "Send the invoice to ' + f"{WHO} {SENTINEL}"

    def parse(raw):
        return json.loads(raw)

    data, *_ = await generate_json_resilient(
        Replies([cut, '{"rewritten": "ok"}']), "system", "prompt", parse, stage="noumeno")
    assert data == {"rewritten": "ok"}
    log.clean(SENTINEL, WHO, "invoice")
    assert log.of("truncated_response") == [
        f"stage=noumeno event=truncated_response attempt=1 attempts=2 chars={len(cut)} — "
        "retrying"]


async def test_an_intent_class_outside_the_vocabulary_is_named_by_its_case(log):
    payload = dict(PERFECT_JSON, intent_class=f"{SENTINEL}_{WHO}")
    analyzer = IntentAnalyzer(backend=StubBackend(response=json.dumps(payload)),
                              prompts_dir=PROMPTS_DIR)
    result = await analyzer.analyze(make_noumeno_result())
    assert result.intent_class == "ACTION_REQUEST"              # from the SYSTEM tag
    log.clean(SENTINEL, WHO, "Odile")
    assert log.of("intent_class_fallback") == [
        "stage=ner event=intent_class_fallback llm=out_of_vocabulary coerced=ACTION_REQUEST"]


async def test_an_intent_class_the_model_left_unknown_is_named_by_its_case(log):
    payload = dict(PERFECT_JSON, intent_class="UNKNOWN")
    analyzer = IntentAnalyzer(backend=StubBackend(response=json.dumps(payload)),
                              prompts_dir=PROMPTS_DIR)
    await analyzer.analyze(make_noumeno_result())
    assert log.of("intent_class_fallback") == [
        "stage=ner event=intent_class_fallback llm=UNKNOWN coerced=ACTION_REQUEST"]


async def test_a_tool_the_model_invented_is_counted_and_not_named(log):
    invented = f"transfer_to_{SENTINEL}_odile_varnes"
    backend = ScriptedToolCallingBackend([_tool_turn(invented, {}), {"content": "I cannot."}])
    ctx = ego_ctx()
    await EgoStage().process(ctx, backend, StubDispatcher.with_tools("add_income"),
                             system_prompt=SYS)
    assert ctx.ego_result.tools_executed[0].tool == invented     # the trace keeps the name
    log.clean(SENTINEL, "odile", "varnes")
    assert log.of("unknown_tool") == [
        f"stage=ego event=unknown_tool step=0 name_len={len(invented)}"]


async def test_a_tool_of_the_catalogue_is_still_named(log):
    """The control of the test above: a name the HOST declared is not text of the model's."""
    backend = ScriptedToolCallingBackend([_tool_turn("add_income", {"amount": 40}),
                                          {"content": "Recorded."}])
    await EgoStage().process(ego_ctx(), backend, StubDispatcher.with_tools("add_income"),
                             system_prompt=SYS)
    assert log.of("step") == ["stage=ego event=step step=0 tool=add_income ok=True "
                              "side_effect=False"]
    assert log.of("start") and log.of("done")


async def test_a_circling_streak_that_is_not_a_count_logs_its_type(log):
    assert _as_streak({"note": f"{SENTINEL} {WHO}"}) == 0
    log.clean(SENTINEL, WHO)
    assert log.of("circling_streak_unusable") == [
        "stage=ego event=circling_streak_unusable type=dict — hint skipped"]


async def test_an_id_no_read_returned_logs_its_length_and_the_arguments_name(log):
    class Source:
        def tools_schema(self):
            return []

        async def execute(self, name, arguments):
            return ToolResult(output="never reached")

    stale = f"appt-{SENTINEL}-0042"
    guard = IdProvenanceDispatcher(Source(), guarded={"cancel_appointment": "appointment_id"})
    result = await guard.execute("cancel_appointment", {"appointment_id": stale})
    assert result.ok is False and stale in result.error          # the executor is still told
    log.clean(SENTINEL, "0042")
    assert log.of("id_provenance_refused") == [
        f"event=id_provenance_refused tool=cancel_appointment arg=appointment_id "
        f"id_len={len(stale)}"]


def test_control_the_capture_sees_a_line_that_carries_the_sentinel(log):
    """``clean`` over an empty capture passes anything. This is the line as it was."""
    logging.getLogger("cogno_anima.superego").warning(
        "stage=superego event=judge approved=false branch=%s critique=%s", "execution",
        f"{WHO} {SENTINEL}"[:80])
    with pytest.raises(AssertionError, match="reached the log"):
        log.clean(SENTINEL)
    with pytest.raises(AssertionError, match="reached the log"):
        log.clean("odile")                                     # folded, like a reader's grep
    assert len(log.of("judge")) == 1
