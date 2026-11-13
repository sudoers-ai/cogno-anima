"""Nothing LISTED without a read: the voice's deterministic net for clock times.

## The measured turn (a scheduling demo; anonymised — no person, tenant or identity below)

The contact asked which times a professional had free. ``check_availability`` FAILED (the
executor passed the wrong id), the judge rejected and the correction budget ran out, and the
reply the contact got offered three times. No call of the turn had returned a time. Every rule
against it was already in the voice prompt — the executor data says "relay THIS, do NOT invent
alternatives" — and the judge's critique, quoted in the verdict section, NAMED the invented
times.

## What is pinned here

* THE TWINS, on what the contact RECEIVES (``voice().response``): a time nobody read never
  reaches it — re-voiced once with the reason, and when the second reply still states one, the
  statements carrying it are taken out (or, when nothing would be left, the times masked). On the
  base tree this file fails by ASSERTION, never by ImportError: nothing new is imported.
* THE CONTROLS: a read that returned times, a time the contact named, a time the executor asked
  about, a duration, the persona's own hours, the host's clock, a turn with no tool at all — each
  one voice call and the reply untouched.
* THE PROMPT of the first call is the base tree's, byte for byte (digests taken on ``f9f10d7``):
  the correction rides only the re-voice, appended after ``# Task`` with no header.
* MUTATIONS: the net switched off; the critique admitted as evidence; the gate dropped; the
  header rule of the floor dropped — each kills a named twin and leaves the controls green.
"""

from __future__ import annotations

import hashlib

import pytest

from cogno_anima import metakeys as mk
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.types import (EgoResult, EgoStep, IntentResult, NoumenoResult, PipelineContext,
                               StageMetrics, ToolExecution)

USER = "quais horários o Dr. Quostral tem amanhã?"
FAILED = ToolExecution(tool="check_availability", arguments={"host_id": "1", "date": "2026-11-11"},
                       result="", error="unknown host_id", ok=False, side_effect=False)
DATE = ToolExecution(tool="resolve_date", arguments={"text": "amanhã"},
                     result="2026-11-11 (quarta-feira)", ok=True, side_effect=False)
CRITIQUE = ("check_availability failed and the draft lists 09:00, 10:30 and 14:00, which no "
            "successful call returned.")
INVENTED = ("Não consegui confirmar a agenda agora. Amanhã o Dr. Quostral tem 09:00, 10:30 e "
            "14:00 livres. Quer marcar algum?")
HONEST = ("Não consegui consultar a agenda do Dr. Quostral agora. Quer que eu tente de novo, ou "
          "prefere me dizer o horário que quer?")


class Scripted:
    def __init__(self, replies):
        self.replies = list(replies)
        self.model = "stub-voice"
        self.calls: "list[dict]" = []

    async def generate(self, system, prompt):
        self.calls.append({"system": system, "prompt": prompt})
        return (self.replies.pop(0) if self.replies else ""), 5, 3


def _m(stage: str) -> StageMetrics:
    return StageMetrics(stage=stage, elapsed_ms=1.0, tokens_in=1, tokens_out=1, model="t")


def _ctx(calls=(FAILED, DATE), *, user: str = USER, rejected: bool = True,
         context: str = "", draft: str = INVENTED) -> PipelineContext:
    ctx = PipelineContext(user_input=user)
    ctx.noumeno = NoumenoResult(
        original=user, rewritten=user, context_turn="", language="pt", canonical_language="en",
        drift_score=0.0, drift_tag="PASS_THROUGH", changed=False, confidence=1.0,
        change_subject=False, subject_similarity=1.0, context_used=False, preserved_terms=[],
        rewrite_warnings=[], metrics=_m("noumeno"))
    ctx.intent = IntentResult(intent_class="INFORMATION_REQUEST", sentiment="NEUTRAL",
                              confidence=0.9, temporal_class="RECENT", triad_signal="EGO",
                              goal="list the professional's free times tomorrow",
                              metrics=_m("ner"))
    steps = [EgoStep(index=0, path="native", tool_calls=list(calls))] if calls else []
    steps.append(EgoStep(index=len(steps), path="native", assistant_text=draft))
    ctx.ego_result = EgoResult(steps=steps, metrics=_m("ego"))
    ctx.turn_executions = list(calls)
    if rejected:
        ctx.metadata[mk.VOICE_CORRECTION] = {"reason": CRITIQUE, "kind": "not_executed"}
        ctx.metadata[mk.JUDGE_VERDICT] = {"approved": False, "attempts": 1}
    else:
        ctx.metadata[mk.JUDGE_VERDICT] = {"approved": True, "attempts": 1}
    if context:
        ctx.metadata[mk.EGO_CONTEXT] = context
    return ctx


async def _voice(ctx, replies, voice_prompt: str = "Você é a secretária da clínica."):
    backend = Scripted(replies)
    result = await SuperegoStage().voice(ctx, backend, voice_prompt=voice_prompt)
    return result, backend


_TIMES = ("09:00", "10:30", "14:00", "9h", "10h30", "14h")


def _states_a_time_nobody_read(text: str) -> bool:
    return any(t in text for t in _TIMES)


# ── presence first: the measured prompt holds the times only in the CRITIQUE ──────────────

def test_the_prompt_carries_the_invented_times_only_in_the_critique():
    ctx = _ctx()
    stage = SuperegoStage()
    payload = stage._tool_payload(ctx)
    prompt = stage._build_voice_prompt(ctx, payload, [], [])
    assert "09:00" in prompt                        # the critique, quoted in the verdict
    assert "09:00" not in payload and "09:00" not in USER


# ── the twins ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_failed_read_and_invented_times_are_revoiced_once():
    result, backend = await _voice(_ctx(), [INVENTED, HONEST])
    assert result.response == HONEST
    assert len(backend.calls) == 2
    second = backend.calls[1]["prompt"]
    assert "HARD RULE — REWRITE" in second and "09:00" in second and "check_availability" in second
    assert "voice:unread_times" in result.adjustments
    assert "voice:revoiced_unread_times" in result.adjustments


@pytest.mark.asyncio
async def test_the_same_net_holds_on_an_approved_turn():
    """Not an exhaustion rule: a time nobody read is invented whoever signed the draft."""
    result, backend = await _voice(_ctx(rejected=False), [INVENTED, HONEST])
    assert result.response == HONEST and len(backend.calls) == 2


@pytest.mark.asyncio
async def test_a_second_reply_that_still_lists_them_loses_those_statements():
    stubborn = ("Não consegui confirmar a agenda agora.\n"
                "Horários que costumam estar livres:\n"
                "- 09:00\n"
                "- 10:30\n"
                "\n"
                "Quer que eu tente de novo?")
    result, _ = await _voice(_ctx(), [INVENTED, stubborn])
    assert not _states_a_time_nobody_read(result.response)
    assert result.response == "Não consegui confirmar a agenda agora.\n\nQuer que eu tente de novo?"
    assert "voice:unread_times_removed" in result.adjustments


@pytest.mark.asyncio
async def test_prose_loses_the_sentence_and_keeps_the_others():
    stubborn = "Não consegui confirmar a agenda. Tenho 09:00 ou 14h. Quer tentar outro dia?"
    result, _ = await _voice(_ctx(), [INVENTED, stubborn])
    assert result.response == "Não consegui confirmar a agenda. Quer tentar outro dia?"


@pytest.mark.asyncio
async def test_when_nothing_would_be_left_the_times_are_masked():
    only = "09:00, 10:30 ou 14:00?"
    result, _ = await _voice(_ctx(), [only, only])
    assert result.response == "[…], […] ou […]?"
    assert "voice:unread_times_masked" in result.adjustments


# ── the controls: one call, the reply untouched ────────────────────────────────────────────

_FREE = ToolExecution(tool="check_availability", arguments={"host_id": "dr_quostral",
                                                            "date": "2026-11-11"},
                      result="Free: 09:00, 10:30, 14:00", ok=True, side_effect=False)
_ASKED = ToolExecution(tool="check_availability", arguments={"host_id": "1", "time": "15:00"},
                       result="", error="unknown host_id", ok=False, side_effect=False)

_CONTROLS = {
    # a read RETURNED times: the reply works within what was read
    "read_returned_times": (lambda: _ctx((_FREE,), rejected=False),
                            "Amanhã há 09:00, 10:30 e 14:00."),
    # ...and once a read returned times, another time is the judge's to weigh, not a pattern's
    "read_returned_times_other": (lambda: _ctx((_FREE,), rejected=False),
                                  "Amanhã há 09:00 e também 16:00."),
    "contact_named_it": (lambda: _ctx(user="tem às 15 amanhã?"),
                         "Não consegui confirmar se às 15h está livre."),
    "executor_asked_it": (lambda: _ctx((_ASKED,)), "Não consegui verificar as 15:00 agora."),
    "a_duration": (lambda: _ctx(), "Não consegui ver a agenda. A consulta dura 1h."),
    "the_hosts_clock": (lambda: _ctx(context="[TODAY] 2026-11-10 (terça-feira) 09:12"),
                        "Agora são 09:12 e não consegui ver a agenda."),
    "no_tool_ran": (lambda: _ctx(calls=(), rejected=False), "Podemos falar às 16h?"),
    "no_time_at_all": (lambda: _ctx(), HONEST),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(_CONTROLS))
async def test_a_reply_whose_times_were_read_or_told_is_left_alone(name):
    build, reply = _CONTROLS[name]
    result, backend = await _voice(build(), [reply, "SHOULD NOT BE CALLED"])
    assert len(backend.calls) == 1
    assert result.response == reply
    assert not any(a.startswith("voice:unread_times") for a in result.adjustments)


@pytest.mark.asyncio
async def test_the_persona_s_own_hours_ground_a_time():
    reply = "Atendemos das 8h às 18h, mas não consegui ver a agenda agora."
    result, backend = await _voice(_ctx(), [reply, "SHOULD NOT BE CALLED"],
                                   voice_prompt="Você é a secretária. Atendemos das 8h às 18h.")
    assert len(backend.calls) == 1 and result.response == reply


# ── the first call's prompt: the base tree's bytes ───────────────────────────────────────

def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# Regenerated 2026-11-12 (figures privacy scrub): the `_DERIVED_FROM_EVIDENCE` example renders in
# every first voice prompt; control — reverting it to the old reproduced each digest.
_FIRST_PROMPT_BASE = {
    'a_duration': '7c40df5d73f8b554',
    'contact_named_it': 'f312dd75b0ac31cb',
    'executor_asked_it': '09c523985f86adae',
    'measured': '7c40df5d73f8b554',
    'no_time_at_all': '7c40df5d73f8b554',
    'no_tool_ran': 'd92ac4d4795d5312',
    'read_returned_times': 'b146475cda66b1d4',
    'read_returned_times_other': 'b146475cda66b1d4',
    'the_hosts_clock': '4f81fb04a4133879',
}


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(["measured", *_CONTROLS]))
async def test_the_first_voice_prompt_did_not_move(name):
    build = _ctx if name == "measured" else _CONTROLS[name][0]
    reply = INVENTED if name == "measured" else _CONTROLS[name][1]
    _, backend = await _voice(build(), [reply, HONEST])
    assert _sha(backend.calls[0]["system"] + "\n\x00\n" + backend.calls[0]["prompt"]) \
        == _FIRST_PROMPT_BASE[name]
