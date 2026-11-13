"""A write made with another VALUE than the one the user stated is shown to the judge as such.

## The measured turn (a scheduling demo; anonymised — no person, tenant or identity below)

The contact had moved the hour twice («às 10?» → «às 14» → «às 15») and answered «às 15». The
executor called ``book_appointment`` with the time ``14:00`` — the hour of an earlier turn — and
the judge APPROVED: criterion #1 asks "did it do exactly what was asked (X, not Y)", and nothing in
the prompt put the two values side by side. The draft reported 14:00, i.e. it was truthful about
the wrong action, so nothing else in the prompt looked wrong either.

## What is pinned here

* THE TWIN. The demo's shape renders a paragraph, after the criteria, naming the write, the
  value the user stated (``15:00``) and the value the call used (``14:00``), with the rule.
  On the base tree this file fails by ASSERTION, never by ImportError: nothing new is imported.
* THE CONTROLS, byte for byte. Every rendering with no computed mismatch is the prompt of the base
  tree — whole-prompt digests (system + user half) taken on ``f9f10d7`` with these fixtures: the
  same hour, two hours in the request, no hour in the request, the hour on a READ, a write with no
  hour in it, a message text carrying the same hour, the same date, across the branches, with and
  without limits and context.
* WHAT THE CODE DOES NOT DECIDE. «remind me one hour before the 15:00 appointment» renders the
  paragraph too: the code cannot tell a derived value from a wrong one, so the model reads the
  exception in the rule. That half is `tests/integration/test_superego.py` (cloud, n≥3).
* MUTATIONS: the splice removed → the twins go red, the controls stay green; ``tool_mutating``
  dropped from the write test → the failed-write twin goes red.
"""

from __future__ import annotations

import hashlib

import pytest

from cogno_anima import metakeys as mk
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.types import (EgoResult, EgoStep, IntentResult, PipelineContext, StageMetrics,
                               ToolExecution)
from tests.unit._judge_0115 import as_0114

_OPENING = "A WRITE USED A VALUE THE USER DID NOT STATE"


def _m(stage: str) -> StageMetrics:
    return StageMetrics(stage=stage, elapsed_ms=1.0, tokens_in=1, tokens_out=1, model="t")


def _intent(cls: str = "ACTION_REQUEST") -> IntentResult:
    return IntentResult(intent_class=cls, sentiment="NEUTRAL", confidence=0.9,
                        temporal_class="RECENT", triad_signal="EGO",
                        goal="book the appointment at the time the user chose", metrics=_m("ner"))


def _book(time: str = "14:00", date: str = "2026-11-11", *, ok: bool = True,
          side_effect: bool = True, mutating=None, tool: str = "book_appointment") -> ToolExecution:
    return ToolExecution(tool=tool, arguments={"host_id": "dr_quostral", "date": date,
                                               "time": time},
                         result=(f"Booked {date} {time}." if ok else ""),
                         error=(None if ok else "slot not available"), ok=ok,
                         side_effect=side_effect, tool_mutating=mutating)


def _ctx(user: str, calls, *, draft: str = "Agendado.", conversational: bool = False,
         context: str = "", cls: str = "ACTION_REQUEST") -> PipelineContext:
    ctx = PipelineContext(user_input=user)
    ctx.intent = _intent(cls)
    ctx.ego_result = EgoResult(steps=[EgoStep(index=0, path="native", tool_calls=list(calls)),
                                      EgoStep(index=1, path="native", assistant_text=draft)],
                               metrics=_m("ego"))
    if conversational:
        ctx.metadata[mk.JUDGE_CONVERSATIONAL] = True
    if context:
        ctx.metadata[mk.EGO_CONTEXT] = context
    return ctx


def _sent(ctx: PipelineContext, limits: str = "") -> str:
    """The two messages the judge is sent, as one text."""
    return (SuperegoStage._judge_system(ctx) + "\n\x00\n"
            + SuperegoStage()._build_judge_prompt(ctx, limits))


def _sent_0115(ctx: PipelineContext, limits: str = "") -> str:
    """…with 0.1.15's deliberate changes undone (`_judge_0115.as_0114`) — the digests below were
    taken on the base, before them."""
    return as_0114(_sent(ctx, limits))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ── the twins ─────────────────────────────────────────────────────────────────────────────

def test_the_demo_shape_puts_the_two_hours_side_by_side():
    ctx = _ctx("pode ser às 15", [_book("14:00")], draft="Agendado para amanhã às 14h.")
    prompt = _sent(ctx)
    assert _OPENING in prompt
    line = next(x for x in prompt.splitlines() if x.startswith("- book_appointment: "))
    assert "15:00" in line and "14:00" in line
    # After the criteria and before TRUST THE TOOLS: it is a rule about the criteria.
    assert prompt.index("# Judge the EXECUTION") < prompt.index(_OPENING) \
        < prompt.index("TRUST THE TOOLS")
    rule = prompt[prompt.index(_OPENING):prompt.index("TRUST THE TOOLS")]
    assert "REJECT it" in rule
    # A PERSON is not compared, so the rule names none: a name's FORM is the 0.1.14 rule's
    # (`_PERSON_NAME_IS_A_VALUE`), and "or person" here would invite a rejection by it.
    assert "person" not in rule.lower()


def test_a_failed_write_with_the_other_hour_is_shown_too():
    """A write attempted with the wrong value is the wrong action whether or not the server
    took it — and a dispatcher may stamp only ``tool_mutating`` on a failure."""
    call = _book("14:00", ok=False, side_effect=False, mutating=True)
    prompt = _sent(_ctx("às 15 por favor", [call], draft="14h não está disponível."))
    assert _OPENING in prompt


def test_another_date_than_the_one_stated_is_shown():
    prompt = _sent(_ctx("marca dia 12 às 15", [_book("15:00", "2026-11-13")]))
    line = next(x for x in prompt.splitlines() if x.startswith("- book_appointment: "))
    assert "12/11" not in line and "day 12" in line and "13/11" in line


def test_the_derived_value_still_renders_the_comparison():
    """The code cannot tell «one hour before the 15:00 appointment» from a wrong hour; it shows
    the comparison and the rule's exception is the model's to read (integration test)."""
    remind = ToolExecution(tool="remind_me", arguments={"at": "2026-11-11T14:00:00-03:00"},
                           result="Reminder set.", ok=True, side_effect=True)
    prompt = _sent(_ctx("me lembra 1h antes da consulta das 15", [remind]))
    assert _OPENING in prompt
    assert "derive the written one from it" in prompt


# ── the controls: byte for byte the base tree's prompt ──────────────────────────────────

_NOTIFY = ToolExecution(tool="notify_user", arguments={"to": "reception",
                                                       "text": "a consulta é às 15h"},
                        result="Sent.", ok=True, side_effect=True)
_READ = ToolExecution(tool="check_availability", arguments={"date": "2026-11-11"},
                      result="Free: 14:00, 16:00.", ok=True, side_effect=False)
_EXPENSE = ToolExecution(tool="record_expense", arguments={"amount": "50", "note": "táxi"},
                         result="Recorded 50.", ok=True, side_effect=True)

_CONTROLS = {
    "same_hour": lambda: _ctx("pode ser às 15", [_book("15:00")]),
    "same_hour_iso": lambda: _ctx("às 15", [_book("2026-11-11T15:00:00-03:00")]),
    "two_hours": lambda: _ctx("muda das 14 para as 15", [_book("14:00")]),
    "two_hours_second": lambda: _ctx("muda das 14 para as 15", [_book("15:00")]),
    "no_hour": lambda: _ctx("marca amanhã com o dr", [_book("14:00")]),
    "hour_on_a_read": lambda: _ctx("tem às 15?", [_READ], cls="INFORMATION_REQUEST"),
    "write_without_hour": lambda: _ctx("anota 50 do táxi às 15", [_EXPENSE]),
    "message_same_hour": lambda: _ctx("avisa a recepção que a consulta é às 15", [_NOTIFY]),
    "same_date": lambda: _ctx("dia 11 às 14", [_book("14:00", "2026-11-11")]),
    "conversational": lambda: _ctx("pode ser às 15", [], conversational=True),
    "with_context": lambda: _ctx("pode ser às 15", [_book("15:00")],
                                 context="[TODAY] 2026-11-10 (terça-feira) 09:12"),
}

# Taken on the base tree (`f9f10d7`), system + user half, with and without a limits slot.
# Regenerated 2026-11-12 (figures privacy scrub): the worked example in `_DERIVED_FROM_EVIDENCE`
# moved (not `conversational`, which reads no grounding constant); control — reverting it to the
# old reproduced each digest.
_BASE = {
    ('conversational', ''): '8180c9f8625b486a',
    ('conversational', 'Never book on Sundays.'): '54d4e32283f495a7',
    ('hour_on_a_read', ''): 'd950c3bf386bd56c',
    ('hour_on_a_read', 'Never book on Sundays.'): '0a421ecb1c9ca189',
    ('message_same_hour', ''): '333425a359ad6bc7',
    ('message_same_hour', 'Never book on Sundays.'): 'edc91d00a976b9d5',
    ('no_hour', ''): '32e3eed57fb99887',
    ('no_hour', 'Never book on Sundays.'): 'dba5d906a8edf358',
    ('same_date', ''): '3fce5182e2b3ecbc',
    ('same_date', 'Never book on Sundays.'): '8ff546844a17a0cb',
    ('same_hour', ''): '86315725f4006a43',
    ('same_hour', 'Never book on Sundays.'): '1bfec5350da2e324',
    ('same_hour_iso', ''): 'df560a504fda8129',
    ('same_hour_iso', 'Never book on Sundays.'): 'cd7694033ccaca76',
    ('two_hours', ''): 'dbae57da56393d05',
    ('two_hours', 'Never book on Sundays.'): '51e60151b12b36ea',
    ('two_hours_second', ''): '7f2909768a3b2ea1',
    ('two_hours_second', 'Never book on Sundays.'): 'c76ab4a7666c967c',
    ('with_context', ''): '318d6b28f6ae0773',
    ('with_context', 'Never book on Sundays.'): '0be36e7397352fa7',
    ('write_without_hour', ''): 'a402b61eb54031d0',
    ('write_without_hour', 'Never book on Sundays.'): '0e345d2057ec7c8d',
}


@pytest.mark.parametrize("name", sorted(_CONTROLS))
@pytest.mark.parametrize("limits", ["", "Never book on Sundays."])
def test_no_computed_mismatch_is_the_prompt_of_before(name, limits):
    ctx = _CONTROLS[name]()
    prompt = _sent_0115(ctx, limits)
    assert _OPENING not in prompt
    assert _sha(prompt) == _BASE[(name, limits)]
