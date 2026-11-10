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
    assert "REJECT it" in prompt[prompt.index(_OPENING):prompt.index("TRUST THE TOOLS")]


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
_BASE = {
    ('conversational', ''): '8180c9f8625b486a',
    ('conversational', 'Never book on Sundays.'): '54d4e32283f495a7',
    ('hour_on_a_read', ''): '569bb481b0298d32',
    ('hour_on_a_read', 'Never book on Sundays.'): 'ed286a74fde44ebe',
    ('message_same_hour', ''): '0154cde050a11d71',
    ('message_same_hour', 'Never book on Sundays.'): '2f8d874e4a3d3296',
    ('no_hour', ''): 'cdfb8463606e0064',
    ('no_hour', 'Never book on Sundays.'): '0489a7c701aa1fb8',
    ('same_date', ''): '3732b39ad074e033',
    ('same_date', 'Never book on Sundays.'): 'cc77d69eb0bb0f52',
    ('same_hour', ''): '796f1621654c2d21',
    ('same_hour', 'Never book on Sundays.'): '4cb0a838ef7fbc1c',
    ('same_hour_iso', ''): '805c318cd6dddb6b',
    ('same_hour_iso', 'Never book on Sundays.'): '1dbd68d570be702b',
    ('two_hours', ''): 'f82c6e34982a8c96',
    ('two_hours', 'Never book on Sundays.'): '2d0892f4e6edbb6c',
    ('with_context', ''): '11ce3617f2fe5c4c',
    ('with_context', 'Never book on Sundays.'): '859ad10ecceed6b6',
    ('write_without_hour', ''): 'a0a881ee3eca8b4c',
    ('write_without_hour', 'Never book on Sundays.'): '6c175ca0191f68f7',
}


@pytest.mark.parametrize("name", sorted(_CONTROLS))
@pytest.mark.parametrize("limits", ["", "Never book on Sundays."])
def test_no_computed_mismatch_is_the_prompt_of_before(name, limits):
    ctx = _CONTROLS[name]()
    prompt = _sent(ctx, limits)
    assert _OPENING not in prompt
    assert _sha(prompt) == _BASE[(name, limits)]
