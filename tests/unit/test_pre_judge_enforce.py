"""``PreJudgeDispatcher(enforce=…)`` — the pre-verdict COUNTS, for the tools the caller names.

The shadow (``test_pre_judge_shadow.py``) never changes a call. Enforcement is opt-in and per
tool, and the properties below are each proven in the world where they could fail:

* **the twin** — a WRONG proposal on an enforced tool does NOT run: the executor gets the caller's
  PROPOSAL (gate C, ``needs_confirmation``), and the inner dispatcher is never called; the RIGHT
  proposal runs, once;
* **fail open, counted** — no verdict (``error``/``timeout``) runs the call as today, and the
  record says ``executed_fail_open``;
* **a confirmed call runs unjudged — only THAT call** — the caller's ``confirmed`` predicate is
  asked with the call's own arguments, so a yes to a proposal about one target never waves
  through a call on another;
* **one judgement per call** — an enforced call is judged once, and its record is the shadow's
  record with two more keys;
* **the controls** — without ``enforce`` the wrapper is the shadow byte for byte; a tool
  ``enforce`` does not name stays in shadow; a read is never judged; a ``confirm`` answer that is
  not a proposal is replaced, so a held call can never be read as a write.

Tool and persona names are invented.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from cogno_anima.tools import (PRE_OUTCOME_EXECUTED, PRE_OUTCOME_FAIL_OPEN, PRE_OUTCOME_HELD,
                               PRE_OUTCOMES, PreJudgeDispatcher, PreJudgeSink, PreJudgment,
                               Proposal)
from cogno_anima.tools.base import ToolPolicyDispatcher
from cogno_anima.types import StageMetrics, ToolResult

_MOVE = "hand_over_to"          # the enforced write
_BOOK = "book_slot"             # a write the caller did NOT enforce
_READ = "list_slots"
_RIGHT = {"target": "LIBRARIAN"}
_WRONG = {"target": "GARDENER"}


class _Source:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def tools_schema(self) -> list[dict]:
        return [{"type": "function", "function": {"name": n, "description": f"{n} tool",
                                                  "parameters": {}}}
                for n in (_MOVE, _BOOK, _READ)]

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        self.calls.append((name, dict(arguments)))
        return ToolResult(output=f"ran {name}", ok=True, side_effect=name != _READ)

    def is_mutating(self, name: str) -> bool:
        return name in (_MOVE, _BOOK)

    def requires_confirmation(self, name: str) -> bool:
        return False


class _Judge:
    """Approves ``_RIGHT`` (and any ``_BOOK``); critiques the rest. Can hang, or raise."""

    model = "judge-fake"

    def __init__(self, *, hang: bool = False, raise_exc: bool = False) -> None:
        self.seen: list[Proposal] = []
        self._hang = hang
        self._raise = raise_exc

    async def __call__(self, proposal: Proposal) -> PreJudgment:
        self.seen.append(proposal)
        if self._hang:
            await asyncio.sleep(3600)
        if self._raise:
            raise ValueError("judge exploded")
        ok = proposal.tool == _BOOK or proposal.arguments == _RIGHT
        return PreJudgment("approved" if ok else "critique",
                           StageMetrics(stage="whatever", elapsed_ms=1.0, tokens_in=90,
                                        tokens_out=6, model=self.model))


def _asks(proposal: Proposal) -> ToolResult:
    """The CALLER's proposal — the words are the caller's (the host names the destination)."""
    return ToolResult(output=f"Hand over to {proposal.arguments.get('target')}?", ok=False,
                      error="needs_confirmation", side_effect=False, needs_confirmation=True)


def _wrap(*, judge=None, enforce=frozenset({_MOVE}).__contains__, confirm=_asks,
          confirmed=None, **kw):
    inner, judge = _Source(), judge or _Judge()
    sink = PreJudgeSink()
    d = PreJudgeDispatcher(inner, judge=judge, sink=sink, enforce=enforce, confirm=confirm,
                           confirmed=confirmed, **kw)
    return inner, judge, sink, d


# ── the twin ──────────────────────────────────────────────────────────────────────

async def test_twin_a_WRONG_proposal_on_an_enforced_tool_does_not_run_it_is_proposed():
    inner, _, sink, d = _wrap()
    result = await d.execute(_MOVE, dict(_WRONG))
    assert inner.calls == []                                   # the call NEVER reached the tool
    assert result.needs_confirmation and not result.ok and not result.side_effect
    assert result.output == "Hand over to GARDENER?"           # the caller's own words
    await sink.settle()
    assert sink.records == [{"tool": _MOVE, "verdict": "critique", "ms": sink.records[0]["ms"],
                             "committed": False, "enforced": True, "outcome": PRE_OUTCOME_HELD}]


async def test_twin_the_RIGHT_proposal_runs_once_after_the_verdict():
    inner, _, sink, d = _wrap()
    result = await d.execute(_MOVE, dict(_RIGHT))
    assert inner.calls == [(_MOVE, _RIGHT)] and result.ok
    await sink.settle()
    assert [(r["verdict"], r["outcome"], r["committed"]) for r in sink.records] == \
        [("approved", PRE_OUTCOME_EXECUTED, True)]


async def test_an_enforced_call_WAITS_for_its_verdict():
    """The shadow returns before the judge answers; an enforced call cannot — that is the whole
    difference, and it is measured: the judge had answered before the tool ran."""
    class _Slow(_Judge):
        async def __call__(self, proposal):
            await asyncio.sleep(0.2)
            self.answered = time.perf_counter()
            return await super().__call__(proposal)

    judge = _Slow()
    inner, _, _, d = _wrap(judge=judge)
    ran: list[float] = []
    real = inner.execute

    async def _timed(name, arguments):
        ran.append(time.perf_counter())
        return await real(name, arguments)

    inner.execute = _timed                                     # type: ignore[method-assign]
    await d.execute(_MOVE, dict(_RIGHT))
    assert ran and judge.answered < ran[0]


# ── fail open, and counted ────────────────────────────────────────────────────────

async def test_a_judge_that_ERRORS_lets_the_call_run_and_the_record_says_fail_open():
    inner, _, sink, d = _wrap(judge=_Judge(raise_exc=True))
    result = await d.execute(_MOVE, dict(_WRONG))
    assert inner.calls == [(_MOVE, _WRONG)] and result.ok
    await sink.settle()
    assert [(r["verdict"], r["outcome"]) for r in sink.records] == \
        [("error", PRE_OUTCOME_FAIL_OPEN)]


async def test_a_judge_past_the_ENFORCE_ceiling_is_a_timeout_and_the_call_runs():
    inner, _, sink, d = _wrap(judge=_Judge(hang=True), enforce_timeout_s=0.05)
    t0 = time.perf_counter()
    await d.execute(_MOVE, dict(_WRONG))
    assert time.perf_counter() - t0 < 1.0                      # bounded by the ceiling
    assert inner.calls == [(_MOVE, _WRONG)]
    await sink.settle()
    assert [(r["verdict"], r["outcome"]) for r in sink.records] == \
        [("timeout", PRE_OUTCOME_FAIL_OPEN)]


# ── a confirmed call runs unjudged — only THAT call ───────────────────────────────

async def test_a_call_the_contact_CONFIRMED_runs_without_being_judged_again():
    confirmed = lambda tool, args: tool == _MOVE and args == _WRONG      # noqa: E731
    inner, judge, sink, d = _wrap(confirmed=confirmed)
    await d.execute(_MOVE, dict(_WRONG))
    assert inner.calls == [(_MOVE, _WRONG)]
    assert judge.seen == [] and sink.records == []


async def test_a_yes_to_ONE_target_does_not_wave_through_a_call_on_ANOTHER():
    """The contact confirmed a hand-over to GARDENER; the executor now calls it with ANOTHER
    target. The predicate is asked with the call's own arguments, answers no, and the call is
    judged like any other — here it is critiqued and held, never executed."""
    confirmed = lambda tool, args: args == {"target": "GARDENER"}        # noqa: E731
    inner, judge, _, d = _wrap(confirmed=confirmed)
    result = await d.execute(_MOVE, {"target": "PAINTER"})
    assert inner.calls == [] and result.needs_confirmation
    assert [p.arguments for p in judge.seen] == [{"target": "PAINTER"}]


async def test_a_confirmed_predicate_that_raises_confirms_nothing():
    def _boom(tool, args):
        raise RuntimeError("store down")

    inner, judge, _, d = _wrap(confirmed=_boom)
    await d.execute(_MOVE, dict(_WRONG))
    assert inner.calls == [] and len(judge.seen) == 1


# ── one judgement per call ────────────────────────────────────────────────────────

async def test_an_enforced_call_is_judged_ONCE():
    _, judge, sink, d = _wrap()
    await d.execute(_MOVE, dict(_RIGHT))
    await sink.settle(grace_s=1.0)
    assert len(judge.seen) == 1 and d.pre_judged == 1 and len(sink.records) == 1
    assert [m.stage for m in sink.metrics] == ["judge_pre"]


# ── the controls ──────────────────────────────────────────────────────────────────

async def test_control_WITHOUT_enforce_the_wrapper_is_the_shadow_byte_for_byte():
    inner, _, sink, d = _wrap(enforce=None)
    result = await d.execute(_MOVE, dict(_WRONG))
    assert inner.calls == [(_MOVE, _WRONG)] and result.ok        # it went out, as in the shadow
    await sink.settle(grace_s=1.0)
    assert set(sink.records[0]) == {"tool", "verdict", "ms", "committed"}   # no new keys


async def test_control_a_write_enforce_does_NOT_name_stays_in_shadow():
    inner, _, sink, d = _wrap()
    await d.execute(_BOOK, {"date": "2026-03-12"})
    assert inner.calls == [(_BOOK, {"date": "2026-03-12"})]
    await sink.settle(grace_s=1.0)
    assert set(sink.records[0]) == {"tool", "verdict", "ms", "committed"}


async def test_control_a_READ_is_never_judged_even_if_enforce_names_it():
    inner, judge, sink, d = _wrap(enforce=lambda name: True)
    await d.execute(_READ, {})
    assert inner.calls == [(_READ, {})] and judge.seen == [] and sink.records == []


@pytest.mark.parametrize("answer", [
    ToolResult(output="done", ok=True, side_effect=True),                     # a WRITE
    ToolResult(output="x", ok=False, side_effect=False, needs_confirmation=False),
    ToolResult(output="x", ok=True, side_effect=False, needs_confirmation=True),
    None,
])
async def test_a_confirm_answer_that_is_not_a_PROPOSAL_is_replaced(answer):
    inner, _, _, d = _wrap(confirm=lambda proposal: answer)
    result = await d.execute(_MOVE, dict(_WRONG))
    assert inner.calls == []
    assert result.needs_confirmation and not result.ok and not result.side_effect


async def test_no_confirm_callback_still_holds_with_a_neutral_proposal():
    inner, _, _, d = _wrap(confirm=None)
    result = await d.execute(_MOVE, dict(_WRONG))
    assert inner.calls == [] and result.needs_confirmation and not result.ok


async def test_an_enforce_policy_that_raises_enforces_nothing():
    def _boom(name):
        raise RuntimeError("policy down")

    inner, _, _, d = _wrap(enforce=_boom)
    await d.execute(_MOVE, dict(_WRONG))
    assert inner.calls == [(_MOVE, _WRONG)]                     # back to the shadow


async def test_every_enforced_record_speaks_the_closed_alphabet_and_carries_no_text():
    _, _, sink, d = _wrap()
    await d.execute(_MOVE, dict(_WRONG))
    await d.execute(_MOVE, dict(_RIGHT))
    await sink.settle()
    for r in sink.records:
        assert set(r) == {"tool", "verdict", "ms", "committed", "enforced", "outcome"}
        assert r["outcome"] in PRE_OUTCOMES
    assert "GARDENER" not in repr(sink.records) and "LIBRARIAN" not in repr(sink.records)


def test_the_policy_is_still_forwarded_only_when_the_source_has_one():
    class _NoPolicy:
        def tools_schema(self):
            return []

        async def execute(self, name, arguments):
            return ToolResult(output="ok")

    judge = _Judge()
    assert isinstance(PreJudgeDispatcher(_Source(), judge=judge, sink=PreJudgeSink(),
                                         enforce=lambda n: True), ToolPolicyDispatcher)
    assert not isinstance(PreJudgeDispatcher(_NoPolicy(), judge=judge, sink=PreJudgeSink(),
                                             enforce=lambda n: True), ToolPolicyDispatcher)
