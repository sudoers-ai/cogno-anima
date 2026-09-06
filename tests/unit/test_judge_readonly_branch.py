"""The READ-ONLY judge branch — chosen, not argued.

A turn whose every tool call SUCCEEDED and whose every call was a READ has no mutation to
verify, so criteria #1 (GOAL<->EXECUTION) and #3 (COMPLETENESS) have nothing to bind to and
decay into "does the reply satisfy the user" — a judgement about the DRAFT wearing the
vocabulary of execution. Measured over 730 production traces carrying a judge block: a turn
that WROTE was rejected at least once 14.1% of the time [7.0-24.4], a turn that only READ
65.9% [61.0-70.6], and half of the latter were never approved at all.

**The claim this file pins is not "a clause was missing".** A deterministic probe over the
rendered prompt (7 clauses x 3 cases = 21 cells) showed NOTHING TO DO and MID-FLOW — the two
clauses that already describe this exact outcome — are UNCONDITIONAL and were present in the
prompts that produced those rejections. The claim is that the RIGHT clauses are not being
APPLIED to this shape of turn, and that the branch SWAPS the criteria instead of repeating
them, exactly as ``JUDGE_CONVERSATIONAL`` did (measured 0/3 -> 3/3).

The half a model must answer — does the judge now approve the truthful read and still reject
the inventing one — lives in ``tests/integration/test_superego.py``, as a PAIR.
"""

from __future__ import annotations

import pytest

from cogno_anima import metakeys as mk
from cogno_anima.stages.superego import (
    JUDGE_CONVERSATIONAL_BRANCH,
    JUDGE_EXECUTION,
    JUDGE_READONLY,
    SuperegoStage,
)
from cogno_anima.types import (EgoResult, EgoStep, PipelineContext, StageMetrics,
                               ToolExecution)


def _m(stage: str) -> StageMetrics:
    return StageMetrics(stage=stage, elapsed_ms=1.0, tokens_in=1, tokens_out=1, model="t")


def _ctx(calls=(), *, draft="Here is what I found.", interrupted=False, held=(),
         conversational=False):
    ctx = PipelineContext(user_input="e as aulas do mês que vem?")
    steps = [EgoStep(index=0, path="native", tool_calls=list(calls)),
             EgoStep(index=1, path="native", assistant_text=draft)]
    ctx.ego_result = EgoResult(steps=steps, interrupted=interrupted,
                               pending_confirmation=list(held), metrics=_m("ego"))
    if conversational:
        ctx.metadata[mk.JUDGE_CONVERSATIONAL] = True
    return ctx


def _read(tool="get_schedule", *, ok=True, side_effect=False, mutating=None,
          result="No classes found."):
    return ToolExecution(tool=tool, arguments={"month": "2026-10"}, result=result, ok=ok,
                         side_effect=side_effect, tool_mutating=mutating)


BRANCH = SuperegoStage._judge_branch


# ── which branch ──────────────────────────────────────────────────────────────────────
def test_a_clean_read_takes_the_readonly_branch():
    assert BRANCH(_ctx([_read()])) == JUDGE_READONLY


def test_a_turn_that_WROTE_keeps_the_execution_criteria():
    """THE mutation guard: point this branch at a turn that acted and the whole thing is a
    licence to skip verifying the action. ``side_effect`` is the fact about the CALL — it is
    what "committed" is built on everywhere else in this codebase, and it is what says no."""
    assert BRANCH(_ctx([_read("record_expense", side_effect=True)])) == JUDGE_EXECUTION


def test_one_write_among_reads_is_still_a_write():
    assert BRANCH(_ctx([_read(), _read("record_expense", side_effect=True), _read()])) \
        == JUDGE_EXECUTION


def test_a_tool_DECLARED_mutating_keeps_the_execution_criteria():
    """The dispatcher-forgot case, and why both writing facts are read rather than one.

    ``side_effect`` is a property of the call, set after it ran; ``tool_mutating`` is a
    property of the NAME, declared before. A source that fills in the second and forgets the
    first would otherwise hand this branch a write."""
    assert BRANCH(_ctx([_read("cancel_appointment", mutating=True)])) == JUDGE_EXECUTION


def test_a_failed_call_keeps_the_execution_criteria():
    """A read that failed tells you nothing about the world, only that the read failed — so a
    draft built on it is exactly the claim the execution criteria exist to catch."""
    assert BRANCH(_ctx([_read(ok=False)])) == JUDGE_EXECUTION


def test_an_interrupted_loop_keeps_the_execution_criteria():
    assert BRANCH(_ctx([_read()], interrupted=True)) == JUDGE_EXECUTION


def test_a_held_proposal_keeps_the_execution_criteria():
    """A turn holding a write for confirmation is PROPOSING one — reads plus an unexecuted
    mutation is not the same shape as reads alone."""
    assert BRANCH(_ctx([_read()], held=[_read("cancel_appointment", mutating=True)])) \
        == JUDGE_EXECUTION


def test_a_turn_that_executed_nothing_is_not_readonly():
    """Nothing ran, so nothing was read. That turn is either conversational (the host says so)
    or an executor that chose not to act — and the second must stay judged in full."""
    assert BRANCH(_ctx([])) == JUDGE_EXECUTION


def test_no_ego_result_is_not_readonly():
    assert BRANCH(PipelineContext(user_input="oi")) == JUDGE_EXECUTION


def test_the_hosts_conversational_signal_wins():
    """Precedence, and it is what keeps every conversational turn byte-identical to before:
    whether a persona was offered any tool at all is knowledge only the host has."""
    assert BRANCH(_ctx([_read()], conversational=True)) == JUDGE_CONVERSATIONAL_BRANCH


# ── what the prompt actually says ─────────────────────────────────────────────────────
_GOAL_CRITERION = "1. GOAL↔EXECUTION: did it do exactly what was asked"
_COMPLETENESS = "3. COMPLETENESS: was the goal fully met"


def _prompt(ctx):
    return SuperegoStage()._build_judge_prompt(ctx, "")


def test_the_branch_SWAPS_the_criteria_it_does_not_add_to_them():
    """The whole point. Appending a fourth paragraph to a prompt whose third is already
    ignored is the weak move; this asserts the execution criteria are GONE, not out-argued."""
    out = _prompt(_ctx([_read()]))
    assert "executed READS ONLY" in out
    assert _GOAL_CRITERION not in out
    assert _COMPLETENESS not in out
    assert "not available to you as a finding" in out


def test_grounding_is_promoted_not_relaxed():
    """The condition that stops this being a free pass: there ARE tool results here, so the
    fabrication rule is criterion #1 and an empty read is declared to ground a NEGATIVE only."""
    out = _prompt(_ctx([_read()]))
    assert "1. FABRICATION / GROUNDING" in out
    assert "grounds a NEGATIVE answer and nothing else" in out
    assert "CONTRADICTS THE READ" in out


def test_a_write_turn_still_gets_the_original_criteria_verbatim():
    """No collateral: a turn that acted is judged exactly as it was before this branch."""
    out = _prompt(_ctx([_read("record_expense", side_effect=True)]))
    assert _GOAL_CRITERION in out and _COMPLETENESS in out
    assert "executed READS ONLY" not in out


@pytest.mark.parametrize("ctx", [
    _ctx([_read()]),
    _ctx([_read("record_expense", side_effect=True)]),
    _ctx([_read()], conversational=True),
])
def test_the_unconditional_tail_survives_in_every_branch(ctx):
    """NOTHING TO DO and MID-FLOW say what a valid OUTCOME is, which does not vary with the
    shape of the turn — and the probe that motivated this branch is only honest if they stay
    where it found them."""
    out = _prompt(ctx)
    assert "NOTHING TO DO is a VALID outcome" in out
    assert "MID-FLOW is a VALID outcome" in out
    assert "TRUST THE TOOLS" in out


def test_the_three_branch_labels_are_distinct():
    """A closed alphabet: these land in a log line that a reader uses to tell which criteria
    produced a rejection, because the rendered prompt is deliberately never persisted."""
    assert len({JUDGE_EXECUTION, JUDGE_CONVERSATIONAL_BRANCH, JUDGE_READONLY}) == 3


def test_an_unreadable_trace_falls_to_the_STRICTER_branch():
    """The safe direction, and the red that taught it.

    Seven tests in ``test_judge_learns_the_duty.py`` build ``ego_result`` as a partial
    ``SimpleNamespace`` — no ``interrupted``, no ``pending_confirmation`` — and the first cut of
    this predicate raised ``AttributeError`` on all of them, taking the whole turn down to
    answer a question about which criteria to use. ``_format_unavailable`` already carries the
    rule that a judge prompt must never be the reason a turn dies; for a RELAXATION the safe
    degradation is the opposite one — refuse to relax.
    """
    from types import SimpleNamespace as NS

    ctx = PipelineContext(user_input="oi")
    ctx.ego_result = NS(tools_executed=[_read()], draft="…")     # no interrupted/pending
    assert BRANCH(ctx) == JUDGE_EXECUTION

    class Hostile:
        @property
        def tools_executed(self):
            raise RuntimeError("trace is not readable")

    ctx.ego_result = Hostile()
    assert BRANCH(ctx) == JUDGE_EXECUTION


def test_a_write_on_a_DISCARDED_attempt_still_blocks_the_branch():
    """Two readings of one fact, and the second was wrong — found by review, not by a test.

    ``write_attempted_this_turn`` (``types.py``) asks the same per-call question this branch
    needs — ``side_effect is True or tool_mutating is True`` — but it walks BOTH execution
    lists: ``ctx.turn_executions`` in UNION with ``ego_result.tools_executed``. The first cut
    of ``_is_readonly_turn`` walked only the second, so a turn whose attempt 1 WROTE and whose
    surviving attempt shows clean reads answered ``readonly``: the judge would have been told
    "there was no mutation to verify" about a turn that mutated. Measured 1 divergence in 3
    shapes before, 0 after. It is the survivor-attempt-read-as-the-turn defect, and the cure
    is the one ``committed_this_turn`` prescribes — one definition, never a second reading.
    """
    ctx = _ctx([_read()])                       # o sobrevivente só leu
    ctx.turn_executions = [_read("record_expense", side_effect=True, mutating=True), _read()]
    assert BRANCH(ctx) == JUDGE_EXECUTION


def test_the_two_readings_of_the_write_fact_cannot_disagree():
    """The gate that keeps them one: whatever ``write_attempted_this_turn`` calls a write, this
    branch must refuse. Asserted over the shapes, not over the source text."""
    from cogno_anima.types import write_attempted_this_turn

    shapes = []
    for surv, acc in (([_read()], []),
                      ([_read("record_expense", side_effect=True)], []),
                      ([_read("cancel_appointment", mutating=True)], []),
                      ([_read()], [_read("record_expense", side_effect=True, mutating=True)]),
                      ([_read()], [_read()])):
        ctx = _ctx(surv)
        if acc:
            ctx.turn_executions = acc
        shapes.append((write_attempted_this_turn(ctx), BRANCH(ctx)))
    assert shapes, "denominador vazio"
    bad = [s for s in shapes if s[0] and s[1] == JUDGE_READONLY]
    assert not bad, f"{len(bad)}/{len(shapes)} formas relaxam sobre uma escrita: {bad}"
