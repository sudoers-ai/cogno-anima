"""When the approved draft and the tool data both carry figures, which of them is a CONFLICT.

A production turn raised the question (`turn_traces` id=1691, demo box, host `771cce4`,
2026-09-08 21:00:09Z, row un-rewritten). The judge APPROVED the executor's draft at the first
attempt — *"Base: R$ 120,00 por hora"*, the tenant's own configured rate — while one of the
thirteen tool results carried a knowledge-graph edge reading *"Senhor Barriga
--[ASSOCIATES_WITH]--> R$ 45,00"*. The contact was told the rate is R$ 45,00 per hour.

**That turn is NOT evidence of a missing precedence rule; it predates the fix.** Its
`superego.prompt_blocks` are `user_request, context, executor_data, traits, signals, task` —
no `draft` block at all, which is the exact shape `_draft_section`'s docstring names as the
defect it closes, on a box whose anima was still older than that commit. Replayed offline
against the current code with the row's real texts, both halves fire: the approved draft now
renders as the voice's PRIMARY source, and `_draft_divergence` returns
``lost=['12000', '3000', '4000']``, which forces the one deterministic re-voice.

**And the counting is what says not to add a suppression rule on top.** Over the 973
judge-approved turns of that box, 603 are provable here (draft under its 4000-char cap, draft +
tools + reply all present; a CAPPED tool result is deliberately KEPT — the claim being made is
positive, "this figure IS in a tool result", and truncation can only hide such a figure, never
invent one). In those 603:

* **81** turns have a figure in the approved draft AND a figure in the tool results;
* **34** of those carry a tool figure the draft does not state — the syntactic definition of
  "two sanctioned sources with different figures";
* **1** of those 34 saw the reply take the tool-only figure. It is id=1691.

Reading all 34 says why the other 33 must be left alone: every one is a BOOKKEEPER turn whose
tool-only figure is another LINE of the same `get_summary` block — `Income: R$ 0,00` beside the
`Expense: R$ 45,00` the draft reports. Two different facts, both true, both from the same
result. A rule that let the draft's figure SUPPRESS a tool-only figure would fire on 34 turns to
correct one that is already corrected, and on id=1777 it would forbid a reply from stating an
income of R$ 300,00 that the contact had just registered.

So the pair below pins the boundary rather than moving it: the displacement fires, the second
fact does not.
"""

from __future__ import annotations

import pytest

from cogno_anima import metakeys as mk
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.types import EgoResult, EgoStep, ToolExecution
from tests.unit.test_superego import ScriptedBackend, _ctx, _m

# id=1691's three figures and the graph edge that displaced them. Neutral stand-ins for the
# tool NAMES only — the figures and their relationship are the row's.
RATE_DRAFT = ("A remuneração é calculada por hora-aula. Base: R$ 120,00 por hora. "
              "IBOPE entre 80% e 89%: adicional de R$ 30,00 por hora. "
              "IBOPE acima de 90%: adicional de R$ 40,00 por hora.")
GRAPH_RESIDUE = ("knowledge_search: [Knowledge Graph]\n"
                 "- Senhor Barriga --[ASSOCIATES_WITH]--> R$ 45,00")
DISPLACED_REPLY = ("Oi! Sobre a parte financeira, a remuneração para as aulas que você "
                   "ministra é de R$ 45,00 por hora.")

# The 33-turn shape: one tool block, two lines, two different facts.
SUMMARY = ("get_summary: Income:  R$ 300,00 (1 entries)\n"
           "Expense: R$ 45,00 (1 entries)\nNet:     R$ 255,00")


def _turn(draft, result, *, tool="lookup"):
    ctx = _ctx(user="quanto eu receberia por mês?")
    ctx.ego_result = EgoResult(
        steps=[EgoStep(index=0, path="native", assistant_text=draft,
                       tool_calls=[ToolExecution(tool=tool, arguments={}, result=result,
                                                 ok=True, side_effect=False)])],
        metrics=_m("ego"))
    ctx.metadata[mk.JUDGE_VERDICT] = {"approved": True, "attempts": 1}
    return ctx


# ── the pair ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_tool_figure_that_DISPLACES_the_approved_answer_is_re_voiced():
    """id=1691 replayed. The reply drops every figure review approved and states one that the
    tool data happens to carry: `lost` is what names it, and it is what the turn never had."""
    ctx = _turn(RATE_DRAFT, GRAPH_RESIDUE, tool="knowledge_search")
    payload = SuperegoStage._tool_payload(ctx)
    lost, invented = SuperegoStage._draft_divergence(
        RATE_DRAFT, payload, DISPLACED_REPLY, ctx.user_input)
    assert lost == ["12000", "3000", "4000"]
    # NOT fabrication, and the distinction is the whole point: R$ 45,00 IS in a tool result,
    # so the anti-fabrication half is silent and only the displacement half speaks.
    assert invented == []

    b = ScriptedBackend([DISPLACED_REPLY, "A base é de R$ 120,00 por hora."])
    out = await SuperegoStage().voice(ctx, b, voice_prompt="persona")
    assert len(b.calls) == 2, "the deterministic net must force exactly one re-voice"
    assert "voice:diverged_from_approved_draft" in out.adjustments
    assert "Every figure in the approved executor's answer" in b.calls[1]["prompt"]


@pytest.mark.asyncio
async def test_a_SECOND_FACT_from_the_same_tool_block_is_not_a_conflict():
    """The 33 turns the measurement found, and the muzzle this must not become.

    `Income: R$ 300,00` is not a rival value for `Expense: R$ 45,00` — it is the next line of
    the same block, and a reply that reports both is more complete, not less grounded. Nothing
    fires, no re-voice is spent, and a precedence rule that let the draft suppress a tool-only
    figure would have deleted a true statement about money the contact had just registered."""
    ctx = _turn("Suas despesas neste mês totalizam R$ 45,00.", SUMMARY, tool="get_summary")
    payload = SuperegoStage._tool_payload(ctx)
    reply = "Suas despesas somam R$ 45,00 e suas entradas, R$ 300,00."
    assert SuperegoStage._draft_divergence(
        "Suas despesas neste mês totalizam R$ 45,00.", payload, reply, ctx.user_input) == ([], [])

    b = ScriptedBackend([reply])
    out = await SuperegoStage().voice(ctx, b, voice_prompt="persona")
    assert len(b.calls) == 1, "no re-voice: nothing was displaced"
    assert "voice:diverged_from_approved_draft" not in out.adjustments


def test_omitting_a_tool_figure_is_not_a_divergence_either():
    """The mirror of the same boundary, stated as a property rather than a scenario: the net
    is about what the APPROVED ANSWER states, never about what the tool data also contains. A
    reply that answers the question asked and leaves the block's other lines out is complete."""
    assert SuperegoStage._draft_divergence(
        "Suas despesas neste mês totalizam R$ 45,00.", SUMMARY,
        "Suas despesas neste mês totalizam R$ 45,00.") == ([], [])
