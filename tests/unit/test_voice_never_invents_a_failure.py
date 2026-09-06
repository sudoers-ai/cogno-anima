"""The voice may not tell a contact that something failed when nothing was tried.

Measured 2026-09-06 on production traces (demo box, a 1094-row snapshot of ``turn_traces``,
1049 of them of a provable era under the ``xmin`` rewrite filter): the contact
asked for an expense to be recorded, the EGO called only ``resolve_date`` and asked for
confirmation in prose, the judge rejected the turn with the RIGHT critique ("only asked for
confirmation without recording"), and the voice — holding that critique — wrote *"Não consegui
registrar a despesa"*.

Nothing was attempted. Nothing failed. And a person was told of a failure, which is worse than
doing nothing: an invented failure is a FALSE statement about the world, and they act on it —
they assume the system tried, and may not ask again. In the measured scenario the expense was
simply never recorded.

225 turns of that snapshot carry the shape (judge rejection + nothing committed + not one call
to a writing tool); 9 shipped a sentence claiming the requested ACTION could not be done, while
16 more truthfully reported a READ that came up empty — those must keep saying so.

The twins below are the whole point, and the SECOND is what keeps the fix from becoming a
muzzle: a turn where the write really was attempted and really did fail MUST still be allowed
to say so. "Never say you failed" would delete a truth the contact needs.

Deterministic: the assertions are on the RENDERED voice prompt, not on a model's reply.
"""

from __future__ import annotations

import pytest

from cogno_anima import metakeys as mk
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.types import (
    EgoResult, EgoStep, ToolExecution, write_attempted_this_turn,
)
from tests.unit.test_superego import ScriptedBackend, _ctx, _m

# The critique the judge actually wrote on the measured turn.
CRITIQUE = "only asked for confirmation without recording"

# What the clause forbids, and what it demands instead. Both are asserted on the PROMPT: the
# reply itself needs a model, and this file is the deterministic half of the pair.
FORBIDS = "MUST NOT write that the requested action was attempted and did not work"
DEMANDS = "the confirmation this request is still waiting for, or the ONE question"
NEVER_CLAIM_DONE = "MUST NOT claim, imply or narrate that any action was performed"


def _ego(*calls: ToolExecution) -> EgoResult:
    return EgoResult(
        steps=[EgoStep(index=0, path="native",
                       assistant_text="posso registrar a despesa de R$45?",
                       tool_calls=list(calls))],
        metrics=_m("ego"))


def _read_only_turn():
    """The measured turn: one READ ran, no writing tool was ever called."""
    ctx = _ctx(with_ego=False)
    ctx.ego_result = _ego(ToolExecution(tool="resolve_date", arguments={}, result="2026-09-06",
                                        ok=True, side_effect=False, tool_mutating=False))
    return ctx


def _rendered(ctx, *, kind=None, reason=CRITIQUE):
    carrier = {"reason": reason}
    if kind is not None:
        carrier["kind"] = kind
    ctx.metadata[mk.VOICE_CORRECTION] = carrier
    return SuperegoStage()._build_voice_prompt(ctx, "resolve_date: 2026-09-06",
                                               ["general:review"])


# ── the two twins ────────────────────────────────────────────────────

def test_nothing_tried_the_voice_is_told_not_to_report_a_failure():
    """POSITIVE twin: judge critique + ZERO calls to a writing tool."""
    prompt = _rendered(_read_only_turn())
    assert FORBIDS in prompt, (
        "the voice was handed a rejection over a turn that attempted no write and nothing "
        "told it that 'I could not do it' is a false statement here")
    # ...and it is told what to write INSTEAD — a rule that only forbids leaves the model to
    # pick, and what it picked was the invented failure.
    assert DEMANDS in prompt
    # the OPPOSITE fabrication stays forbidden too: "registei" is exactly as false
    assert NEVER_CLAIM_DONE in prompt


def test_a_write_that_really_failed_may_still_be_reported_as_a_failure():
    """NEGATIVE twin — the one that stops this from becoming a muzzle.

    ``ToolResult(ok=False)`` on a tool with an effect: the booking WAS attempted and the server
    refused it. Telling the contact it did not go through is the truth, and the truth is what
    they need in order to try something else.
    """
    ctx = _ctx(with_ego=False)
    ctx.ego_result = _ego(ToolExecution(
        tool="record_expense", arguments={"amount": 45}, result="", ok=False,
        error="the ledger rejected the entry", side_effect=True, tool_mutating=True))
    prompt = _rendered(ctx)
    assert FORBIDS not in prompt, (
        "a write was attempted and failed — forbidding the failure sentence here would delete "
        "a truth the contact needs")
    assert NEVER_CLAIM_DONE in prompt        # still may not claim it succeeded


def test_the_modern_dispatcher_convention_still_counts_as_an_attempt():
    """The dispatchers shipped since 2026-09-01 stamp ``side_effect=False`` on their FAILURE
    branch (see ``EgoStage._warn_if_effect_without_success``), so ``side_effect`` alone cannot
    recognise a write that was attempted and refused. ``tool_mutating`` is what carries it.

    This is not a hypothetical shape. Two production turns of the corpus swept for this change
    carry it exactly (`turn_traces` id=448 and id=711, both verified as of their own era):
    every call to the writing tool came back ``ok=False, side_effect=False,
    tool_mutating=True``, and each reply told the contact — truthfully, and with the reason —
    that the thing they asked for had not gone through. A predicate reading only
    ``side_effect`` would have called both "never attempted" and gagged them.
    """
    ctx = _ctx(with_ego=False)
    ctx.ego_result = _ego(ToolExecution(
        tool="book_appointment", arguments={}, result="", ok=False, error="slot taken",
        side_effect=False, tool_mutating=True))
    assert write_attempted_this_turn(ctx) is True
    assert FORBIDS not in _rendered(ctx)


# ── the predicate is about what was EXECUTED ─────────────────────────

def test_the_predicate_reads_both_execution_sources():
    """A write recorded ONLY in the turn accumulator counts. `ego_result` is the surviving
    attempt; a write from an earlier attempt of the same turn does not un-happen."""
    ctx = _ctx(with_ego=False)
    ctx.ego_result = _ego(ToolExecution(tool="resolve_date", arguments={}, result="x",
                                        ok=True, side_effect=False, tool_mutating=False))
    assert write_attempted_this_turn(ctx) is False
    ctx.turn_executions = [ToolExecution(tool="record_expense", arguments={}, result="",
                                         ok=False, error="refused", side_effect=False,
                                         tool_mutating=True)]
    assert write_attempted_this_turn(ctx) is True
    assert FORBIDS not in _rendered(ctx)


def test_having_writing_tools_on_the_table_is_not_an_attempt():
    """The predicate the fix must NOT use. A persona offered twenty writing tools and called
    none of them attempted nothing; an ACTION_REQUEST intent is not an attempt either."""
    ctx = _read_only_turn()
    assert ctx.intent.intent_class == "ACTION_REQUEST"
    assert write_attempted_this_turn(ctx) is False
    assert FORBIDS in _rendered(ctx)


def test_an_unreadable_carrier_leaves_the_behaviour_exactly_as_it_was():
    """A rule that RESTRICTS what the voice may say must not fire on a carrier it cannot read.
    False there would gag a failure over a turn nobody can vouch for."""
    class Broken:
        metadata: dict = {}
        user_input = "x"
        noumeno = intent = id_result = None

        @property
        def turn_executions(self):
            raise RuntimeError("boom")

        @property
        def ego_result(self):
            raise RuntimeError("boom")

    assert write_attempted_this_turn(Broken()) is True
    # ...and the sibling commit predicate keeps ITS direction (never release on a broken read)
    from cogno_anima.types import committed_this_turn
    assert committed_this_turn(Broken()) is False


# ── scope: one condition on one variant ──────────────────────────────

def test_the_clause_is_purely_additive_on_a_turn_that_did_attempt():
    """Remove the clause and the prompt IS the pre-change prompt — byte for byte, header
    included. No new section: `_VOICE_BLOCKS` (and the inventory the host persists) is
    untouched, which is why this is a CONDITION on the execution verdict and not a fourth
    ``kind`` the orchestrator would have to learn.
    """
    attempted = _rendered(_ctx())            # the default fixture commits `record_expense`
    tried = _rendered(_read_only_turn())
    clause = next(line for line in tried.split("\n")
                 if line.startswith("NOTHING WAS EVEN TRIED"))
    assert tried.replace(clause + "\n", "") == attempted
    assert "# Execution verdict (HARD RULE)" in tried
    assert sum(h.startswith("# ") for h in tried.split("\n")) == \
           sum(h.startswith("# ") for h in attempted.split("\n"))


@pytest.mark.parametrize("kind", ["repeated_reply", "unverified_claim"])
def test_the_other_two_rejection_variants_are_untouched(kind):
    """`repeated_reply` is the host's anti-repeat guard (the content was fine) and
    `unverified_claim` is a turn where nothing ran at all, whose own text already tells the
    voice to drop the claim and admit the limit. Neither is the measured defect, and widening
    the clause into them would collide with wording those branches were measured into."""
    assert FORBIDS not in _rendered(_read_only_turn(), kind=kind)


def test_a_rejection_without_a_reason_renders_no_verdict_and_no_clause():
    """The clause rides the verdict section; it may not appear on a turn that has none."""
    ctx = _read_only_turn()
    ctx.metadata[mk.VOICE_CORRECTION] = {"reason": ""}
    prompt = SuperegoStage()._build_voice_prompt(ctx, "p", ["general:review"])
    assert "# Execution verdict (HARD RULE)" not in prompt and FORBIDS not in prompt


def test_an_approved_turn_never_sees_the_clause():
    """No rejection at all — the overwhelming majority of turns. The clause must be reachable
    ONLY through the verdict section."""
    prompt = SuperegoStage()._build_voice_prompt(_read_only_turn(), "p", ["general:review"])
    assert FORBIDS not in prompt


# ── the path is really reached ───────────────────────────────────────

@pytest.mark.asyncio
async def test_the_clause_reaches_the_prompt_voice_actually_sends():
    """`_build_voice_prompt` is a helper; what ships is what `voice()` hands the backend."""
    ctx = _read_only_turn()
    ctx.metadata[mk.VOICE_CORRECTION] = {"reason": CRITIQUE}
    backend = ScriptedBackend(["Posso registrar a despesa de R$45? É só confirmar."])
    result = await SuperegoStage().voice(ctx, backend, voice_prompt="persona")
    assert FORBIDS in backend.calls[0]["prompt"]
    assert FORBIDS in (result.prompt_text or "")
