"""The four sentences of the defect, each RED on the tree before the fix for the reason it names.

This module imports NOTHING the fix added (no ``cogno_anima.verdict``, no new field, no new
metakey), on purpose: a twin that fails with an ``ImportError`` or an ``AttributeError`` on the
old tree has not shown the defect, it has shown that the test is newer than the code. Every
assertion here is about a DECISION or about two RECORDS, both of which exist in either world:

1. the JUDGE read ``"approved": "false"`` — a string — as APPROVED (``bool("false") is True``),
   a fail-CLOSED gate failing open on the field that decides it;
2. the scope GUARD read ``"blocked": "false"`` as BLOCKED, refusing a contact the classifier
   had just allowed;
3. the GUARD's ALLOW over a reply it could not parse left the same record, byte for byte, as an
   ALLOW the classifier gave;
4. a verdict key written TWICE read as its last value, so ``false`` then ``true`` approved — in
   the judge and in the pre-judge, which was already strict about the type.

Measured on the tree before the fix: 4 failed. The controls, the adverse shapes and the marks
are in ``test_verdict_is_a_json_boolean.py``.
"""

from __future__ import annotations

from cogno_anima.stages.proposal_judge import ProposalJudge
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.tools.pre_judge import PRE_APPROVED, Proposal
from tests.unit.test_superego import ScriptedBackend, _ctx

_SCOPE = "You handle scheduling and study support for the school."
_USER = "what can I use to study for the exam?"


async def test_red_twin_the_judge_does_not_approve_a_string_false():
    """Fail-CLOSED means the opposite of approving the execution a reply just rejected."""
    backend = ScriptedBackend(['{"approved": "false", "critique": "booked the wrong day"}'])
    result = await SuperegoStage().evaluate(_ctx(), backend, limits_prompt="")
    assert len(backend.calls) == 1, "the premise: the judge was asked"
    assert result.approved is False


async def test_red_twin_the_guard_does_not_block_on_a_string_false():
    """A fail-OPEN guard must not refuse a contact over its own misreading of an ALLOW."""
    backend = ScriptedBackend(['{"blocked": "false", "refusal_message": ""}'])
    ctx = _ctx(user=_USER, intent_class="INFORMATION_REQUEST")
    result = await SuperegoStage().check_input_scope(ctx, backend, scope_prompt=_SCOPE)
    assert len(backend.calls) == 1, "the premise: the classifier was asked"
    assert result.blocked is False


async def _guard_record(raw: str) -> "tuple[bool, dict]":
    """One consulted guard turn: the decision, and EVERYTHING the turn leaves behind about the
    call minus the wall clock — the result as a host would serialise it and the per-turn
    metadata keys the guard stamps."""
    backend = ScriptedBackend([raw], ti=23, to=5)
    ctx = _ctx(user=_USER, intent_class="INFORMATION_REQUEST")
    result = await SuperegoStage().check_input_scope(ctx, backend, scope_prompt=_SCOPE)
    assert len(backend.calls) == 1, "the premise: the classifier was asked"
    body = result.model_dump()
    body["metrics"].pop("elapsed_ms")
    stamped = {k: v for k, v in ctx.metadata.items() if k.startswith("scope_")}
    return result.blocked, {"result": body, "metadata": stamped}


async def test_red_twin_an_unread_guard_reply_does_not_leave_the_record_of_a_read_allow():
    """Both turns are ALLOWED, and that is the contract (fail-OPEN). What must differ is the
    RECORD: "the classifier read this and let it through" and "we could not read what the
    classifier said" were one fact."""
    read_blocked, read = await _guard_record('{"blocked": false, "refusal_message": ""}')
    unread_blocked, unread = await _guard_record(
        "I would let this one through, it is about studying.")
    assert (read_blocked, unread_blocked) == (False, False)
    assert read != unread


async def test_red_twin_a_verdict_written_twice_is_not_read_as_its_last_value():
    """``json.loads`` keeps the last of two keys in silence: a rejection followed by an
    approval in the same object approved, in both judges."""
    raw = '{"approved": false, "critique": "booked the wrong day", "approved": true}'
    result = await SuperegoStage().evaluate(_ctx(), ScriptedBackend([raw]), limits_prompt="")
    assert result.approved is False
    pre = ProposalJudge(ScriptedBackend([raw]), request="book it for tomorrow at ten")
    verdict = (await pre(Proposal("book_slot", {"day": "tomorrow", "time": "10:00"}))).verdict
    assert verdict != PRE_APPROVED
