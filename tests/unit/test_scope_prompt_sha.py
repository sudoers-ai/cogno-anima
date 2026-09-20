"""The scope guard records the digest of the WHOLE prompt it sent — not of the slot alone.

Measured on a downstream host: the same guard input classified ALLOW 4/4 and, forty-five
minutes later, BLOCK 7/7 at ``temperature=0``, stable inside each period and across process
restarts of the same build. The host could prove its own contribution byte-identical, because
it digests the slot it writes — and read that as "the prompt did not change". It is not the
same claim: the decision rule, the layout, the contact's sentence, the task and the examples
are assembled HERE, and never leave this function.

``prompt_sha`` closes that gap with the ONE digest algorithm in the ecosystem
(:func:`cogno_anima.prompts.prompt_digest`), over the exact two arguments handed to
``backend.generate``, in that order. Two properties carry it:

* **identity** — the recorded digest IS the digest of the bytes that were sent, and NOT the
  digest of the slot, of the user half, or of the system half (the slice implementations);
* **sensitivity** — every block in the closed ``_SCOPE_BLOCKS`` table changes it, enumerated
  from that table so a block added tomorrow cannot be quietly left uncovered.

And the rule the whole family shares: ``None``/absent means NO PROMPT WAS BUILT.
"""

from __future__ import annotations

import json

import pytest

from cogno_anima import SCOPE_TOOL_TABLE_HEADER, metakeys as mk
from cogno_anima.prompts import prompt_digest
from cogno_anima.stages.superego import (
    SCOPE_TENANT_FACTS_HEADER, SuperegoStage, _SCOPE_SYSTEM,
)
from tests.unit.test_superego import RaisingBackend, ScriptedBackend, _ctx

_ALLOW = '{"blocked": false, "refusal_message": ""}'

_DEFINITION = "You handle scheduling and study support for the school."
_FACTS = f"{SCOPE_TENANT_FACTS_HEADER}\n- Course catalogue 2026"
_TABLE = (f"{SCOPE_TOOL_TABLE_HEADER}\n"
          "A request one of these serves IS in scope.\n"
          "- consult_material — reads the tenant's published corpus")

#: A slot that renders EVERY row of the closed table — the enumeration below asserts that,
#: so a block introduced later either renders here or fails this file loudly.
_SLOT = f"{_DEFINITION}\n\n{_FACTS}\n\n{_TABLE}"

_USER = "que materiais posso usar para estudar?"


async def _guard(*, slot=_SLOT, user=_USER, language="pt-BR", backend=None):
    """One consulted turn. Returns ``(result, ctx, backend)``."""
    backend = backend or ScriptedBackend([_ALLOW])
    ctx = _ctx(user=user, intent_class="INFORMATION_REQUEST", language=language)
    result = await SuperegoStage().check_input_scope(ctx, backend, scope_prompt=slot)
    return result, ctx, backend


def test_the_absence_is_None_on_the_type_itself_and_not_an_empty_string():
    """The default a caller inherits when it builds a result without one. ``""`` is a VALUE —
    it compares equal to another ``""`` and, in a store, is indistinguishable from a digest
    that was recorded as blank; ``None`` is the claim the record actually makes, which is that
    no prompt was built. The same rule ``prompt_blocks`` carries with ``[]``, and the one
    ``StageMetrics.system_fingerprint`` carries beside it."""
    from cogno_anima.types import ScopeCheckResult, StageMetrics

    empty = ScopeCheckResult(metrics=StageMetrics(stage="superego_scope", elapsed_ms=0.0,
                                                  tokens_in=0, tokens_out=0, model="m"))
    assert empty.prompt_sha is None
    assert empty.prompt_blocks == []


# ── identity: the digest is of the bytes that were SENT ───────────────────────────────────

@pytest.mark.asyncio
async def test_the_digest_is_of_the_bytes_that_were_actually_sent():
    """Identity, not membership. ``prompt_digest`` joins its parts with a newline, so the
    canonical form is ``_SCOPE_SYSTEM`` then the user half — exactly the two arguments
    ``backend.generate`` received, in that order."""
    result, _, backend = await _guard()
    sent = backend.calls[0]
    assert sent["system"] == _SCOPE_SYSTEM
    assert result.prompt_sha == prompt_digest(sent["system"], sent["prompt"])


@pytest.mark.asyncio
async def test_the_digest_is_not_of_the_slot_nor_of_either_half_alone():
    """The three implementations that would look right and answer the wrong question. The
    slot one is the measured defect itself: the host already has that digest, and it is what
    proved "identical" while the classifier flipped."""
    result, _, backend = await _guard()
    sent = backend.calls[0]
    assert result.prompt_sha != prompt_digest(_SLOT)
    assert result.prompt_sha != prompt_digest(sent["prompt"])
    assert result.prompt_sha != prompt_digest(sent["system"])


@pytest.mark.asyncio
async def test_the_same_inputs_give_the_same_digest():
    """It is a LABEL: two turns asking the same thing of the same configuration group
    together, or it groups nothing."""
    first, _, _ = await _guard()
    second, _, _ = await _guard()
    assert first.prompt_sha == second.prompt_sha


# ── sensitivity: every block, enumerated from the closed table ────────────────────────────

@pytest.mark.asyncio
async def test_every_block_of_the_closed_table_changes_the_digest():
    """A digest that ignores a block is the exact blind spot this record exists to close.

    Enumerated from ``_SCOPE_BLOCKS`` rather than from a hand-written list, so a block added
    tomorrow arrives here: the fixture must render every row (asserted first), and one byte
    inserted inside each block's own span must move the digest.
    """
    result, _, backend = await _guard()
    sent = backend.calls[0]
    rendered = [b["block"] for b in result.prompt_blocks]
    assert set(rendered) == {slug for _, slug in SuperegoStage._SCOPE_BLOCKS}, (
        "the fixture no longer renders every block — extend `_SLOT`, do not trim the table")
    # The IDENTITY anchor, repeated here on purpose: "a byte inside this block moves the
    # recorded digest" is only a statement about coverage once the recorded digest is known to
    # be the digest of these bytes. Without it a slot-only implementation would pass the loop
    # below trivially — every mutation differs from a digest of something else.
    assert result.prompt_sha == prompt_digest(sent["system"], sent["prompt"])

    for header, slug in SuperegoStage._SCOPE_BLOCKS:
        at = sent["prompt"].find(header)
        assert at >= 0, slug
        cut = at + len(header)
        mutated = f"{sent['prompt'][:cut]}X{sent['prompt'][cut:]}"
        assert prompt_digest(sent["system"], mutated) != result.prompt_sha, (
            f"one byte inside `{slug}` did not move the digest")

    # ...and the system half, which is not a block and is sent on every call.
    assert prompt_digest(f"{_SCOPE_SYSTEM}X", sent["prompt"]) != result.prompt_sha


@pytest.mark.asyncio
async def test_one_byte_of_the_contacts_sentence_moves_it():
    base, _, _ = await _guard()
    other, _, _ = await _guard(user=f"{_USER}?")
    assert other.prompt_sha != base.prompt_sha


@pytest.mark.asyncio
async def test_one_byte_of_the_hosts_slot_moves_it_on_either_half():
    """The half the host CAN digest, and the half it renders beside it — both are inside."""
    base, _, _ = await _guard()
    definition, _, _ = await _guard(
        slot=f"{_DEFINITION} And enrolment.\n\n{_FACTS}\n\n{_TABLE}")
    table, _, _ = await _guard(
        slot=f"{_DEFINITION}\n\n{_FACTS}\n\n{_TABLE}\n- book_slot — books a slot")
    facts, _, _ = await _guard(
        slot=f"{_DEFINITION}\n\n{_FACTS}\n- Term dates\n\n{_TABLE}")
    assert len({base.prompt_sha, definition.prompt_sha,
                table.prompt_sha, facts.prompt_sha}) == 4


@pytest.mark.asyncio
async def test_the_refusal_language_directive_moves_it():
    """A block assembled HERE out of an input the host never sees as prompt text: the
    language only reaches the guard through the NOUMENO, and it rewrites the task section."""
    base, _, _ = await _guard()
    other, _, _ = await _guard(language="es")
    assert other.prompt_sha != base.prompt_sha


# ── the rule the family shares: absence means NO PROMPT WAS BUILT ─────────────────────────

@pytest.mark.asyncio
async def test_a_bypass_records_None_on_the_result_and_NOTHING_in_the_metadata():
    """``None`` is the same claim the empty ``prompt_blocks`` makes: the model was never
    consulted. Both states are pinned — present after a real call, absent after a bypass —
    because "absent" is only a fact if the other case is visibly different."""
    for kwargs, slot in ((dict(intent_class="SOCIAL"), _SLOT),
                         (dict(intent_class="UNKNOWN", goal_status="ONGOING"), _SLOT),
                         (dict(intent_class="INFORMATION_REQUEST"), "")):
        backend = ScriptedBackend([])
        ctx = _ctx(user="oi", **kwargs)
        result = await SuperegoStage().check_input_scope(ctx, backend, scope_prompt=slot)
        assert backend.calls == []
        assert result.prompt_sha is None
        assert mk.SCOPE_PROMPT_SHA not in ctx.metadata
        assert ctx.metadata[mk.SCOPE_PROMPT_BLOCKS] == []


@pytest.mark.asyncio
async def test_a_bypass_after_a_real_call_does_not_leave_the_earlier_digest_behind():
    """A per-turn fact on a carrier that may hold metadata over. Absence has to be PRODUCED:
    a bypassed turn wearing the previous turn's digest is precisely the lie the record exists
    to prevent, and it would read as "the same prompt was sent" on a turn that sent none."""
    result, ctx, _ = await _guard()
    assert ctx.metadata[mk.SCOPE_PROMPT_SHA] == result.prompt_sha

    ctx.intent.intent_class = "SOCIAL"                      # NER-assisted bypass, same ctx
    bypassed = await SuperegoStage().check_input_scope(
        ctx, ScriptedBackend([]), scope_prompt=_SLOT)
    assert bypassed.prompt_sha is None
    assert mk.SCOPE_PROMPT_SHA not in ctx.metadata


@pytest.mark.asyncio
async def test_the_fail_open_path_records_the_digest_of_the_prompt_it_had_built():
    """"The call blew up" and "the classifier read this and allowed" both come back
    ``blocked=False``. The prompt was BUILT on this path, so it is recorded — the same
    decision the inventory beside it makes, and the digest is the one a clean call produces
    from the same inputs."""
    ctx = _ctx(user=_USER, intent_class="INFORMATION_REQUEST", language="pt-BR")
    result = await SuperegoStage().check_input_scope(
        ctx, RaisingBackend(), scope_prompt=_SLOT)
    clean, _, _ = await _guard()
    assert result.blocked is False
    assert result.prompt_sha == clean.prompt_sha
    assert ctx.metadata[mk.SCOPE_PROMPT_SHA] == result.prompt_sha


@pytest.mark.asyncio
async def test_the_digest_reaches_a_trace_writer_and_not_only_the_result():
    """``ScopeCheckResult`` is consumed by the orchestrator and dropped, so a record living
    only on the result has no reader — and this is the gate that ends turns."""
    result, ctx, _ = await _guard()
    assert ctx.metadata[mk.SCOPE_PROMPT_SHA] == result.prompt_sha
    # A per-turn fact on a serializable carrier a host writes into JSONB.
    assert json.loads(json.dumps(ctx.metadata))[mk.SCOPE_PROMPT_SHA] == result.prompt_sha


# ── only the digest leaves: the canary, with its control ──────────────────────────────────

@pytest.mark.asyncio
async def test_not_one_byte_of_the_prompt_leaves_with_the_digest():
    """The rendered prompt holds the tenant's rules and the contact's own sentence. The
    CONTROL comes first — an absence test over an input that never carried the value proves
    nothing — and it asserts on the PRODUCED prompt, upstream of anything that could sanitize
    it on the way out."""
    rule_canary = "CANARY_TENANT_RULE_7f3a"
    contact_canary = "CANARY_CONTACT_SENTENCE_91b2"

    backend = ScriptedBackend([_ALLOW])
    result, ctx, backend = await _guard(
        slot=f"{_DEFINITION} {rule_canary}\n\n{_FACTS}\n\n{_TABLE}",
        user=f"{_USER} {contact_canary}", backend=backend)

    sent = backend.calls[0]["prompt"]
    for canary in (rule_canary, contact_canary):
        assert canary in sent, "the control failed: the canary never reached the prompt"

    blob = json.dumps({"result": result.model_dump(), "metadata": ctx.metadata},
                      sort_keys=True, default=str)
    for canary in (rule_canary, contact_canary):
        assert canary not in blob, f"{canary} left with the record"
    # The digest itself is hex and nothing else — no room for a byte of anybody's text.
    assert result.prompt_sha is not None
    assert all(c in "0123456789abcdef" for c in result.prompt_sha)
