"""A draft review REJECTED may not be handed to the voice as the executor's own data.

This is not a contradiction between two clauses. It is a door with NO LABEL silencing the door
that has a gate.

## The three lines

``cogno_anima/stages/superego.py``:

1. :meth:`SuperegoStage._tool_payload` ended in ``"\\n".join(parts) or (ctx.ego_result.draft or
   "(no data)")`` — with no tool record at all, **the DRAFT becomes the payload**, and the
   payload is rendered under ``# Data gathered by the executor (ground figures/dates ONLY in
   this)``;
2. :meth:`SuperegoStage._draft_section` opens with ``if not draft or draft in payload: return
   ""`` — the fallback makes the draft's own section fall silent, satisfied that the work is
   already done;
3. its very next line is ``if rejection is not None: return ""``, commented "a rejected draft
   is handled by rejection_section, not re-offered".

So on a turn with NO tool AND a review rejection, the section that knows the draft is poison
closes itself **twice** while the draft reaches the voice anyway, through the only door nobody
gated — **wearing the executor's label**. That is why "do not repeat the rejected claim" and
"reproduce everything in the executor data" point in opposite directions on such a turn: the
rejected claim HAD BEEN PUT in the executor data.

## The pair

The sentences below are the real ones: the contact asked «voces integram com o Bling?» and the
draft answered «Sim, o Cogno integra com o Bling e com o TOTVS.» — an invented integration,
flagged by review (``kind="unverified_claim"``). The fixture puts them in the SHAPE this change
is about, and that shape is stated as a shape rather than as a count: a rejection AND not one
renderable record. It is deliberately NOT "a persona with no tools" — ``_draft_section``'s own
docstring notes that essential tools such as ``resolve_date`` ride every turn for every role,
so a toolless persona still tends to fill ``parts``, and the fixtures for the 2026-08-03 turn
in ``test_superego.py`` carry exactly such a record. No production count is claimed for the
shape; the case for the branch is structural — on a turn that HAS it, the prompt asserted two
incompatible things about the same sentence.

## The fix, and what it deliberately is NOT

When the turn carries a JUDGE rejection the fallback returns ``"(no data)"``. Nothing else.
With the draft out of the payload the exhaustive-reproduction rule no longer reaches it — the
contradiction dies at the root instead of earning an exception — and ``rejection_section`` is
again the only authority over the rejected text, which is what line 3's comment already
promises.

**The predicate is the NARROW one.** ``_rejection`` also answers True for the host's anti-repeat
guard (``kind="repeated_reply"``), where nothing was refused: the draft there is the executor's
NEW answer, written after it obeyed that critique, and ``# Already said (HARD RULE)`` is asking
for exactly that. Under the broad predicate a no-record ``repeated_reply`` turn reaches the
voice with no draft anywhere — the twin below pins it.

**The turn WITHOUT a rejection is untouched, on purpose**, and the branch that justifies it is
the NON-conversational one: with no record, no rejection and no approved verdict
``_draft_section`` returns ``""``, so emptying the payload leaves the voice with nothing at all
— the failure measured on the CLOSER on 2026-08-03, where the model returned the user's own
question. (On a CONVERSATIONAL turn the same emptying would move the draft into
``# Executor's answer`` instead, which is its correct label; that half is a relabelling, not a
repair, and is not attempted here.) Parked by name:
``o-rascunho-viaja-por-duas-portas-e-uma-nao-tem-rotulo``. The byte-identity twin below is that
boundary, stated as an assertion.

Deterministic: every assertion is on the RENDERED voice prompt, never on a model's reply.
"""

from __future__ import annotations

import hashlib

from cogno_anima import metakeys as mk
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.types import EgoResult, EgoStep, ToolExecution

from tests.unit.test_superego import _ctx, _m

# The measured pair, verbatim.
QUESTION = "voces integram com o Bling?"
DRAFT = "Sim, o Cogno integra com o Bling e com o TOTVS."
# The reviewer's critique. It deliberately does NOT quote the draft, so "the rejected text is
# nowhere outside the review section" can be asserted over the whole prompt.
CRITIQUE = "a integração afirmada não está suportada por nenhum dado desta persona"

DATA_HEADER = "# Data gathered by the executor"
REVIEW_HEADER = "# Review verdict (HARD RULE)"


def _turn():
    """The measured turn: a persona with NO tool, and a draft that invented an integration.

    ``draft`` and ``tools_executed`` are PROPERTIES derived from ``steps`` — passing them to
    the constructor neither sets them nor raises, so the draft goes where the model put it:
    the last step's ``assistant_text``.
    """
    ctx = _ctx(user=QUESTION, intent_class="INFORMATION_REQUEST",
               goal="know whether the product integrates with Bling", with_ego=False)
    ctx.ego_result = EgoResult(
        steps=[EgoStep(index=0, path="native", assistant_text=DRAFT, tool_calls=[])],
        metrics=_m("ego"))
    return ctx


def _rendered(ctx, *, kind, reason=CRITIQUE):
    """The prompt the voice receives, over the payload ``voice()`` itself would build."""
    ctx.metadata[mk.VOICE_CORRECTION] = {"reason": reason, "kind": kind}
    return SuperegoStage()._build_voice_prompt(
        ctx, SuperegoStage._tool_payload(ctx), ["general:review"])


def _outside(prompt: str, slug: str) -> str:
    """The prompt with one section removed — what the voice reads everywhere ELSE."""
    section = SuperegoStage.voice_prompt_block(prompt, slug)
    assert section, f"the {slug} section did not render; this helper would prove nothing"
    return prompt.replace(section, "")


# ── the control: this scaffold CAN put the draft in the data block ───
#
# It is not decoration. A test that asserts an ABSENCE has to prove first that it knows how to
# produce the PRESENCE — otherwise the absence may belong to the fixture, and the neighbouring
# `test_voice_does_not_adopt_the_critiques_remedy` records a version of exactly that mistake
# passing over its own mutation. The same fixture, with the rejection removed, is the control:
# the fallback fires and the draft lands INSIDE `# Data gathered by the executor`. That is
# also the branch this change deliberately leaves alone.

def test_the_scaffold_CAN_show_the_draft_inside_the_executor_data():
    ctx = _turn()
    assert SuperegoStage._tool_payload(ctx) == DRAFT, (
        "the fallback did not fire on this fixture — the twin below would be asserting an "
        "absence the scaffold could never have produced")
    prompt = SuperegoStage()._build_voice_prompt(
        ctx, SuperegoStage._tool_payload(ctx), ["general:review"])
    assert DRAFT in SuperegoStage.voice_prompt_block(prompt, "executor_data")


# ── the twin: the rejected draft is not executor data ────────────────

def test_a_rejected_draft_does_not_become_the_executor_data():
    """The defect, on the prompt the voice actually received.

    Mutation: restore ``"\\n".join(parts) or (ctx.ego_result.draft or "(no data)")`` and this
    dies — the payload becomes the rejected sentence again.
    """
    ctx = _turn()
    prompt = _rendered(ctx, kind="unverified_claim")
    assert SuperegoStage._tool_payload(ctx) == "(no data)", (
        "the rejected draft is still the executor data")

    data = SuperegoStage.voice_prompt_block(prompt, "executor_data")
    assert data.startswith(DATA_HEADER)
    assert DRAFT not in data, (
        "the sentence review refused is being presented under 'ground figures/dates ONLY in "
        "this'")

    # ...and nowhere else either: the review section is the only authority over that text.
    assert REVIEW_HEADER in prompt
    assert DRAFT not in _outside(prompt, "review_verdict"), (
        "the rejected claim reached the voice through some other section")


def test_the_draft_section_is_still_the_one_that_withholds_it():
    """The fix does not move the gate — it stops the ungated door from making the gate moot.

    `_draft_section` still returns "" on a rejected turn, so the `draft` block never renders;
    what changed is that the payload no longer carries the same text under another header.
    """
    prompt = _rendered(_turn(), kind="unverified_claim")
    slugs = [b["block"] for b in SuperegoStage.voice_prompt_inventory(prompt)]
    assert "draft" not in slugs
    assert "executor_data" in slugs          # the block still renders, with "(no data)"


def test_the_execution_verdict_variant_is_covered_too():
    """The fallback is gated on `_rejection`, not on a ``kind`` — so the EXECUTION verdict
    (any kind that is not ``unverified_claim``/``repeated_reply``) is covered by the same
    line. A gate per kind would have to be re-derived for the next one."""
    prompt = _rendered(_turn(), kind="not_executed")
    assert "# Execution verdict (HARD RULE)" in prompt
    assert DRAFT not in prompt


def test_a_carrier_with_no_reason_is_not_a_rejection_here_either():
    """A carrier the verdict section does not render must not empty the payload either, or
    the two would disagree about what a rejection is."""
    ctx = _turn()
    ctx.metadata[mk.VOICE_CORRECTION] = {"kind": "unverified_claim", "reason": "  "}
    assert SuperegoStage._tool_payload(ctx) == DRAFT


# ── the anti-repeat guard is NOT a rejection, and that is the narrow predicate's whole job ──

def test_the_anti_repeat_guard_still_gets_the_draft():
    """`repeated_reply` rides the same metadata key and says the OPPOSITE thing: the content
    was fine, it had already been sent. The draft on such a turn is the executor's NEW answer,
    written after it obeyed that critique — and `# Already said (HARD RULE)` is asking the
    voice for precisely something the contact has not received yet.

    `_draft_section` withholds the draft on ANY rejection, so this fallback is its only door.
    Under the broad `_rejection` the prompt came out with the new draft nowhere in it, leaving
    the voice the user's sentence and the Context — where the already-sent reply lives. That
    is the collapse `# Already said` exists to prevent.

    Mutation: swap `_judge_rejection` back to `_rejection` in `_tool_payload` and this dies.
    """
    ctx = _turn()
    ctx.metadata[mk.VOICE_CORRECTION] = {"kind": "repeated_reply",
                                         "reason": "byte-identical to the previous reply"}
    assert SuperegoStage._tool_payload(ctx) == DRAFT, (
        "the anti-repeat guard refused nothing — emptying the payload deletes the only copy "
        "of the answer the executor rewrote")
    prompt = SuperegoStage()._build_voice_prompt(
        ctx, SuperegoStage._tool_payload(ctx), ["general:review"])
    assert "# Already said (HARD RULE)" in prompt
    assert DRAFT in prompt


# ── the boundary: a turn with no tool and NO rejection is byte-identical ──
#
# Measured on the base revision (origin/main e4e7dde2) by rendering `_turn()` with no
# `voice_correction` at all. Recompute the same way on the revision you are comparing against;
# a failure here means the voice prompt moved for a turn this change is not about.
_MAIN_NO_REJECTION = "1b50d8f51e575666"


def test_a_turn_with_no_rejection_renders_byte_for_byte_as_before():
    ctx = _turn()
    prompt = SuperegoStage()._build_voice_prompt(
        ctx, SuperegoStage._tool_payload(ctx), ["general:review"])
    assert DRAFT in prompt, "the parked branch stopped handing the draft over"
    assert hashlib.sha256(prompt.encode()).hexdigest()[:16] == _MAIN_NO_REJECTION, (
        "the no-rejection rendering is not main's — this change must not touch it "
        "(`o-rascunho-viaja-por-duas-portas-e-uma-nao-tem-rotulo`)")


def test_a_turn_that_ran_a_tool_is_untouched_by_the_rejection():
    """The new branch sits AFTER the records are rendered: with any record to render, the
    rejection changes nothing about the payload. That is the whole reach of this change."""
    ctx = _turn()
    ctx.ego_result = EgoResult(
        steps=[EgoStep(index=0, path="native", assistant_text=DRAFT,
                       tool_calls=[ToolExecution(tool="resolve_date", arguments={},
                                                 result="2026-09-23", ok=True,
                                                 side_effect=False, tool_mutating=False)])],
        metrics=_m("ego"))
    without = SuperegoStage._tool_payload(ctx)
    ctx.metadata[mk.VOICE_CORRECTION] = {"kind": "unverified_claim", "reason": CRITIQUE}
    assert SuperegoStage._tool_payload(ctx) == without == "resolve_date: 2026-09-23"


def test_no_new_header_so_the_persisted_inventory_does_not_move():
    """Same shape every voice change of this family took: nothing is added to `_VOICE_BLOCKS`,
    so `test_voice_blocks_sync` and the host's persisted inventory stay where they are.

    Asserted as the EXACT sequence, not as membership of the table: `voice_prompt_inventory`
    emits slugs FROM `_VOICE_BLOCKS` and never from the text it matched (that closed alphabet
    is its stated safety property), so "every slug is a known slug" is true of any string ever
    passed to it — a tautology downstream of the sanitizer, which is the shape this repo has
    already been caught by. The table's length is pinned beside it, so a new row fails here.
    """
    prompt = _rendered(_turn(), kind="unverified_claim")
    slugs = [b["block"] for b in SuperegoStage.voice_prompt_inventory(prompt)]
    assert slugs == ["user_request", "executor_data", "review_verdict", "signals", "task"]
    # 10 when this was written; 11 since the contact's note (`contact_memo`) got its own
    # section — a row that renders ONLY on a turn carrying a note, so this turn's slugs above
    # are untouched.
    assert len(SuperegoStage._VOICE_BLOCKS) == 11, (
        "a block was added or removed; this change adds no header, so the table must not move")
