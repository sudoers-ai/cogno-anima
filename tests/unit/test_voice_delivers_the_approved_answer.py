"""The voice may not DENY, drop or replace what an APPROVED draft affirms.

EGO=executor, SUPEREGO=locutor — and on an execution turn the locutor never saw what the
executor wrote. `_draft_section` surfaced the draft only on a CONVERSATIONAL turn; on every
other turn the voice was handed the tool payload and the retrieved context and nothing else.

Measured on the demo box over 1369 traces carrying a SUPEREGO block: **650 turns** where review
APPROVED an execution and the approved draft never entered the voice prompt. On 13 of them a
figure the draft states is missing from the delivered reply; on 22 the reply carries a figure
found in NEITHER the draft NOR the tool data; on 54 the reply admits a limit the draft did not.

One trace is all three at once. Under a delegation lock the specialist drafted the tenant's own
configured rate, review approved it at the FIRST attempt (`attempts: 1`), and the reply the
contact read was "não consegui encontrar informações" — twice in three runs — and, on the third,
a rate that appears only in the retrieved memories. The voice prompt for those turns carried
`user_request, context, executor_data, traits, signals, task`, and the whole of `executor_data`
was one line: a lookup tool answering that it found no records. **The voice was structurally
incapable of delivering the approved answer.** No sharpening of its instructions could have
changed that, and the missing `execution_verdict` block was not the cause: that section is
conditional on a judge REJECTION, and these turns were approved.

Three properties are pinned here:

* the approved draft reaches the voice as the PRIMARY source (and the prose that says so);
* the floor does NOT move — an unapproved draft, or one from a loop with a FAILED call, is
  still withheld, which is what `test_voice_surfaces_a_failed_read_so_it_cannot_fabricate`
  has always guarded;
* the deterministic net under the new promise: a reply that loses the approved figures or
  invents one is re-voiced ONCE, both outcomes counted.
"""

from __future__ import annotations

import pytest

from cogno_anima import metakeys as mk
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.types import EgoResult, EgoStep, ToolExecution
from tests.unit.test_superego import ScriptedBackend, _ctx, _m

# The shape of the measured turn: one read that SUCCEEDED and found nothing, and a draft whose
# figures come from the tenant's own configured rules (which live in the EXECUTOR's prompt, not
# in any tool result). Neutral stand-ins — this repo is public.
_DRAFT = ("The configured rate is R$ 120,00 per hour. Bonus: R$ 30,00/hour above 80%, "
          "R$ 40,00/hour above 90%.")
_EMPTY_READ = "lookup_rules: No records found."


def _turn(*, approved=True, ok=True, draft=_DRAFT, interrupted=False):
    ctx = _ctx(user="what is the hourly rate and the bonus rules?")
    ctx.ego_result = EgoResult(
        steps=[EgoStep(index=0, path="native", assistant_text=draft,
                       tool_calls=[ToolExecution(tool="lookup_rules", arguments={},
                                                 result="No records found." if ok else "",
                                                 error="" if ok else "lookup failed",
                                                 ok=ok, side_effect=False)])],
        interrupted=interrupted, metrics=_m("ego"))
    if approved:
        ctx.metadata[mk.JUDGE_VERDICT] = {"approved": True, "attempts": 1}
    return ctx


async def _voice(ctx, replies):
    b = ScriptedBackend(list(replies))
    result = await SuperegoStage().voice(ctx, b, voice_prompt="persona")
    return result, b


# ── (a) the approved draft is the voice's PRIMARY source ─────────────────────────────

@pytest.mark.asyncio
async def test_the_approved_draft_reaches_the_voice_on_an_execution_turn():
    result, b = await _voice(_turn(), [_DRAFT])
    prompt = b.calls[0]["prompt"]
    assert "R$ 120,00 per hour" in prompt, (
        "the approved draft is not in the voice prompt — this is the 650-turn defect")
    assert "REVIEWED AND APPROVED" in prompt
    assert "PRIMARY" in prompt and "background" in prompt
    # ...and the inventory the host persists says so, so a turn can be diagnosed after the fact
    assert "draft" in [b_["block"] for b_ in result.prompt_blocks]


@pytest.mark.asyncio
async def test_silence_in_the_tool_data_is_not_a_contradiction():
    """The measured turn's whole executor_data was a read that found NOTHING. Without this
    clause the existing precedence sentence ("the executor data wins on any figure") reads as
    licence to treat that silence as a refutation of the approved answer — which is exactly the
    reply the contact got."""
    _, b = await _voice(_turn(), [_DRAFT])
    prompt = b.calls[0]["prompt"]
    assert "SILENCE is not a contradiction" in prompt
    assert "you must state it anyway" in prompt


# ── (b) the voice may not admit a limit the executor does not have ───────────────────

@pytest.mark.asyncio
async def test_the_voice_is_forbidden_from_claiming_a_limit_the_draft_disproves():
    """The half that is worse than a wrong figure: "I could not find that information" over a
    draft that found it. The contact acts on it and may not ask again."""
    _, b = await _voice(_turn(), [_DRAFT])
    prompt = b.calls[0]["prompt"]
    assert "MUST NOT" in prompt
    assert "could not find, do not have, or cannot access" in prompt
    assert "costs them the answer" in prompt


@pytest.mark.asyncio
async def test_the_answer_still_loses_to_tool_data_that_CONTRADICTS_it():
    """The other half of the same section, and it is not symmetry — it is the price of the
    change, measured.

    Handing the voice an approved draft makes review's verdict load-bearing, so the honest
    question is what happens when the verdict is WRONG. Measured with the served voicer
    (`gpt-4o-mini`, n=5) on a draft offering slots that `check_availability` had said did not
    exist, with the approval stamped by hand: with the first wording of this section the
    fabrication shipped **4/5**; naming the contradiction case explicitly, and putting it LAST
    where a hard rule belongs, took it to **0/5** — with the real defect still at 5/5 on both
    the served voicer and the preset's `gpt-4.1-nano`.

    That case is a counterfactual and this test does not pretend otherwise: the served judge
    (`gpt-5.6-luna`, n=5) approves that draft **0/5**, critique naming the contradiction, while
    it approves the real turn's draft **5/5**. The gate is real. This clause is what stands
    under it anyway."""
    _, b = await _voice(_turn(), ["ok"])
    prompt = b.calls[0]["prompt"]
    assert "where the executor data does SPEAK about the very thing this answer offers" in prompt
    assert "then the executor data is what actually happened and this answer is wrong" in prompt
    assert "drop the part that contradicts it" in prompt


# ── the floor does NOT move ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_an_unapproved_draft_is_still_withheld():
    """The mutation twin of the first test. Without review's approval the draft is exactly what
    it was before this change: an optimistic narration, and not grounding."""
    _, b = await _voice(_turn(approved=False), [_DRAFT])
    assert "R$ 120,00 per hour" not in b.calls[0]["prompt"]
    assert "REVIEWED AND APPROVED" not in b.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_a_draft_from_a_loop_with_a_failed_call_is_still_withheld():
    """The anti-fabrication floor, and it needs its OWN test rather than the inherited green.

    `test_voice_surfaces_a_failed_read_so_it_cannot_fabricate` stamps no verdict, so its draft
    is withheld by the APPROVAL gate and it never reaches the clean-loop one: measured — remove
    `_loop_ran_clean` from `_approved_draft` and that test still passes. This one stamps the
    approval AND fails the call, which is the combination the floor actually has to survive."""
    _, b = await _voice(_turn(ok=False), [_DRAFT])
    assert "R$ 120,00 per hour" not in b.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_an_interrupted_loop_is_still_withheld():
    _, b = await _voice(_turn(interrupted=True), [_DRAFT])
    assert "REVIEWED AND APPROVED" not in b.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_a_malformed_or_absent_verdict_changes_nothing():
    """Absence of the carrier makes the rule STRICTER, never off — and a merely truthy value
    is not a verdict. A host that stamps nothing gets the prompt it had before."""
    for verdict in (None, {}, {"approved": False}, {"approved": "yes"}, "approved", 1):
        ctx = _turn(approved=False)
        if verdict is not None:
            ctx.metadata[mk.JUDGE_VERDICT] = verdict
        _, b = await _voice(ctx, [_DRAFT])
        assert "REVIEWED AND APPROVED" not in b.calls[0]["prompt"], verdict


@pytest.mark.asyncio
async def test_a_rejected_draft_is_still_not_re_offered_on_an_execution_turn():
    """A rejection outranks an approval stamp from an earlier attempt: the verdict section is
    there to make the voice DROP the draft, and handing it back as approved content in the same
    prompt would undo that."""
    ctx = _turn()
    ctx.metadata[mk.VOICE_CORRECTION] = {"reason": "did not do what was asked",
                                         "kind": "not_executed"}
    _, b = await _voice(ctx, ["ok"])
    assert "REVIEWED AND APPROVED" not in b.calls[0]["prompt"]
    assert "Execution verdict (HARD RULE)" in b.calls[0]["prompt"]


# ── (c) the deterministic net under the new promise ──────────────────────────────────

@pytest.mark.asyncio
async def test_a_reply_that_denies_the_approved_answer_is_re_voiced_once():
    """The exact reply two runs in three delivered."""
    result, b = await _voice(_turn(), ["Infelizmente, não consegui encontrar informações.",
                                      "A taxa é R$ 120,00 por hora, com bónus de R$ 30,00 e "
                                      "R$ 40,00."])
    assert len(b.calls) == 2, "the net did not re-voice"
    assert "REWRITE" in b.calls[1]["prompt"]
    assert "R$ 120,00" in result.response
    assert "voice:diverged_from_approved_draft" in result.adjustments
    assert "voice:revoiced_from_approved_draft" in result.adjustments


@pytest.mark.asyncio
async def test_a_figure_that_is_in_neither_the_draft_nor_the_tool_data_is_caught():
    """The third run: a rate that exists only in the retrieved memories. The context is
    BACKGROUND — deliberately not a source of figures for this net."""
    result, b = await _voice(_turn(), ["O valor é R$ 45,00 por hora.",
                                       "A taxa é R$ 120,00 por hora, mais R$ 30,00 e R$ 40,00."])
    assert len(b.calls) == 2
    assert "voice:diverged_from_approved_draft" in result.adjustments
    assert "45,00" not in result.response


@pytest.mark.asyncio
async def test_a_faithful_reply_is_left_alone():
    """The discrimination twin. A net that fires on a correct reply is the mechanism."""
    result, b = await _voice(_turn(), ["A taxa é de R$ 120,00 por hora; o bónus é R$ 30,00 "
                                       "acima de 80% e R$ 40,00 acima de 90%."])
    assert len(b.calls) == 1, "the net re-voiced a faithful reply"
    assert not [a for a in result.adjustments if a.startswith("voice:diverged")]


@pytest.mark.asyncio
async def test_a_rounded_restatement_is_not_a_divergence():
    """"R$ 120" carries "R$ 120,00". Comparing full digit strings alone would re-voice this."""
    result, b = await _voice(_turn(), ["São R$ 120 por hora, com bónus de R$ 30 e R$ 40."])
    assert len(b.calls) == 1
    assert not [a for a in result.adjustments if a.startswith("voice:diverged")]


@pytest.mark.asyncio
async def test_a_worse_re_voice_is_not_shipped():
    """A net that can make a turn worse is not a net. The first reply is kept, and the trace
    still records that the predicate fired — the firing is the count, the swap is the action,
    and reading them apart is how "is this net doing work" gets a denominator."""
    result, b = await _voice(_turn(), ["A taxa é R$ 120,00 e o bónus R$ 30,00.",
                                       "Não sei."])
    assert len(b.calls) == 2
    assert "R$ 120,00" in result.response, "the worse re-voice was shipped"
    assert "voice:diverged_from_approved_draft" in result.adjustments
    assert "voice:revoiced_from_approved_draft" not in result.adjustments


@pytest.mark.asyncio
async def test_the_net_only_guards_the_turns_the_promise_was_made_on():
    """Scope, and it is load-bearing. Judged against a draft the voice never saw, this
    predicate would have fired on all 650 measured turns at once — that is not a net, it is a
    second defect. No approval → no promise → no net."""
    result, b = await _voice(_turn(approved=False), ["Não consegui encontrar informações."])
    assert len(b.calls) == 1
    assert not [a for a in result.adjustments if a.startswith("voice:diverged")]


@pytest.mark.asyncio
async def test_both_voicer_calls_are_billed():
    """The re-voice is a real LLM call; a net whose cost is invisible is a net that gets
    blamed on the model."""
    one, _ = await _voice(_turn(), ["A taxa é R$ 120,00, R$ 30,00 e R$ 40,00."])
    two, _ = await _voice(_turn(), ["Não consegui encontrar.",
                                    "A taxa é R$ 120,00, R$ 30,00 e R$ 40,00."])
    assert two.metrics.tokens_in == 2 * one.metrics.tokens_in
    assert two.metrics.tokens_out == 2 * one.metrics.tokens_out


# ── the pure predicate, exercised directly ───────────────────────────────────────────

def test_the_figure_reader_ignores_counts_and_date_fragments():
    """`_NUM_RE` matches the "2" of "2 items" and both halves of "08/09". A net built on it
    would re-voice a correct reply for writing a date in words."""
    keys = SuperegoStage._figure_keys
    assert keys("2 items on 08/09 at 10h") == {}
    assert set(keys("R$ 1.234,56 and 120,00")) == {"123456", "12000"}


def test_the_divergence_is_symmetric_and_grounded():
    d = SuperegoStage._draft_divergence
    assert d("R$ 120,00", "", "R$ 120,00") == ([], [])
    assert d("R$ 120,00", "", "não encontrei") == (["12000"], [])
    assert d("R$ 120,00", "", "R$ 120,00 e R$ 45,00") == ([], ["4500"])
    # a figure the TOOL returned is grounded, not invented
    assert d("R$ 120,00", "total: R$ 45,00", "R$ 120,00 e R$ 45,00") == ([], [])
    # ...and so is one the CONTACT themselves typed
    assert d("R$ 120,00", "", "R$ 120,00 e R$ 45,00", "eu paguei R$ 45,00") == ([], [])
