"""One rule about DERIVED values, and it has to reach both doors.

## The two live cases

* **V2** — an executor read the tenant's configured rate and the hours worked; the reply said
  ``R$ 120,00 x 4 h = R$ 480,00``. The judge refused it under a "quote, never calculate"
  reading of its own grounding criterion.
* **X02** — a tool answered that the business is *"closed Saturdays and Sundays"*; the reply
  said *"Monday to Friday"*. Refused for the same reason: those three words appear in no tool
  result, in no rule and in no context block. They were WORKED OUT.

Both refusals are correct applications of what the judge was told. ``_GROUNDING_SOURCES`` —
the constant BOTH judging branches read — ends *"Reject a figure, name, date, policy or claim
that appears in NONE of the three"*, and a derived value appears, character for character, in
none of the three: it was computed, not quoted.

## Why this is a RECONCILIATION and not a new permission

The permission already existed, one constant away and one day older.
``_FIGURES_HAVE_A_SOURCE`` (the voice's ``# Task`` rule) has said since 2026-09-08 that a
figure *"you WORK OUT from those is allowed ONLY if you show the calculation in the reply"*,
and it gives ``"4 h x R$ 120,00 = R$ 480,00"`` as its own worked example. The judge was never
told. The comment on that constant already names this shape as a defect — *"a voice held to a
NARROWER set than the judge approves is the two-doors defect"* — and this is the same defect
with the doors swapped, which is the worse arrangement: a narrow VOICE only says the answer
less well, a narrow JUDGE rejects it and starts the correction loop.

**And the two doors were not merely disagreeing in prose — the deterministic net enforced the
narrow reading.** ``_draft_divergence`` classified a figure as ``invented`` when it was in
neither the approved draft nor the tool data nor the contact's message, which a derived total
never is. Run against the worked example printed inside ``_FIGURES_HAVE_A_SOURCE`` it returned
``invented=['48000']`` and fired a re-voice whose instruction reads *"no figure may appear that
is not in that answer or in the executor data"*. **The rule's own example was refused by the
check beside it.** That is the mutation this file's twins are built on.

## The permission is granted by the ARITHMETIC, not by the format

A format-only rule ("there is an ``=``, let it through") would admit the fabrication this repo
already measured: ``turn_traces`` 1795 shipped *"Totalizando 20h, o que resultaria em
R$ 2.400,00"* — a sentence that shows its work over inputs whose product is not 2 400. So
``_shown_derivations`` requires three things and each of its inverses is pinned below: the
operation written down, every operand present in the evidence, and a result the operation
actually produces.

Deterministic throughout — no model is called. The prompt half asserts on the RENDERED prompt
(a test on a symbol keeps passing over a sentence rewritten into meaninglessness) and the
arithmetic half on the shipped predicate.
"""

from __future__ import annotations

import pytest

from cogno_anima import metakeys as mk
from cogno_anima.stages.superego import (
    JUDGE_CONVERSATIONAL_BRANCH, JUDGE_EXECUTION, JUDGE_READONLY,
    SuperegoStage, _DERIVED_FROM_EVIDENCE, _FIGURES_HAVE_A_SOURCE, _GROUNDING_SOURCES)
from cogno_anima.types import EgoResult, EgoStep, ToolExecution
from tests.unit.test_superego import _ctx, _m

# ── V2, as data ───────────────────────────────────────────────────────────────────────
# Neutral stand-ins for the live turn — this repo is public. The rate is in the tool data,
# the hours are in the approved draft, and the product is in NEITHER.
_DRAFT = "The configured rate is R$ 120,00 per hour and you taught 4 hours."
_PAYLOAD = "lookup_rules: 'Class - R$ 120,00 per hour'"
_SHOWN = "You taught 4 h at R$ 120,00 per hour: 4 x R$ 120,00 = R$ 480,00."


def _diverges(reply: str, *, draft: str = _DRAFT, payload: str = _PAYLOAD,
              user_input: str = "") -> "tuple[list[str], list[str]]":
    return SuperegoStage._draft_divergence(draft, payload, reply, user_input)


# ── TWIN 1 (V2): a shown calculation over grounded inputs is not an invention ─────────

def test_the_worked_example_of_the_voice_rule_is_not_called_invented():
    """The rule prints ``4 h x R$ 120,00 = R$ 480,00`` as ALLOWED. The check beside it used to
    answer ``invented=['48000']`` over that exact sentence and fire a re-voice."""
    assert "4 h x R$ 120,00 = R$ 480,00" in _FIGURES_HAVE_A_SOURCE
    lost, invented = _diverges(_SHOWN)
    assert invented == [], f"the rule's own worked example was refused: {invented}"
    assert lost == []


def test_a_sum_of_two_grounded_figures_is_not_an_invention():
    """Not only multiplication: the operator set is whatever the reply wrote, as long as it
    wrote ONE kind of it."""
    assert _diverges("R$ 120,00 + R$ 120,00 = R$ 240,00.") == ([], [])


def test_the_derived_total_still_has_to_be_stated_with_its_inputs():
    """The relaxation is scoped to the SENTENCE, not to the turn: a reply may derive a figure
    and still lose one the approved draft states, and the ``lost`` half is untouched."""
    lost, invented = _diverges("4 x R$ 120,00 = R$ 480,00, and nothing else applies.",
                               draft=_DRAFT + " The bonus is R$ 30,00.")
    assert invented == []
    assert lost == ["3000"], "a figure the approved draft states is still owed to the contact"


# ── TWIN 1's INVERSES: each of the three conditions, refused on its own ───────────────

def test_a_calculation_whose_arithmetic_does_not_come_out_is_still_invented():
    """The measured fabrication (``turn_traces`` 1795): a sentence that SHOWS ITS WORK and
    whose work is wrong. A format-only rule would have shipped it."""
    _, invented = _diverges("Totalling 20 h, which gives 4 x R$ 120,00 = R$ 2.400,00.")
    assert "240000" in invented


def test_a_bare_total_with_no_calculation_shown_is_still_invented():
    """The other pole the constant refuses: the right number, unverifiable."""
    _, invented = _diverges("The total is R$ 480,00.")
    assert invented == ["48000"]


def test_a_calculation_whose_operands_are_not_in_the_evidence_is_still_invented():
    """The arithmetic is impeccable (9 x 130 = 1 170) and NEITHER input is on the page."""
    _, invented = _diverges("9 x R$ 130,00 = R$ 1.170,00.")
    assert "117000" in invented and "13000" in invented


def test_a_mixed_operator_expression_is_not_guessed_at():
    """Reading precedence out of ``400 + 80 x 0`` is guessing, and a wrong guess here ADMITS a
    figure. The parser declines and the figure stays refused.

    The case is built so that ONLY this guard can refuse it, which is what a mutation has to
    be able to prove: folded left-to-right with the FIRST operator the expression comes to
    ``400 + 80 + 0 = 480``, exactly the total claimed — so the arithmetic check downstream is
    satisfied and would let it through. Read with real precedence it is ``400 + 0 = 400``, and
    the figure ``480,00`` is in neither the draft nor the payload. A first version of this test
    used ``2 + 3 x 120 = 360``, where every fold is wrong and the arithmetic check killed it
    first: the mutation removing this guard changed nothing and the test passed for the wrong
    reason."""
    draft = "The base is R$ 400,00, the supplement is R$ 80,00 and the surcharge is 0."
    _, invented = _diverges("R$ 400,00 + R$ 80,00 x 0 = R$ 480,00.", draft=draft)
    assert "48000" in invented


def test_the_helper_can_only_ever_remove_a_figure_from_the_invented_set():
    """The structural property behind every inverse above: whatever ``_shown_derivations``
    answers, it is intersected OUT of a set the old code had already computed."""
    for reply in (_SHOWN, "The total is R$ 480,00.", "9 x R$ 130,00 = R$ 1.170,00.",
                  "2 + 3 x R$ 120,00 = R$ 360,00."):
        _, invented = _diverges(reply)
        old = sorted(k for k in SuperegoStage._figure_keys(reply)
                     if k not in SuperegoStage._figure_keys(_DRAFT)
                     and k not in SuperegoStage._figure_keys(_PAYLOAD))
        assert set(invented) <= set(old), reply


# ── the arithmetic reader, on its own ────────────────────────────────────────────────

@pytest.mark.parametrize("token,value", [
    ("4", 4.0), ("120,00", 120.0), ("2.400,00", 2400.0), ("1.250", 1250.0), ("480.00", 480.0),
])
def test_a_numeral_is_read_the_way_the_rest_of_this_stage_reads_numerals(token, value):
    """Digit-strings, like ``_figure_keys``: two trailing decimals make a FIGURE, anything else
    is a whole number with its grouping stripped. A second convention here would disagree with
    ``_figure_keys`` about the very tokens the two are asked about together."""
    assert SuperegoStage._numeral_value(token) == value


def test_the_evidence_offers_an_operand_in_both_of_its_readings():
    """``R$ 120,00`` in the payload has to ground an operand the reply wrote as ``120`` — the
    same widening the ``lost`` half already does for the reply."""
    forms = SuperegoStage._numeral_forms("4 hours at R$ 120,00")
    assert {"4", "120", "12000"} <= forms


# ── TWIN 2 (X02): the judge is told the same rule, on both branches it judges with ────

def _turn(*, wrote: bool = False, conversational: bool = False):
    """A turn whose read SUCCEEDED and whose draft restates what it returned."""
    ctx = _ctx(user="what days are you open?", intent_class="INFORMATION_REQUEST",
               goal="find out the opening days", with_ego=False)
    ctx.ego_result = EgoResult(
        steps=[
            EgoStep(index=0, path="native", assistant_text="", tool_calls=[
                ToolExecution(tool="get_schedule_settings", arguments={},
                              result="closed Saturdays and Sundays", ok=True,
                              side_effect=wrote, tool_mutating=wrote)]),
            EgoStep(index=1, path="native", assistant_text="We are open Monday to Friday."),
        ], metrics=_m("ego"))
    if conversational:
        ctx.metadata[mk.JUDGE_CONVERSATIONAL] = True
    return ctx


def _prompt(**kw) -> "tuple[str, str]":
    ctx = _turn(**kw)
    st = SuperegoStage()
    return st._judge_branch(ctx), st._build_judge_prompt(ctx, "# Persona limits\nBe helpful.")


@pytest.mark.parametrize("kw,expect", [({}, JUDGE_READONLY), ({"wrote": True}, JUDGE_EXECUTION)])
def test_the_judge_is_told_that_a_derivation_over_evidence_is_grounded(kw, expect):
    """X02's half. The judge that refused "Monday to Friday" was applying its criteria
    correctly; the criteria had to stop asserting the opposite."""
    branch, prompt = _prompt(**kw)
    assert branch == expect
    assert _DERIVED_FROM_EVIDENCE in prompt
    assert "restatement that follows necessarily from a fact that is in it" in prompt


def test_the_rule_carries_its_own_refusal_and_not_only_its_permission():
    """A relaxation whose text says only what is now allowed is a licence. Both conditions
    travel in the same sentence the model reads."""
    assert "must show the work" in _DERIVED_FROM_EVIDENCE
    assert "A bare result whose inputs are not on this page is NOT grounded" in _DERIVED_FROM_EVIDENCE
    assert "neither is one whose arithmetic does not come out" in _DERIVED_FROM_EVIDENCE


@pytest.mark.parametrize("kw", [{}, {"wrote": True}])
def test_a_fact_in_none_of_the_sources_is_still_fabrication(kw):
    """The sentence the reconciliation must NOT have deleted. A derivation is admitted; an
    invention is refused exactly as hard as before."""
    assert "appears in NONE of the three" in _prompt(**kw)[1]


def test_the_rule_is_written_once_and_reaches_both_doors():
    """``_ADMITTING_A_LIMIT``'s rule applied a third time — and this one has ALREADY diverged
    once, which is the whole reason the constant exists."""
    assert _DERIVED_FROM_EVIDENCE in _FIGURES_HAVE_A_SOURCE
    assert _DERIVED_FROM_EVIDENCE in _GROUNDING_SOURCES
    ro, ex = _prompt()[1], _prompt(wrote=True)[1]
    assert ro.count(_DERIVED_FROM_EVIDENCE) == 1 and ex.count(_DERIVED_FROM_EVIDENCE) == 1


def test_the_conversational_branch_is_untouched():
    """It reads neither constant. A persona with no tool executes nothing, so there is no
    executor evidence to derive FROM, and this branch must render as it always did."""
    branch, prompt = _prompt(conversational=True)
    assert branch == JUDGE_CONVERSATIONAL_BRANCH
    assert _GROUNDING_SOURCES not in prompt and _DERIVED_FROM_EVIDENCE not in prompt


# ── the guards can fail ──────────────────────────────────────────────────────────────

def test_the_assertions_would_catch_the_sentence_being_removed():
    """A guard that cannot fail passes for the wrong reason. Feed every predicate above the
    PREVIOUS text of each constant and each one must reject it."""
    before_judge = (
        "What counts as GROUNDED: the tool results are not the only ground truth in this "
        "prompt. Reject a figure, name, date, policy or claim that appears in NONE of the "
        "three.")
    assert _DERIVED_FROM_EVIDENCE not in before_judge
    assert "restatement that follows necessarily" not in before_judge
    assert "appears in NONE of the three" in before_judge     # the half that must SURVIVE

    before_voice = (
        "A figure you WORK OUT from those is allowed ONLY if you show the calculation in the "
        "reply, every input and the operation, so the reader can check it.")
    assert "arithmetic does not come out" not in before_voice
    assert "must show the work" not in before_voice
