"""The judge's GROUNDING enumeration may name only the sections THIS prompt actually carries.

## What was measured

`_GROUNDING_SOURCES` (#156, `bdbbf6f`) widened the closed set of ground truth from "the tool
results" to "the tool results, `# Persona limits`, `# Context`", because a tenant's own
configured rate was being called a fabrication when three reads came back empty
(`turn_traces` id=1440). That correction is right and this file does not touch it.

What it did not carry is a guard on its own evidence. Both sections are OPTIONAL — `limits`
renders only for a non-blank `limits_prompt`, `context` only when the host injected one — and
on a turn where NEITHER rendered, every clause of the widened sentence is FALSE about the
prompt the judge is holding. Worse than false: "a search that returned nothing does NOT prove
that a fact stated in those sections is invented" hands a fabricating draft an alibi, and the
sections it points at are not there to be checked.

That is not hypothetical. The nightly canary's read-only twin —
`tests/integration/test_superego.py::test_judge_still_rejects_a_read_whose_draft_invents`, an
empty `get_schedule` plus a draft listing "Algebra" and "Physics", written with the docstring
"the one that must die if it goes lax" — APPROVED on qwen3:8b, `approved=True, critique=None`,
on run 35207468672 (2026-09-17) and on every scheduled run back to 2026-09-09. A deterministic
probe over the RENDERED prompt for that exact turn found its only top-level sections to be
`# User request`, `# Active goal`, `# What the EGO executed`, `# EGO draft` and the criteria:
`# Persona limits` and `# Context` were both absent.

So the fix is the rule `_OUT_OF_REACH` already follows — render the opening only when the block
it depends on is really there — applied to the clause that needed it just as badly.

## Why the assertions read the RENDERED prompt

The change is PROSE and it is CONDITIONAL, so there are two renderings and a test on a symbol
would see neither. Every assertion below reads what `_build_judge_prompt` actually produced,
and the last test proves these predicates can fail.
"""

from __future__ import annotations

import pytest

from cogno_anima import metakeys as mk
from cogno_anima.stages.superego import (
    JUDGE_EXECUTION, JUDGE_READONLY, SuperegoStage,
    _GROUNDING_SOURCE_SET, _GROUNDING_SOURCE_SET_NONE)
from cogno_anima.types import EgoResult, EgoStep, ToolExecution
from tests.unit.test_superego import _ctx, _m

#: The tenant shape #156 was measured on: rules rendered into the judge's `limits` slot.
LIMITS = (
    "Answer within the persona's scope.\n\n"
    "# Tenant rules (legitimate grounding)\n"
    "# Valores Financeiros\n - Aula - R$ 120,00 por hora\n"
)

#: The other optional grounding section — host-injected clock/memories/history.
CONTEXT = "[TODAY] 2026-09-17\nThe contact was told the rate is R$ 120,00/h last week."

#: The canary turn, reduced: one SUCCESSFUL read that came back empty, and a draft that fills
#: the emptiness with content no tool returned.
INVENTING_DRAFT = "You have Algebra on 5 October at 09:00 and Physics on 12 October at 14:00."


def _turn(*, wrote: bool = False, context: bool = False):
    ctx = _ctx(user="e as aulas do mês que vem?", intent_class="INFORMATION_REQUEST",
               goal="list the classes scheduled for next month", with_ego=False)
    ctx.ego_result = EgoResult(
        steps=[EgoStep(index=0, path="native", assistant_text=INVENTING_DRAFT, tool_calls=[
            ToolExecution(tool="get_schedule", arguments={"month": "2026-10"},
                          result="No classes found.", ok=True,
                          side_effect=wrote, tool_mutating=wrote)])],
        metrics=_m("ego"))
    if context:
        ctx.metadata[mk.EGO_CONTEXT] = CONTEXT
    return ctx


def _prompt(*, limits: str = "", wrote: bool = False, context: bool = False):
    ctx = _turn(wrote=wrote, context=context)
    st = SuperegoStage()
    return st._judge_branch(ctx), st._build_judge_prompt(ctx, limits)


def _headers(prompt: str) -> set[str]:
    """The top-level sections this prompt RENDERED, by line start.

    Both criterion variants name `'# Persona limits'` and `'# Context'` inside their own
    sentences, so `"# Context" in prompt` is true even on the prompt that carries neither —
    which is precisely the question these tests exist to ask. Headers only."""
    return {line for line in prompt.splitlines() if line.startswith("#")}


# ── half one: neither section rendered → the closed set is a set of ONE ───────────────

@pytest.mark.parametrize("kw,expect_branch", [
    ({}, JUDGE_READONLY),
    ({"wrote": True}, JUDGE_EXECUTION),
])
def test_with_neither_section_the_criterion_stops_naming_them(kw, expect_branch):
    """The measured defect. A prompt with no `# Persona limits` and no `# Context` must not
    tell the judge that either one might be holding the draft up."""
    branch, prompt = _prompt(**kw)
    assert branch == expect_branch
    # The sections really are absent — without this the test could pass over a prompt where
    # they rendered, which is the confusion the guard exists to prevent. Checked as HEADERS,
    # at the start of a line: both criterion variants QUOTE these names in their own prose, so
    # a substring test over the whole prompt answers a different question than it looks like.
    assert not _headers(prompt) & {"# Persona limits", "# Context (authoritative"}
    assert _GROUNDING_SOURCE_SET not in prompt, (
        f"[{branch}] the criterion still offers two sections this prompt does not carry")
    assert _GROUNDING_SOURCE_SET_NONE in prompt, branch


@pytest.mark.parametrize("kw", [{}, {"wrote": True}])
def test_the_alibi_sentence_goes_with_the_sections_it_names(kw):
    """The half that made it a licence: an empty read cannot be excused by sections that are
    not in the prompt. Both halves asserted, the way `_GROUNDING_SOURCES`' own tests do."""
    _, prompt = _prompt(**kw)
    assert "does NOT prove that a fact stated in those sections is invented" not in prompt
    assert "the tool results above are the ONLY ground truth this reply has" in prompt
    assert "an empty read leaves NOTHING for the draft to have read it from" in prompt


@pytest.mark.parametrize("kw", [{}, {"wrote": True}])
def test_the_set_is_still_CLOSED(kw):
    """The floor, unchanged in force and re-counted in wording: a claim grounded in nothing is
    still rejected. Drop this and the substitution stops being a correction."""
    _, prompt = _prompt(**kw)
    assert "Reject a figure, name, date, policy or claim that appears in NONE of the tool " \
           "results above" in prompt


@pytest.mark.parametrize("kw", [{}, {"wrote": True}])
def test_the_derived_values_rule_is_NOT_dropped_with_the_source_list(kw):
    """`_DERIVED_FROM_EVIDENCE` is appended to the same constant and is orthogonal to WHICH
    sections exist — arithmetic over figures that ARE on the page stays grounded. Taking it
    out with the enumeration would trade one over-rejection for another."""
    _, prompt = _prompt(**kw)
    assert "DERIVED VALUES AND FACTS" in prompt
    assert "the reply must show the work" in prompt


# ── half two: the #156 turn is untouched — the discrimination ────────────────────────

@pytest.mark.parametrize("kw,expect_branch", [
    ({"limits": LIMITS}, JUDGE_READONLY),
    ({"limits": LIMITS, "wrote": True}, JUDGE_EXECUTION),
    ({"context": True}, JUDGE_READONLY),
    ({"limits": LIMITS, "context": True}, JUDGE_READONLY),
])
def test_a_turn_that_HAS_a_grounding_section_keeps_the_widened_set(kw, expect_branch):
    """The other half, and the one that keeps this from being #156 reverted. Whenever EITHER
    section is on the page the criterion enumerates both exactly as it did — this substitution
    can reach only the turn where there is nothing to enumerate."""
    branch, prompt = _prompt(**kw)
    assert branch == expect_branch
    assert _GROUNDING_SOURCE_SET in prompt, (
        f"[{branch}] #156's widened source list was dropped from a turn that carries evidence")
    assert _GROUNDING_SOURCE_SET_NONE not in prompt, branch
    assert "appears in NONE of the three" in prompt


def test_the_tenant_rule_that_MOTIVATED_the_widening_still_grounds_an_empty_read():
    """id=1440 itself: reads empty, the figure configured in the persona's own limits. The
    criterion must still say the empty search does not prove it invented."""
    _, prompt = _prompt(limits=LIMITS)
    assert "Aula - R$ 120,00 por hora" in prompt
    assert "does NOT prove that a fact stated in those sections is invented" in prompt
    assert "it proves only that the search found nothing" in prompt


# ── the guard can fail ───────────────────────────────────────────────────────────────

def test_the_assertions_would_catch_the_substitution_being_removed():
    """A guard that cannot fail passes for the wrong reason. Feed the predicates the text the
    no-section prompt carried BEFORE this fix, and every one of them must reject it."""
    before = _GROUNDING_SOURCE_SET
    assert "does NOT prove that a fact stated in those sections is invented" in before
    assert "the tool results above are the ONLY ground truth this reply has" not in before
    assert "appears in NONE of the tool results above" not in before
    # …and the two variants are genuinely different strings, so `in`/`not in` can discriminate
    assert _GROUNDING_SOURCE_SET != _GROUNDING_SOURCE_SET_NONE
