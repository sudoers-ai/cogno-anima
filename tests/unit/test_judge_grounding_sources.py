"""The judge's GROUNDING criterion must name every source of ground truth in its own prompt.

## What was measured, and what it refutes

`turn_traces` id=1440 on the live box, host `b1901a6` — i.e. WITH the framing block that
carries a tenant's configured rules to the judge already shipped. An EMPLOYEE asked what a
teaching hour pays and what the bonus rules are; the tenant's own `custom_rules` for that role
configure exactly those figures; the executor answered them correctly; the three tools it
called were reads that came back empty ("No faculty records found.", "Nothing recorded
about…"). The judge rejected the turn:

    "The draft fabricates the hourly rate, bonus amounts, eligibility rules, invoice deadline,
     and payment date. The successful searches found no faculty records or knowledge about
     teacher rates and bonus rules…"

Two incompatible explanations fit that trace, and only one of them is true:

  (i)  the tenant's rules never reached the judge's prompt — a wiring defect;
  (ii) they reached it and the CRITERIA overrode them — a prompt defect.

A deterministic probe over the RENDERED prompt settled it: the tenant's own
``- Aula - R$ 120,00 por hora`` was present, verbatim, under ``# Tenant rules (legitimate
grounding)``, 120 lines above the criteria — so (i) is refuted and (ii) is what happened. The
read-only branch's criterion #1 said "the reads are the ONLY ground truth this reply has" and
"must trace to a tool result above", and the execution branch's #4 asked whether everything is
"backed by the tool results". A fail-CLOSED judge reading an EXCLUSIVE claim inside its own
numbered REJECT list resolves the contradiction against a framing paragraph it read earlier —
which is exactly the failure mode ``JUDGE_READONLY`` was created to answer, arriving by the
other door.

The correction was already in the file, in the third branch: ``_CONVERSATIONAL_CRITERIA`` has
always enumerated three sources ("NOT in the Context above, in the persona's limits, or in what
the user said"). ``_GROUNDING_SOURCES`` gives the other two branches the same enumeration, once.

## Why these tests are written against the RENDERED prompt

The change is PROSE. A test that asserted on a symbol would keep passing over a sentence
rewritten into meaninglessness, so every assertion below reads the string
``_build_judge_prompt`` actually produced, and the last test proves the assertions can fail.
"""

from __future__ import annotations

import re

import pytest

from cogno_anima import metakeys as mk
from cogno_anima.stages.superego import (
    JUDGE_CONVERSATIONAL_BRANCH, JUDGE_EXECUTION, JUDGE_READONLY,
    SuperegoStage, _CONVERSATIONAL_CRITERIA, _GROUNDING_SOURCES)
from cogno_anima.types import EgoResult, EgoStep, ToolExecution
from tests.unit.test_superego import _ctx, _m

# A real tenant shape: business rules the host renders into the judge's `limits` slot, framed
# there as legitimate grounding. The figure is what the draft below states and what NO tool
# returns.
LIMITS = (
    "Answer within the persona's scope.\n\n"
    "# Tenant rules (legitimate grounding)\n"
    "The tenant configured the business rules below. An execution/answer grounded in them is "
    "CORRECTLY grounded — do not reject it as fabricated, ungrounded or off-goal.\n"
    "# Valores Financeiros\n - Aula - R$ 120,00 por hora\n"
)


def _turn(*, wrote: bool = False, conversational: bool = False):
    """A turn whose reads all came back EMPTY and whose draft answers from the tenant rules."""
    ctx = _ctx(user="qual o valor da hora aula?", intent_class="INFORMATION_REQUEST",
               goal="find out the hourly rate", with_ego=False)
    ctx.ego_result = EgoResult(
        steps=[
            EgoStep(index=0, path="native", assistant_text="", tool_calls=[
                ToolExecution(tool="knowledge_search", arguments={"query": "hourly rate"},
                              result="Nothing recorded about 'hourly rate'.", ok=True,
                              side_effect=wrote, tool_mutating=wrote)]),
            EgoStep(index=1, path="native",
                    assistant_text="A aula é R$ 120,00 por hora."),
        ], metrics=_m("ego"))
    if conversational:
        ctx.metadata[mk.JUDGE_CONVERSATIONAL] = True
    return ctx


def _prompt(**kw) -> "tuple[str, str]":
    ctx = _turn(**kw)
    st = SuperegoStage()
    return st._judge_branch(ctx), st._build_judge_prompt(ctx, LIMITS)


def _grounding_clause(prompt: str) -> str:
    """The numbered criterion that decides GROUNDING, whichever branch rendered."""
    for mark in ("1. FABRICATION / GROUNDING", "4. GROUNDING", "1. FABRICATION"):
        if mark in prompt:
            i = prompt.index(mark)
            return prompt[i:prompt.index("\n", i)]
    raise AssertionError("no grounding criterion in the rendered judge prompt")


# ── the branches this fixes ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("kw,expect_branch", [
    ({}, JUDGE_READONLY),
    ({"wrote": True}, JUDGE_EXECUTION),
])
def test_the_grounding_criterion_names_the_persona_limits_as_a_source(kw, expect_branch):
    """The measured defect. `# Persona limits` carries the tenant's rules, and the criterion
    that decides fabrication has to say so — the framing paragraph 120 lines up did not win."""
    branch, prompt = _prompt(**kw)
    assert branch == expect_branch
    clause = _grounding_clause(prompt)
    assert "'# Persona limits' section" in clause, (
        f"[{branch}] the grounding criterion does not name the section that carries the "
        f"tenant's own configured rules:\n{clause}")
    # …and the tenant's figure really is in the prompt this criterion is judging. Without this
    # the test above could pass over a prompt where the rules never arrived — the (i)/(ii)
    # confusion, rebuilt inside the guard that exists to prevent it.
    assert "Aula - R$ 120,00 por hora" in prompt


@pytest.mark.parametrize("kw", [{}, {"wrote": True}])
def test_a_search_that_found_nothing_no_longer_proves_invention(kw):
    """The sentence the live critique turned on: three empty reads were read as PROOF that a
    configured figure was made up."""
    _, prompt = _prompt(**kw)
    clause = _grounding_clause(prompt)
    # BOTH halves. The first alone is satisfied by "does NOT prove anything at all", which
    # deletes the precision that makes this safe — an empty search says something quite
    # definite about ITS OWN subject, and a criterion that stops saying so trades one
    # over-rejection for a licence. (Caught by mutation: the opening survived on its own.)
    assert "A search that returned nothing does NOT prove that a fact stated in those " \
           "sections is invented" in clause, clause
    assert "it proves only that the search found nothing" in clause, clause


@pytest.mark.parametrize("kw", [{}, {"wrote": True}])
def test_the_tool_results_are_no_longer_claimed_to_be_the_ONLY_ground_truth(kw):
    """The half that had to be REMOVED, not added. Both exclusive phrasings are gone — leaving
    either one in place would put the criteria back in contradiction with themselves."""
    _, prompt = _prompt(**kw)
    clause = _grounding_clause(prompt)
    assert "only ground truth this reply has" not in clause, clause
    assert "backed by the tool results" not in clause, clause


# ── the floor: this is not a licence to invent ────────────────────────────────────────

@pytest.mark.parametrize("kw", [{}, {"wrote": True}])
def test_a_fact_in_none_of_the_sources_is_still_fabrication(kw):
    """The set of sources is WIDER and still CLOSED, and every member of it is inside this
    prompt. Drop this line and the change stops being a correction and becomes a free pass."""
    _, prompt = _prompt(**kw)
    clause = _grounding_clause(prompt)
    assert "appears in NONE of the three" in clause, clause


def test_an_empty_read_still_grounds_only_a_negative_answer_about_what_it_covers():
    """The read-only branch's own teeth, kept — narrowed to the tool's SCOPE rather than
    deleted. An empty appointment list still cannot be filled with plausible appointments."""
    branch, prompt = _prompt()
    assert branch == JUDGE_READONLY
    clause = _grounding_clause(prompt)
    assert "grounds a NEGATIVE answer about WHAT THAT TOOL COVERS and nothing else" in clause
    assert "fills that emptiness with plausible content is fabricating" in clause


# ── the sources are the ones IN this prompt ──────────────────────────────────────────

def test_every_named_source_is_a_section_of_the_judge_prompt_itself():
    """The closure is what makes this a correction and not a licence, and it is checkable: each
    source the criterion names is a HEADER of the very prompt it is judging, and the count in
    the closing line matches (tool results + those two = "the three")."""
    named = re.findall(r"'(#[^']+)'", _GROUNDING_SOURCES)
    assert named == ["# Persona limits", "# Context"], named
    headers = [h for h, _ in SuperegoStage._JUDGE_BLOCKS]
    for n in named:
        assert any(h.startswith(n) for h in headers), (
            f"the criterion offers {n!r} as grounding and no such section is rendered — a "
            f"source the judge cannot read is worse than none, it is an invitation to guess")
    assert "NONE of the three" in _GROUNDING_SOURCES


@pytest.mark.parametrize("kw", [{}, {"wrote": True}])
def test_the_judge_is_never_told_its_OWN_knowledge_is_grounding(kw):
    """The dangerous direction of this change, and it is pinned because a mutation found it:
    widening the source set to something that is NOT in the prompt turns the fix into a free
    pass, and it reads almost identically to the fix. The prompt's own tail already forbids it
    ("Do NOT re-derive them from your own reasoning"), so a grounding clause that offered it
    would put the criteria back in contradiction with themselves — which is the whole defect
    this change exists to end, arriving from the opposite side."""
    _, prompt = _prompt(**kw)
    assert "Do NOT re-derive them from your own reasoning" in prompt
    clause = _grounding_clause(prompt).lower()
    for hazard in ("your own", "common sense", "general knowledge", "what you know"):
        assert hazard not in clause, (
            f"the grounding criterion offers {hazard!r} as a source: that is not in this "
            f"prompt, so nothing can check it")


# ── written once ──────────────────────────────────────────────────────────────────────

def test_the_sentence_is_written_once_and_reaches_both_branches():
    """`_ADMITTING_A_LIMIT`'s rule, applied again: a copy of a shared sentence in the other
    branch is a contract that diverges silently, and this file already keeps two such pins."""
    ro = _prompt()[1]
    ex = _prompt(wrote=True)[1]
    assert _GROUNDING_SOURCES in ro and _GROUNDING_SOURCES in ex
    assert ro.count(_GROUNDING_SOURCES) == 1 and ex.count(_GROUNDING_SOURCES) == 1


def test_the_conversational_branch_is_untouched():
    """It already enumerated all three sources, and it is the PRECEDENT this change follows —
    so it must render byte-identically, without the new sentence bolted on beside it."""
    branch, prompt = _prompt(conversational=True)
    assert branch == JUDGE_CONVERSATIONAL_BRANCH
    assert _CONVERSATIONAL_CRITERIA in prompt
    assert _GROUNDING_SOURCES not in prompt
    assert "in the persona's limits" in _grounding_clause(prompt)


# ── the guard can fail ────────────────────────────────────────────────────────────────

def test_the_assertions_would_catch_the_sentence_being_removed():
    """A guard that cannot fail passes for the wrong reason. Feed the same predicates the
    PREVIOUS text and every one of them must reject it."""
    before = ("1. FABRICATION / GROUNDING - the reads are the only ground truth this reply "
              "has. Every figure the draft states must trace to a tool result above.")
    assert "'# Persona limits' section" not in before
    assert "A search that returned nothing does NOT prove" not in before
    assert "appears in NONE of the three" not in before
    assert "only ground truth this reply has" in before      # the phrase the fix removes
