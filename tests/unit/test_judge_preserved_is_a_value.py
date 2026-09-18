"""A preserved term is a VALUE, and "exactly" is about the value — not about the spelling.

The NOUMENO rewrites every request into canonical English before the executor ever sees it, so
the EGO's draft is written in English BY DESIGN. Showing the judge the user's own Portuguese
term under a header reading *"must be reproduced verbatim"* asks the draft for something the
pipeline deliberately removed: it judges the TRANSLATION and calls the verdict grounding.

**Measured in the rehearsal tenant on 2026-09-18, against the SERVED code — not in
production.** The contact asked for a course syllabus by its Portuguese name,
``consult_material`` returned the document ``ok=True``, the draft answered from it correctly
and in full, and the judge rejected it with ONE reason: the preserved term *"should have been
reproduced exactly"* while the draft used the English name. The correction budget is 1 on every
plan, so there was no retry; the draft was dropped and the contact was told the system could
not access the syllabus. The right answer was already written. The production corpus has never
produced this shape (0 of 218 turns carrying a judge block) — the rehearsal tenant produced it,
which is the rehearsal tenant doing its job.

**Narrowed, not removed, and the argument is in the alternative.** `_preserved_mutated` — the
only other preserved-term guard in the tree — is FLAG-ONLY, fires only on a mutation of a term
ALREADY PRESENT in the executor payload, and reads the VOICED text, i.e. after the decision to
ship. Dropping the criterion would trade the sole REJECTING check on a mangled figure, email or
URL for a trace adjustment. So the criterion keeps its question and loses the half that was
never true, over the scope this file already uses for the purpose: ``_CRITICAL_TERM_RE``.

Deterministic throughout: every assertion is on the RENDERED judge prompt or on the module's
own AST. The model half is the pair in ``tests/integration/test_superego.py``.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from cogno_anima import metakeys as mk
from cogno_anima.stages.superego import (
    _CONVERSATIONAL_CRITERIA, _EXECUTION_CRITERIA, _PRESERVED_CLAUSE, _PRESERVED_IS_A_VALUE,
    _READONLY_CRITERIA, SuperegoStage,
)
from tests.unit.test_superego import _ctx

# The p0's own shape: a course name. No figure, no `@`, no scheme — nothing that can be
# CORRUPTED, only spelt differently.
A_PHRASE = "Modelagem de Dados"
ASKED = f"qual e a ementa de {A_PHRASE}?"
# …and the three kinds that CAN be corrupted, which is why the block still exists.
A_FIGURE = "1234.56"
AN_EMAIL = "aluno@example.com"
A_URL = "https://example.com/ementas.md"


def _prompt(*terms: str, user: str = ASKED, limits: str = "") -> str:
    ctx = _ctx(user=user)
    ctx.noumeno.preserved_terms = list(terms)
    return SuperegoStage()._build_judge_prompt(ctx, limits)


def _block(prompt: str) -> str:
    """The `# Preserved terms` section of a rendered judge prompt, or ``""``."""
    head = next((line for line, slug in SuperegoStage._JUDGE_BLOCKS
                 if slug == "preserved_terms"), None)
    assert head, "the judge table no longer has a preserved_terms row — this test is blind"
    i = prompt.find("\n" + head)
    if i < 0:
        return ""
    rest = prompt[i + 1:]
    end = rest.find("\n# ", 1)
    return rest if end < 0 else rest[:end]


def _ast_names(fn) -> "set[str]":
    """Every bare NAME the function's body references. AST, not a substring of the file: a
    comment claiming two readers share a definition is not the two sharing it."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


# ── the block: values only ───────────────────────────────────────────

def test_a_phrase_the_draft_only_TRANSLATES_never_reaches_the_judge():
    """THE p0. The preserved term is a course name; the block that demanded it verbatim is
    gone, and so is the criterion clause that would have asked after it.

    The EXECUTION branch, where the word disappears from the prompt ENTIRELY. What the other
    two branches keep is asserted separately below, and the distinction is not pedantry — the
    measured turn was judged under READ-ONLY.
    """
    prompt = _prompt(A_PHRASE)
    assert "# Preserved terms" not in prompt, (
        "a term that can only be spelt differently, never corrupted, was still being handed to "
        "the judge under a 'reproduce verbatim' header")
    assert _PRESERVED_CLAUSE not in prompt, (
        "criterion #4 still asks after preserved values on a prompt that lists none — the "
        "defect `_GROUNDING_SOURCE_SET_NONE` was written for, pointed at this block")
    # …and on this branch nothing at all is left that could be read as a demand
    assert "preserved" not in prompt.lower()
    # the phrase itself is NOT lost: it was never the block's contribution — the user's own
    # words are in `# User request` verbatim, which is where the judge should read them.
    assert A_PHRASE in prompt


def test_what_the_OTHER_branches_keep_is_the_value_sentence_and_no_term_list():
    """The claim above is branch-local, and this is the honest statement of the rest.

    `_CONVERSATIONAL_CRITERIA` and `_READONLY_CRITERIA` each carry their own sentence — "A
    preserved term reproduced INCORRECTLY (a mangled figure, email or URL) counts as
    fabrication too" — which is UNCONDITIONAL and stays. It was written value-scoped from the
    start, names the three corruptible kinds explicitly, and is untouched by this change, so it
    cannot ask for a spelling; the measured turn's own branch (READ-ONLY) is proof that the
    LIST, not that sentence, was what the judge reached for. What must not survive on any
    branch is the list, and that is what is asserted here.

    This test exists because the whole-prompt assertion above was written first and read as if
    it held everywhere. It does not, and a test that overclaims is a test nobody can use to
    reason about the next change.
    """
    ctx = _ctx(user=ASKED, intent_class="INFORMATION_REQUEST", with_ego=False)
    ctx.noumeno.preserved_terms = [A_PHRASE]
    from cogno_anima.types import EgoResult, EgoStep, ToolExecution
    from tests.unit.test_superego import _m
    ctx.ego_result = EgoResult(steps=[EgoStep(
        index=0, path="native", assistant_text="The Data Modeling syllabus is 60 hours.",
        tool_calls=[ToolExecution(tool="consult_material", arguments={}, result="60 hours",
                                  ok=True, side_effect=False, tool_mutating=False)])],
        metrics=_m("ego"))
    readonly = SuperegoStage()._build_judge_prompt(ctx, "")
    ctx.metadata[mk.JUDGE_CONVERSATIONAL] = True
    conversational = SuperegoStage()._build_judge_prompt(ctx, "")
    for prompt in (readonly, conversational):
        assert "# Preserved terms" not in prompt and A_PHRASE not in prompt.split("# Active")[1]
        assert _PRESERVED_CLAUSE not in prompt
        # what remains, and all that remains: the value-scoped sentence, naming its own scope
        assert prompt.lower().count("preserved") == 1
        assert ("A preserved term reproduced INCORRECTLY (a mangled figure, email or URL)"
                in prompt)


@pytest.mark.parametrize("term", [A_FIGURE, AN_EMAIL, A_URL])
def test_a_value_that_can_be_CORRUPTED_still_reaches_the_judge(term):
    """The control that keeps this a narrowing rather than a removal. A mangled figure, email
    or URL is the defect the criterion exists for, and it must still be rejectable."""
    prompt = _prompt(term)
    assert "# Preserved terms" in prompt and term in prompt
    assert _PRESERVED_CLAUSE in prompt
    assert _PRESERVED_IS_A_VALUE in prompt


def test_a_mixed_list_keeps_the_values_and_drops_the_phrase():
    body = _block(_prompt(A_PHRASE, A_FIGURE))
    assert A_FIGURE in body and A_PHRASE not in body
    assert _PRESERVED_CLAUSE in _prompt(A_PHRASE, A_FIGURE)


def test_the_block_says_the_value_is_the_criterion_not_the_language():
    """A rule that only LISTS leaves the model to decide what 'exactly' ranges over, and what
    it decided was the spelling. The block states the scope where the values are."""
    text = _block(_prompt(A_FIGURE))
    assert "never about spelling, wording or language" in text
    assert "ENGLISH BY DESIGN" in text
    assert "never reject a translation" in text


# ── one definition of "critical", two readers ────────────────────────

def test_the_block_filters_with_the_SAME_regex_the_output_backstop_uses():
    """`_preserved_mutated` has always ignored a non-critical term — correctly, since "Acme"
    written "Acmee" is a typo and not a corrupted answer — while this block told the judge that
    same name had to be verbatim. One definition, or they diverge again."""
    assert "_CRITICAL_TERM_RE" in _ast_names(SuperegoStage._format_preserved), (
        "the block re-derived what 'critical' means instead of reading the one definition")
    assert "_CRITICAL_TERM_RE" in _ast_names(SuperegoStage._preserved_mutated), (
        "the backstop moved off the shared definition; this test is now measuring nothing")


def test_the_flag_only_backstop_is_untouched_by_the_narrowing():
    """The narrowing is about the JUDGE. The output backstop's verdicts are byte-identical —
    it was already value-scoped, which is the whole argument for narrowing to its scope."""
    assert SuperegoStage._preserved_mutated([A_FIGURE], f"x: {A_FIGURE}", "I sent 1234.5") is True
    assert SuperegoStage._preserved_mutated([A_PHRASE], f"x: {A_PHRASE}", "Data Modeling") is False


# ── the splice, and the trap it is written against ───────────────────

def test_the_preserved_clause_substitution_matches():
    """``str.replace`` that matches nothing does not raise — it returns the string unchanged,
    which is how `_NO_OTHER_SOURCES` one screen up describes the same trap. What is pinned here
    is the ONE thing that would spring it: criterion #4 must go on reading the constant.

    MEASURED, and the first draft of this docstring was WRONG about it. The sabotage it claimed
    — "reword `_PRESERVED_CLAUSE` without re-splicing" — was run on 2026-09-18 and the whole
    suite stayed GREEN, because the clause is spliced BY REFERENCE (``+ _PRESERVED_CLAUSE +``)
    and not copied: rewording the constant rewords the criterion with it, so `replace` cannot
    miss. That is a property of the code, not of this test, and a test cannot take credit for
    it.

    The real sabotage, and the one that turns this red: inline into `_EXECUTION_CRITERIA` a
    hand-written copy of the sentence that DRIFTS from the constant — a word dropped is enough.
    The rendered criterion still reads correctly to a human, `count` falls to 0, the conditional
    silently stops removing anything, and `test_a_phrase_...` goes red beside this one. A
    byte-identical copy is deliberately NOT claimed to be caught: it cannot be, by any string
    test, and it also does no harm — `replace` still matches it. Splicing the clause TWICE is
    caught by the same assertion, from the other side.
    """
    assert _EXECUTION_CRITERIA.count(_PRESERVED_CLAUSE) == 1
    without = _EXECUTION_CRITERIA.replace(_PRESERVED_CLAUSE, "")
    assert without != _EXECUTION_CRITERIA
    # …and the criterion still reads: the clause is a whole sentence between two others
    assert "(no invented data)? What counts as GROUNDED" in without


def test_the_other_two_branches_are_untouched():
    """They already scoped it to the VALUE — "a mangled figure, email or URL" — which is the
    precedent this change follows rather than invents. Touching them would be unmeasured churn,
    and the READ-ONLY branch is the one the p0 turn itself was judged under."""
    for criteria in (_CONVERSATIONAL_CRITERIA, _READONLY_CRITERIA):
        assert "A preserved term reproduced INCORRECTLY (a mangled figure, email or URL)" \
            in criteria
        assert _PRESERVED_CLAUSE not in criteria


def test_the_block_is_dropped_in_every_branch():
    """The block is rendered ONCE, before the criteria are chosen, so a phrase-only turn drops
    it whichever branch it lands in — and the p0 landed in READ-ONLY, whose criterion was
    already value-scoped. That is what makes the BLOCK the load-bearing half of this fix."""
    for conversational in (False, True):
        ctx = _ctx(user=ASKED)
        ctx.noumeno.preserved_terms = [A_PHRASE]
        if conversational:
            ctx.metadata[mk.JUDGE_CONVERSATIONAL] = True
        assert "# Preserved terms" not in SuperegoStage()._build_judge_prompt(ctx, "")


# ── the inventory contract ───────────────────────────────────────────

def test_the_header_keeps_its_row_in_the_judge_inventory():
    """The host persists `judge_prompt_inventory`; a renamed section that silently stopped
    being recognised would drop a column nobody notices. The slug is stable, the text is not.

    The PAIR is the measurement: present for a value, absent for a phrase. The first alone
    would pass over a table that matches everything.
    """
    def blocks(prompt):
        return {row["block"] for row in SuperegoStage.judge_prompt_inventory(prompt)}
    assert "preserved_terms" in blocks(_prompt(A_FIGURE))
    assert "preserved_terms" not in blocks(_prompt(A_PHRASE))
