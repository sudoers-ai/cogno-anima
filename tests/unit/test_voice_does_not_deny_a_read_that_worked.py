"""The voice may not tell a contact that a lookup failed when the lookup returned the thing.

The sibling of ``test_voice_never_invents_a_failure``, one step earlier in the sentence. That
clause forbids inventing a failed ACTION ("I could not record it" over a turn that called no
writing tool); this one forbids inventing a failed ACCESS ("I could not get it" over data the
executor is holding). Both are false statements about the world said to a person who cannot
check them and will act on them.

**The p0, and where it was measured.** Rehearsal tenant, 2026-09-18, SERVED code: the contact
asked for a course syllabus, ``consult_material`` returned the document ``ok=True``, the draft
answered from it correctly — and the judge rejected that draft over the SPELLING of a preserved
term (the defect ``_PRESERVED_IS_A_VALUE`` closes). The correction budget is 1 on every plan, so
there was no retry; holding a critique about WORDING, the voice wrote *"I could not access the
syllabus"*. The system had reached it, read it, and had the answer written.

**THE SCOPE IS COUNTED, NOT PREFERRED — and this is the half that decides whether the fix is
worth shipping.** Over 218 production turns carrying a judge block (``xmin`` rewrite filter,
rehearsal tenant excluded) the p0's exact shape is **0**. The BROAD family — a successful read,
a rejection, and a failure sentence delivered — is **13**; 4 are the canonical exhaustion
handoff and the other 9 were classified one by one against their traces:

  * **5 honest capability limits** — ids 360, 362, 366, 467, 661. Id 360 is the clearest:
    *"I have no access to financial information"* from a persona whose twenty tools include no
    financial one. True.
  * **3 honest empty reads** — ids 859, 1039, 1754. A directory lookup returning three names
    and not the one asked for; a summary returning ``Income R$0,00 (0 entries)``. True.
  * **1 undecidable** — id 855. Its persisted tool result is truncated at 246 chars, so whether
    the named resource was inside it cannot be read off the trace at all. It is left
    UNCLASSIFIED here rather than pushed to whichever side would flatter this change; the
    truncation is a separate defect with its own queue position.
  * **0 falsehoods.**

So eight of the nine are replies a BROAD prohibition would have deleted, every one of them
true. The clause is therefore tied to *a resource the executor data actually contains*, and the
two measured classes are carved out of it by name. `read_succeeded_this_turn` opens the door
and refuses on any turn where something failed; the containment question — *is the thing they
were told about IN the data* — is the model's, because it is the only reader that can ask it.

Deterministic: every assertion below is on the RENDERED voice prompt.
"""

from __future__ import annotations

import pytest

from cogno_anima import metakeys as mk
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.types import (
    EgoResult, EgoStep, ToolExecution, read_succeeded_this_turn,
)
from tests.unit.test_superego import ScriptedBackend, _ctx, _m

# What the clause forbids, what it permits, and the SCOPE that separates them. All three are
# asserted on the PROMPT: the reply itself needs a model, and this file is the deterministic
# half of the pair.
FORBIDS = "MUST NOT tell the contact that you could not access, find, obtain"
SCOPE = "something that IS in that data"
ALLOWS_EMPTY = "read came back EMPTY or did not contain what they asked for"
ALLOWS_LIMIT = "not a capability you have here"

# Critiques do not decide anything here (nothing in the tree classifies critique prose), but
# the measured turn's was about WORDING, which is what made the invented failure so stark.
CRITIQUE = "the preserved term should have been reproduced exactly"


def _ego(*calls: ToolExecution, draft: str = "Here is what I found.") -> EgoResult:
    return EgoResult(
        steps=[EgoStep(index=0, path="native", assistant_text="", tool_calls=list(calls)),
               EgoStep(index=1, path="native", assistant_text=draft)],
        metrics=_m("ego"))


def _read(tool: str, result: str) -> ToolExecution:
    return ToolExecution(tool=tool, arguments={}, result=result, ok=True,
                         side_effect=False, tool_mutating=False)


def _failed(tool: str, error: str = "upstream timeout") -> ToolExecution:
    return ToolExecution(tool=tool, arguments={}, result="", ok=False, error=error,
                         side_effect=False, tool_mutating=False)


def _turn(*calls: ToolExecution, user: str = "qual e a ementa do curso?"):
    ctx = _ctx(user=user, intent_class="INFORMATION_REQUEST", with_ego=False)
    ctx.ego_result = _ego(*calls)
    return ctx


def _rendered(ctx, *, kind=None, reason=CRITIQUE, payload="consult_material: (document)"):
    carrier = {"reason": reason}
    if kind is not None:
        carrier["kind"] = kind
    ctx.metadata[mk.VOICE_CORRECTION] = carrier
    return SuperegoStage()._build_voice_prompt(ctx, payload, ["general:review"])


# ── THE FALSEHOOD: forbidden (the rehearsal p0) ──────────────────────

def test_the_p0_the_voice_is_told_the_syllabus_was_retrieved():
    """``consult_material`` returned the document. "I could not access it" is false."""
    ctx = _turn(_read("consult_material", "Data Modeling — 60 hours. Contents: …"))
    prompt = _rendered(ctx)
    assert read_succeeded_this_turn(ctx) is True
    assert FORBIDS in prompt, (
        "the voice was handed a rejection over a turn whose lookup SUCCEEDED and nothing told "
        "it that 'I could not access that' is a false statement here")
    # …and it is told what to write instead. A rule that only forbids leaves the model to pick,
    # and on the measured turn what it picked was the invented failure.
    assert "Say what was read" in prompt
    # the OPPOSITE fabrication stays forbidden too, from the sentence that was already there
    assert "MUST NOT claim, imply or narrate that any action was performed" in prompt


# ── THE EIGHT TRUE REPLIES: still sayable ────────────────────────────
#
# These are the production turns, by id, that a broad prohibition would have deleted. The
# assertion is that the clause carries their carve-out IN THE SAME BREATH as the prohibition —
# a prompt that forbids on one line and permits on another, three paragraphs apart, is the
# "contradicted prompt" this file's neighbours already have a diagnosis for.

@pytest.mark.parametrize("trace_id,tool,result", [
    # a directory lookup that returned rows — just not the one asked for
    (859, "tenant_directory", "BOOKKEEPER, SCHEDULER, RECEPTION"),
    (1039, "knowledge_search", "no matching documents"),
    # a summary that answered, and the answer was zero
    (1754, "get_summary", "Income R$0,00 (0 entries)"),
])
def test_an_honest_EMPTY_read_is_still_sayable(trace_id, tool, result):
    """Ids 859 / 1039 / 1754. The read SUCCEEDED and came back without the thing; telling the
    contact so is the correct answer and the most common one on this kind of turn."""
    ctx = _turn(_read(tool, result))
    prompt = _rendered(ctx, payload=f"{tool}: {result}")
    assert FORBIDS in prompt            # the clause is live on this turn…
    assert ALLOWS_EMPTY in prompt, (
        f"turn {trace_id} reported truthfully that a read returned nothing; the clause must "
        f"carry that carve-out, or it deletes a true reply")


@pytest.mark.parametrize("trace_id", [360, 661])
def test_an_honest_CAPABILITY_LIMIT_is_still_sayable(trace_id):
    """Ids 360 / 661. Twenty tools on the table and not one of them financial: *"I have no
    access to financial information"* is true, and it is what `_ADMITTING_A_LIMIT` has said is
    a COMPLETE answer since the out-of-reach clause was written."""
    ctx = _turn(_read("resolve_date", "2026-09-18"), user="quanto gastei este mes?")
    prompt = _rendered(ctx, payload="resolve_date: 2026-09-18")
    assert FORBIDS in prompt
    assert ALLOWS_LIMIT in prompt, (
        f"turn {trace_id} admitted a capability the persona does not have; forbidding that "
        f"sentence would be a muzzle, not a fix")


def test_the_prohibition_is_SCOPED_to_what_the_data_contains():
    """The property the eight true replies depend on, pinned where it can die.

    SABOTAGE: reword the clause to forbid the failure sentence outright (drop
    "something that IS in that data") -> red here, while every other test in this file stays
    green. That is the mutation this assertion exists to catch, because it is the one that
    would look like a stronger fix.
    """
    prompt = _rendered(_turn(_read("consult_material", "…")))
    sentence = next(s for s in prompt.split(" — ") if FORBIDS in s)
    assert SCOPE in prompt
    # the prohibition and its scope are ONE sentence, not two paragraphs
    assert prompt.index(FORBIDS) < prompt.index(SCOPE) < prompt.index(FORBIDS) + len(sentence) + 80


# ── THE MUZZLE GUARDS: a lookup that really failed ───────────────────
#
# No production turn has this shape with a delivered failure sentence (the broad family
# required `ok=True`), so these are mechanism twins and are labelled as such. They are what
# keeps the predicate from ever firing beside a true report of a failure.

def test_a_lookup_that_really_FAILED_leaves_the_clause_off():
    """The single failed call is the whole evidence. Telling the contact the lookup did not
    work is the truth, and the truth is what lets them try something else."""
    ctx = _turn(_failed("consult_material"))
    assert read_succeeded_this_turn(ctx) is False
    assert FORBIDS not in _rendered(ctx)


def test_one_failed_call_beside_a_successful_one_leaves_the_clause_off():
    """The mixed turn, and the reason the predicate asks TWO questions rather than one. A
    successful `resolve_date` does not make "I could not reach the syllabus" false."""
    ctx = _turn(_read("resolve_date", "2026-09-18"), _failed("consult_material"))
    assert read_succeeded_this_turn(ctx) is False
    assert FORBIDS not in _rendered(ctx)


def test_a_successful_WRITE_is_not_evidence_that_a_LOOKUP_worked():
    """A write is not a read. Counting one would let a booking vouch for a lookup nobody made."""
    ctx = _ctx(with_ego=False)
    ctx.ego_result = _ego(ToolExecution(tool="record_expense", arguments={}, result="Recorded",
                                        ok=True, side_effect=True, tool_mutating=True))
    assert read_succeeded_this_turn(ctx) is False
    assert FORBIDS not in _rendered(ctx)


def test_a_turn_that_ran_nothing_leaves_the_clause_off():
    """No call, no evidence. This is the `unverified_claim` world, and it has its own variant."""
    ctx = _ctx(with_ego=False)
    ctx.ego_result = _ego()
    assert read_succeeded_this_turn(ctx) is False
    assert FORBIDS not in _rendered(ctx)


def test_an_unreadable_carrier_leaves_the_behaviour_exactly_as_it_was():
    """Same principle as `write_attempted_this_turn`'s ``unreadable=True``, opposite constant:
    a carrier this rule cannot read leaves the prompt as it was. There the rule is suppressed
    by answering True, here by answering False — the polarity differs, the rule does not.

    MEASURED WEAKNESS, recorded rather than hidden: no SINGLE mutation of
    `read_succeeded_this_turn` turns this assertion red. The direction is carried twice —
    question 1 (``unreadable=True``, "did anything fail, or can we not tell") short-circuits
    before question 2 (``unreadable=False``) is asked — so flipping either one alone leaves
    the answer False. Both mutations were run (2026-09-18): each left the whole suite green.
    Killing it takes flipping BOTH, which is not a unit mutation. The assertion is kept because
    the property is real and a future simplification to ONE question would make it the only
    thing standing between a broken carrier and a clause that forbids a true sentence; it is
    not claimed to be doing mutation work it is not doing. The SIBLING assertion below is
    single-sourced and does die to a mutation of `write_attempted_this_turn`'s constant.
    """
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

    assert read_succeeded_this_turn(Broken()) is False
    # …and the sibling keeps ITS direction, which suppresses ITS clause on the same carrier
    from cogno_anima.types import write_attempted_this_turn
    assert write_attempted_this_turn(Broken()) is True


def test_the_predicate_reads_both_execution_sources():
    """A failure recorded ONLY in the turn accumulator still counts. `ego_result` is the
    surviving attempt, and a lookup that failed on an earlier attempt of the same turn does not
    un-fail — the survivor-attempt-read-as-the-turn defect this family already carries a
    docstring against."""
    ctx = _turn(_read("consult_material", "…"))
    assert read_succeeded_this_turn(ctx) is True
    ctx.turn_executions = [_failed("consult_material")]
    assert read_succeeded_this_turn(ctx) is False
    assert FORBIDS not in _rendered(ctx)


# ── scope: one condition on one variant ──────────────────────────────

def test_the_clause_is_purely_additive():
    """Remove the clause and the prompt IS the pre-change prompt — no new section, so
    `_VOICE_BLOCKS` and the inventory the host persists are untouched. The same shape
    `nothing_tried` took, and the reason this is a CONDITION rather than a fourth ``kind``:
    `cogno_soma` splits its kind on whether ANY tool ran, which cannot separate these worlds.
    """
    off = _rendered(_turn(_failed("consult_material")))
    on = _rendered(_turn(_read("consult_material", "…")))
    clause = next(line for line in on.split("\n") if line.startswith("THE LOOKUPS WORKED"))
    assert on.replace(clause + "\n", "") == off
    assert sum(h.startswith("# ") for h in on.split("\n")) == \
           sum(h.startswith("# ") for h in off.split("\n"))


@pytest.mark.parametrize("kind", ["repeated_reply", "unverified_claim"])
def test_the_other_two_rejection_variants_are_untouched(kind):
    """`repeated_reply` is the host's anti-repeat guard (the content was fine) and
    `unverified_claim` is a turn where nothing ran at all — whose own text already tells the
    voice to drop the claim. Neither is the measured defect."""
    assert FORBIDS not in _rendered(_turn(_read("consult_material", "…")), kind=kind)


def test_an_approved_turn_never_sees_the_clause():
    """No rejection at all — the overwhelming majority of turns. The clause must be reachable
    ONLY through the verdict section."""
    ctx = _turn(_read("consult_material", "…"))
    assert FORBIDS not in SuperegoStage()._build_voice_prompt(ctx, "p", ["general:review"])


def test_a_rejection_without_a_reason_renders_no_verdict_and_no_clause():
    ctx = _turn(_read("consult_material", "…"))
    ctx.metadata[mk.VOICE_CORRECTION] = {"reason": ""}
    prompt = SuperegoStage()._build_voice_prompt(ctx, "p", ["general:review"])
    assert "# Execution verdict (HARD RULE)" not in prompt and FORBIDS not in prompt


def test_both_conditions_can_hold_at_once_and_do_not_contradict():
    """The p0 turn attempted no write AND read successfully, so BOTH clauses render. They are
    the same family pointing at two different inventions — an ACTION that was never tried, and
    an ACCESS that never failed — and neither licenses what the other forbids."""
    prompt = _rendered(_turn(_read("consult_material", "…")))
    assert "NOTHING WAS EVEN TRIED" in prompt and "THE LOOKUPS WORKED" in prompt


# ── the path is really reached ───────────────────────────────────────

@pytest.mark.asyncio
async def test_the_clause_reaches_the_prompt_voice_actually_sends():
    """`_build_voice_prompt` is a helper; what ships is what `voice()` hands the backend."""
    ctx = _turn(_read("consult_material", "Data Modeling — 60 hours."))
    ctx.metadata[mk.VOICE_CORRECTION] = {"reason": CRITIQUE}
    backend = ScriptedBackend(["A ementa tem 60 horas."])
    result = await SuperegoStage().voice(ctx, backend, voice_prompt="persona")
    assert FORBIDS in backend.calls[0]["prompt"]
    assert FORBIDS in (result.prompt_text or "")
