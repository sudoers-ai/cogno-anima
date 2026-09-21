"""`# Review verdict (HARD RULE)` may not tell the voice that nothing ran, when something did.

The third clause of the same family, on the one variant the first two did not reach.
``test_voice_never_invents_a_failure`` forbids inventing a failed ACTION and
``test_voice_does_not_deny_a_read_that_worked`` forbids inventing a failed ACCESS — both on the
EXECUTION verdict. This file is the REVIEW verdict, the variant the host stamps
``kind="unverified_claim"``, whose opening sentence asserts a fact it does not carry.

**The measured turn** (persisted trace, read 2026-09-19, described generically — the contact asked
how long a class lasts). Scope ALLOWED the turn; a knowledge-read tool ran and returned
``ok=True`` with the timetable; the draft answered from it, verbatim; review APPROVED it. A
downstream host's anti-fabrication net then flagged the reply anyway — it judges provenance by
WHICH tool was called (only one vertical's own reads legitimise a schedule claim) rather than by
what the tool RETURNED, so the read holding those very times did not count — and re-voiced the
turn under ``unverified_claim``. The voice obeyed this section word for word: told that *nothing
was executed this turn* and steered to *say plainly that you do not have that information*, it
replied that the information was unavailable. **The first reply had been correct and fully
grounded.** The provenance-by-tool-name rule is the host's and is a separate, larger fix; this
section has to be honest about the turn either way.

**Why the host's ``kind`` cannot decide this.** That repair fires when the net flags a reply AND
the turn did not COMMIT — and a read-only turn never commits, so EVERY flagged read turn arrives
wearing a kind whose text was written for the opposite world: a persona with no tools asserting
an integration that does not exist. That world is real and its wording is untouched here, byte
for byte (``test_the_twin_that_must_not_break``).

**Why a CONDITION and not a fourth kind** — the reasoning ``nothing_tried`` already wrote:
asking the host to re-derive a fact the core can read off the trace is the re-derivation this
repo keeps paying for, and here the host's kind has just been measured wrong about that very
fact. ``read_succeeded_this_turn`` answers it from the executed records; its boundary is the
point, and it is pinned below — ANY failed record makes it False, because then "I could not get
that" may be TRUE of that call and the section stays exactly as it was.

No new header: the slug is still ``review_verdict`` and the persisted inventory does not move.

Deterministic: every assertion is on the RENDERED voice prompt.
"""

from __future__ import annotations

import hashlib
import inspect
import re

import pytest

from cogno_anima import metakeys as mk
from cogno_anima.stages import superego as _se
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.types import EgoResult, EgoStep, ToolExecution, read_succeeded_this_turn
from tests.unit.test_superego import ScriptedBackend, _ctx, _m

# ── the fixture: the measured turn ───────────────────────────────────
#
# A knowledge read that RETURNED the answer, and a draft built from it. The reply that shipped
# said the information was unavailable.
USER = "quanto tempo dura a aula?"
TIMETABLE = "Wednesday 19h-22h30; Tuesday 8h-11h30"
PAYLOAD = f"consult_material: {TIMETABLE}"
CRITIQUE = "the reply states class times without a scheduling read"

# What the section must STOP saying on this turn, and what it must START saying. Asserted on the
# prompt: the reply itself needs a model, and the integration pair in `tests/integration` is that
# half.
NOTHING_RAN = "nothing was executed this turn"
OLD_DEFAULT = "say plainly that you do not have that information"
NEVER_DENY = "MUST NOT tell the contact that you could not access, find, obtain"
# The new section's rule, in the order it must be read: the DATA is the authority, what it holds
# is stated even when flagged, and only what it does not hold is dropped.
ONLY_AUTHORITY = "the executor data above is the ONLY authority here"
STATE_WHAT_IT_CONTAINS = ("If the executor data answers the request, write the reply from what "
                          "it DOES contain")
# EXHAUSTIVE, not "whatever answers": the narrowed wording let the model pick "the answer" and
# derive it (CI canary, qwen3:8b: "A aula dura 3 horas e 30 minutos", both times dropped).
EVERY_VALUE_IN_IT = "reproducing every figure, time, date, name or identifier in it"
EVEN_WHEN_FLAGGED = "even when it is part of what review flagged"
DROP_ONLY = ("Whatever the flagged claim says that the data does NOT contain, you MUST NOT say, "
             "restate, soften or hedge: drop it")
NOTHING_ANSWERS = ("When nothing in the data answers the request, do not build a reply out of "
                   "what was retrieved")
# What the first cut said and code review refuted (2026-09-21): an assertion the core cannot
# know, and an absolute prohibition placed BEFORE the scope that qualified it.
FALSE_PREMISE = "this data does not support"
ABSOLUTE_PROHIBITION = "You MUST NOT repeat the rejected claim"

# `# Review verdict (HARD RULE)` exactly as `main` renders it for a turn that executed NOTHING —
# the world this kind was written for. Copied from a rendering of the base revision, with
# ``reason="R"``, and compared byte for byte rather than by substring: a twin pinned with `in`
# checks survives a rewrite that keeps the words and changes the meaning.
MAIN_REVIEW_VERDICT = (
    "# Review verdict (HARD RULE)\n"
    "The draft below was REJECTED by review as UNVERIFIED — nothing was executed this turn, "
    "so the draft is a claim, not a result.\n"
    "Reviewer critique: R\n"
    "That critique is a complaint about the draft, NOT evidence: nothing executed it and "
    "nothing verified it. Do not carry any figure, total, date or fact from it into your reply "
    "— not even one it states confidently, and not even when its arithmetic looks right. Use "
    "it to understand what was wrong, never as a source.\n"
    "You MUST NOT repeat the rejected claim, or any softened version of it. Say ONLY what the "
    "Context above supports; when it supports nothing, say plainly that you do not have that "
    "information — admitting a limit is a COMPLETE answer and is always preferable to "
    "repeating an unverified one. You may then ask ONE question to move forward.\n\n"
)


def _ego(*calls: ToolExecution, draft: str = "The class runs Wednesday 19h-22h30.") -> EgoResult:
    return EgoResult(
        steps=[EgoStep(index=0, path="native", assistant_text="", tool_calls=list(calls)),
               EgoStep(index=1, path="native", assistant_text=draft)],
        metrics=_m("ego"))


def _read(tool: str = "consult_material", result: str = TIMETABLE) -> ToolExecution:
    return ToolExecution(tool=tool, arguments={}, result=result, ok=True,
                         side_effect=False, tool_mutating=False)


def _failed(tool: str = "check_availability") -> ToolExecution:
    return ToolExecution(tool=tool, arguments={}, result="", ok=False, error="upstream timeout",
                         side_effect=False, tool_mutating=False)


def _write() -> ToolExecution:
    return ToolExecution(tool="book_slot", arguments={}, result="Booked", ok=True,
                         side_effect=True, tool_mutating=True)


def _turn(*calls: ToolExecution):
    ctx = _ctx(user=USER, intent_class="INFORMATION_REQUEST", with_ego=False)
    ctx.ego_result = _ego(*calls)
    return ctx


def _render(ctx, *, kind: "str | None" = "unverified_claim", reason: str = CRITIQUE,
            payload: str = PAYLOAD) -> str:
    if kind is not None:
        carrier = {"reason": reason, "kind": kind}
        ctx.metadata[mk.VOICE_CORRECTION] = carrier
    return SuperegoStage()._build_voice_prompt(ctx, payload, ["general:review"])


def _sentences(text: str) -> "list[str]":
    """Sentences of a rendered section — split at a full stop AND at a line break, because a
    clause that follows a constant ending in ``.\n`` starts a line, not a ``". "``."""
    return [x for x in re.split(r"(?<=\.)\s+|\n", text) if x]


def _verdict(ctx, **kw) -> str:
    """The `review_verdict` section alone, sliced by the SAME closed table the host reads."""
    return SuperegoStage.voice_prompt_block(_render(ctx, **kw), "review_verdict")


# ── PRESENCE FIRST: the harness can produce the sentence this file is about to assert away ──

def test_the_control_the_same_harness_renders_the_no_execution_wording():
    """An absence test that never proved it could produce the presence is a test of nothing.

    Same builder, same kind, same critique — only the executed records differ. This turn ran
    NOTHING, so the section says so, and steers to the limit. That is correct here and it is
    what the next test asserts is gone.
    """
    ctx = _turn()
    assert read_succeeded_this_turn(ctx) is False
    section = _verdict(ctx)
    assert NOTHING_RAN in section and OLD_DEFAULT in section
    assert STATE_WHAT_IT_CONTAINS not in section


# ── THE MEASURED TURN ────────────────────────────────────────────────

def test_the_measured_turn_a_read_returned_the_answer_and_the_voice_was_told_otherwise():
    """The read ran, succeeded, and its result is in the executor data. Both halves matter:
    the section must stop asserting the false premise AND must say what to write instead — a
    rule that only forbids leaves the model to pick, and what it picked was the denial."""
    ctx = _turn(_read())
    assert read_succeeded_this_turn(ctx) is True
    section = _verdict(ctx)
    # identity: this is the REVIEW verdict branch, not the execution one
    assert section.startswith("# Review verdict (HARD RULE)\n")
    assert SuperegoStage.voice_prompt_block(_render(_turn(_read())), "execution_verdict") == ""
    # the false premise is gone…
    assert NOTHING_RAN not in section, (
        "the voice was told nothing ran on a turn whose read returned the answer")
    # …and so is the steer that produced the delivered reply: the limit may still be SAID, but
    # only inside the sentence that makes it conditional on nothing in the data answering
    limit = next(x for x in _sentences(section) if OLD_DEFAULT in x)
    assert limit.startswith(NOTHING_ANSWERS), (
        "'you do not have that information' is back as the DEFAULT outcome on a turn that had it")
    # …replaced by what is true, and by an instruction
    assert _se._EVERY_TOOL_SUCCEEDED in section
    assert STATE_WHAT_IT_CONTAINS in section
    assert NEVER_DENY in section
    # the opposite fabrication stays forbidden from the same section — scoped, not absolute
    assert DROP_ONLY in section
    assert _se._CRITIQUE_IS_NOT_EVIDENCE.strip() in section


def test_the_instruction_is_scoped_to_a_claim_the_data_does_NOT_contain():
    """The scope that keeps the instruction from deleting the answer.

    On the measured turn the executor data holds every figure the draft states; what the net
    flagged was the PROVENANCE of the claim, not a figure missing from the data. A section
    that said "drop the claim" unqualified would tell the voice to drop the very times it is
    also told to reproduce.

    SABOTAGE: widen the drop sentence so it no longer names "that the data does NOT
    contain" -> red here.
    """
    section = _verdict(_turn(_read()))
    sentence = next(s for s in _sentences(section) if "MUST NOT say, restate, soften" in s)
    assert "that the data does NOT contain" in sentence, (
        "the drop instruction lost its scope: it now reads as 'drop the claim', which on this "
        "turn covers the data the very next sentence orders reproduced")
    # …and the reproduce half is present, with the verbatim demand
    assert "exactly as written there" in _verdict(_turn(_read()))
    assert "If what is left answers the request, that IS the answer" in section


def test_the_limit_is_the_LAST_resort_and_not_the_default():
    """Admitting the data did not hold it stays a COMPLETE reply — but only when that is true.
    The measured defect is the ordering, not the sentence."""
    section = _verdict(_turn(_read()))
    assert NOTHING_ANSWERS in section
    assert section.index(STATE_WHAT_IT_CONTAINS) < section.index(NOTHING_ANSWERS)
    # the limit sentence is said ONCE, and it is the conditional one
    assert section.count(OLD_DEFAULT) == 1
    assert section.index(NOTHING_ANSWERS) < section.index(OLD_DEFAULT)


# ── CODE REVIEW (2026-09-21): THE DATA DECIDES, NOT THE CRITIQUE ──────
#
# Two findings, measured on the merge of `main` and this branch. The first cut told the voice
# that review had flagged "a CLAIM in the draft that this data does not support" — which the
# core cannot know, and which is FALSE on the live shape below — and it placed an absolute "MUST
# NOT repeat the rejected claim" BEFORE the colon that scoped it, so when the flagged claim IS a
# figure in the data the two rules pointed opposite ways.

WORKLOAD = "Data Modeling - total workload: 60h"


def test_the_60h_twin_what_the_data_holds_is_stated_even_when_review_flagged_it():
    """The measured live failure. The read returned "60h", the draft said "60 horas", and a
    downstream net flagged the reply by tool NAME. The data SUPPORTS the claim; the core cannot
    tell, so it must not say otherwise — and the rule that follows must let the figure through.

    Mutations this dies to: restoring the "does not support" premise; putting the absolute
    "MUST NOT repeat the rejected claim" back in front of the scope.
    """
    ctx = _turn(_read("consult_material", WORKLOAD))
    ctx.ego_result = _ego(_read("consult_material", WORKLOAD),
                          draft="A carga horaria total e de 60 horas.")
    section = _verdict(ctx, reason="the reply states a workload no scheduling read confirmed")
    # the core asserts nothing about whether the data supports the flagged claim…
    assert FALSE_PREMISE not in section, (
        "the section asserts the data does not support a claim — on this turn it does, and "
        "the core cannot know either way")
    assert ONLY_AUTHORITY in section
    # …and the rule that states what the data holds says so EVEN WHEN it was flagged, as ONE
    # sentence, by identity — not two sentences that happen to be near each other
    sentence = next(x for x in _sentences(section) if EVEN_WHEN_FLAGGED in x)
    assert sentence.startswith(STATE_WHAT_IT_CONTAINS)
    # …and no absolute prohibition comes before it to contradict it
    assert ABSOLUTE_PROHIBITION not in section
    assert section.index(EVEN_WHEN_FLAGGED) < section.index("MUST NOT")


def test_once_the_data_answers_EVERY_value_in_it_is_reproduced_not_a_chosen_one():
    """Measured on the model-backed canary, not guessed. The review rewrite said "WHATEVER in the
    executor data answers the request, state exactly" — and on the class-duration turn (data:
    two class times) qwen3:8b at temperature 0 replied *"A aula dura 3 horas e 30 minutos"*: a
    bare DERIVED figure, both times the contact could check dropped. The sentence before it,
    "reproducing every figure, time, date, name or identifier IN IT", had passed that same test.

    So the relevance judgement gates the WHOLE sentence and the reproduction stays exhaustive.
    Asserted by identity on ONE rendered sentence: the condition, the exhaustive reproduction
    and the "even when flagged" clause are the same sentence.

    MUTATION: restore the "Whatever … answers the request, state exactly" wording -> red here.
    """
    section = _verdict(_turn(_read()))
    sentence = next(x for x in _sentences(section) if EVEN_WHEN_FLAGGED in x)
    assert sentence.startswith(STATE_WHAT_IT_CONTAINS)
    assert EVERY_VALUE_IN_IT in sentence, (
        "the reproduction was narrowed to the values the model decides 'answer' the request — "
        "the shape that shipped a bare derived duration instead of the times")
    assert "Whatever in the executor data answers the request" not in section


def test_the_CLOSER_with_only_resolve_date_run_gets_the_new_section_and_its_two_limits():
    """A "does it integrate with X?" turn whose only call was a universal `resolve_date`. The
    read is visible, so the NEW section renders — not the legacy one — and the two rules that
    keep it honest must both be there: drop what the data does not contain (it contains a
    date, not an integration), and when nothing in the data answers the request, do not build
    a reply out of what was retrieved.

    WHY `resolve_date` STILL COUNTS. Excluding "utility" tools from the gate was proposed and
    measured wrong: on a turn whose question IS a date, the resolved date is the answer, and
    an excluded turn would fall back to the legacy text's "say plainly that you do not have
    that information" — the denial this branch exists to stop. Relevance cannot be decided by
    tool NAME; it is the model's call, and the wording asks for it. The twin that pins the
    other side is `test_the_date_question_with_only_resolve_date_run_is_answered_from_the_date`.
    """
    ctx = _ctx(user="voces integram com o ACME ERP?", intent_class="INFORMATION_REQUEST",
               with_ego=False)
    ctx.ego_result = _ego(_read("resolve_date", "today: 2026-09-21 (Monday)"),
                          draft="Yes, the product integrates with ACME ERP.")
    assert read_succeeded_this_turn(ctx) is True
    assert SuperegoStage._payload_shows_a_read(ctx) is True
    section = _verdict(ctx, reason="no source confirms an ACME ERP integration")
    assert NOTHING_RAN not in section and _se._EVERY_TOOL_SUCCEEDED in section
    assert DROP_ONLY in section
    limit = next(x for x in _sentences(section) if OLD_DEFAULT in x)
    assert limit.startswith(NOTHING_ANSWERS)
    assert "mentioning a lookup only when it was a lookup FOR what they asked" in limit


def test_the_date_question_with_only_resolve_date_run_is_answered_from_the_date():
    """The twin that stops finding 1 from being "fixed" by excluding the tool.

    The contact asked which day next Tuesday is; `resolve_date` answered. The date is in the
    prompt the voice receives and the new section tells it to state what answers the request.
    Exclude `resolve_date` from the gate and this turn falls back to the legacy wording —
    "nothing was executed", "say plainly that you do not have that information" — over a date
    the system resolved.
    """
    ctx = _ctx(user="que dia cai a proxima terca?", intent_class="INFORMATION_REQUEST",
               with_ego=False)
    ctx.ego_result = _ego(_read("resolve_date", "next Tuesday: 2026-09-22"),
                          draft="Next Tuesday is 2026-09-22.")
    payload = SuperegoStage._tool_payload(ctx)
    prompt = _render(ctx, reason="the date was not confirmed by a scheduling read",
                     payload=payload)
    section = SuperegoStage.voice_prompt_block(prompt, "review_verdict")
    assert "2026-09-22" in SuperegoStage.voice_prompt_block(prompt, "executor_data")
    assert NOTHING_RAN not in section
    assert STATE_WHAT_IT_CONTAINS in section and NEVER_DENY in section


# ── THE TWIN THAT MUST NOT BREAK ─────────────────────────────────────

def test_the_twin_that_must_not_break_is_byte_identical_to_main():
    """A persona that executed nothing and asserted an integration that does not exist. That
    is the world this kind was written for, it is measured and live, and its wording does not
    move by one byte."""
    ctx = _turn()
    assert read_succeeded_this_turn(ctx) is False
    assert _verdict(ctx, reason="R") == MAIN_REVIEW_VERDICT


# ── THE BOUNDARY ─────────────────────────────────────────────────────

def test_one_FAILED_call_beside_the_read_keeps_todays_wording():
    """The predicate's second question, and the reason it is not widened: with a failed call
    in the trace, "I could not get that" may be TRUE of that call, and a rule that fires next
    to a true sentence is a rule that will delete it."""
    ctx = _turn(_read(), _failed())
    assert read_succeeded_this_turn(ctx) is False
    assert _verdict(ctx, reason="R") == MAIN_REVIEW_VERDICT


def test_a_successful_WRITE_is_not_evidence_that_a_LOOKUP_worked():
    """A write is not a read; counting one would let a booking vouch for a lookup nobody made."""
    ctx = _turn(_write())
    assert read_succeeded_this_turn(ctx) is False
    assert _verdict(ctx, reason="R") == MAIN_REVIEW_VERDICT


def test_the_consulted_specialists_read_does_NOT_open_the_clause():
    """REPLACES `test_the_consulted_specialists_read_counts`, which pinned the defect as if it
    were the feature.

    That test asserted that a read performed by the CONSULTED specialist opens this section,
    on the true premise that `read_succeeded_this_turn` walks `ctx.consult_result`. What it
    missed is that the section makes a claim about THIS PROMPT — "what they returned is in the
    executor data above" — and `_tool_payload` renders only `ctx.ego_result.tools_executed`.
    The specialist's result is nowhere in the prompt, so the old behaviour told the voice the
    data was above AND forbade it the honest "I did not find that": a false premise plus a
    muzzle, the exact pair this family exists to prevent.

    The predicate is right and unchanged — the TURN did read successfully, and other readers
    depend on that answer. What changed is the clause's gate. Widening the payload to carry
    another persona's reads is a larger change with provenance questions of its own and was
    deliberately not made.
    """
    ctx = _shape_consult_only()
    assert read_succeeded_this_turn(ctx) is True          # the TURN read…
    assert SuperegoStage._payload_shows_a_read(ctx) is False   # …this PROMPT does not show it
    assert _verdict(ctx, reason="R") == MAIN_REVIEW_VERDICT


def test_a_read_from_a_DISCARDED_attempt_does_not_open_the_clause_either():
    """The same defect through the other source. `ctx.turn_executions` accumulates EVERY
    attempt of the turn; `_tool_payload` renders the SURVIVING one. A read that succeeded on
    an attempt the correction loop then replaced is not in front of the voice, so the premise
    would be false about it too."""
    ctx = _shape_discarded_only()
    assert read_succeeded_this_turn(ctx) is True
    assert SuperegoStage._payload_shows_a_read(ctx) is False
    assert _verdict(ctx, reason="R") == MAIN_REVIEW_VERDICT


# ── NO NEW HEADER, NO NEW SLUG ───────────────────────────────────────

def test_the_persisted_inventory_does_not_move():
    """Same slugs, same order, same count — the host persists this list, and a new section
    would be a new column nobody asked for. The condition is a CONDITION, not a block."""
    on = SuperegoStage.voice_prompt_inventory(_render(_turn(_read())))
    off = SuperegoStage.voice_prompt_inventory(_render(_turn()))
    assert [b["block"] for b in on] == [b["block"] for b in off]
    assert "review_verdict" in [b["block"] for b in on]
    assert "execution_verdict" not in [b["block"] for b in on]


# ── EVERY OTHER RENDERING IS MAIN'S, AND THE KINDS COME FROM THE CODE ──
#
# Section-scoped digests, measured on the base revision. Scoped to the verdict SECTION rather
# than the whole prompt so an unrelated change elsewhere in the voice prompt cannot fail this
# test with a message about the wrong thing; recompute by rendering the same fixtures on the
# revision you are comparing against.
_MAIN_SECTIONS = {
    "no_exec|repeated_reply|already_said": "aef13fbe8b469365",
    "no_exec|unverified_claim|review_verdict": "e90b3d53d76ee533",
    "no_exec|other|execution_verdict": "b2990b63fcf8392d",
    "read_ok|repeated_reply|already_said": "aef13fbe8b469365",
    "read_ok|other|execution_verdict": "4c65c7761eb5e82e",
    "read_plus_failed|repeated_reply|already_said": "aef13fbe8b469365",
    "read_plus_failed|unverified_claim|review_verdict": "e90b3d53d76ee533",
    "read_plus_failed|other|execution_verdict": "b2990b63fcf8392d",
    "write|repeated_reply|already_said": "aef13fbe8b469365",
    "write|unverified_claim|review_verdict": "e90b3d53d76ee533",
    "write|other|execution_verdict": "91faad424d9ef189",
}

def _shape_no_exec():
    return _turn()


def _shape_read_ok():
    """(0) the control: the SURVIVING attempt read, so the result IS in the rendered payload."""
    return _turn(_read())


def _shape_read_plus_failed():
    return _turn(_read(), _failed())


def _shape_write():
    return _turn(_write())


def _shape_discarded_only():
    """(i) the only successful read belongs to an attempt the turn later replaced."""
    ctx = _turn()
    ctx.turn_executions = [_read()]
    return ctx


def _shape_consult_only():
    """(ii) the only successful read is the consulted specialist's."""
    ctx = _turn()
    ctx.consult_result = _ego(_read())
    return ctx


_SHAPES = {"no_exec": _shape_no_exec, "read_ok": _shape_read_ok,
           "read_plus_failed": _shape_read_plus_failed, "write": _shape_write,
           "discarded_only": _shape_discarded_only, "consult_only": _shape_consult_only}


def _kinds_from_the_code() -> "list[str]":
    """The rejection kinds `_build_voice_prompt` BRANCHES on, read out of the function itself.

    Hand-listing them is how a future kind gets forgotten: it would ship with no byte-identity
    test at all and nobody would notice, which is precisely how the variant this file fixes
    went three weeks asserting a fact it does not carry.
    """
    src = inspect.getsource(SuperegoStage._build_voice_prompt)
    return sorted(set(re.findall(r'\(rejection\.get\("kind"\) or ""\) == "([a-z_]+)"', src)))


def test_the_kinds_are_read_from_the_code_and_every_one_is_covered():
    kinds = _kinds_from_the_code()
    assert kinds == ["repeated_reply", "unverified_claim"], (
        f"the rejection kinds changed: {kinds}. A new kind needs its own row in "
        "_MAIN_SECTIONS, measured against the revision before it")
    # "other" is the else branch — the execution verdict, reached by any kind not listed above
    covered = {k.split("|")[1] for k in _MAIN_SECTIONS}
    assert covered == set(kinds) | {"other"}


@pytest.mark.parametrize("label", sorted(_MAIN_SECTIONS))
def test_every_other_rendering_is_exactly_mains(label):
    """The whole change is ONE cell of this table, and that cell is deliberately not in it:
    ``read_ok|unverified_claim|review_verdict``. Everything else — both other variants, all
    four execution shapes, the write-attempted condition on either side — renders byte for
    byte as it did."""
    shape, kind, slug = label.split("|")
    ctx = _SHAPES[shape]()
    prompt = _render(ctx, kind="not_executed" if kind == "other" else kind, reason="R")
    section = SuperegoStage.voice_prompt_block(prompt, slug)
    assert section, f"{label}: the section did not render at all"
    assert hashlib.sha256(section.encode()).hexdigest()[:16] == _MAIN_SECTIONS[label]


def test_the_only_cell_that_moved_is_the_one_this_change_is_about():
    """Stated as an assertion rather than as a comment: the changed rendering is NOT main's."""
    section = _verdict(_turn(_read()), reason="R")
    assert hashlib.sha256(section.encode()).hexdigest()[:16] != "e90b3d53d76ee533"


# ── the two sentences are SHARED, not copied ─────────────────────────

def test_the_premise_and_the_prohibition_are_spliced_into_both_verdicts():
    """One definition, two readers — the rule `_ADMITTING_A_LIMIT` and `_PRESERVED_CLAUSE`
    already follow here. Both sections make the same claim about the same fact; a hand-written
    second copy is a contract that diverges the first time either is reworded."""
    review = _verdict(_turn(_read()))
    execution = SuperegoStage.voice_prompt_block(
        _render(_turn(_read()), kind="not_executed"), "execution_verdict")
    for shared in (_se._EVERY_TOOL_SUCCEEDED, _se._NEVER_DENY_WHAT_WAS_READ):
        assert shared in review and shared in execution
    # …and there is exactly ONE copy of each in the module source: the definition.
    src = inspect.getsource(_se)
    assert src.count("every tool that ran this turn SUCCEEDED") == 1
    assert src.count("could not access, find, obtain, consult") == 1


def test_the_remedy_is_NOT_shared_because_the_two_branches_diverge_there():
    """What the voice is told to DO next is per branch and must stay so: the execution verdict
    sends it to "say what was read, correcting what the critique says was wrong"; the review
    verdict sends it to "drop the one claim the data does not carry and say the rest"."""
    review = _verdict(_turn(_read()))
    execution = SuperegoStage.voice_prompt_block(
        _render(_turn(_read()), kind="not_executed"), "execution_verdict")
    assert "Say what was read (correcting whatever the critique says" in execution
    assert "Say what was read (correcting whatever the critique says" not in review
    assert DROP_ONLY in review and DROP_ONLY not in execution


# ── the path is really reached ───────────────────────────────────────

@pytest.mark.asyncio
async def test_the_condition_reaches_the_prompt_voice_actually_sends():
    """`_build_voice_prompt` is a helper; what ships is what `voice()` hands the backend."""
    ctx = _turn(_read())
    ctx.metadata[mk.VOICE_CORRECTION] = {"reason": CRITIQUE, "kind": "unverified_claim"}
    backend = ScriptedBackend(["A aula e as quartas, das 19h as 22h30."])
    result = await SuperegoStage().voice(ctx, backend, voice_prompt="persona")
    sent = backend.calls[0]["prompt"]
    assert STATE_WHAT_IT_CONTAINS in sent and NOTHING_RAN not in sent
    assert STATE_WHAT_IT_CONTAINS in (result.prompt_text or "")
    assert "review_verdict" in [b["block"] for b in result.prompt_blocks]


# ── THE PREMISE IS A CLAIM ABOUT THIS PROMPT, NOT ABOUT THE TURN ─────
#
# Both verdict sections tell the voice that what the tools returned "is in the executor data
# above" and then forbid it from reporting a failure to find it. `read_succeeded_this_turn`
# answers a TURN question and walks three sources; `_tool_payload` renders ONE of them. The
# three rows below are the measured shapes, for BOTH kinds — the review verdict (this file's
# variant) and the execution verdict (the clause that landed first and had the same defect).

_KIND_TO_SLUG = {"unverified_claim": "review_verdict", "not_executed": "execution_verdict"}


@pytest.mark.parametrize("kind", sorted(_KIND_TO_SLUG))
def test_row_0_the_surviving_attempts_read_opens_the_clause(kind):
    """(0) The control. The read is in the payload the voice receives, so the premise is true
    and both clauses fire — this is the behaviour the whole family exists for."""
    ctx = _shape_read_ok()
    assert SuperegoStage._payload_shows_a_read(ctx) is True
    section = SuperegoStage.voice_prompt_block(_render(ctx, kind=kind, reason="R"),
                                               _KIND_TO_SLUG[kind])
    assert _se._EVERY_TOOL_SUCCEEDED in section


@pytest.mark.parametrize("kind", sorted(_KIND_TO_SLUG))
@pytest.mark.parametrize("shape", ["discarded_only", "consult_only"])
def test_rows_i_and_ii_a_read_this_prompt_cannot_SHOW_leaves_the_section_as_it_was(shape, kind):
    """(i) and (ii). The turn read successfully and the prompt does not carry the result, so
    the sentence "it is in the executor data above" would be FALSE and the prohibition that
    follows it would be a muzzle over an honest "I did not find that".

    Pinned as byte equality against the rendering of a turn with no read at all — not by
    substring — so a reworded clause that still leaks into these shapes cannot pass.
    """
    ctx = _SHAPES[shape]()
    assert read_succeeded_this_turn(ctx) is True           # the TURN did read…
    assert SuperegoStage._payload_shows_a_read(ctx) is False   # …this PROMPT shows nothing
    slug = _KIND_TO_SLUG[kind]
    got = SuperegoStage.voice_prompt_block(_render(ctx, kind=kind, reason="R"), slug)
    none_at_all = SuperegoStage.voice_prompt_block(
        _render(_shape_no_exec(), kind=kind, reason="R"), slug)
    assert got == none_at_all, f"{shape}/{kind}: the section is not the no-read rendering"
    assert _se._EVERY_TOOL_SUCCEEDED not in got


# ── THE INVARIANT, STATED DIRECTLY ───────────────────────────────────

@pytest.mark.parametrize("kind", sorted(_KIND_TO_SLUG))
@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_whenever_the_section_claims_the_data_is_above_it_really_IS(shape, kind):
    """The property both clauses depend on, over every shape and both kinds — and asserted
    against the payload `_tool_payload` actually builds, not the fixture string the other
    tests pass in.

    The canary is inside the READ's result, so "the data is above" is checked as *this
    successful read's result reached the prompt*, which is what the sentence promises.
    """
    ctx = _SHAPES[shape]()
    payload = SuperegoStage._tool_payload(ctx)
    prompt = _render(ctx, kind=kind, reason="R", payload=payload)
    section = SuperegoStage.voice_prompt_block(prompt, _KIND_TO_SLUG[kind])
    if _se._EVERY_TOOL_SUCCEEDED in section:
        assert TIMETABLE in payload, (
            f"{shape}/{kind}: the section says the result is in the executor data and the "
            f"payload does not carry it — payload={payload!r}")


def test_the_gate_and_the_payload_share_a_source():
    """`_payload_shows_a_read` and `_tool_payload` must never disagree about WHICH executions
    the prompt shows. They read one list (`_payload_records`); this is the assertion that
    fails the day they stop doing so.

    WHAT IT DOES AND DOES NOT CATCH, measured rather than asserted — the first version of this
    docstring named the wrong mutation and the run said so. Repointing `_payload_records`
    itself at `ctx.turn_executions` leaves this test GREEN, and correctly: both readers move
    together, which is the whole point of the shared source (what that mutation breaks is the
    CLAUSE, and `test_rows_i_and_ii…` and the invariant test go red for it — 40 failures).
    What kills THIS test is the readers DIVERGING: iterating `ctx.turn_executions` inside
    `_payload_shows_a_read` while the renderer keeps reading `_payload_records` turns it red
    (verified 2026-09-19), which is exactly the shape of the defect it guards — a gate that
    believes the prompt shows something the prompt does not show.
    """
    for name, build in sorted(_SHAPES.items()):
        ctx = build()
        assert SuperegoStage._payload_shows_a_read(ctx) is (TIMETABLE in
                                                            SuperegoStage._tool_payload(ctx)), (
            f"{name}: the gate and the rendered payload disagree about the read")


@pytest.mark.asyncio
async def test_the_invariant_holds_through_the_path_that_ships():
    """`voice()` is where the payload and the prompt are built together; a helper-level
    invariant that the real path can break is not an invariant."""
    for shape in ("read_ok", "discarded_only", "consult_only"):
        ctx = _SHAPES[shape]()
        ctx.metadata[mk.VOICE_CORRECTION] = {"reason": CRITIQUE, "kind": "unverified_claim"}
        backend = ScriptedBackend(["ok"])
        result = await SuperegoStage().voice(ctx, backend, voice_prompt="persona")
        sent = backend.calls[0]["prompt"]
        if _se._EVERY_TOOL_SUCCEEDED in sent:
            assert TIMETABLE in sent, f"{shape}: the prompt claims data it does not carry"
        assert (result.prompt_text or "") == sent
