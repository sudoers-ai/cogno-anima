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
SAY_WHAT_WAS_READ = "Write the reply from what the executor data DOES contain"
DROP_ONLY = "DROP the claim the data does not contain"
NEVER_DENY = "MUST NOT tell the contact that you could not access, find, obtain"
STILL_FORBIDDEN = "You MUST NOT repeat the rejected claim, or any softened version of it"

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
    assert SAY_WHAT_WAS_READ not in section


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
    # …and so is the steer that produced the delivered reply
    assert OLD_DEFAULT not in section, (
        "'you do not have that information' was the DEFAULT outcome on a turn that had it")
    # …replaced by what is true, and by an instruction
    assert _se._EVERY_TOOL_SUCCEEDED in section
    assert SAY_WHAT_WAS_READ in section
    assert NEVER_DENY in section
    # the opposite fabrication stays forbidden from the same section
    assert STILL_FORBIDDEN in section
    assert _se._CRITIQUE_IS_NOT_EVIDENCE.strip() in section


def test_the_instruction_is_scoped_to_a_claim_the_data_does_NOT_contain():
    """The scope that keeps the instruction from deleting the answer.

    On the measured turn the executor data holds every figure the draft states; what the net
    flagged was the PROVENANCE of the claim, not a figure missing from the data. A section
    that said "drop the claim" unqualified would tell the voice to drop the very times it is
    also told to reproduce.

    SABOTAGE: widen ``DROP the claim the data does not contain`` to ``DROP the claim`` -> red
    here, green everywhere else in this file.
    """
    section = _verdict(_turn(_read()))
    sentence = next(s for s in section.split(". ") if "DROP the claim" in s)
    assert "the data does not contain" in sentence, (
        "the drop instruction lost its scope: it now reads as 'drop the claim', which on this "
        "turn covers the data the very next sentence orders reproduced")
    # …and the reproduce half is present, with the verbatim demand
    assert "exactly as written there" in _verdict(_turn(_read()))
    assert "If what is left answers the request, that IS the answer" in section


def test_the_limit_is_the_LAST_resort_and_not_the_default():
    """Admitting the data did not hold it stays a COMPLETE reply — but only when that is true.
    The measured defect is the ordering, not the sentence."""
    section = _verdict(_turn(_read()))
    assert "Only when the data holds nothing relevant to the request" in section
    assert "COMPLETE and honest" in section
    assert section.index("Write the reply from what the executor data") < \
        section.index("Only when the data holds nothing relevant")


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


def test_the_consulted_specialists_read_counts():
    """The intra-turn CONSULT is the turn's third execution source, and the predicate walks it:
    a hub whose specialist did the reading has the data in front of it exactly as if it had
    read it itself. A predicate blind to that source would leave the hub denying her answer."""
    ctx = _turn()                       # the hub itself ran nothing
    ctx.consult_result = _ego(_read())  # the specialist did
    assert read_succeeded_this_turn(ctx) is True
    section = _verdict(ctx)
    assert NOTHING_RAN not in section and SAY_WHAT_WAS_READ in section


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

_SHAPES = {"no_exec": (), "read_ok": (_read,), "read_plus_failed": (_read, _failed),
           "write": (_write,)}


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
    ctx = _turn(*(call() for call in _SHAPES[shape]))
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
    assert SAY_WHAT_WAS_READ in sent and NOTHING_RAN not in sent
    assert SAY_WHAT_WAS_READ in (result.prompt_text or "")
    assert "review_verdict" in [b["block"] for b in result.prompt_blocks]
