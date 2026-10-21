"""A person's NAME is a value too: the judge does not reject the name the USER used.

The defect (measured downstream, a reference host's booking benches: 17 of 94 judge rejections
in 8 runs): the contact books «com o Dr. Ruivo», the executor acts on the right professional,
the tool answers with the full record («Dr. Ruivo Tavares», or a name that differs in form),
and the draft says «Dr. Ruivo» — the name the CONTACT used, for the SAME person. The judge
rejected it as a grounding failure: the draft «names Dr. Ruivo but the record says Dr. Ruivo
Tavares». The same family as the preserved term the judge read as a spelling instead of a value
(`_PRESERVED_IS_A_VALUE`, 2026-09-18).

The rule, ``_PERSON_NAME_IS_A_VALUE``: a name is judged by WHO it designates, never by its
form — the name the user used, or a shorter or fuller form of that same name, for the person the
execution acted on is CORRECT; only a name the evidence ties to a DIFFERENT person is rejected.

**It travels with its evidence.** It renders only on a turn whose request named somebody —
``intent.entities_people``, the NER's reading of the contact's message, with at least one
non-blank entry. It renders no name: the judge already reads the request under
``# User request``, so there is nothing of the contact's to add. A turn with no person named
gets the prompt it always got, byte for byte: the digests below were taken on the base of this
change (anima #225, ``b714083``) — a landing-time proof, regenerated when the judge's text
changes on purpose.

The model half — does the judge then approve the short name and still reject another person —
is ``tests/integration/test_superego.py`` (cloud spec, written, not run here).
"""

from __future__ import annotations

import hashlib

import pytest

from cogno_anima import metakeys as mk
from cogno_anima.stages import superego as sg
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.types import (EgoResult, EgoStep, IntentResult, PipelineContext, StageMetrics,
                               ToolExecution)


def _m(stage: str) -> StageMetrics:
    return StageMetrics(stage=stage, elapsed_ms=1.0, tokens_in=1, tokens_out=1, model="t")


def _intent(people) -> IntentResult:
    return IntentResult(intent_class="ACTION_REQUEST", sentiment="NEUTRAL", confidence=0.9,
                        temporal_class="RECENT", triad_signal="EGO",
                        goal="book a consultation with the doctor the user named",
                        entities_people=list(people), metrics=_m("ner"))


_BOOK = ToolExecution(tool="book_appointment",
                      arguments={"host_id": "dr_ruivo", "date": "2026-10-22", "time": "09:00"},
                      result="Booked: Dr. Ruivo Tavares, 2026-10-22 09:00.", ok=True,
                      side_effect=True)
_READ = ToolExecution(tool="list_appointments", arguments={"host_id": "dr_ruivo"},
                      result="Dr. Ruivo Tavares: 2026-10-22 09:00 (CONFIRMED).", ok=True,
                      side_effect=False)


def _ctx(branch: str, people=(), *, context: str = "") -> PipelineContext:
    ctx = PipelineContext(user_input="marca amanhã às 9 com o Dr. Ruivo")
    ctx.intent = _intent(people)
    call = _READ if branch == "readonly" else _BOOK
    steps = [EgoStep(index=0, path="native", tool_calls=[call]),
             EgoStep(index=1, path="native",
                     assistant_text="Booked with Dr. Ruivo tomorrow at 09:00.")]
    ctx.ego_result = EgoResult(steps=steps, metrics=_m("ego"))
    if branch == "conversational":
        ctx.metadata[mk.JUDGE_CONVERSATIONAL] = True
    if context:
        ctx.metadata[mk.EGO_CONTEXT] = context
    return ctx


def _sent(ctx: PipelineContext, limits: str = "") -> str:
    """The two messages the judge is sent, as one text."""
    return SuperegoStage._judge_system(ctx) + "\n\x00\n" + \
        SuperegoStage()._build_judge_prompt(ctx, limits)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


CELLS = [(b, people, limits, context)
         for b in ("execution", "readonly", "conversational")
         for people in ((), ("   ",))
         for limits in ("", "Agendamentos só em dias úteis.")
         for context in ("", "[HOJE] 2026-10-21")]

#: sha256[:16] of what the judge is sent for each cell with NO person named — taken on anima
#: #225 (`b714083`), before this change. Filled by `_regenerate` below.
BASE: "dict[str, str]" = {
    "execution|people=0|limits=False|context=False": "0d4a6a32988bb092",
    "execution|people=0|limits=False|context=True": "f124b69c1962909f",
    "execution|people=0|limits=True|context=False": "e0453bb64e044515",
    "execution|people=0|limits=True|context=True": "1e80b2f2186d9078",
    "execution|people=1|limits=False|context=False": "0d4a6a32988bb092",
    "execution|people=1|limits=False|context=True": "f124b69c1962909f",
    "execution|people=1|limits=True|context=False": "e0453bb64e044515",
    "execution|people=1|limits=True|context=True": "1e80b2f2186d9078",
    "readonly|people=0|limits=False|context=False": "dadb5f105f1bbb66",
    "readonly|people=0|limits=False|context=True": "79770bd6e8e5e36c",
    "readonly|people=0|limits=True|context=False": "b89aaa37c36b316b",
    "readonly|people=0|limits=True|context=True": "53d8e1feea5f3c0a",
    "readonly|people=1|limits=False|context=False": "dadb5f105f1bbb66",
    "readonly|people=1|limits=False|context=True": "79770bd6e8e5e36c",
    "readonly|people=1|limits=True|context=False": "b89aaa37c36b316b",
    "readonly|people=1|limits=True|context=True": "53d8e1feea5f3c0a",
    "conversational|people=0|limits=False|context=False": "2a0dab60322feea5",
    "conversational|people=0|limits=False|context=True": "620f42169aeb3bb2",
    "conversational|people=0|limits=True|context=False": "1fa67b2e2c5e310d",
    "conversational|people=0|limits=True|context=True": "112d145316d907dc",
    "conversational|people=1|limits=False|context=False": "2a0dab60322feea5",
    "conversational|people=1|limits=False|context=True": "620f42169aeb3bb2",
    "conversational|people=1|limits=True|context=False": "1fa67b2e2c5e310d",
    "conversational|people=1|limits=True|context=True": "112d145316d907dc",
}


def _cell_id(cell) -> str:
    b, people, limits, context = cell
    return f"{b}|people={len(people)}|limits={bool(limits)}|context={bool(context)}"


@pytest.mark.parametrize("cell", CELLS, ids=_cell_id)
def test_CONTROL_with_no_person_named_the_judge_is_sent_the_bytes_of_before(cell):
    b, people, limits, context = cell
    assert _digest(_sent(_ctx(b, people, context=context), limits)) == BASE[_cell_id(cell)]


def test_PAR_the_digests_tell_the_cells_apart():
    """A digest that every cell shared would agree with any prompt. Twelve distinct values:
    the blank-name cells are the no-name cells, every other axis moves the bytes."""
    assert len(set(BASE.values())) == 12


@pytest.mark.parametrize("branch", ["execution", "readonly", "conversational"])
def test_TWIN_a_request_that_names_somebody_carries_the_rule(branch):
    """On the base: absent in all three branches."""
    sent = _sent(_ctx(branch, ("Dr. Ruivo",)))
    assert sg._PERSON_NAME_IS_A_VALUE in sent
    assert sent.count(sg._PERSON_NAME_IS_A_VALUE) == 1


@pytest.mark.parametrize("branch", ["execution", "readonly", "conversational"])
def test_the_rule_is_the_ONLY_difference_and_renders_no_name(branch):
    named = _sent(_ctx(branch, ("Dr. Ruivo",)))
    plain = _sent(_ctx(branch, ()))
    assert named.replace(sg._PERSON_NAME_IS_A_VALUE, "", 1) == plain
    assert "Ruivo" not in sg._PERSON_NAME_IS_A_VALUE


def test_the_rule_says_value_not_form_and_keeps_another_person_a_rejection():
    rule = sg._PERSON_NAME_IS_A_VALUE
    assert "REJECT" in rule and "DIFFERENT person" in rule
    assert "the name the USER used" in rule


def test_the_inventory_of_the_judge_prompt_does_not_move():
    """No header: the rule is a paragraph after the criteria, like the memo rule — the
    inventory's rows are the same with and without it."""
    ctx_named, ctx_plain = _ctx("execution", ("Dr. Ruivo",)), _ctx("execution", ())
    rows = [SuperegoStage.judge_prompt_inventory(SuperegoStage()._build_judge_prompt(c, ""))
            for c in (ctx_named, ctx_plain)]
    assert [r["block"] for r in rows[0]] == [r["block"] for r in rows[1]]


def _regenerate() -> "dict[str, str]":  # pragma: no cover — run by hand on the base tree
    return {_cell_id(c): _digest(_sent(_ctx(c[0], c[1], context=c[3]), c[2])) for c in CELLS}
