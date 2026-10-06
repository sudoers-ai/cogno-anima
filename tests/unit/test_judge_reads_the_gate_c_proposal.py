"""A gate-C PROPOSAL is shown to the judge as what it is — and its output grounds ITS OWN held message.

Gate C (``stages/ego.py``): the skill RAN, committed nothing, and asked about THIS call
(``ToolResult(needs_confirmation=True)``). The core records it ``ok=False`` — a proposal must never
count as a write — and the judge's execution block rendered every ``ok=False`` as ``→ ERROR``. So a
held message composed by that very skill, quoting the figures the skill read («R$ 1.920,00»), was
judged against an execution block that said the only read had FAILED, under rules that read figures
off calls marked OK: no source at all. Measured deterministically on a downstream host's templated
e-mail (``send_email``, E1): the figures appeared once, inside the held message, and the execution
read ``(no tools executed)`` under gate B or ``→ ERROR`` under gate C.

The fix is narrow on purpose:

* the proposal renders ``→ PROPOSED (held for the user's confirmation; nothing executed)`` with its
  output — never ``ERROR``; a call held by NAME (gate B, never executed) and a real failure render
  exactly as before;
* criterion (d) of the held-message rule counts that output as a READ for the held message of THAT
  SAME call and nothing else — not the draft, not another message;
* ``committed_this_turn`` and its family do not move: the call stays ``ok=False``;
* a prompt with no proposal is byte for byte what ``main`` rendered (digests taken there, below,
  with a control that sees the change enter).

Names, figures and texts are invented.
"""

from __future__ import annotations

import hashlib

from cogno_anima import is_skill_proposal
from cogno_anima import metakeys as mk
from cogno_anima.stages.superego import PROPOSED_STATUS, SuperegoStage
from cogno_anima.types import (HELD_BY_NAME_PREFIX, EgoResult, EgoStep, ToolExecution,
                               committed_this_turn, write_attempted_this_turn)
from tests.unit.test_superego import _ctx, _m

#: What the e-mail skill READ and proposed — the output of the gate-C call.
PROPOSAL = ("PROPOSTA — não enviado. Lido da agenda: 4 aulas × 4 h × R$ 120,00 = R$ 1.920,00; "
            "valor por aula 4 h × R$ 120,00 = R$ 480,00.\n"
            "Assunto: Emissão de nota fiscal — setembro/2026")
#: The e-mail the same call holds for the user's «sim» — its figures are the proposal's.
RENDERED = ("Emissão de nota fiscal — setembro/2026\nOlá, Prof Alfa,\n\n"
            "valor por aula: R$ 480,00\nSem bônus: R$ 1.920,00")
#: A real failure of the same tool, for the control.
FAILED = "the mail server did not accept the message"


def _proposal(result: str = PROPOSAL, tool: str = "send_email", rendered: str = RENDERED):
    return ToolExecution(tool=tool, arguments={"recipient": "Prof Alfa", "rendered": rendered},
                         ok=False, error="needs_confirmation", side_effect=False,
                         tool_mutating=True, result=result)


def _held_by_name(tool: str = "notify_user", message: str = "Olá Joana, a aula mudou."):
    return ToolExecution(tool=tool, arguments={"target": "Joana", "message": message}, ok=False,
                         error="needs_confirmation", tool_mutating=True,
                         result=f"{HELD_BY_NAME_PREFIX} '{tool}' is destructive and was NOT "
                                "executed; it needs explicit user confirmation first.")


def _turn(*calls, held=(), draft="", declared=None):
    ctx = _ctx(user="manda o e-mail da nota fiscal de setembro para o Prof Alfa",
               goal="send the invoice e-mail", with_ego=False)
    ctx.ego_result = EgoResult(
        steps=[EgoStep(index=0, path="native", assistant_text=draft, tool_calls=list(calls))],
        pending_confirmation=list(held), metrics=_m("ego"))
    if declared is not None:
        ctx.metadata[mk.HELD_DELIVERED_TEXT] = declared
    return ctx


def _prompt(ctx) -> str:
    return SuperegoStage()._build_judge_prompt(ctx, "")


# ── the predicate ──────────────────────────────────────────────────────────────────────────

def test_the_predicate_tells_a_proposal_from_a_name_hold_and_a_failure():
    assert is_skill_proposal(_proposal()) is True
    assert is_skill_proposal(_held_by_name()) is False                       # gate B
    failed = ToolExecution(tool="send_email", ok=False, error="smtp_refused", result=FAILED)
    assert is_skill_proposal(failed) is False                                # a real failure
    assert is_skill_proposal(ToolExecution(tool="x", ok=True, result="ok")) is False
    # a held call with NO output wrote nothing to read — the shape of every gate-B test double
    assert is_skill_proposal(_proposal(result="")) is False
    assert is_skill_proposal(_proposal(result="   ")) is False
    assert is_skill_proposal(object()) is False


# ── the twin: PROPOSED, with its output, and the clause for ITS held message ─────────────────

def test_the_twin_a_proposal_renders_PROPOSED_with_its_output_and_grounds_its_message():
    p = _prompt(_turn(_proposal(), held=[_proposal()], declared={"send_email": "rendered"}))
    assert f"→ {PROPOSED_STATUS}:\n<tool_output name=\"send_email\">\n{PROPOSAL}" in p
    assert "→ ERROR" not in p
    assert "A held message of a call marked PROPOSED above (`send_email`)" in p
    assert "It supports NOTHING else: not the EGO draft" in p
    # the figure has a source on the page beside the message now: the proposal's own output
    out = p[p.index('<tool_output name="send_email">'):p.index("</tool_output>")]
    held = p[p.index('<held_message name="send_email">'):p.index("</held_message>")]
    assert "R$ 1.920,00" in out and "R$ 1.920,00" in held


def test_control_the_same_figure_in_the_DRAFT_gets_no_new_source():
    """The clause names the held message of the proposed call and refuses everything else in the
    same sentence — the draft keeps today's criteria, word for word."""
    p = _prompt(_turn(_proposal(), held=[_proposal()], declared={"send_email": "rendered"},
                      draft="O total do Prof Alfa é R$ 1.920,00."))
    clause_start = p.index("A held message of a call marked PROPOSED above")
    clause = p[clause_start:p.index("\n", clause_start)]
    assert "not the EGO draft" in clause
    assert "successful tool result" in p          # the draft's grounding rule stays as it was


def test_control_a_real_failure_still_renders_ERROR_and_widens_nothing():
    failed = ToolExecution(tool="send_email", arguments={"rendered": RENDERED}, ok=False,
                           error="smtp_refused", result=FAILED)
    p = _prompt(_turn(failed, held=[], declared={"send_email": "rendered"}))
    assert "→ ERROR:" in p and PROPOSED_STATUS not in p
    assert "marked PROPOSED above" not in p


def test_control_a_call_held_by_NAME_renders_as_before():
    held = _held_by_name()
    p = _prompt(_turn(held, held=[held], declared={"notify_user": "message"}))
    assert "→ ERROR:" in p and PROPOSED_STATUS not in p
    assert "marked PROPOSED above" not in p


def test_the_clause_names_only_proposed_calls_that_deliver_a_held_message():
    """A proposal whose tool delivers no declared text has no held message to ground."""
    p = _prompt(_turn(_proposal(tool="book_room"), held=[_proposal(tool="book_room")],
                      declared={"send_email": "rendered"}))
    assert f"→ {PROPOSED_STATUS}" in p
    assert "marked PROPOSED above" not in p
    # …and beside a proposal that DOES deliver one, the clause names that one alone.
    both = [_proposal(tool="book_room"), _proposal()]
    p = _prompt(_turn(*both, held=both, declared={"send_email": "rendered"}))
    assert "A held message of a call marked PROPOSED above (`send_email`) is judged" in p


def test_the_proposal_is_still_NOT_a_write():
    ctx = _turn(_proposal(), held=[_proposal()], declared={"send_email": "rendered"})
    assert committed_this_turn(ctx) is False
    assert write_attempted_this_turn(ctx) is True          # it IS an attempt, as before


# ── the behaviour that CHANGES: the draft no longer has to say «pendente» ───────────────────

def test_a_proposal_tells_the_judge_the_call_is_held_and_the_asking_is_the_hosts():
    """Measured downstream (M2, gpt-5.6-luna, n=5): over an empty draft beside a held proposal,
    the judge of main REJECTED 5/5 asking the draft to say the e-mail was pending, and the branch
    APPROVED 5/5. The two prompts differ only in the status word: `→ ERROR` (main) against
    `→ PROPOSED (held for the user's confirmation; nothing executed)` — and the held-message rule
    already says the asking is the HOST's (`_HELD_ASKING_IS_THE_HOSTS`, #199). With PROPOSED the
    two facts are on the page together, so a draft that does not say «pendente» is no longer a
    defect. This pins the PROMPT half of that change; the model half is the downstream M2."""
    ctx = _turn(_proposal(), held=[_proposal()], declared={"send_email": "rendered"})
    p = _prompt(ctx)
    assert f"→ {PROPOSED_STATUS}:" in p
    assert "ADDED BY THE HOST, AFTER this judgement" in p
    import cogno_anima.stages.superego as sg
    real = sg.is_skill_proposal
    try:
        sg.is_skill_proposal = lambda call: False          # the main's rendering of the call
        before = _prompt(_turn(_proposal(), held=[_proposal()],
                               declared={"send_email": "rendered"}))
    finally:
        sg.is_skill_proposal = real
    assert "→ ERROR:" in before and PROPOSED_STATUS not in before
    assert "ADDED BY THE HOST, AFTER this judgement" in before   # the rule was already there


# ── byte for byte when there is no proposal (digests taken on origin/main 02e1850) ───────────

def _digest(ctx) -> str:
    return hashlib.sha256(_prompt(ctx).encode("utf-8")).hexdigest()


def _fixtures():
    ok = ToolExecution(tool="daily_checks", arguments={}, ok=True,
                       result="Aula 19h, sala 4: revisão do capítulo 3.")
    failed = ToolExecution(tool="send_email", arguments={"rendered": RENDERED}, ok=False,
                           error="smtp_refused", result=FAILED)
    held = _held_by_name()
    return {
        "read_only": _turn(ok, draft="A aula é às 19h na sala 4."),
        "failure": _turn(failed, draft="Não consegui enviar."),
        "gate_b_held_message": _turn(ok, held, held=[held], declared={"notify_user": "message"}),
        "nothing": _turn(draft="Olá!"),
    }


#: ``sha256`` of each fixture's judge prompt on ``origin/main`` (02e1850), before this change.
_ORIGIN_MAIN = {
    "read_only": "1b4e040efa8703d402e60bcd3adf41daac6b07ae31500b970ac8bbe7dd7f8c04",
    "failure": "c627d58efb24d7310d3eba3cee8c9d9e61ea2d94eec968d90c77d71bd4c2f15c",
    "gate_b_held_message": "4f0262b6aad451aba8874bb1408384d1209dce682530720f9af329e34924ab6e",
    "nothing": "ad4f1c07f825f0ada5a86d1b07bfd201315bee197f51532afebf8ed44fd54070",
}


def test_without_a_proposal_every_prompt_is_byte_for_byte_main():
    for name, ctx in _fixtures().items():
        assert _digest(ctx) == _ORIGIN_MAIN[name], name


def test_control_the_digest_sees_the_change_enter():
    """The instrument's own control: a proposal in the same fixture moves the digest, so the
    equality above is a property of these prompts and not of a hash that cannot move."""
    ctx = _fixtures()["read_only"]
    ctx.ego_result.steps[0].tool_calls.append(_proposal())
    assert _digest(ctx) != _ORIGIN_MAIN["read_only"]
