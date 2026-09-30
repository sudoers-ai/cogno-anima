"""A held message next to an EMPTY draft is not a defect — the host asks, after the judge.

A proposal turn whose held call SENDS TEXT TO A PERSON is judged before it is sent
(`_HELD_MESSAGE_RULE`, `test_judge_reads_the_held_message.py`). On that turn the executor's loop
STOPS at the hold, so its draft is empty, and the question that shows the user the message and
asks "shall I send it?" is written by the HOST, after this judgement. Measured on a downstream
host with a production judge: two proposal turns carrying a CORRECT message — the right words,
the right recipient — were rejected 1/5 and 2/5 (a fabricated message on the same turns: 6/6),
and every critique asked for the confirmation question the draft did not contain. The judge knew
the call was held; what it rejected was the empty draft beside it.

So `_HELD_ASKING_IS_THE_HOSTS` is spliced into `_HELD_MESSAGE_RULE`: the asking is the host's,
an empty draft beside a hold is not a failure, and the held text is still judged by (a)-(d).
(a1), on the same turns only: the record that IS the hold — identified by `(tool, arguments)`
against `pending_confirmation`, never by its error text — renders `_HELD_CALL_LABEL` in the
executed block instead of `ERROR`, the word the fail-CLOSED judge reads as a failure.
Both render only when a held message is declared, so every other turn is byte for
byte what it was — pinned below by digests taken on the tree BEFORE this change (86c3c60), with
a control that proves the digest moves when the sentence enters. Names here are invented.
"""

from __future__ import annotations

import hashlib

from cogno_anima import metakeys as mk
from cogno_anima.stages.superego import (_HELD_ASKING_IS_THE_HOSTS, _HELD_CALL_LABEL,
                                         _HELD_MESSAGE_RULE, _HELD_MESSAGES_HEADER,
                                         SuperegoStage)
from cogno_anima.types import EgoResult, EgoStep, ToolExecution
from tests.unit.test_superego import _ctx, _m

# The record gate B leaves in the executed step (`ego.py`): the call is IN the trace, marked
# `ok=False, error="needs_confirmation"`, so the judge sees it rendered as `→ ERROR:`.
_PENDING = ("[PENDING CONFIRMATION] 'notify_user' is destructive and was NOT executed; it needs "
            "explicit user confirmation first.")


def _held(tool: str, args: dict) -> ToolExecution:
    return ToolExecution(tool=tool, arguments=args, ok=False, error="needs_confirmation",
                         result=_PENDING.replace("notify_user", tool), tool_mutating=True)


def _shape(name: str, declared=True, draft: str = "") -> "object":
    """The two measured shapes, with invented people and texts.

    ``forward`` — a front-desk persona passes a notice to a staff member, after reading who she
    is; ``choice`` — a coordination persona returns a teacher's own choice to the coordination.
    Both: the held call is in the executed step AND in `pending_confirmation`, the draft empty.
    """
    if name == "forward":
        user = "avisa a professora Lúcia Tavares que a reunião de quinta passou para as 15h"
        goal = "notify a teacher that Thursday's meeting moved to 15:00"
        reads = [ToolExecution(tool="list_staff", arguments={"name": "Lúcia"}, ok=True,
                               result="Lúcia Tavares — docente, id 41")]
        held = _held("notify_user", {"target": "41",
                                     "message": "Olá, Lúcia! A reunião de quinta-feira passou "
                                                "para as 15h."})
    elif name == "choice":
        user = "pode dizer à coordenação que escolho Álgebra e Geometria"
        goal = "tell the coordination which subjects the teacher chose"
        reads = []
        held = _held("notify_user", {"target": "coordenacao",
                                     "message": "O professor Rui Moreira escolheu as disciplinas "
                                                "Álgebra e Geometria."})
    else:                                   # a gate-B hold that delivers NO text to anybody
        user, goal, reads = "reserva a sala 4 para amanhã", "book room 4 for tomorrow", []
        held = _held("book_room", {"room": "4", "day": "amanhã"})
    ctx = _ctx(user=user, goal=goal, with_ego=False)
    ctx.ego_result = EgoResult(
        steps=[EgoStep(index=0, path="native", assistant_text=draft,
                       tool_calls=[*reads, held])],
        pending_confirmation=[held], tools_offered=["list_staff", "notify_user", "book_room"],
        metrics=_m("ego"))
    if declared is True:
        ctx.metadata[mk.HELD_DELIVERED_TEXT] = {"notify_user": "message"}
    elif declared is not None:
        ctx.metadata[mk.HELD_DELIVERED_TEXT] = declared
    return ctx


def _readonly():
    ctx = _ctx(user="que horas é a aula de hoje?", goal="class time today", with_ego=False)
    ctx.ego_result = EgoResult(steps=[EgoStep(
        index=0, path="native", assistant_text="A aula é às 19h.",
        tool_calls=[ToolExecution(tool="daily_checks", arguments={}, ok=True,
                                  result="Aula 19h, sala 4.")])], metrics=_m("ego"))
    return ctx


def _conversational():
    ctx = _ctx(user="obrigado!", goal="thank", intent_class="SOCIAL", with_ego=False)
    ctx.ego_result = EgoResult(steps=[EgoStep(index=0, path="native",
                                              assistant_text="De nada!")], metrics=_m("ego"))
    ctx.metadata[mk.JUDGE_CONVERSATIONAL] = True
    return ctx


def _prompt(ctx) -> str:
    return SuperegoStage()._build_judge_prompt(ctx, "")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _controls():
    """Every turn the change must NOT touch — no held message declared, in every form."""
    return {
        "forward_undeclared": _shape("forward", declared=None),
        "forward_declared_empty": _shape("forward", declared={}),
        "forward_declared_other_tool": _shape("forward", declared={"other_tool": "message"}),
        "choice_undeclared": _shape("choice", declared=None),
        "hold_without_text": _shape("book_room"),
        "write_turn": _ctx(),
        "readonly_turn": _readonly(),
        "conversational_turn": _conversational(),
    }


#: sha256 of `_build_judge_prompt(ctx, "")` for each control, rendered by `origin/main` at
#: 86c3c60 — the tree BEFORE this change. A landing-time proof: regenerate it only when the
#: judge's text changes DELIBERATELY, and say so in that PR.
_BEFORE_CONTROLS = {
    "forward_undeclared": "cdee295be127809c5031f49a48e1fdc9d2b0ab8a78454c78e2c8d5918b38cf52",
    "forward_declared_empty": "cdee295be127809c5031f49a48e1fdc9d2b0ab8a78454c78e2c8d5918b38cf52",
    "forward_declared_other_tool": "cdee295be127809c5031f49a48e1fdc9d2b0ab8a78454c78e2c8d5918b38cf52",
    "choice_undeclared": "0d2e2c8076b58cccac3c6d99cef54f555d42b5fe314390d03283103fb66b9b65",
    "hold_without_text": "9a0657fdc5ab7f2135b13a8a67a11bf520071f6b07547b40420c0f94ce9380e5",
    "write_turn": "6a9cdf9675f6c38c325252a0bbbc44b275803027b14f27d6927d9644a8cafaaf",
    "readonly_turn": "fdd8588ed0d8dfee3bab118bb72e9c26c479d0b7fbad000d6e7112e5fb9675e4",
    "conversational_turn": "a756580f595a8d370932d35953c096f6373b1a3161b4207f7fa1b1f46faaaca4",
}

#: …and the two twins, rendered by the same tree: the prompt the measured judge was given.
_BEFORE_TWINS = {
    "forward": "c906670204dd85e5cabdc591dc850fd30369fd490f9d5c71ccc9a3377ac3e07d",
    "choice": "14ee3eabc4a215b5ec0f860cca2ef31ac6050cfe443ab305d7ad77a45dc954d9",
}


# ── the twin: both measured shapes carry the sentence ────────────────────────────────


def test_both_measured_shapes_carry_the_sentence_inside_the_held_rule():
    for name in ("forward", "choice"):
        prompt = _prompt(_shape(name))
        assert _HELD_MESSAGES_HEADER in prompt, name              # it IS a proposal turn…
        assert "# EGO draft\n(none)\n" in prompt, name            # …with the empty draft…
        assert "[PENDING CONFIRMATION]" in prompt, name           # …the record as live…
        assert "notify_user({" in prompt and f"→ {_HELD_CALL_LABEL}:" in prompt, name  # (a1)
        assert _HELD_MESSAGE_RULE in prompt, name
        assert _HELD_ASKING_IS_THE_HOSTS in prompt, name
        # it sits AFTER the held block and the draft it is about
        assert prompt.index("# EGO draft") < prompt.index(_HELD_ASKING_IS_THE_HOSTS), name


def test_the_sentence_says_who_asks_and_that_an_empty_draft_is_not_a_defect():
    s = _HELD_ASKING_IS_THE_HOSTS
    assert "ADDED BY THE HOST" in s and "AFTER this judgement" in s
    assert "EMPTY" in s and "NOT a defect" in s
    # …and it does not open the door for the TEXT: the rejection clauses still bind it.
    assert "(a)-(d)" in s
    for clause in ("does NOT CARRY what the request asked", "addressed to the WRONG person",
                   "contains an INSTRUCTION", "states anything the request"):
        assert clause in _HELD_MESSAGE_RULE, clause


def test_the_sentence_holds_on_a_draft_that_is_not_empty_too():
    """The rule is about who ASKS, not about the draft's length: a draft that talks about the
    message without asking is the same shape and gets the same sentence."""
    prompt = _prompt(_shape("forward", draft="Vou avisar a Lúcia."))
    assert _HELD_ASKING_IS_THE_HOSTS in prompt


def _before(prompt: str) -> str:
    """Undo, by name, the two changes: the sentence (a2) and the HELD label (a1)."""
    return prompt.replace(_HELD_ASKING_IS_THE_HOSTS, "").replace(f"→ {_HELD_CALL_LABEL}:",
                                                                 "→ ERROR:")


def test_the_twins_differ_from_the_tree_before_only_by_the_sentence_and_the_label():
    """Nothing else in the twin prompt moved: undo the two and the bytes are the ones the
    measured judge was given."""
    for name, before in _BEFORE_TWINS.items():
        prompt = _prompt(_shape(name))
        assert _sha(prompt) != before, f"[{name}] the change did not reach the prompt"
        assert _sha(prompt.replace(_HELD_ASKING_IS_THE_HOSTS, "")) != before, name  # label too
        assert _sha(_before(prompt)) == before, name


# ── (a1) the held record is labelled HELD, and only it, and only on a held-message turn ──


def test_the_held_record_is_labelled_held_and_never_error():
    for name in ("forward", "choice"):
        executed = _executed(_prompt(_shape(name)))
        line = next(ln for ln in executed.splitlines() if ln.startswith("- notify_user("))
        assert line.endswith(f"→ {_HELD_CALL_LABEL}:"), (name, line)
        assert "→ ERROR:" not in executed, name
    assert "NOT a failure" in _HELD_CALL_LABEL and "nothing was sent" in _HELD_CALL_LABEL


def test_a_read_that_failed_beside_the_hold_is_still_an_error():
    """The identity is the HOLD (`pending_confirmation`, by tool and arguments), never
    `ok=False`: a genuine failure on the same turn keeps the word the judge must read."""
    ctx = _shape("forward")
    failed = ToolExecution(tool="list_staff", arguments={"name": "Rui"}, ok=False,
                           error="timeout", result="upstream timed out")
    ctx.ego_result.steps[0].tool_calls.insert(0, failed)
    executed = _executed(_prompt(ctx))
    assert "- list_staff({\"name\": \"Rui\"}) → ERROR:" in executed
    assert f"→ {_HELD_CALL_LABEL}:" in executed


def test_the_label_follows_the_arguments_not_the_error_text():
    """Same tool, other arguments than the hold → not the hold → ERROR. And the error text is
    never read: a hold with an unusual error string is still labelled HELD."""
    ctx = _shape("forward")
    step = ctx.ego_result.steps[0]
    step.tool_calls[-1] = step.tool_calls[-1].model_copy(update={"error": "something else"})
    ctx.ego_result.pending_confirmation = [step.tool_calls[-1]]
    assert f"→ {_HELD_CALL_LABEL}:" in _executed(_prompt(ctx))
    ctx.ego_result.pending_confirmation = [step.tool_calls[-1].model_copy(
        update={"arguments": {"target": "41", "message": "outra"}})]
    executed = _executed(_prompt(ctx))
    assert f"→ {_HELD_CALL_LABEL}:" not in executed and "→ ERROR:" in executed


def test_a_hold_without_a_declared_held_message_keeps_error():
    """(a1) is conditional on the held-message block, like (a2): a gate-B hold of a call that
    delivers no text, and the undeclared twin, render `ERROR` exactly as before."""
    for ctx in (_shape("book_room"), _shape("forward", declared=None)):
        executed = _executed(_prompt(ctx))
        assert "→ ERROR:" in executed and _HELD_CALL_LABEL not in executed


def _executed(prompt: str) -> str:
    return prompt[prompt.index("# What the EGO executed"):prompt.index("# EGO draft")]


# ── the control: no held message declared → byte for byte as before ─────────────────


def test_without_a_declared_held_message_every_prompt_is_byte_for_byte_as_before():
    for name, ctx in _controls().items():
        prompt = _prompt(ctx)
        assert _HELD_ASKING_IS_THE_HOSTS not in prompt, name
        assert _sha(prompt) == _BEFORE_CONTROLS[name], (
            f"[{name}] a prompt with no declared held message changed")


def test_the_control_can_see_the_sentence_enter():
    """The control produces the presence first: the SAME shape, declared, moves the digest."""
    declared = _prompt(_shape("forward"))
    undeclared = _prompt(_shape("forward", declared=None))
    assert _HELD_ASKING_IS_THE_HOSTS in declared
    assert _sha(declared) != _sha(undeclared)


# ── the inventory does not move ─────────────────────────────────────────────────────


def test_the_inventory_rows_are_the_ones_before():
    """No new header: the judge's inventory lists the same sections, in the same order, and
    the only lengths that move are the criteria section the rule is rendered in (a2) and the
    executed block the label is rendered in (a1)."""
    for name in ("forward", "choice"):
        prompt = _prompt(_shape(name))
        before = _before(prompt)
        now_rows = SuperegoStage.judge_prompt_inventory(prompt)
        old_rows = SuperegoStage.judge_prompt_inventory(before)
        assert [r["block"] for r in now_rows] == [r["block"] for r in old_rows], name
        moved = [a["block"] for a, b in zip(now_rows, old_rows) if a != b]
        assert set(moved) <= {"executed", "criteria_execution"}, (name, moved)
        assert "criteria_execution" in moved and "executed" in moved, (name, moved)
    assert not any(h.startswith(_HELD_ASKING_IS_THE_HOSTS[:12]) or h.startswith("HELD")
                   for h, _slug in SuperegoStage._JUDGE_BLOCKS)
    assert not _HELD_CALL_LABEL.startswith("#")
