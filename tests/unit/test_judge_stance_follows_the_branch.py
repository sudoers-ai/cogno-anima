"""The judge stops contradicting itself (0.1.15): the stance follows the branch; a hold is HELD.

## The shapes (measured downstream over 30 days; anonymised)

* 31 turns ended with the judge exhausted. 14 were on the READ-ONLY branch, every call a
  successful read, and 4 on the CONVERSATIONAL one: 18 of 31 on the two branches whose criteria
  open "APPROVE BY DEFAULT … This overrides the general 'do not approve what you cannot verify'
  stance". The stance they override is the judge's SYSTEM message, the same on every branch:
  "Default to NOT approving when you cannot verify the criteria." An instruction-tuned model
  ranks the system message above the user half, so the prompt contradicted itself on its
  strongest side.
* A held message the owner called perfect was rejected twice. Criterion (d) of the held-message
  rule was an EXCLUSIVE list ("anything the request and the successful tool results above do not
  support"), narrower than the GROUNDING criterion of the same prompt, and the critiques named
  values from the conversation history and a total the message showed its parts for.
* «pode enviar» with no live proposal: the executor re-issued the message, the gate held it as it
  must, the call rendered `→ ERROR`, and the judge rejected because "the message was not sent".

## What is pinned here

* THE TWINS, red on the base tree by ASSERTION (nothing new is imported from the library): the
  read-only and conversational system messages carry no "Default to NOT approving"; a held call
  renders HELD; (d) points at the GROUNDING criterion; the hold-is-the-outcome sentence renders.
* THE CONTROLS, byte for byte the base tree's (`f9f10d7`): the EXECUTION branch's whole request
  (system + user half) over a grid of turns; the USER half of the read-only and conversational
  branches; a call that RAN and failed still renders `ERROR`.
* The helper every older digest proof now hashes through (`_judge_0115.as_0114`) undoes exactly
  these changes, and is the identity on the execution branch with no held call.
"""

from __future__ import annotations

import hashlib

import pytest

from cogno_anima import metakeys as mk
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.types import (EgoResult, EgoStep, IntentResult, PipelineContext, StageMetrics,
                               ToolExecution)

_DEFAULT_NOT = "Default to NOT approving"


def _m(stage: str) -> StageMetrics:
    return StageMetrics(stage=stage, elapsed_ms=1.0, tokens_in=1, tokens_out=1, model="t")


def _ctx(calls=(), *, user: str = "que horas abre amanhã?", draft: str = "Abre às 8h.",
         conversational: bool = False, held=(), declared=None, interrupted: bool = False,
         rules: str = "", context: str = "", cls: str = "INFORMATION_REQUEST") -> PipelineContext:
    ctx = PipelineContext(user_input=user)
    ctx.intent = IntentResult(intent_class=cls, sentiment="NEUTRAL", confidence=0.9,
                              temporal_class="RECENT", triad_signal="EGO",
                              goal="answer the user", metrics=_m("ner"))
    steps = [EgoStep(index=0, path="native", tool_calls=list(calls))] if calls else []
    steps.append(EgoStep(index=len(steps), path="native", assistant_text=draft))
    ctx.ego_result = EgoResult(steps=steps, pending_confirmation=list(held),
                               interrupted=interrupted, metrics=_m("ego"))
    if conversational:
        ctx.metadata[mk.JUDGE_CONVERSATIONAL] = True
    if declared is not None:
        ctx.metadata[mk.HELD_DELIVERED_TEXT] = declared
    if rules:
        ctx.metadata[mk.PERSONA_RULES] = rules
    if context:
        ctx.metadata[mk.EGO_CONTEXT] = context
    return ctx


_READ = ToolExecution(tool="opening_hours", arguments={}, result="Opens 08:00.", ok=True,
                      side_effect=False)
_FAILED_READ = ToolExecution(tool="opening_hours", arguments={}, result="", error="timeout",
                             ok=False, side_effect=False)
_WRITE = ToolExecution(tool="book_room", arguments={"room": "4"}, result="Booked room 4.",
                       ok=True, side_effect=True)
_FAILED_WRITE = ToolExecution(tool="book_room", arguments={"room": "4"}, result="",
                              error="room taken", ok=False, side_effect=False, tool_mutating=True)
_MESSAGE = "Olá Belmira! A reunião de amanhã é às 15h, na sala 4. Total: 2 x R$ 50,00 = R$ 100,00."


def _held_call(prefixed: bool) -> ToolExecution:
    return ToolExecution(tool="notify_user", arguments={"to": "Belmira", "message": _MESSAGE},
                         ok=False, error="needs_confirmation", tool_mutating=True,
                         result=("[PENDING CONFIRMATION] 'notify_user' is destructive and was "
                                 "NOT executed; it needs explicit user confirmation first."
                                 if prefixed else ""))


def _held_turn(prefixed: bool = True) -> PipelineContext:
    call = _held_call(prefixed)
    return _ctx((_READ, call), user="pode enviar", draft="", held=(call,),
                declared={"notify_user": "message"}, cls="ACTION_REQUEST")


def _system(ctx) -> str:
    return SuperegoStage._judge_system(ctx)


def _user(ctx, limits: str = "") -> str:
    return SuperegoStage()._build_judge_prompt(ctx, limits)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ── the twins ─────────────────────────────────────────────────────────────────────────────

#: The host's switch, by its VALUE — so the twins run on the base tree (which has no such
#: metakey) and fail there by assertion. `test_the_switch_is_the_metakey` ties it to the name.
_SWITCH = "judge_stance_follows_branch"


def _switched(ctx, value=True):
    ctx.metadata[_SWITCH] = value
    return ctx


def test_the_switch_is_the_metakey():
    assert mk.JUDGE_STANCE_FOLLOWS_BRANCH == _SWITCH


@pytest.mark.parametrize("ctx", [_ctx((_READ,)), _ctx(conversational=True)],
                         ids=["readonly", "conversational"])
def test_the_branches_that_approve_by_default_are_not_told_the_opposite(ctx):
    """With the host's switch ON (`mk.JUDGE_STANCE_FOLLOWS_BRANCH`)."""
    _switched(ctx)
    system, user = _system(ctx), _user(ctx)
    assert "APPROVE BY DEFAULT" in user                    # the criteria are the same text
    assert _DEFAULT_NOT not in system
    assert "REPLY" in system and "approve it unless" in system
    assert "no evidence in the prompt supports" in system   # fabrication is still named


def test_the_answer_stance_carries_the_business_rules_like_the_other():
    system = _system(_switched(_ctx((_READ,), rules="- Abrimos às 8h.")))
    assert _DEFAULT_NOT not in system and "<business_rules>" in system


@pytest.mark.parametrize("prefixed", [True, False], ids=["by_name", "no_output"])
def test_a_held_call_renders_HELD_never_ERROR(prefixed):
    user = _user(_held_turn(prefixed))
    line = next(x for x in user.splitlines() if x.startswith("- notify_user("))
    assert "→ HELD for the user's confirmation" in line and "→ ERROR" not in line


def test_criterion_d_points_at_the_grounding_criterion_and_says_nothing_about_totals():
    """Sources only. A sentence about totals was measured net-negative on the production judge
    model (the wrong total passed MORE), so the DERIVED VALUES rule applies as before."""
    user = _user(_held_turn())
    assert "is not GROUNDED by the GROUNDING criterion above" in user
    assert "anything the request and the successful tool results above do not support" not in user
    d = user[user.index("(d) states"):user.index("The MID-FLOW and confirmation allowances")]
    assert "total" not in d and "sum" not in d


def test_the_hold_is_the_outcome_sentence_renders_on_a_held_message_turn():
    user = _user(_held_turn())
    assert "never reject because a held message was not sent" in user
    assert "is the CORRECT outcome of this turn" in user


# ── the controls ──────────────────────────────────────────────────────────────────────────

_EXECUTION = {
    "write": lambda: _ctx((_WRITE,), cls="ACTION_REQUEST", draft="Reservei a sala 4."),
    "failed_read": lambda: _ctx((_FAILED_READ,), draft="Não consegui ver o horário."),
    "failed_write": lambda: _ctx((_FAILED_WRITE,), cls="ACTION_REQUEST",
                                 draft="A sala 4 já está ocupada."),
    "read_and_failed_read": lambda: _ctx((_READ, _FAILED_READ)),
    "interrupted_read": lambda: _ctx((_READ,), interrupted=True),
    "no_call": lambda: _ctx(draft="Abre às 8h."),
    "write_with_rules": lambda: _ctx((_WRITE,), cls="ACTION_REQUEST", rules="- Sala 4 só às terças."),
    "write_with_context": lambda: _ctx((_WRITE,), cls="ACTION_REQUEST",
                                       context="[TODAY] 2026-11-10 (terça-feira)"),
}
_ANSWER = {
    "readonly": lambda: _ctx((_READ,)),
    "readonly_with_rules": lambda: _ctx((_READ,), rules="- Abrimos às 8h."),
    "conversational": lambda: _ctx(conversational=True),
}

# Taken on the base tree (`f9f10d7`), by rendering these fixtures there. The grounding entries
# were regenerated 2026-11-12 (figures privacy scrub): the worked example in
# `_DERIVED_FROM_EVIDENCE` moved, so every execution entry and the read-only answer entries moved;
# `conversational` (no grounding constant) and `_BASE_FAILED_LINE` did not. Control: reverting the
# new example to the old inside each rendering reproduced every value below, so nothing else moved.
_BASE_EXECUTION: "dict[tuple[str, str], str]" = {
    ('failed_read', ''): '9face61d74a47235',
    ('failed_read', 'Nunca prometa descontos.'): 'ca89a502036532ea',
    ('failed_write', ''): '71060021387f8eef',
    ('failed_write', 'Nunca prometa descontos.'): '309315488b83aadf',
    ('interrupted_read', ''): '9ead0d6343a0f04a',
    ('interrupted_read', 'Nunca prometa descontos.'): '483b0047ea416e92',
    ('no_call', ''): '44fdf80f42f642b7',
    ('no_call', 'Nunca prometa descontos.'): 'ae26c05e9ab63af3',
    ('read_and_failed_read', ''): '113a280e21f5f6c0',
    ('read_and_failed_read', 'Nunca prometa descontos.'): '63147f5771c75f34',
    ('write', ''): 'd5f554223b7c1dba',
    ('write', 'Nunca prometa descontos.'): '23787b9c4c5f1189',
    ('write_with_context', ''): '67fea29ca268c75f',
    ('write_with_context', 'Nunca prometa descontos.'): 'a7ba0f3a2d81b22b',
    ('write_with_rules', ''): 'bb70ca91caab0be1',
    ('write_with_rules', 'Nunca prometa descontos.'): '729e481a3544a502',
}
_BASE_ANSWER_USER: "dict[tuple[str, str], str]" = {
    ('conversational', ''): 'cd78af5cf0dfb76f',
    ('conversational', 'Nunca prometa descontos.'): 'a4b979bd8c22c772',
    ('readonly', ''): 'eb6bb4d6eb0ede36',
    ('readonly', 'Nunca prometa descontos.'): '1877542fe4fa5497',
    ('readonly_with_rules', ''): 'd7ee7b6acb6f19e7',
    ('readonly_with_rules', 'Nunca prometa descontos.'): 'd98edeb0c9133fec',
}
_BASE_ANSWER_WHOLE: "dict[str, str]" = {
    'conversational': 'c8fec463852b3cce',
    'readonly': '5fa4070eb71a38a4',
    'readonly_with_rules': '9f2fe7ef034c8605',
}
_BASE_FAILED_LINE = 'ce50c0defd4584f3'


@pytest.mark.parametrize("name", sorted(_EXECUTION))
@pytest.mark.parametrize("limits", ["", "Nunca prometa descontos."])
def test_the_execution_branch_is_the_base_tree_s_whole_request(name, limits):
    ctx = _EXECUTION[name]()
    assert SuperegoStage._judge_branch(ctx) == "execution"
    assert _sha(_system(ctx) + "\n\x00\n" + _user(ctx, limits)) == _BASE_EXECUTION[(name, limits)]


@pytest.mark.parametrize("name", sorted(_ANSWER))
@pytest.mark.parametrize("limits", ["", "Nunca prometa descontos."])
def test_the_answer_branches_moved_only_their_system_message(name, limits):
    assert _sha(_user(_ANSWER[name](), limits)) == _BASE_ANSWER_USER[(name, limits)]


@pytest.mark.parametrize("value", [None, False, 1, "true", "yes", [True]],
                         ids=["absent", "false", "one", "str_true", "yes", "list"])
@pytest.mark.parametrize("name", sorted(_ANSWER))
def test_without_the_switch_every_branch_is_told_what_it_was_told_before(name, value):
    """OFF by default, and only a JSON `True` switches it on: absent, `False` or a truthy
    non-boolean gives the read-only and conversational branches the execution stance — the
    system message of before — so their WHOLE request is the base tree's."""
    ctx = _ANSWER[name]()
    if value is not None:
        _switched(ctx, value)
    assert _DEFAULT_NOT in _system(ctx)
    assert _sha(_system(ctx) + "\n\x00\n" + _user(ctx)) == _BASE_ANSWER_WHOLE[name]


def test_a_call_that_ran_and_failed_still_renders_ERROR():
    user = _user(_EXECUTION["failed_write"]())
    line = next(x for x in user.splitlines() if x.startswith("- book_room("))
    assert "→ ERROR:" in line and _sha(line) == _BASE_FAILED_LINE


# ── the helper the older proofs hash through ───────────────────────────────────────────────

def test_the_helper_is_the_identity_where_nothing_changed_and_undoes_the_rest():
    from tests.unit._judge_0115 import as_0114
    for build in _EXECUTION.values():
        ctx = build()
        text = _system(ctx) + _user(ctx)
        assert as_0114(text) == text
    held = _user(_held_turn())
    assert as_0114(held) != held and "→ ERROR:" in as_0114(held)
    assert _DEFAULT_NOT in as_0114(_system(_switched(_ctx((_READ,)))))


def test_a_carrier_with_metadata_alone_gets_the_execution_stance_and_never_raises():
    """`_judge_system` read only the metadata until the stance followed the branch, and callers
    build it over a stand-in that carries nothing else (a downstream host's test double does).
    Such a carrier gets the bytes it always got. Green on the base; red on the first cut of this
    change, which raised `AttributeError` from the branch's read of `ego_result`."""
    from types import SimpleNamespace
    assert _DEFAULT_NOT in SuperegoStage._judge_system(SimpleNamespace(metadata={}))
    with_rules = SuperegoStage._judge_system(
        SimpleNamespace(metadata={mk.PERSONA_RULES: "- Abrimos às 8h."}))
    assert with_rules.startswith(SuperegoStage._judge_system(SimpleNamespace(metadata={})))
    assert "<business_rules>" in with_rules

