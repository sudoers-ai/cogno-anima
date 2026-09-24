"""A value the business DECLARED in the persona's configuration is KNOWN to the voice.

The shape (values invented; no tenant's): a persona's rules carry the tenant's fixed values — a
rent, an hourly rate, a late fee — and the persona answers a question by quoting them, with no
tool, because none is needed. The judge already had the rules in `# Persona limits` and was told
in prose that they ground a statement; it ALSO had the persona's own factory limit saying
financial data comes only from a tool call, and a fail-CLOSED judge resolved the two against the
rules. The correction budget ran out, and the exhausted voice told the contact the value was not
available.

The host now declares the VALUES (`mk.PERSONA_DECLARED_VALUES`) for the readers that do not read
prose; the JUDGE gets the rules whole instead (`test_persona_rules_reach_the_judge.py`). This
file pins what the voice does with the values:

* the voice is told, in the `# Task` beside the source rule, that they are known — ONLY when
  there are any, and without a new header (the persisted inventory does not move);
* the voice's deterministic figure net does not re-voice a reply for stating one;
* and the text the model reads says the list grounds a VALUE, never an ACT, and that a value
  NOT on it — or COMPUTED from it — is judged as before.
"""

from __future__ import annotations

import hashlib

import pytest

from cogno_anima import metakeys as mk
from cogno_anima.stages.superego import (MAX_DECLARED_VALUES, SuperegoStage,
                                         _declared_voice_clause)
from cogno_anima.types import EgoResult, EgoStep, ToolExecution
from tests.unit.test_superego import _ctx, _m

VALUES = ["R$ 10,00", "R$ 120,00", "4 h", "2%"]
LIMITS = ("Every financial data operation went through a tool call.\n\n"
          "# Tenant rules (legitimate grounding)\nAluguel da sala: R$ 10,00 por dia.")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _read_ctx(declared=None, *, readonly=True):
    ctx = _ctx(user="quanto é o aluguel da sala?", intent_class="INFORMATION_REQUEST",
               goal="know the room rent", with_ego=False)
    ctx.ego_result = EgoResult(steps=[
        EgoStep(index=0, path="native", assistant_text="", tool_calls=[
            ToolExecution(tool="resolve_date", arguments={}, result="2026-10-05", ok=True,
                          side_effect=not readonly, tool_mutating=not readonly)]),
        EgoStep(index=1, path="native", assistant_text="O aluguel da sala é R$ 10,00 por dia.")],
        metrics=_m("ego"))
    if declared is not None:
        ctx.metadata[mk.PERSONA_DECLARED_VALUES] = declared
    return ctx


def _judge(ctx) -> str:
    return SuperegoStage()._build_judge_prompt(ctx, LIMITS)


def _voice(ctx) -> str:
    return SuperegoStage()._build_voice_prompt(ctx, payload="resolve_date: 2026-10-05",
                                               adjustments=[])


# ── the voice ─────────────────────────────────────────────────────────────────────────────
def test_the_voice_is_told_the_declared_values_are_known():
    prompt = _voice(_read_ctx(VALUES))
    task = SuperegoStage.voice_prompt_block(prompt, "task")
    assert "VALUES DECLARED IN THIS PERSONA'S CONFIGURATION: R$ 10,00; R$ 120,00; 4 h; 2%." in task
    assert "never tell the contact you do not have it" in task
    assert "never an action" in task
    assert task.rstrip().endswith("Reply with the message text only.")


def test_the_voice_clause_rides_the_exhausted_turn_too():
    """The measured shape: the correction budget ran out and the voice denied the value."""
    ctx = _read_ctx(VALUES)
    ctx.metadata[mk.VOICE_CORRECTION] = {"kind": "not_executed",
                                         "reason": "valor citado sem chamada de ferramenta"}
    prompt = _voice(ctx)
    assert "# Execution verdict (HARD RULE)" in prompt
    assert "VALUES DECLARED IN THIS PERSONA'S CONFIGURATION" in (
        SuperegoStage.voice_prompt_block(prompt, "task"))


@pytest.mark.parametrize("empty", [None, [], "R$ 10,00", [None, 3, ""]])
def test_nothing_declared_is_the_voice_prompt_byte_for_byte(empty):
    assert _sha(_voice(_read_ctx(empty))) == _sha(_voice(_read_ctx()))


def test_the_voice_inventory_does_not_move():
    with_values = [r["block"] for r in SuperegoStage.voice_prompt_inventory(_voice(_read_ctx(VALUES)))]
    without = [r["block"] for r in SuperegoStage.voice_prompt_inventory(_voice(_read_ctx()))]
    assert with_values == without


def test_the_voice_clause_is_empty_without_values():
    assert _declared_voice_clause(()) == ""


# ── the voice's figure net ────────────────────────────────────────────────────────────────
def test_a_declared_value_is_not_an_invented_figure_in_the_voice_net():
    draft = "O aluguel da sala é por dia."
    reply = "O aluguel da sala é R$ 10,00 por dia."
    _, invented = SuperegoStage._draft_divergence(draft, "resolve_date: 2026-10-05", reply)
    assert invented, "the control: with nothing declared the net calls this figure invented"
    _, invented = SuperegoStage._draft_divergence(draft, "resolve_date: 2026-10-05", reply,
                                                  "", VALUES)
    assert invented == []


@pytest.mark.parametrize("reply", [
    "O aluguel da sala fica em R$ 300,00 por mês.",                 # 30 x R$ 10,00, bare
    "Por mês: 30 x R$ 10,00 = R$ 300,00.",                          # the same, work shown
])
def test_the_voice_does_not_sum_or_derive_from_declared_values(reply):
    """Declared «R$ 10,00 por dia»; the reply says «R$ 300,00 por mês». The declaration grounds
    the value WRITTEN, not one worked out from it: the monthly figure is written nowhere, so the
    figure net calls it invented — bare, and even with the work shown (the operands a shown
    derivation may use are the evidence on the page: the draft, the tool data, the contact's
    words; the declared list admits a VALUE, not an operand)."""
    # BOTH operands declared ("R$ 10,00" and "30 dias"), so the only thing between the shown
    # work and an admitted figure is that a declared value is not an operand.
    _, invented = SuperegoStage._draft_divergence(
        "O aluguel da sala é por dia.", "resolve_date: 2026-10-05", reply, "",
        ["R$ 10,00", "30 dias"])
    assert "30000" in invented
    # …and the declared value itself, beside it, is not what the net objects to
    assert "1000" not in invented


def test_the_task_carries_exactly_the_values_the_host_declared_and_nothing_else():
    """The core cannot tell whose values these are — the HOST resolves them for THIS persona and
    THIS role tab (and pins that another persona's never enter). What the core guarantees is that
    the Task lists exactly what it was handed: a value that was not handed over is not there."""
    task = SuperegoStage.voice_prompt_block(_voice(_read_ctx(["R$ 10,00", "2%"])), "task")
    assert "VALUES DECLARED IN THIS PERSONA'S CONFIGURATION: R$ 10,00; 2%." in task
    assert "R$ 999,00" not in task
    assert "VALUES DECLARED" not in SuperegoStage.voice_prompt_block(_voice(_read_ctx()), "task")


def test_an_undeclared_figure_is_still_invented_in_the_voice_net():
    _, invented = SuperegoStage._draft_divergence(
        "O aluguel é por dia.", "", "O aluguel é R$ 12,00 por dia.", "", VALUES)
    assert invented


# ── the carrier is sanitized, never trusted ───────────────────────────────────────────────
def test_the_carrier_is_sanitized():
    ctx = _read_ctx(["R$ 10,00", "R$ 10,00", "  R$\n120,00 ", 7, "", "# EGO draft",
                     "x" * 200, "2%"])
    assert SuperegoStage._declared_values(ctx) == ("R$ 10,00", "R$ 120,00", "2%")


def test_the_carrier_is_bounded():
    ctx = _read_ctx([f"R$ {n},00" for n in range(MAX_DECLARED_VALUES + 30)])
    assert len(SuperegoStage._declared_values(ctx)) == MAX_DECLARED_VALUES


def test_the_values_never_reach_the_judge():
    """The judge reads the RULES themselves (`mk.PERSONA_RULES`), never this list: a list of
    figures would tell a prose reader less than the text it was extracted from."""
    ctx = _read_ctx(["R$ 37,50", "13%"])
    asked = SuperegoStage._judge_system(ctx) + _judge(ctx)
    assert "37,50" not in asked and "13%" not in asked
    # the control: the same values DO reach the voice
    assert "R$ 37,50" in _voice(ctx)
