"""A figure reaches the contact from its evidence, or it does not reach them at all.

Measured on two production turns of the SAME conversation, minutes apart, same host revision
(``turn_traces`` 1779 and 1795 on the demo box, scope ``019da7d2…/800915080190``, host
``f251c29``; both rows verified un-rewritten by the ``xmin`` filter). A professor asked what he
earns. Both turns ended ``judge_rejected_all`` → ``last_draft_voiced``, so both rendered
``# Execution verdict (HARD RULE)`` and both had their draft withheld — leaving the reviewer's
critique as the only prose in the prompt that spoke about money.

* turn 3 — critique: *"Remova esse valor calculado; informe apenas R$ 120 por hora e o mínimo
  de 4 horas por aula."* Delivered: *"não tenho informações específicas sobre o valor que um
  professor recebe por hora"* — while ``- Aula - R$ 120,00 por hora`` sat in the tenant's own
  configured rules, inside this stage's system prompt.
* turn 4 — critique: *"…totalizando 20h = R$ 2.400,00…"*. The executor's draft said the
  opposite (*"permanece não determinado"*). Delivered: *"Totalizando 20h, o que resultaria em
  R$ 2.400,00."* ``2.400`` is in no tool result, no draft and no rule — only in the critique.

n=1 on each side; no regression is claimed and none is needed. The mechanism is one, and the
twins below are a PAIR on purpose: a fix that only stops the invented figure re-buys turn 3,
and a fix that only frees the verified one re-buys turn 4. That trade is literally what the
two turns show the system doing to itself.

Deterministic throughout: every assertion is on the RENDERED prompt or on a pure predicate.
The model half belongs to the integration suite.
"""

from __future__ import annotations

import pytest

from cogno_anima import metakeys as mk
from cogno_anima.stages.superego import (
    _CRITIQUE_IS_NOT_EVIDENCE, _FIGURES_HAVE_A_SOURCE, SuperegoStage,
)
from tests.unit.test_superego import _ctx

# The two measured critiques, trimmed to the sentence that carries the figure.
CRITIQUE_FORBIDDING = "Remova esse valor calculado; informe apenas R$ 120,00 por hora"
CRITIQUE_DEMANDING = "setembro inclui NoSQL (16h) e Workshop (4h), totalizando 20h = R$ 2.400,00"

# The tenant's own configured rules, as the host appends them to the VOICE slot.
PERSONA_RULES = ("Você é a Sofia.\n\n# Tenant-specific direction\n"
                 "- Aula - R$ 120,00 por hora, sendo o mínimo 4h por aula.")

LABEL = "Reviewer critique:"


def _rendered(ctx, *, reason: str = "", kind: str | None = None) -> str:
    if reason:
        carrier: dict = {"reason": reason}
        if kind is not None:
            carrier["kind"] = kind
        ctx.metadata[mk.VOICE_CORRECTION] = carrier
    return SuperegoStage()._build_voice_prompt(ctx, "get_professor_schedule: 08/09 · DE_09",
                                               ["general:review"])


# ── the pair ─────────────────────────────────────────────────────────

def test_the_verified_figure_is_allowed_to_reach_the_contact():
    """POSITIVE twin (turn 3). The rule must NAME the persona's own configured rules as a
    source, and must allow a total worked out from sources — on the condition that the reply
    shows the calculation.

    Both halves are load-bearing. Drop the first and the rate the tenant configured can never
    be told to the professor it is about, which is the loss that was measured. Drop the second
    and "what do I earn for a class?" has no answer either, because the answer is a product.
    """
    prompt = _rendered(_ctx())
    assert _FIGURES_HAVE_A_SOURCE in prompt
    assert "this persona's own configured rules and limits" in prompt, (
        "the voice was held to a NARROWER source set than review approves against — the "
        "two-doors defect `_GROUNDING_SOURCES` closed for the judge, still open for the voice")
    assert "allowed ONLY if you show the calculation" in prompt, (
        "a blanket ban on arithmetic answers 'how much per class?' with silence")
    assert "a bare total, or one whose inputs are not on this page, is not" in prompt


def test_the_critique_is_never_a_source_of_figures():
    """NEGATIVE twin (turn 4). The critique renders — the voice needs to know what was wrong —
    but it renders WITH the clause that says it is not evidence."""
    prompt = _rendered(_ctx(), reason=CRITIQUE_DEMANDING)
    assert CRITIQUE_DEMANDING in prompt, "the voice still has to be told what was rejected"
    assert _CRITIQUE_IS_NOT_EVIDENCE in prompt
    # ...and the clause sits with the critique, not paragraphs away in another block.
    assert prompt.index(_CRITIQUE_IS_NOT_EVIDENCE) - prompt.index(CRITIQUE_DEMANDING) \
        < len(CRITIQUE_DEMANDING) + 8


# ── the pairing is a mechanism, not a habit ──────────────────────────

@pytest.mark.parametrize("kind", [None, "unverified_claim"])
def test_every_rendered_critique_carries_the_clause(kind):
    """Counted, not spot-checked: a fourth rejection variant that prints the reviewer's words
    without the clause is the measured defect wearing a new label."""
    prompt = _rendered(_ctx(), reason=CRITIQUE_DEMANDING, kind=kind)
    assert prompt.count(LABEL) == 1
    assert prompt.count(LABEL) == prompt.count(_CRITIQUE_IS_NOT_EVIDENCE)


def test_the_anti_repeat_guard_is_not_a_critique_and_carries_no_clause():
    """``repeated_reply`` is the host's anti-repeat guard: the content was fine, it had already
    been sent. It prints no ``Reviewer critique:`` label and nothing was rejected as unfounded,
    so the clause would be a statement about a critique that is not there."""
    prompt = _rendered(_ctx(), reason="já enviada", kind="repeated_reply")
    assert LABEL not in prompt
    assert _CRITIQUE_IS_NOT_EVIDENCE not in prompt


def test_a_turn_with_no_rejection_carries_no_clause():
    prompt = _rendered(_ctx())
    assert _CRITIQUE_IS_NOT_EVIDENCE not in prompt
    assert _FIGURES_HAVE_A_SOURCE in prompt      # the figures rule is unconditional


# ── the counted half ─────────────────────────────────────────────────

def _flag(reply: str, *, reason: str, payload: str = "(no execution)",
          system: str = "Você é a Sofia.", context: str = "") -> "list[str]":
    ctx = _ctx(user="qual seria o q tenho a receber?")
    ctx.metadata[mk.VOICE_CORRECTION] = {"reason": reason}
    if context:
        ctx.metadata[mk.EGO_CONTEXT] = context
    return SuperegoStage._figures_from_the_critique(ctx, reply, payload, system)


def test_the_backstop_counts_the_measured_leak():
    """Turn 4, byte for byte: the figure is in the critique and nowhere else the voice held."""
    assert _flag("Totalizando 20h, o que resultaria em R$ 2.400,00.",
                 reason=CRITIQUE_DEMANDING) == ["240000"]


def test_a_figure_the_persona_rules_state_is_not_a_leak():
    """The muzzle guard, and turn 3's other half: the critique names R$ 120,00 and so do the
    tenant's rules. Saying it back is the CORRECT answer, and a net that counted it would
    argue for deleting the answer again."""
    assert _flag("O valor é R$ 120,00 por hora.", reason=CRITIQUE_FORBIDDING,
                 system=PERSONA_RULES) == []


def test_a_figure_the_tool_data_states_is_not_a_leak():
    assert _flag("Foram R$ 2.400,00 no período.", reason=CRITIQUE_DEMANDING,
                 payload="get_summary: Income R$ 2.400,00") == []


def test_a_figure_the_contact_typed_is_not_a_leak():
    ctx = _ctx(user="paguei R$ 2.400,00 no mês passado")
    ctx.metadata[mk.VOICE_CORRECTION] = {"reason": CRITIQUE_DEMANDING}
    assert SuperegoStage._figures_from_the_critique(
        ctx, "Confirmando: R$ 2.400,00.", "(no execution)", "Você é a Sofia.") == []


def test_the_context_block_is_evidence_the_voice_could_see():
    """Not a source the RULE admits — but the question this predicate answers is narrower and
    must stay honest: *did this figure come from the critique?* A figure the context also
    carries did not, whatever else is wrong with quoting it. ``_draft_divergence`` is the net
    that owns the context-as-source defect, and two nets claiming one finding would double-count
    it."""
    assert _flag("R$ 2.400,00.", reason=CRITIQUE_DEMANDING,
                 context="[MEMORY] o professor recebeu R$ 2.400,00 em agosto") == []


def test_no_rejection_means_nothing_to_count():
    ctx = _ctx()
    assert SuperegoStage._figures_from_the_critique(
        ctx, "R$ 2.400,00", "(no execution)", "s") == []


def test_a_critique_with_no_figure_costs_nothing():
    assert _flag("R$ 2.400,00", reason="não respondeu à pergunta") == []


def test_the_backstop_only_flags_and_never_masks():
    """Flag-only by construction: it fires on a turn the judge already rejected and whose
    correction budget the orchestrator has already spent. A re-voice there would be this stage
    arguing with an exhaustion path it does not own — the prompt clause is the prevention."""
    reply = "Totalizando 20h, o que resultaria em R$ 2.400,00."
    ctx = _ctx()
    ctx.metadata[mk.VOICE_CORRECTION] = {"reason": CRITIQUE_DEMANDING}
    flagged = SuperegoStage._figures_from_the_critique(ctx, reply, "(no execution)", "s")
    assert flagged and reply == "Totalizando 20h, o que resultaria em R$ 2.400,00."
