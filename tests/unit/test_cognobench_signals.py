"""
Unit tests (no Ollama) for the CognoBench wiring of previously-dropped NER
signals + the act_confirm read-only / confirmation gates: the harness injects
constraints/negation/parole onto the hand-built IntentResult and flips
ego_readonly, and the new cases exist with the scoring intent that validates the
stage behavior end-to-end.
"""

from cognobench.dimensions import _ego_ctx, _superego_ctx
from cognobench.ego_cases import EGO_CASES, EgoCase
from cognobench.superego_cases import SUPEREGO_CASES, SuperegoCase


# ── harness wires the new signals into the context ───────────────────

def test_ego_ctx_sets_readonly_metadata():
    ctx = _ego_ctx(EgoCase("c", "d", "maybe record 50?", readonly=True))
    assert ctx.metadata.get("ego_readonly") is True


def test_ego_ctx_no_readonly_by_default():
    ctx = _ego_ctx(EgoCase("c", "d", "Record 50 for lunch."))
    assert "ego_readonly" not in ctx.metadata


def test_superego_ctx_propagates_constraints_negation_parole():
    case = SuperegoCase("c", "judge", "registra 50, mas não categorize",
                        goal="record 50", negation=["do not categorize"],
                        constraints=["only this month"], parole="ACADEMICO")
    ctx = _superego_ctx(case)
    assert ctx.intent.negation == ["do not categorize"]
    assert ctx.intent.constraints == ["only this month"]
    assert ctx.intent.parole == "ACADEMICO"


# ── the new cases exist with the right scoring intent ───────────────

def test_ego_has_readonly_and_destructive_cases():
    ro = next(c for c in EGO_CASES if c.id == "readonly_propose")
    assert ro.readonly is True and ro.expect_no_mutation is True
    dz = next(c for c in EGO_CASES if c.id == "destructive_needs_confirmation")
    assert dz.expect_pending == "delete_all_records" and dz.expect_no_mutation is True


def test_judge_negation_case_expects_rejection():
    viol = next(c for c in SUPEREGO_CASES if c.id == "judge_violates_negation")
    assert viol.kind == "judge" and viol.negation
    assert viol.expect_approved is False     # cognobench must REACT to negation

    honored = next(c for c in SUPEREGO_CASES if c.id == "judge_honors_constraint")
    assert honored.constraints and honored.expect_approved is True


def test_voice_register_case_present():
    reg = next(c for c in SUPEREGO_CASES if c.id == "voice_academic_register")
    assert reg.kind == "voice" and reg.parole == "ACADEMICO"
    assert reg.expect_contains                # still grounded (no regression)
