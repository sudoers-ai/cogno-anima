"""
Phase 1 contract tests for the ID stage (Stage 3) additions:

  - IdResult is a validated pydantic model with the expected fields/defaults;
  - PipelineContext wires `id_result` into id_metrics / stage_metrics / totals;
  - the ID closed vocabularies (goal status, complexity) exist in cogno_core.vocab.

No behavior is exercised yet (the IDStage lands in a later phase) — this only
locks the data contract so the rest of the build can depend on it.
"""

import pytest

from cogno_core import IdResult, StageMetrics, PipelineContext
from cogno_core import vocab


def _metrics(stage: str = "id", **kw) -> StageMetrics:
    base = dict(stage=stage, elapsed_ms=1.0, tokens_in=0, tokens_out=0, model="heuristic")
    base.update(kw)
    return StageMetrics(**base)


# ──────────────────────────────────────────────────────────────────────
#  IdResult model
# ──────────────────────────────────────────────────────────────────────

def test_id_result_minimal_construction_and_defaults():
    res = IdResult(triad_route="EGO", metrics=_metrics())
    assert res.triad_route == "EGO"
    assert res.goal_status == "NEW"
    assert res.goal_similarity == 1.0
    assert res.active_goal is None
    assert res.active_intentions == []
    assert res.attention_focus == []
    assert res.blocked is False
    assert res.block_reason is None
    assert res.turn_number == 1
    assert res.temporal_class is None
    assert res.emotional_override is None
    assert res.complexity == "LOW"


def test_id_result_requires_triad_route_and_metrics():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        IdResult(metrics=_metrics())          # missing triad_route
    with pytest.raises(ValidationError):
        IdResult(triad_route="EGO")           # missing metrics


def test_id_result_default_lists_are_independent():
    """Field(default_factory=list) must not share a mutable instance across instances."""
    a = IdResult(triad_route="EGO", metrics=_metrics())
    b = IdResult(triad_route="EGO", metrics=_metrics())
    a.active_intentions.append("x")
    a.attention_focus.append("y")
    assert b.active_intentions == []
    assert b.attention_focus == []


def test_id_result_carries_embedding_telemetry():
    res = IdResult(
        triad_route="EGO",
        metrics=_metrics(embedding_tokens=48, embedding_calls=2),
    )
    assert res.metrics.embedding_tokens == 48
    assert res.metrics.embedding_calls == 2
    # ID has no LLM call → token cost is purely the embedding cost.
    assert res.metrics.tokens_total == 48


# ──────────────────────────────────────────────────────────────────────
#  PipelineContext wiring
# ──────────────────────────────────────────────────────────────────────

def test_pipeline_context_id_result_defaults_none():
    ctx = PipelineContext(user_input="hi")
    assert ctx.id_result is None
    assert ctx.id_metrics is None


def test_id_metrics_and_stage_metrics_include_id():
    ctx = PipelineContext(user_input="hi")
    ctx.id_result = IdResult(
        triad_route="EGO",
        metrics=_metrics(embedding_tokens=10, embedding_calls=2),
    )
    assert ctx.id_metrics is ctx.id_result.metrics
    assert ctx.id_result.metrics in ctx.stage_metrics
    # totals fold the ID embedding cost in.
    assert ctx.total_tokens == 10
    assert ctx.total_embedding_tokens == 10
    assert ctx.total_llm_tokens == 0


# ──────────────────────────────────────────────────────────────────────
#  Vocab (single source for the heuristic ID stage)
# ──────────────────────────────────────────────────────────────────────

def test_id_vocab_values():
    assert vocab.VALID_GOAL_STATUS == {"NEW", "ONGOING", "COMPLETED", "ABANDONED"}
    assert vocab.VALID_COMPLEXITY == {"LOW", "MEDIUM", "HIGH", "EXPERT"}


def test_id_vocab_uses_code_truth_not_stale_doc_terms():
    """architecture.md used IN_PROGRESS/ACHIEVED; the code truth is ONGOING/COMPLETED."""
    assert "IN_PROGRESS" not in vocab.VALID_GOAL_STATUS
    assert "ACHIEVED" not in vocab.VALID_GOAL_STATUS
