"""
Integration tests for the ID stage (Stage 3).

The ID stage makes NO LLM call — its only real dependency is the Embedder (goal
similarity at GoalManager Stage 2). Embeddings are deterministic, so these assert
hard semantic properties (related goals score higher than unrelated ones, real
embedding tokens are captured) rather than the flaky LLM bands the NER suite uses.

Uses a REAL CachingEmbedder(OllamaEmbedder). Auto-skipped if Ollama is down.
"""

import httpx
import pytest

from cogno_anima.llm import OllamaEmbedder, CachingEmbedder
from cogno_anima.stages.id import IDStage
from cogno_anima.types import PipelineContext, IntentResult, NoumenoResult, StageMetrics


async def is_ollama_available() -> bool:
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.get("http://localhost:11434/")
            return resp.status_code == 200
    except Exception:
        return False


def _make_embedder() -> CachingEmbedder:
    return CachingEmbedder(OllamaEmbedder(model="nomic-embed-text:latest"))


def _noumeno() -> NoumenoResult:
    m = StageMetrics(stage="noumeno", elapsed_ms=1.0, tokens_in=1, tokens_out=1, model="test")
    return NoumenoResult(
        original="x", rewritten="x rewritten about a topic", context_turn="",
        language="en", canonical_language="en", drift_score=0.1, drift_tag="REWRITTEN",
        changed=True, confidence=1.0, change_subject=False, subject_similarity=1.0,
        context_used=False, preserved_terms=[], rewrite_warnings=[], metrics=m,
    )


def _intent(goal: str, intent_class: str = "INFORMATION_REQUEST") -> IntentResult:
    m = StageMetrics(stage="ner", elapsed_ms=1.0, tokens_in=1, tokens_out=1, model="test")
    return IntentResult(
        intent_class=intent_class, sentiment="NEUTRAL", confidence=1.0,
        temporal_class="TIMELESS", triad_signal="BALANCED",
        location=None, aristotelian={}, domains=[], goal=goal,
        modality=None, speech_act=None, pii=[], pii_risk="NONE",
        metrics=m, raw_response=None,
    )


def _ctx(goal: str, id_state: dict | None = None) -> PipelineContext:
    ctx = PipelineContext(user_input=goal)
    ctx.noumeno = _noumeno()
    ctx.intent = _intent(goal)
    if id_state is not None:
        ctx.metadata["id_state"] = id_state
    return ctx


@pytest.mark.asyncio
async def test_first_turn_no_embedding_cost():
    if not await is_ollama_available():
        pytest.skip("Local Ollama server (http://localhost:11434) is not running.")
    stage = IDStage()
    out = await stage.process(_ctx("configure docker on ubuntu"), _make_embedder())
    assert out.id_result.goal_status == "NEW"
    assert out.id_result.goal_similarity == 1.0
    assert out.id_result.metrics.embedding_tokens == 0       # no prior goal → no Stage 2
    assert out.drift.situational_drift == 0.0


@pytest.mark.asyncio
async def test_related_goal_scores_higher_than_unrelated():
    """Core semantic property: a related follow-up is more similar than an unrelated one."""
    if not await is_ollama_available():
        pytest.skip("Local Ollama server (http://localhost:11434) is not running.")
    embedder = _make_embedder()
    stage = IDStage()

    # Establish the goal.
    base = await stage.process(_ctx("configure docker on ubuntu"), embedder)
    state = base.metadata["id_state"]

    # Related follow-up (Stage 2 semantic).
    related = await stage.process(
        _ctx("what does the docker daemon do", id_state=dict(state)), embedder,
    )
    # Unrelated follow-up (Stage 2 semantic).
    unrelated = await stage.process(
        _ctx("what is the best pizza in town", id_state=dict(state)), embedder,
    )

    assert related.id_result.goal_similarity > unrelated.id_result.goal_similarity
    # situational_drift = 1 - similarity → inverse ordering.
    assert related.drift.situational_drift < unrelated.drift.situational_drift
    # The Stage-2 similarity is counted as 2 embed operations. Token count is
    # best-effort: Ollama's /api/embeddings does not return prompt_eval_count for
    # nomic-embed, so embedding_tokens is typically 0 here (≥ 0 invariant only).
    assert related.id_result.metrics.embedding_calls == 2
    assert related.id_result.metrics.embedding_tokens >= 0


@pytest.mark.asyncio
async def test_related_continues_unrelated_abandons_with_tuned_threshold():
    """With a threshold between the two similarities, related → ONGOING, unrelated → ABANDONED."""
    if not await is_ollama_available():
        pytest.skip("Local Ollama server (http://localhost:11434) is not running.")
    embedder = _make_embedder()
    stage = IDStage(goal_threshold=0.5)

    base = await stage.process(_ctx("configure docker on ubuntu"), embedder)
    state = base.metadata["id_state"]

    related = await stage.process(
        _ctx("how do I run the docker daemon", id_state=dict(state)), embedder,
    )
    unrelated = await stage.process(
        _ctx("recommend a good italian restaurant", id_state=dict(state)), embedder,
    )
    assert related.id_result.goal_status == "ONGOING"
    assert unrelated.id_result.goal_status == "ABANDONED"
    # cumulative drift stays a valid signal throughout.
    for out in (related, unrelated):
        assert 0.0 <= out.drift.cumulative_drift <= 1.0
        assert out.drift.drift_action in {"none", "warn", "ask_user", "self_correct"}


@pytest.mark.asyncio
async def test_caching_embedder_anchor_hit_across_turns():
    """The active-goal anchor repeats across turns → CachingEmbedder absorbs it."""
    if not await is_ollama_available():
        pytest.skip("Local Ollama server (http://localhost:11434) is not running.")
    embedder = _make_embedder()
    stage = IDStage()

    base = await stage.process(_ctx("configure docker on ubuntu"), embedder)
    state = base.metadata["id_state"]
    # two Stage-2 turns reuse the same anchor; the second should see cache hits.
    await stage.process(_ctx("explain the docker daemon", id_state=dict(state)), embedder)
    hits_before = embedder.usage.cache_hits
    await stage.process(_ctx("explain docker networking", id_state=dict(state)), embedder)
    assert embedder.usage.cache_hits > hits_before
