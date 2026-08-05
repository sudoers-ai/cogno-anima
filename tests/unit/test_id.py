"""Unit tests for cogno_anima.stages.id.IDStage (heuristic + embedder, async)."""

import pytest

from cogno_anima.types import PipelineContext
from cogno_anima.stages.id import IDStage
from cogno_anima.stages.drift import DriftCalculator, DriftThresholds
from tests.unit.test_drift import make_intent_result, make_noumeno_result


class UsageEmbedder:
    """Usage-aware embedder double: fixed similarity, fixed tokens per call."""
    model = "fake"

    def __init__(self, sim: float = 0.9, tokens: int = 24) -> None:
        self._sim = sim
        self._tokens = tokens

    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]

    async def similarity(self, a: str, b: str) -> float:
        return self._sim

    async def similarity_with_usage(self, a: str, b: str) -> tuple[float, int]:
        return self._sim, self._tokens


class PlainEmbedder:
    """Embedder without similarity_with_usage (exercises the getattr fallback)."""
    model = "plain"

    def __init__(self, sim: float = 0.9) -> None:
        self._sim = sim

    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]

    async def similarity(self, a: str, b: str) -> float:
        return self._sim


def make_ctx(intent=None, **meta) -> PipelineContext:
    ctx = PipelineContext(user_input="x")
    ctx.noumeno = make_noumeno_result("orig", "rewritten text about things")
    ctx.intent = intent or make_intent_result()
    ctx.metadata.update(meta)
    return ctx


def _intent(**over):
    """Build an IntentResult and apply overrides (model is mutable)."""
    base = make_intent_result(
        intent_class=over.pop("intent_class", "UNKNOWN"),
        sentiment=over.pop("sentiment", "NEUTRAL"),
        temporal_class=over.pop("temporal_class", "TIMELESS"),
        goal=over.pop("goal", None),
    )
    for k, v in over.items():
        setattr(base, k, v)
    return base


# ── Guards ────────────────────────────────────────────────────────────────

async def test_requires_noumeno_and_intent():
    stage = IDStage()
    ctx = PipelineContext(user_input="x")    # nothing populated
    with pytest.raises(ValueError, match="NOUMENO and NER"):
        await stage.process(ctx, PlainEmbedder())


# ── Routing table ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("intent_kw,expected,blocked", [
    (dict(pii_risk="CRITICAL"), "SUPEREGO", True),
    (dict(pii_risk="HIGH"), "SUPEREGO", False),
    (dict(intent_class="CREATIVE_TASK"), "SUPEREGO", False),
    (dict(intent_class="ACTION_REQUEST", mandatory_tags=["NER.SYSTEM"]), "EGO", False),
    (dict(intent_class="SOCIAL"), "SUPEREGO", False),
    (dict(triad_signal="EGO"), "EGO", False),
    (dict(triad_signal="BOGUS"), "BALANCED", False),
])
async def test_routing_table(intent_kw, expected, blocked):
    stage = IDStage()
    ctx = make_ctx(_intent(**intent_kw))
    out = await stage.process(ctx, PlainEmbedder())
    assert out.id_result.triad_route == expected
    assert out.id_result.blocked is blocked
    if blocked:
        assert "CRITICAL" in out.id_result.block_reason


@pytest.mark.parametrize("kw,expected", [
    (dict(intent_class="ACTION_REQUEST", speech_act="INTERROGATIVE"), True),
    (dict(intent_class="ACTION_REQUEST", modality="UNCERTAIN"), True),
    (dict(intent_class="ACTION_REQUEST", modality="POSSIBLE"), True),
    (dict(intent_class="ACTION_REQUEST", speech_act="DIRECTIVE", modality="CERTAIN"), False),
    # info requests never flag, even when tentative (answering is not committing)
    (dict(intent_class="INFORMATION_REQUEST", speech_act="INTERROGATIVE"), False),
    (dict(intent_class="ACTION_REQUEST"), False),
])
async def test_needs_confirmation_signal(kw, expected):
    """ID flags a tentatively-framed ACTION — SIGNAL only, never forces the EGO."""
    stage = IDStage()
    ctx = make_ctx(_intent(**kw))
    out = await stage.process(ctx, PlainEmbedder())
    assert out.id_result.needs_confirmation is expected
    # the ID must NOT touch the EGO's read-only switch — that's the host's call
    assert "ego_readonly" not in ctx.metadata


# ── Cross-stage doubt signals (2R-C / 2R-D) ────────────────────────────────

@pytest.mark.parametrize("noumeno_conf,ner_conf,expected", [
    (0.9, 0.4, True),    # rewrite looked clean but intent murky → diverge
    (0.4, 0.9, True),    # the inverse also diverges
    (0.9, 0.85, False),  # both confident → no signal
    (0.5, 0.5, False),   # equal (even if low) → no DISAGREEMENT
])
async def test_confidence_divergence_signal(noumeno_conf, ner_conf, expected):
    """The ID flags DISAGREEMENT between NOUMENO/NER confidence (not the absolute
    value — the core distrusts that). SIGNAL only."""
    stage = IDStage(confidence_divergence_threshold=0.4)
    ctx = make_ctx(_intent())
    ctx.noumeno.confidence = noumeno_conf
    ctx.intent.confidence = ner_conf
    out = await stage.process(ctx, PlainEmbedder())
    assert out.id_result.confidence_divergence is expected


@pytest.mark.parametrize("warnings,expected", [
    (["ambiguous referent: 'ele' unresolved"], True),
    ([], False),
])
async def test_clarification_suggested_signal(warnings, expected):
    """rewrite_warnings (rewriter flagged ambiguity/loss) → clarification signal."""
    stage = IDStage()
    ctx = make_ctx(_intent())
    ctx.noumeno.rewrite_warnings = warnings
    out = await stage.process(ctx, PlainEmbedder())
    assert out.id_result.clarification_suggested is expected


async def test_emotional_override_forces_superego():
    stage = IDStage(frustration_threshold=1)
    ctx = make_ctx(_intent(sentiment="FRUSTRATED", triad_signal="EGO"))
    out = await stage.process(ctx, PlainEmbedder())
    assert out.id_result.emotional_override == "sustained_frustration"
    assert out.id_result.triad_route == "SUPEREGO"


# ── Frustration streak ──────────────────────────────────────────────────────

async def test_frustration_streak_accumulates_and_resets():
    stage = IDStage(frustration_threshold=2)
    # turn 1: FRUSTRATED, streak=1 (< 2) → no override
    ctx1 = make_ctx(_intent(sentiment="FRUSTRATED"))
    out1 = await stage.process(ctx1, PlainEmbedder())
    assert out1.id_result.emotional_override is None

    # turn 2: FRUSTRATED again, streak=2 → override
    ctx2 = make_ctx(_intent(sentiment="FRUSTRATED"))
    ctx2.metadata["id_state"] = ctx1.metadata["id_state"]
    out2 = await stage.process(ctx2, PlainEmbedder())
    assert out2.id_result.emotional_override == "sustained_frustration"

    # turn 3: NEUTRAL → streak resets, no override
    ctx3 = make_ctx(_intent(sentiment="NEUTRAL"))
    ctx3.metadata["id_state"] = ctx2.metadata["id_state"]
    out3 = await stage.process(ctx3, PlainEmbedder())
    assert out3.id_result.emotional_override is None
    assert out3.metadata["id_state"]["frustration_streak"] == 0


async def test_host_can_inject_emotional_override():
    stage = IDStage(frustration_threshold=99)   # never triggers internally
    ctx = make_ctx(_intent(sentiment="NEUTRAL"), emotional_override="host_says_so")
    out = await stage.process(ctx, PlainEmbedder())
    assert out.id_result.emotional_override == "host_says_so"
    assert out.id_result.triad_route == "SUPEREGO"


# ── Turn number ─────────────────────────────────────────────────────────────

async def test_turn_number_host_authoritative():
    stage = IDStage()
    ctx = make_ctx(turn_number=7)
    out = await stage.process(ctx, PlainEmbedder())
    assert out.id_result.turn_number == 7


async def test_turn_number_auto_increments_when_absent():
    stage = IDStage()
    ctx1 = make_ctx()
    out1 = await stage.process(ctx1, PlainEmbedder())
    assert out1.id_result.turn_number == 1
    ctx2 = make_ctx()
    ctx2.metadata["id_state"] = ctx1.metadata["id_state"]
    out2 = await stage.process(ctx2, PlainEmbedder())
    assert out2.id_result.turn_number == 2


# ── Temporal stickiness (must not mutate the NER) ───────────────────────────

async def test_temporal_stickiness_records_on_id_result_not_intent():
    stage = IDStage()
    # turn 1: RECENT, establishes goal in TECH
    ctx1 = make_ctx(_intent(goal="configure docker", intent_class="ACTION_REQUEST",
                            temporal_class="RECENT", domains=["TECH"]))
    out1 = await stage.process(ctx1, PlainEmbedder())
    assert out1.id_result.temporal_class == "RECENT"

    # turn 2: NER says TIMELESS (isolated), but same TECH domain → ONGOING → sticky RECENT
    ctx2 = make_ctx(_intent(goal="fix it", intent_class="ACTION_REQUEST",
                            temporal_class="TIMELESS", domains=["TECH"]))
    ctx2.metadata["id_state"] = ctx1.metadata["id_state"]
    out2 = await stage.process(ctx2, PlainEmbedder())
    assert out2.id_result.goal_status == "ONGOING"
    assert out2.id_result.temporal_class == "RECENT"          # sticky
    assert ctx2.intent.temporal_class == "TIMELESS"           # NER NOT mutated


# ── Embedding token capture ─────────────────────────────────────────────────

async def test_embedding_tokens_captured_at_stage2():
    stage = IDStage()
    # turn 1: first turn, no embedding
    ctx1 = make_ctx(_intent(goal="configure docker", intent_class="ACTION_REQUEST", domains=[]))
    out1 = await stage.process(ctx1, UsageEmbedder(sim=0.9, tokens=24))
    assert out1.id_result.metrics.embedding_tokens == 0

    # turn 2: Stage 2 semantic → embedding runs
    ctx2 = make_ctx(_intent(goal="what does daemon mean", intent_class="INFORMATION_REQUEST", domains=[]))
    ctx2.metadata["id_state"] = ctx1.metadata["id_state"]
    out2 = await stage.process(ctx2, UsageEmbedder(sim=0.9, tokens=24))
    assert out2.id_result.metrics.embedding_tokens == 24
    assert out2.id_result.metrics.embedding_calls == 2
    assert out2.id_result.metrics.tokens_total == 24    # no LLM cost
    assert out2.id_result.goal_status == "ONGOING"


async def test_fast_path_costs_zero_embedding():
    stage = IDStage()
    ctx1 = make_ctx(_intent(goal="configure docker", intent_class="ACTION_REQUEST", domains=["TECH"]))
    await stage.process(ctx1, UsageEmbedder(tokens=24))
    # CLARIFICATION on turn 2 → Stage 0, no embedding
    ctx2 = make_ctx(_intent(goal="huh?", intent_class="CLARIFICATION", domains=[]))
    ctx2.metadata["id_state"] = ctx1.metadata["id_state"]
    out2 = await stage.process(ctx2, UsageEmbedder(tokens=24))
    assert out2.id_result.metrics.embedding_tokens == 0


async def test_plain_embedder_reports_zero_tokens_but_runs():
    stage = IDStage()
    ctx1 = make_ctx(_intent(goal="configure docker", intent_class="ACTION_REQUEST", domains=[]))
    await stage.process(ctx1, PlainEmbedder(sim=0.9))
    ctx2 = make_ctx(_intent(goal="daemon meaning", intent_class="INFORMATION_REQUEST", domains=[]))
    ctx2.metadata["id_state"] = ctx1.metadata["id_state"]
    out2 = await stage.process(ctx2, PlainEmbedder(sim=0.9))
    assert out2.id_result.goal_status == "ONGOING"               # similarity used
    assert out2.id_result.metrics.embedding_tokens == 0          # no usage API
    assert out2.id_result.metrics.embedding_calls == 2


# ── Drift integration ───────────────────────────────────────────────────────

async def test_drift_seeded_and_situational_set():
    stage = IDStage()
    ctx = make_ctx(_intent(goal="configure docker", intent_class="ACTION_REQUEST", domains=[]))
    out = await stage.process(ctx, PlainEmbedder())
    assert out.drift is not None
    assert out.drift.situational_drift is not None
    # first turn goal_similarity = 1.0 → situational drift 0.0
    assert out.drift.situational_drift == 0.0
    assert out.id_result.goal_similarity == 1.0


async def test_situational_drift_from_low_similarity():
    stage = IDStage()
    ctx1 = make_ctx(_intent(goal="configure docker", intent_class="ACTION_REQUEST", domains=[]))
    await stage.process(ctx1, UsageEmbedder(sim=0.2, tokens=10))
    ctx2 = make_ctx(_intent(goal="best pizza", intent_class="INFORMATION_REQUEST", domains=[]))
    ctx2.metadata["id_state"] = ctx1.metadata["id_state"]
    out2 = await stage.process(ctx2, UsageEmbedder(sim=0.2, tokens=10))
    assert out2.id_result.goal_status == "ABANDONED"
    assert out2.id_result.goal_similarity == pytest.approx(0.2)
    assert out2.drift.situational_drift == pytest.approx(0.8)


async def test_downgrade_applied_for_new_goal():
    """High situational drift on a NEW goal should downgrade ask_user→warn."""
    # thresholds tuned so situational drift alone crosses ask_user
    calc = DriftCalculator(thresholds=DriftThresholds(warn=0.3, ask_user=0.5, self_correct=0.9))
    stage = IDStage(drift=calc)
    ctx = make_ctx(_intent(goal="brand new topic", intent_class="ACTION_REQUEST", domains=[]))
    # force epistemological high so cumulative is high on a NEW goal
    ctx.noumeno.drift_score = 0.9
    out = await stage.process(ctx, PlainEmbedder())
    assert out.id_result.goal_status == "NEW"
    assert out.drift.drift_action == "warn"          # downgraded from ask_user


# ── Complexity ──────────────────────────────────────────────────────────────

async def test_complexity_defaults_low_and_high_on_pii():
    stage = IDStage()
    low = await stage.process(make_ctx(_intent()), PlainEmbedder())
    assert low.id_result.complexity == "LOW"
    high = await stage.process(make_ctx(_intent(pii_risk="HIGH")), PlainEmbedder())
    assert high.id_result.complexity == "HIGH"


async def test_complexity_configurable_complex_domains():
    stage = IDStage(complex_domains={"HEALTH"})
    out = await stage.process(make_ctx(_intent(domains=["HEALTH"])), PlainEmbedder())
    assert out.id_result.complexity == "HIGH"
    # core default (no complex domains) would be LOW
    out2 = await IDStage().process(make_ctx(_intent(domains=["HEALTH"])), PlainEmbedder())
    assert out2.id_result.complexity == "LOW"


# ── State round-trip ────────────────────────────────────────────────────────

async def test_goal_state_persists_across_turns():
    stage = IDStage()
    ctx1 = make_ctx(_intent(goal="configure docker", intent_class="ACTION_REQUEST", domains=["TECH"]))
    out1 = await stage.process(ctx1, PlainEmbedder())
    assert out1.id_result.goal_status == "NEW"
    assert out1.id_result.active_goal == "configure docker"

    ctx2 = make_ctx(_intent(goal="fix docker", intent_class="ACTION_REQUEST", domains=["TECH"]))
    ctx2.metadata["id_state"] = out1.metadata["id_state"]
    out2 = await stage.process(ctx2, PlainEmbedder())
    assert out2.id_result.goal_status == "ONGOING"
    assert out2.id_result.active_goal == "configure docker"   # goal persisted


# ── Elliptical-reply guard (short answer to the assistant's pending question) ──

_PENDING_QUESTION_HISTORY = (
    "User: Por hoje eu me cadastro no Cogno.\n"
    "Assistant: Você gostaria de saber mais sobre como configurar isso?"
)


async def test_ellipsis_reply_social_keeps_goal_ongoing():
    # Real regression (CLOSER session, turn 51): assistant asked a question, user said "Sim",
    # NER (context-blind) classified SOCIAL → Rule 1 completed and WIPED the live goal, and
    # the agent restarted its arc. With the guard the goal must survive as ONGOING.
    stage = IDStage()
    ctx1 = make_ctx(_intent(goal="learn how to configure Cogno",
                            intent_class="INFORMATION_REQUEST", domains=["GENERAL"]))
    out1 = await stage.process(ctx1, PlainEmbedder())

    ctx2 = PipelineContext(user_input="Sim")
    ctx2.noumeno = make_noumeno_result("Sim", "Yes.")
    ctx2.intent = _intent(intent_class="SOCIAL", goal=None)
    ctx2.metadata["id_state"] = out1.metadata["id_state"]
    ctx2.metadata["conversation_history"] = _PENDING_QUESTION_HISTORY
    out2 = await stage.process(ctx2, PlainEmbedder())
    assert out2.id_result.goal_status == "ONGOING"
    assert out2.id_result.active_goal == "learn how to configure Cogno"


async def test_long_farewell_after_question_still_completes():
    # The CLOSER ends nearly every reply with a question — a LONG genuine goodbye must still
    # complete the goal (condition 2 of the guard: input ≤ 3 tokens).
    stage = IDStage()
    ctx1 = make_ctx(_intent(goal="learn how to configure Cogno",
                            intent_class="INFORMATION_REQUEST", domains=["GENERAL"]))
    out1 = await stage.process(ctx1, PlainEmbedder())

    ctx2 = PipelineContext(user_input="Gostei muito de conversar com você, obrigado pelo papo!")
    ctx2.noumeno = make_noumeno_result("...", "I really enjoyed talking to you, thanks!")
    ctx2.intent = _intent(intent_class="SOCIAL", goal=None)
    ctx2.metadata["id_state"] = out1.metadata["id_state"]
    ctx2.metadata["conversation_history"] = _PENDING_QUESTION_HISTORY
    out2 = await stage.process(ctx2, PlainEmbedder())
    assert out2.id_result.goal_status == "COMPLETED"
    assert out2.id_result.active_goal is None


@pytest.mark.parametrize("user_input,history,expected", [
    # bare answer + pending question → elliptical
    ("Sim", _PENDING_QUESTION_HISTORY, True),
    ("WhatsApp", _PENDING_QUESTION_HISTORY, True),
    ("Qualquer um deles", _PENDING_QUESTION_HISTORY, True),
    # trailing emoji/markdown after the "?" is tolerated
    ("Sim", "Assistant: Quer saber mais? 😊", True),
    ("Sim", "Assistant: **Quer saber mais?**", True),
    # 4+ words → carries enough signal on its own
    ("entre 10 e 20 por dia", _PENDING_QUESTION_HISTORY, False),
    # assistant's last utterance was not a question
    ("Sim", "User: oi\nAssistant: Entendi, vou registrar isso.", False),
    # no transcript at all (first turn / long gap outside the burst)
    ("Sim", "", False),
    # transcript without an assistant line
    ("Sim", "User: oi", False),
])
async def test_is_ellipsis_reply_detector(user_input, history, expected):
    ctx = PipelineContext(user_input=user_input)
    ctx.metadata["conversation_history"] = history
    assert IDStage._is_ellipsis_reply(ctx) is expected


# ── confidence_divergence must measure DISAGREEMENT, never absence ─────────────────────────

@pytest.mark.asyncio
async def test_a_missing_confidence_key_at_both_ends_is_not_a_divergence():
    """The two stages used to default to DIFFERENT values — 1.0 in NOUMENO, 0.5 in NER — so a
    model that dropped `confidence` at both ends produced |1.0 - 0.5| = 0.50, clearing the 0.4
    threshold exactly. The signal fired on ABSENCE OF INFORMATION, the inverse of what it
    exists to detect, and it was verified firing live on a turn whose only anomaly was the
    missing key.
    """
    from cogno_anima.stages.ner import IntentAnalyzer
    from cogno_anima.stages.noumeno import Noumeno

    class _Emb:
        async def embed(self, text):        # noqa: ANN001
            return [1.0, 0.0, 0.0]

        async def similarity(self, a, b):   # noqa: ANN001
            return 1.0

    class _Backend:
        model = "scripted"

        def __init__(self, payload: str) -> None:
            self._payload = payload

        async def generate(self, system, prompt):   # noqa: ANN001
            return self._payload, 10, 5

    ctx = PipelineContext(user_input="dez por dia")
    # BOTH payloads omit `confidence` — the exact shape that produced a false signal.
    await Noumeno(_Emb()).process(ctx, _Backend('{"rewritten":"Ten a day.","context_turn":""}'))
    await IntentAnalyzer().process(ctx, _Backend(
        '{"intent_class":"INFORMATION_REQUEST","sentiment":"NEUTRAL","domains":["GENERAL"]}'))

    assert ctx.noumeno is not None and ctx.intent is not None
    assert ctx.noumeno.confidence == ctx.intent.confidence   # the property that matters
    out = await IDStage(confidence_divergence_threshold=0.4).process(ctx, _Emb())
    assert out.id_result.confidence_divergence is False


@pytest.mark.asyncio
async def test_a_real_disagreement_still_flags():
    """Contraprova: aligning the defaults must not disarm the signal it exists for."""
    stage = IDStage(confidence_divergence_threshold=0.4)
    ctx = make_ctx(_intent())
    ctx.noumeno.confidence, ctx.intent.confidence = 0.95, 0.30
    out = await stage.process(ctx, PlainEmbedder())
    assert out.id_result.confidence_divergence is True
