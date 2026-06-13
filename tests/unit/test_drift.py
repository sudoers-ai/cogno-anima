import pytest
from cogno_core.types import DriftMetrics, IntentResult, NoumenoResult, StageMetrics
from cogno_core.stages.drift import DriftCalculator


def make_noumeno_result(original: str, rewritten: str) -> NoumenoResult:
    metrics = StageMetrics(stage="noumeno", elapsed_ms=10.0, tokens_in=5, tokens_out=5, model="test")
    return NoumenoResult(
        original=original,
        rewritten=rewritten,
        context_turn="",
        language="pt",
        canonical_language="en",
        drift_score=0.0,
        drift_tag="PASS_THROUGH",
        changed=False,
        confidence=1.0,
        change_subject=False,
        subject_similarity=1.0,
        context_used=False,
        preserved_terms=[],
        rewrite_warnings=[],
        metrics=metrics,
    )


def make_intent_result(
    intent_class="UNKNOWN",
    sentiment="NEUTRAL",
    temporal_class="TIMELESS",
    entities_people=None,
    entities_objects=None,
    entities_concepts=None,
    entities_pronouns=None,
    aristotelian=None,
    goal=None,
) -> IntentResult:
    metrics = StageMetrics(stage="ner", elapsed_ms=10.0, tokens_in=5, tokens_out=5, model="test")
    return IntentResult(
        intent_class=intent_class,
        sentiment=sentiment,
        confidence=1.0,
        temporal_class=temporal_class,
        triad_signal="BALANCED",
        entities_people=entities_people or [],
        entities_pronouns=entities_pronouns or [],
        entities_possessives=[],
        entities_objects=entities_objects or [],
        entities_concepts=entities_concepts or [],
        location=None,
        mandatory_tags=[],
        abstract_tags=[],
        aristotelian=aristotelian or {},
        domains=[],
        goal=goal,
        causal_chain=[],
        parole=None,
        langue=None,
        negation=[],
        constraints=[],
        modality=None,
        speech_act=None,
        verbs=[],
        context_dependent=False,
        is_composite=False,
        is_sequential=False,
        comparatives=[],
        pii=[],
        pii_risk="NONE",
        raw_intent_class=None,
        raw_domains=[],
        raw_goal=None,
        metrics=metrics,
        raw_response=None,
    )


def test_epistemological_drift():
    calc = DriftCalculator()
    noumeno = make_noumeno_result("run the test", "I want to run the test")
    noumeno.drift_score = 0.456
    
    intent = make_intent_result()
    drift = calc.compute(noumeno, intent)
    assert drift.intent_changed is False
    assert drift.sentiment_changed is False
    assert drift.temporal_changed is False
    assert drift.drift_score == 0.456
    assert drift.word_count_original == 3
    assert drift.word_count_noumeno == 6


def test_ontological_drift():
    calc = DriftCalculator()
    noumeno = make_noumeno_result("original", "this is a very specific sentence about gravity")
    
    # High overlap (gravity, specific, sentence)
    intent = make_intent_result(
        entities_concepts=["gravity", "specific"],
        aristotelian={"SUBSTANCE": "SENTENCE | some sentence"},
    )
    drift = calc.compute(noumeno, intent)
    calc.compute_ontological(drift, noumeno, intent)
    # content words in rewritten (filtered by stop words/len > 3): ["very", "specific", "sentence", "gravity"] (len=4)
    # overlap: ["specific", "sentence", "gravity"] (len=3)
    # coverage = 3 / 4 = 0.75 -> drift = 1.0 - 0.75 = 0.25
    assert drift.ontological_drift == pytest.approx(0.25, abs=0.05)


def test_situational_drift():
    calc = DriftCalculator()
    drift = DriftMetrics(
        intent_changed=False, sentiment_changed=False, temporal_changed=False,
        word_count_original=10, word_count_noumeno=10, compression_ratio=1.0,
        aristotelian_coverage=0, drift_score=0.0
    )
    calc.compute_situational(drift, 0.8)
    assert drift.situational_drift == 0.2


def test_execution_drift():
    calc = DriftCalculator()
    drift = DriftMetrics(
        intent_changed=False, sentiment_changed=False, temporal_changed=False,
        word_count_original=10, word_count_noumeno=10, compression_ratio=1.0,
        aristotelian_coverage=0, drift_score=0.0
    )
    
    # 1. No planned skill
    calc.compute_execution(drift, None, "some_skill")
    assert drift.execution_drift == 0.0

    # 2. Plan exists but actual is None
    calc.compute_execution(drift, "some_skill", None)
    assert drift.execution_drift == 0.5

    # 3. Match
    calc.compute_execution(drift, "some_skill", "SOME_SKILL")
    assert drift.execution_drift == 0.0

    # 4. Mismatch
    calc.compute_execution(drift, "some_skill", "other_skill")
    assert drift.execution_drift == 1.0


def test_synthesis_drift():
    calc = DriftCalculator()
    drift = DriftMetrics(
        intent_changed=False, sentiment_changed=False, temporal_changed=False,
        word_count_original=10, word_count_noumeno=10, compression_ratio=1.0,
        aristotelian_coverage=0, drift_score=0.0
    )
    
    # High overlap
    calc.compute_synthesis(drift, "temperature is twenty degrees", "The temperature is twenty degrees celsius")
    # Content words in tool (>3 chars): ["temperature", "twenty", "degrees"] (len=3)
    # All overlap -> coverage = 3/3 = 1.0 -> drift = 0.0
    assert drift.synthesis_drift == 0.0

    # Low overlap
    calc.compute_synthesis(drift, "temperature is twenty degrees", "some unrelated response text")
    assert drift.synthesis_drift == 1.0


def test_cumulative_drift_and_tags():
    calc = DriftCalculator()
    drift = DriftMetrics(
        intent_changed=False, sentiment_changed=False, temporal_changed=False,
        word_count_original=10, word_count_noumeno=5, compression_ratio=0.5,
        aristotelian_coverage=0, drift_score=0.2,
        ontological_drift=0.3, situational_drift=0.1,
        execution_drift=0.0, synthesis_drift=0.4
    )
    calc.compute_cumulative(drift)
    
    # cumulative = 0.15*0.2 + 0.15*0.3 + 0.20*0.1 + 0.25*0.0 + 0.25*0.4
    #            = 0.03 + 0.045 + 0.02 + 0.0 + 0.10 = 0.195 -> 0.195
    assert drift.cumulative_drift == pytest.approx(0.195)
    assert drift.drift_action == "none"

    tags = drift.to_tags()
    # compression_ratio = 0.5 < 0.8 -> COMPRESSED
    assert "NOUMENO.COMPRESSED" in tags
    assert "NOUMENO.DRIFT" not in tags


def _blank_drift(**overrides) -> DriftMetrics:
    base = dict(
        intent_changed=False, sentiment_changed=False, temporal_changed=False,
        word_count_original=10, word_count_noumeno=10, compression_ratio=1.0,
        aristotelian_coverage=0, drift_score=0.0,
    )
    base.update(overrides)
    return DriftMetrics(**base)


def test_situational_drift_clamps_out_of_range_similarity():
    """goal_similarity outside [0,1] is clamped before computing situational drift."""
    calc = DriftCalculator()

    high = _blank_drift()
    calc.compute_situational(high, 1.8)        # clamps to 1.0 -> drift 0.0
    assert high.situational_drift == 0.0

    low = _blank_drift()
    calc.compute_situational(low, -0.5)        # clamps to 0.0 -> drift 1.0
    assert low.situational_drift == 1.0


def test_cumulative_drift_clamps_each_component():
    """Out-of-range component drifts are clamped to [0,1]; cumulative never exceeds 1.0."""
    calc = DriftCalculator()
    drift = _blank_drift(
        drift_score=2.0,            # clamps to 1.0
        ontological_drift=5.0,      # clamps to 1.0
        situational_drift=-3.0,     # clamps to 0.0
        execution_drift=1.5,        # clamps to 1.0
        synthesis_drift=0.0,
    )
    calc.compute_cumulative(drift)
    # weighted: 0.15*1 + 0.15*1 + 0.20*0 + 0.25*1 + 0.25*0 = 0.55
    assert drift.cumulative_drift == pytest.approx(0.55)
    assert 0.0 <= drift.cumulative_drift <= 1.0
    assert drift.drift_action == "warn"   # >= 0.50 threshold


def test_cumulative_drift_all_max_is_capped_at_one():
    """All components at max → cumulative is exactly 1.0 (weights sum to 1.0)."""
    calc = DriftCalculator()
    drift = _blank_drift(
        drift_score=1.0, ontological_drift=1.0, situational_drift=1.0,
        execution_drift=1.0, synthesis_drift=1.0,
    )
    calc.compute_cumulative(drift)
    assert drift.cumulative_drift == pytest.approx(1.0)
    assert drift.drift_action == "self_correct"
