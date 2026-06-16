import pytest
from cogno_anima.types import DriftMetrics, IntentResult, NoumenoResult, StageMetrics
from cogno_anima.stages.drift import (
    DriftCalculator,
    DriftThresholds,
    DEFAULT_CUMULATIVE_WEIGHTS,
)


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


def test_ontological_uncomputed_for_contentless_rewrite():
    """A greeting-like rewrite (<2 content words) leaves ontological drift uncomputed,
    so it is excluded from the renormalized cumulative (no spurious drift action)."""
    calc = DriftCalculator()
    noumeno = make_noumeno_result("oi", "hi")     # 'hi' has no content words
    noumeno.drift_score = 0.3
    intent = make_intent_result(entities_concepts=["greeting"])
    drift = calc.compute(noumeno, intent)
    calc.compute_ontological(drift, noumeno, intent)
    assert drift.ontological_drift is None
    calc.compute_cumulative(drift)
    assert drift.cumulative_drift == pytest.approx(0.3)   # epistemological only
    assert drift.drift_action == "none"


def test_situational_drift():
    calc = DriftCalculator()
    drift = DriftMetrics(
        word_count_original=10, word_count_noumeno=10, compression_ratio=1.0,
        aristotelian_coverage=0, drift_score=0.0
    )
    calc.compute_situational(drift, 0.8)
    assert drift.situational_drift == 0.2


def test_execution_drift():
    calc = DriftCalculator()
    drift = DriftMetrics(
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


def test_cumulative_renormalizes_over_computed_stages():
    """With only NOUMENO+NER populated, cumulative is the mean of epist+onto (full scale)."""
    calc = DriftCalculator()
    # epist via drift_score, onto via compute_ontological; situational/exec/synth = None
    drift = _blank_drift(drift_score=0.6)
    drift.ontological_drift = 1.0   # stage 2 present
    # situational/execution/synthesis remain None (stages not computed)
    calc.compute_cumulative(drift)
    # renormalized: (0.15*0.6 + 0.15*1.0) / (0.15+0.15) = 0.8  → not deflated to 0.24
    assert drift.cumulative_drift == pytest.approx(0.8)
    assert drift.drift_action == "ask_user"   # 0.8 ≥ 0.70


def test_cumulative_epistemological_only():
    """Just after NOUMENO, cumulative equals the epistemological drift itself."""
    calc = DriftCalculator()
    drift = _blank_drift(drift_score=0.45)   # onto/sit/exec/synth all None
    calc.compute_cumulative(drift)
    assert drift.cumulative_drift == pytest.approx(0.45)


def test_cumulative_full_pipeline_matches_static_weights():
    """When all 5 stages are present, renormalization divides by 1.0 → original weighting."""
    calc = DriftCalculator()
    drift = _blank_drift(drift_score=0.2, ontological_drift=0.3, situational_drift=0.1,
                         execution_drift=0.0, synthesis_drift=0.4)
    calc.compute_cumulative(drift)
    assert drift.cumulative_drift == pytest.approx(0.195)


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


# ────────────────────────────────────────────────────────────────────
#  Phase 2: injectable weights / thresholds + goal-aware downgrade
# ────────────────────────────────────────────────────────────────────

def test_default_calculator_unchanged():
    """DriftCalculator() with no args = the historical defaults (regression guard)."""
    calc = DriftCalculator()
    assert calc._weights == DEFAULT_CUMULATIVE_WEIGHTS
    assert calc._thresholds == DriftThresholds(warn=0.50, ask_user=0.70, self_correct=0.85)


def test_injected_weights_change_cumulative():
    """A host risk profile (e.g. FINANCE: execution-heavy) reweights cumulative."""
    finance = {
        "epistemological": 0.10, "ontological": 0.10,
        "situational": 0.30, "execution": 0.50, "synthesis": 0.00,
    }
    calc = DriftCalculator(weights=finance)
    drift = _blank_drift(drift_score=0.0, ontological_drift=0.0, situational_drift=0.0,
                         execution_drift=1.0, synthesis_drift=0.0)
    calc.compute_cumulative(drift)
    # only execution is non-zero: 0.50*1.0 / 1.0 = 0.50
    assert drift.cumulative_drift == pytest.approx(0.50)


def test_injected_thresholds_change_action():
    """Custom thresholds shift the action boundaries."""
    calc = DriftCalculator(thresholds=DriftThresholds(warn=0.10, ask_user=0.20, self_correct=0.30))
    drift = _blank_drift(drift_score=0.25)     # epistemological only
    calc.compute_cumulative(drift)
    assert drift.cumulative_drift == pytest.approx(0.25)
    assert drift.drift_action == "ask_user"    # 0.25 ≥ 0.20 but < 0.30


def test_weights_are_relative_renormalized():
    """Weights need not sum to 1.0 — ratios are what matter after renormalization."""
    calc = DriftCalculator(weights={
        "epistemological": 1, "ontological": 1, "situational": 1,
        "execution": 1, "synthesis": 1,
    })
    drift = _blank_drift(drift_score=0.4, ontological_drift=0.6)  # sit/exec/synth None
    calc.compute_cumulative(drift)
    # equal weights → plain mean of the two present stages
    assert drift.cumulative_drift == pytest.approx(0.5)


@pytest.mark.parametrize("bad,msg", [
    ({"epistemological": 0.5, "ontological": 0.5}, "missing"),         # missing keys
    ({**DEFAULT_CUMULATIVE_WEIGHTS, "bogus": 0.1}, "unknown"),         # extra key
])
def test_weights_validation_rejects_bad_keys(bad, msg):
    with pytest.raises(ValueError, match=msg):
        DriftCalculator(weights=bad)


def test_weights_validation_rejects_negative_and_zero_sum():
    neg = {**DEFAULT_CUMULATIVE_WEIGHTS, "execution": -0.1}
    with pytest.raises(ValueError, match="non-negative"):
        DriftCalculator(weights=neg)
    zero = {k: 0.0 for k in DEFAULT_CUMULATIVE_WEIGHTS}
    with pytest.raises(ValueError, match="positive sum"):
        DriftCalculator(weights=zero)


def test_downgrade_for_intentional_shift_softens_ask_user():
    """ask_user → warn when the user intentionally changed topic (NEW/ABANDONED)."""
    calc = DriftCalculator()
    for status in ("NEW", "ABANDONED"):
        drift = _blank_drift()
        drift.drift_action = "ask_user"
        calc.downgrade_for_intentional_shift(drift, status)
        assert drift.drift_action == "warn"


def test_downgrade_noop_for_ongoing_goal():
    """An ONGOING goal keeps ask_user — the user is mid-task, clarification is warranted."""
    calc = DriftCalculator()
    drift = _blank_drift()
    drift.drift_action = "ask_user"
    calc.downgrade_for_intentional_shift(drift, "ONGOING")
    assert drift.drift_action == "ask_user"


def test_downgrade_only_touches_ask_user():
    """Other actions (none/warn/self_correct) are left untouched regardless of goal_status."""
    calc = DriftCalculator()
    for action in ("none", "warn", "self_correct"):
        drift = _blank_drift()
        drift.drift_action = action
        calc.downgrade_for_intentional_shift(drift, "NEW")
        assert drift.drift_action == action
