from __future__ import annotations

import logging
from typing import Optional

from cogno_core.types import DriftMetrics, IntentResult, NoumenoResult
from cogno_core.utils import (
    clamp01,
    safe_float,
    word_count,
    content_words,
    extend_strings,
)

logger = logging.getLogger("cogno_core.drift")


# Cumulative weights across all 5 stages.
# Must sum to 1.0.
_CUMULATIVE_WEIGHTS: dict[str, float] = {
    "epistemological": 0.15,  # NOUMENO
    "ontological": 0.15,      # NER
    "situational": 0.20,      # ID
    "execution": 0.25,        # EGO
    "synthesis": 0.25,        # SUPEREGO
}

# Action thresholds.
# These are recommendations for the caller/orchestrator.
_THRESHOLD_WARN = 0.50
_THRESHOLD_ASK_USER = 0.70
_THRESHOLD_SELF_CORRECT = 0.85


class DriftCalculator:
    """
    Pure drift utility for the Cogno pipeline.

    Responsibility by stage:

    - NOUMENO:
        Computes epistemological drift during rewriting.
        This calculator consumes `noumeno.drift_score`.

    - NER:
        Feeds ontological drift by comparing `noumeno.rewritten`
        against extracted semantic metadata.

    - ID:
        Feeds situational drift through goal/topic similarity.

    - EGO:
        Feeds execution drift by comparing planned vs actual execution.

    - SUPEREGO:
        Feeds synthesis drift by comparing tool payload vs final response.

    This class does not call LLMs, tools, databases, or external services.
    """

    # ---------------------------------------------------------------------
    # Stage 1: Epistemological drift
    # ---------------------------------------------------------------------

    def compute(self, noumeno: NoumenoResult, intent: IntentResult) -> DriftMetrics:
        """
        Initializes DriftMetrics using the epistemological drift already
        computed by NOUMENO.

        This method no longer performs heuristic intent/sentiment/temporal
        classification. That logic belonged to the old NER-driven drift model.

        Args:
            noumeno: Output from the NOUMENO stage.
            intent: Output from the NER stage.

        Returns:
            DriftMetrics initialized with Stage 1 values and basic coverage stats.
        """
        wc_original = word_count(getattr(noumeno, "original", ""))
        wc_noumeno = word_count(getattr(noumeno, "rewritten", ""))

        compression_ratio = (
            wc_noumeno / wc_original
            if wc_original > 0
            else 1.0
        )

        epistemological_drift = clamp01(
            safe_float(getattr(noumeno, "drift_score", 0.0), default=0.0)
        )

        aristotelian = getattr(intent, "aristotelian", {}) or {}

        return DriftMetrics(
            # Kept for compatibility with the existing DriftMetrics contract.
            # These are no longer computed here because Stage 1 is owned by NOUMENO.
            intent_changed=False,
            sentiment_changed=False,
            temporal_changed=False,

            word_count_original=wc_original,
            word_count_noumeno=wc_noumeno,
            compression_ratio=round(compression_ratio, 3),
            aristotelian_coverage=len(aristotelian),

            # In the existing contract, `drift_score` represents the
            # epistemological drift used in cumulative drift.
            drift_score=round(epistemological_drift, 3),
        )

    # ---------------------------------------------------------------------
    # Stage 2: Ontological drift
    # ---------------------------------------------------------------------

    def compute_ontological(
        self,
        drift: DriftMetrics,
        noumeno: NoumenoResult,
        intent: IntentResult,
    ) -> None:
        """
        Computes ontological drift between the NOUMENO rewrite and the
        semantic metadata extracted by NER.

        MVP implementation:
            Uses deterministic word overlap.

        Future evolution:
            Can be replaced by embedding similarity between
            `noumeno.rewritten` and a serialized semantic package from NER.
        """
        rewritten = getattr(noumeno, "rewritten", "") or ""
        rewritten_words = content_words(rewritten)

        # Degenerate case: a rewrite with almost no content words (greetings,
        # "hi", short social turns) has nothing meaningful for NER to "cover",
        # so ontological drift is not measurable. Leave it UNCOMPUTED (None) so
        # compute_cumulative excludes it from the renormalized average instead of
        # reporting a spurious 1.0 that would false-trigger a drift action.
        if len(rewritten_words) < 2:
            drift.ontological_drift = None
            return

        ner_text_parts: list[str] = []

        extend_strings(ner_text_parts, getattr(intent, "entities_people", []))
        extend_strings(ner_text_parts, getattr(intent, "entities_objects", []))
        extend_strings(ner_text_parts, getattr(intent, "entities_concepts", []))
        extend_strings(ner_text_parts, getattr(intent, "entities_possessives", []))
        extend_strings(ner_text_parts, getattr(intent, "domains", []))
        extend_strings(ner_text_parts, getattr(intent, "verbs", []))
        extend_strings(ner_text_parts, getattr(intent, "constraints", []))
        extend_strings(ner_text_parts, getattr(intent, "comparatives", []))
        extend_strings(ner_text_parts, getattr(intent, "causal_chain", []))
        extend_strings(ner_text_parts, getattr(intent, "abstract_tags", []))
        extend_strings(ner_text_parts, getattr(intent, "mandatory_tags", []))

        location = getattr(intent, "location", None)
        if location:
            ner_text_parts.append(str(location))

        goal = getattr(intent, "goal", None)
        if goal:
            ner_text_parts.append(str(goal))

        negation = getattr(intent, "negation", [])
        extend_strings(ner_text_parts, negation)

        aristotelian = getattr(intent, "aristotelian", {}) or {}
        if isinstance(aristotelian, dict):
            extend_strings(ner_text_parts, aristotelian.keys())
            extend_strings(ner_text_parts, aristotelian.values())

        ner_words = content_words(" ".join(ner_text_parts))

        if not ner_words:
            drift.ontological_drift = 1.0
            return

        overlap = rewritten_words & ner_words
        coverage = len(overlap) / len(rewritten_words)

        drift.ontological_drift = round(clamp01(1.0 - coverage), 2)

    # ---------------------------------------------------------------------
    # Stage 3: Situational drift
    # ---------------------------------------------------------------------

    def compute_situational(
        self,
        drift: DriftMetrics,
        goal_similarity: float,
    ) -> None:
        """
        Computes situational drift from goal/topic similarity.

        Args:
            drift: Mutable DriftMetrics object.
            goal_similarity: Cosine similarity in [0.0, 1.0].
                1.0 = same goal/topic.
                0.0 = completely different goal/topic.
        """
        similarity = clamp01(safe_float(goal_similarity, default=0.0))
        drift.situational_drift = round(1.0 - similarity, 2)

    # ---------------------------------------------------------------------
    # Stage 4: Execution drift
    # ---------------------------------------------------------------------

    def compute_execution(
        self,
        drift: DriftMetrics,
        planned_skill: Optional[str],
        actual_skill: Optional[str],
    ) -> None:
        """
        Computes execution drift between planned and actual execution.

        This method is still named in terms of "skill" because the previous
        contract used planned/actual skill names. In the clean architecture,
        the caller may pass any planned vs actual execution identifier.
        """
        planned = self._normalize_optional(planned_skill)
        actual = self._normalize_optional(actual_skill)

        if planned is None:
            drift.execution_drift = 0.0
        elif actual is None:
            drift.execution_drift = 0.5
        elif planned == actual:
            drift.execution_drift = 0.0
        else:
            drift.execution_drift = 1.0

    # ---------------------------------------------------------------------
    # Stage 5: Synthesis drift
    # ---------------------------------------------------------------------

    def compute_synthesis(
        self,
        drift: DriftMetrics,
        tool_payload: Optional[str],
        response: Optional[str],
    ) -> None:
        """
        Computes synthesis drift between source payload and final response.

        If no source payload exists, drift is 0.0 because there is no grounding
        object to compare against.
        """
        if not tool_payload or not response:
            drift.synthesis_drift = 0.0
            return

        source_words = content_words(tool_payload)
        if not source_words:
            drift.synthesis_drift = 0.0
            return

        response_words = content_words(response)
        if not response_words:
            drift.synthesis_drift = 1.0
            return

        overlap = source_words & response_words
        coverage = len(overlap) / len(source_words)

        drift.synthesis_drift = round(clamp01(1.0 - coverage), 2)

    # ---------------------------------------------------------------------
    # Cumulative drift
    # ---------------------------------------------------------------------

    def compute_cumulative(self, drift: DriftMetrics) -> None:
        """
        Computes weighted cumulative drift and sets a recommendation action.

        Cumulative is a weighted average **renormalized over the stages actually
        computed** (a `None` drift component means that stage has not run). This
        keeps cumulative on a full [0,1] scale — and the action thresholds
        meaningful — whether 2, 3, 4 or 5 stages have populated their drift.
        Epistemological (`drift_score`) is always present.

        The action is a signal for the caller/orchestrator. This class does not
        execute retries, ask the user, call tools, or self-correct by itself.
        """
        raw_components = {
            "epistemological": drift.drift_score,        # always present
            "ontological": drift.ontological_drift,
            "situational": drift.situational_drift,
            "execution": drift.execution_drift,
            "synthesis": drift.synthesis_drift,
        }
        present = {
            stage: clamp01(safe_float(value, default=0.0))
            for stage, value in raw_components.items()
            if value is not None
        }

        total_weight = sum(_CUMULATIVE_WEIGHTS[stage] for stage in present)
        if total_weight > 0:
            cumulative = sum(
                _CUMULATIVE_WEIGHTS[stage] * value for stage, value in present.items()
            ) / total_weight
        else:
            cumulative = 0.0

        drift.cumulative_drift = round(cumulative, 3)

        if cumulative >= _THRESHOLD_SELF_CORRECT:
            drift.drift_action = "self_correct"
        elif cumulative >= _THRESHOLD_ASK_USER:
            drift.drift_action = "ask_user"
        elif cumulative >= _THRESHOLD_WARN:
            drift.drift_action = "warn"
        else:
            drift.drift_action = "none"

        logger.info(
            "drift cumulative=%.3f action=%s over %d stage(s) [%s]",
            drift.cumulative_drift, drift.drift_action, len(present),
            " ".join(f"{s}={v:.2f}" for s, v in present.items()),
        )

    # ---------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------

    def _normalize_optional(self, value: Optional[str]) -> Optional[str]:
        """
        Normalizes optional execution identifiers.
        """
        if value is None:
            return None

        normalized = str(value).strip().lower()
        return normalized or None