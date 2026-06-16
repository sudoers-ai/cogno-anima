from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Optional

from cogno_anima.types import DriftMetrics, IntentResult, NoumenoResult
from cogno_anima.utils import (
    clamp01,
    safe_float,
    word_count,
    content_words,
    extend_strings,
)

logger = logging.getLogger("cogno_anima.drift")


# Default cumulative weights across all 5 stages. Weights are RELATIVE:
# compute_cumulative renormalizes over the stages actually computed, so what
# matters is their ratio (a host may plug a risk profile, e.g. a FINANCE profile
# weighting execution heavily, without forking — see DriftCalculator.__init__).
DEFAULT_CUMULATIVE_WEIGHTS: dict[str, float] = {
    "epistemological": 0.15,  # NOUMENO
    "ontological": 0.15,      # NER
    "situational": 0.20,      # ID
    "execution": 0.25,        # EGO
    "synthesis": 0.25,        # SUPEREGO
}

_STAGE_KEYS = frozenset(DEFAULT_CUMULATIVE_WEIGHTS)


@dataclass(frozen=True)
class DriftThresholds:
    """Cumulative-drift action thresholds — recommendations for the caller.

    The action is a signal only; this library never asks the user, retries, or
    self-corrects on its own.
    """
    warn: float = 0.50
    ask_user: float = 0.70
    self_correct: float = 0.85


DEFAULT_THRESHOLDS = DriftThresholds()


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

    def __init__(
        self,
        weights: Optional[Mapping[str, float]] = None,
        thresholds: Optional[DriftThresholds] = None,
    ) -> None:
        """
        Args:
            weights: Per-stage cumulative weights. Defaults to
                DEFAULT_CUMULATIVE_WEIGHTS. Weights are relative (renormalized
                over the computed stages), so a host may pass a risk profile.
                Validated: all 5 stage keys present, non-negative, positive sum.
            thresholds: Action thresholds. Defaults to DEFAULT_THRESHOLDS.
        """
        w = dict(weights) if weights is not None else dict(DEFAULT_CUMULATIVE_WEIGHTS)
        missing = _STAGE_KEYS - w.keys()
        if missing:
            raise ValueError(f"weights missing stage(s): {sorted(missing)}")
        extra = w.keys() - _STAGE_KEYS
        if extra:
            raise ValueError(f"weights has unknown stage(s): {sorted(extra)}")
        if any(v < 0 for v in w.values()):
            raise ValueError("weights must be non-negative")
        if sum(w.values()) <= 0:
            raise ValueError("weights must have a positive sum")
        self._weights = w
        self._thresholds = thresholds or DEFAULT_THRESHOLDS

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
            word_count_original=wc_original,
            word_count_noumeno=wc_noumeno,
            compression_ratio=round(compression_ratio, 3),
            aristotelian_coverage=len(aristotelian),
            # `drift_score` carries the epistemological drift (from NOUMENO),
            # used as the Stage 1 component of cumulative drift.
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

        total_weight = sum(self._weights[stage] for stage in present)
        if total_weight > 0:
            cumulative = sum(
                self._weights[stage] * value for stage, value in present.items()
            ) / total_weight
        else:
            cumulative = 0.0

        drift.cumulative_drift = round(cumulative, 3)

        t = self._thresholds
        if cumulative >= t.self_correct:
            drift.drift_action = "self_correct"
        elif cumulative >= t.ask_user:
            drift.drift_action = "ask_user"
        elif cumulative >= t.warn:
            drift.drift_action = "warn"
        else:
            drift.drift_action = "none"

        logger.info(
            "drift cumulative=%.3f action=%s over %d stage(s) [%s]",
            drift.cumulative_drift, drift.drift_action, len(present),
            " ".join(f"{s}={v:.2f}" for s, v in present.items()),
        )

    # ---------------------------------------------------------------------
    # Goal-aware action policy
    # ---------------------------------------------------------------------

    def downgrade_for_intentional_shift(
        self,
        drift: DriftMetrics,
        goal_status: str,
    ) -> None:
        """
        Softens an ``ask_user`` action to ``warn`` on an intentional topic change.

        When the user deliberately starts a new objective (goal_status NEW or
        ABANDONED), a high cumulative drift is expected and should not interrupt
        them with a clarification request. compute_cumulative stays goal-agnostic
        (score → action); this policy is invoked explicitly by the caller that
        owns the goal_status (the ID stage), so a host that does not track goals
        is never forced to apply it.
        """
        if goal_status in ("NEW", "ABANDONED") and drift.drift_action == "ask_user":
            drift.drift_action = "warn"

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