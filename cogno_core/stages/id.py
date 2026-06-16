"""
cogno_core.stages.id — IDStage: strategic router & continuity (Stage 3).

Consumes the NER output (`ctx.intent`) and produces an `IdResult`: a routing
decision (triad_route), goal continuity, BDI intentions, attention focus, safety
gate, and cross-turn signals (turn number, sticky temporal, frustration-driven
emotional_override, advisory complexity). It also feeds situational drift into
the shared `DriftCalculator`.

Heuristic — no LLM. The embedder is used only for goal similarity (GoalManager
Stage 2), and only the embedding cost shows up in metrics. The stage is
stateless: all cross-turn state rides in `ctx.metadata["id_state"]` (a
serializable dict the host persists), so it survives a multi-worker HTTP setup
without a live per-session instance.

What stays OUT (host/EGO concern): persona↔MCP binding, skill selection/
execution, model-ladder escalation, session splitting, and the clarification
text — the ID emits signals (drift_action, blocked, complexity), the host acts.
"""

from __future__ import annotations

import time
import logging
from typing import Awaitable, Callable, Optional

from cogno_core.types import PipelineContext, IntentResult, StageMetrics, IdResult
from cogno_core.llm import Embedder
from cogno_core.stages.drift import DriftCalculator
from cogno_core.routing import GoalManager, AttentionFilter, IntentionTracker
from cogno_core.vocab import VALID_TRIAD, VALID_GOAL_STATUS, VALID_COMPLEXITY

logger = logging.getLogger("cogno_core.id")

STAGE_NAME = "id"

# Temporal ranking for stickiness: a follow-up under an ONGOING goal keeps the
# higher (more specific) temporal of the prior turn — NER evaluates each turn in
# isolation and loses this otherwise.
_TEMPORAL_RANK = {"RECENT": 3, "MIXED": 2, "HISTORICAL": 1, "TIMELESS": 0}


class IDStage:
    """The strategic routing layer. One logical stage; state lives in ctx.metadata."""

    name = STAGE_NAME

    def __init__(
        self,
        drift: Optional[DriftCalculator] = None,
        goal_threshold: float = 0.75,
        pii_goal_threshold: float = 0.35,
        attention_top_n: int = 5,
        frustration_threshold: int = 2,
        complex_domains: Optional[set[str]] = None,
    ) -> None:
        self._drift = drift or DriftCalculator()
        self._goal_threshold = goal_threshold
        self._pii_goal_threshold = pii_goal_threshold
        self._attention = AttentionFilter(top_n=attention_top_n)
        self._frustration_threshold = frustration_threshold
        # Domains treated as inherently complex (host policy). Core default: none.
        self._complex_domains = set(complex_domains or ())

    async def process(self, ctx: PipelineContext, embedder: Embedder) -> PipelineContext:
        t0 = time.perf_counter()
        if not ctx.noumeno or not ctx.intent:
            raise ValueError("NOUMENO and NER must be populated before running IDStage")
        intent = ctx.intent
        noumeno = ctx.noumeno

        id_state = dict(ctx.metadata.get("id_state") or {})

        # Turn number: host authoritative (turns.turn_n), else auto-increment.
        turn = ctx.metadata.get("turn_number")
        if turn is None:
            turn = int(id_state.get("turn_number", 0)) + 1
        id_state["turn_number"] = turn

        # Hydrate the continuity helpers from carry-over state.
        acc = {"tokens": 0, "calls": 0}
        goal_mgr = GoalManager(
            similarity_fn=self._make_similarity_fn(embedder, acc),
            similarity_threshold=self._goal_threshold,
            pii_similarity_threshold=self._pii_goal_threshold,
        )
        goal_mgr.from_dict(id_state.get("goal"))
        intentions = IntentionTracker()
        intentions.from_dict(id_state.get("intentions"))

        # Frustration streak → emotional_override (host may inject its own).
        streak = int(id_state.get("frustration_streak", 0))
        streak = streak + 1 if intent.sentiment == "FRUSTRATED" else 0
        id_state["frustration_streak"] = streak
        emotional_override = ctx.metadata.get("emotional_override")
        if emotional_override is None and streak >= self._frustration_threshold:
            emotional_override = "sustained_frustration"

        # Goal continuity (may reach Stage 2 → embedding).
        pii_hint = bool(ctx.metadata.get("pii_session_hint", False))
        goal_status, active_goal, goal_similarity = await goal_mgr.update(
            new_goal=intent.goal,
            intent_class=intent.intent_class,
            domains=intent.domains,
            pii_session_hint=pii_hint,
            context_dependent=intent.context_dependent,
        )

        # Routing + safety gate.
        triad_route = self._resolve_route(intent, emotional_override)
        blocked = intent.pii_risk == "CRITICAL"
        block_reason = (
            f"pii_risk=CRITICAL detected: {intent.pii}" if blocked else None
        )

        # Temporal stickiness → recorded on IdResult (NER stays untouched).
        effective_temporal = self._sticky_temporal(
            id_state.get("prior_temporal"), intent.temporal_class, goal_status,
        )
        id_state["prior_temporal"] = effective_temporal

        # Intentions + attention focus (over host-injected candidates).
        active_intentions = intentions.update(intent, goal_status)
        candidates = list(ctx.metadata.get("attention_candidates") or [])
        attention_focus = self._attention.focus(intent, candidates)

        # Complexity (advisory only — the host scales the model, not the core).
        complexity = self._complexity(intent)

        # Drift: seed once if absent, then situational → cumulative → downgrade.
        drift = ctx.drift
        if drift is None:
            drift = self._drift.compute(noumeno, intent)
            self._drift.compute_ontological(drift, noumeno, intent)
        self._drift.compute_situational(drift, goal_similarity)
        self._drift.compute_cumulative(drift)
        self._drift.downgrade_for_intentional_shift(drift, goal_status)
        ctx.drift = drift

        # Persist cross-turn state back into the carrier.
        id_state["goal"] = goal_mgr.to_dict()
        id_state["intentions"] = intentions.to_dict()
        ctx.metadata["id_state"] = id_state

        # Sanitize against the closed vocab ("never trust" carry-over/hints).
        if triad_route not in VALID_TRIAD:
            triad_route = "BALANCED"
        if goal_status not in VALID_GOAL_STATUS:
            goal_status = "NEW"
        if complexity not in VALID_COMPLEXITY:
            complexity = "LOW"

        elapsed_ms = (time.perf_counter() - t0) * 1000
        metrics = StageMetrics(
            stage=STAGE_NAME,
            elapsed_ms=round(elapsed_ms, 2),
            tokens_in=0,            # ID makes no LLM call
            tokens_out=0,
            embedding_tokens=acc["tokens"],
            embedding_calls=acc["calls"],
            model="heuristic",
        )

        ctx.id_result = IdResult(
            triad_route=triad_route,
            active_goal=active_goal,
            goal_status=goal_status,
            goal_similarity=round(goal_similarity, 4),
            active_intentions=active_intentions,
            attention_focus=attention_focus,
            blocked=blocked,
            block_reason=block_reason,
            turn_number=turn,
            temporal_class=effective_temporal,
            emotional_override=emotional_override,
            complexity=complexity,
            metrics=metrics,
        )
        logger.info(
            "ID turn=%d route=%s goal_status=%s blocked=%s complexity=%s sim=%.2f",
            turn, triad_route, goal_status, blocked, complexity, goal_similarity,
        )
        return ctx

    # ── Embedder usage closure (token accounting) ─────────────────────────────

    @staticmethod
    def _make_similarity_fn(
        embedder: Embedder, acc: dict,
    ) -> Callable[[str, str], Awaitable[float]]:
        """Async similarity that accumulates embedding cost (mirrors NOUMENO).

        Prefers a usage-aware embedder (``similarity_with_usage``); falls back to
        the plain ``similarity`` (0 tokens) so any Embedder still works. Each
        similarity counts as 2 embed operations.
        """
        async def _sim(a: str, b: str) -> float:
            usage_fn = getattr(embedder, "similarity_with_usage", None)
            if usage_fn is not None:
                sim, tokens = await usage_fn(a, b)
                acc["tokens"] += tokens
                acc["calls"] += 2
                return sim
            sim = await embedder.similarity(a, b)
            acc["calls"] += 2
            return sim
        return _sim

    # ── Routing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_route(intent: IntentResult, emotional_override: Optional[str]) -> str:
        """Priority-ordered routing. Safety gates first; then triad fallback."""
        if intent.pii_risk == "CRITICAL":
            return "SUPEREGO"            # blocking handled by caller (blocked=True)
        if intent.pii_risk == "HIGH":
            return "SUPEREGO"            # non-blocking review
        if emotional_override is not None:
            return "SUPEREGO"            # de-escalate sustained frustration
        if intent.intent_class == "CREATIVE_TASK":
            return "SUPEREGO"
        # The EGO is the tool gateway: any request that may need execution OR
        # data (ACTION_REQUEST / INFORMATION_REQUEST) routes to it — the EGO
        # no-ops to a draft when no tool is needed. Pure conversation stays
        # SUPEREGO-direct. (Widened from the old ACTION+SYSTEM rule, which
        # starved tool-requiring info queries like "what's my balance?".)
        if intent.intent_class in ("ACTION_REQUEST", "INFORMATION_REQUEST"):
            return "EGO"
        if intent.intent_class == "SOCIAL":
            return "SUPEREGO"
        signal = (intent.triad_signal or "").upper()
        if signal in VALID_TRIAD:
            return signal
        return "BALANCED"

    # ── Temporal stickiness ─────────────────────────────────────────────────

    @staticmethod
    def _sticky_temporal(
        prior: Optional[str], current: Optional[str], goal_status: str,
    ) -> Optional[str]:
        if goal_status == "ONGOING" and prior:
            if _TEMPORAL_RANK.get(prior, 0) > _TEMPORAL_RANK.get(current or "", 0):
                return prior
        return current

    # ── Complexity (advisory) ─────────────────────────────────────────────────

    def _complexity(self, intent: IntentResult) -> str:
        mandatory_short = [t.split(".")[-1] for t in (intent.mandatory_tags or [])]
        is_creative = intent.intent_class == "CREATIVE_TASK" or "CREATIVE" in mandatory_short
        domains = set(intent.domains or [])
        if is_creative and intent.is_composite:
            return "EXPERT"
        if intent.is_composite and (domains & self._complex_domains):
            return "EXPERT"
        if intent.is_composite or intent.pii_risk in ("HIGH", "CRITICAL"):
            return "HIGH"
        if domains & self._complex_domains:
            return "HIGH"
        if is_creative:
            return "MEDIUM"
        return "LOW"
