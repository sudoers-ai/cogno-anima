"""
cogno_anima.routing.goal — Goal continuity across conversation turns.

Tracks the active conversation goal with a lifecycle:
  NEW       → no active goal, or the previous one was COMPLETED/ABANDONED
  ONGOING   → the same goal continues (continuity check passes)
  COMPLETED → goal resolved (SOCIAL intent after an active goal)
  ABANDONED → user shifted to an unrelated goal while the prior was active

Continuity is evaluated in stages (cheap signals first, embedding last):
  Stage 0   — CLARIFICATION intent → always ONGOING (references prior by definition).
  Stage 1   — domain intersection + goal-oriented intent → ONGOING.
  Stage 1.5 — anaphoric fast-path: pii_session_hint + domain=OTHER/empty → ONGOING.
  Stage 1.6 — context_dependent (NER-detected back-reference) → ONGOING.
  Stage 2   — contextual semantic similarity via the injected async `similarity_fn`
              (cosine in production), with ONE-SIDED enrichment of the active-goal
              anchor; falls back to Jaccard token overlap when no `similarity_fn`
              is configured or it raises.

Pure: no LLM/DB/network. The embedder is reached only through `similarity_fn`,
injected by the caller (the ID stage), which also owns token accounting.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Awaitable, Callable, Optional


def _tokenize(text: str) -> set[str]:
    """Lowercased word tokens (keeps accents and short words), punctuation stripped."""
    return set(re.findall(r"[a-zA-ZÀ-ÿ0-9]+", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets. Returns 1.0 if both empty."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# Intents that signal the ongoing goal has been fulfilled.
_COMPLETION_INTENTS = {"SOCIAL"}

# Intents that can create or continue goals.
_GOAL_INTENTS = {"ACTION_REQUEST", "INFORMATION_REQUEST", "CREATIVE_TASK", "CLARIFICATION"}

SimilarityFn = Callable[[str, str], Awaitable[float]]


class GoalManager:
    """
    Tracks the active conversation goal across turns using staged continuity.

    Args:
        similarity_threshold:     Min similarity for ONGOING at Stage 2 (default 0.75,
                                  calibrated for cosine). Jaccard fallback is well
                                  calibrated lower; the ID stage picks the threshold
                                  per mode.
        pii_similarity_threshold: Lenient threshold when pii_session_hint=True
                                  (default 0.35) for vaguer references to prior PII.
        require_domain_match:     Whether the Stage 1 domain check is active (default True).
        similarity_fn:            Async callable(str, str) -> float for Stage 2.
                                  None → Jaccard fallback (no embedding backend).
        goal_history_size:        Past goals kept for context enrichment (default 3).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        pii_similarity_threshold: float = 0.35,
        require_domain_match: bool = True,
        similarity_fn: Optional[SimilarityFn] = None,
        goal_history_size: int = 3,
    ) -> None:
        self._threshold = similarity_threshold
        self._pii_threshold = pii_similarity_threshold
        self._require_domain_match = require_domain_match
        self._similarity_fn = similarity_fn
        self._history_size = goal_history_size

        self._active_goal: Optional[str] = None
        self._goal_tokens: set[str] = set()
        self._goal_domains: set[str] = set()
        self._goal_intent: str = ""
        self._goal_status: str = "NEW"
        self._goal_history: deque[str] = deque(maxlen=goal_history_size)

    @property
    def active_goal(self) -> Optional[str]:
        return self._active_goal

    @property
    def goal_status(self) -> str:
        return self._goal_status

    @property
    def goal_history(self) -> list[str]:
        """Recent goal strings, newest first (for observability)."""
        return list(self._goal_history)

    async def update(
        self,
        new_goal: Optional[str],
        intent_class: str,
        domains: Optional[list[str]] = None,
        pii_session_hint: bool = False,
        context_dependent: bool = False,
    ) -> tuple[str, Optional[str], float]:
        """
        Process a turn; return (goal_status, active_goal, goal_similarity).

        goal_similarity is 1.0 when continuity is decided without measuring it
        (first turn, fast-paths, no-goal carry-over, completion) and the computed
        value when Stage 2 runs. The caller feeds it to drift.compute_situational
        and surfaces it on IdResult.
        """
        current_domains = set(domains or [])
        prior_active = self._active_goal
        prior_status = self._goal_status

        # Rule 1: SOCIAL intent after an active goal → COMPLETED.
        if intent_class in _COMPLETION_INTENTS and prior_active is not None:
            self._goal_status = "COMPLETED"
            self._active_goal = None
            self._goal_tokens = set()
            self._goal_domains = set()
            self._goal_intent = ""
            return self._goal_status, self._active_goal, 1.0

        # Rule 2: no current goal (fresh start). Only a goal-oriented intent establishes a
        # persistent goal — a SOCIAL greeting ("E aí") must NOT, otherwise the user stating
        # their real request next turn looks like they ABANDONED the "greeting goal".
        if prior_active is None:
            if new_goal and intent_class in _GOAL_INTENTS:
                self._set_active_goal(new_goal, intent_class, current_domains)
            self._goal_status = "NEW"
            return self._goal_status, self._active_goal, 1.0

        # Rule 3: evaluate continuity with the active goal.
        if new_goal:
            is_ongoing, similarity = await self._is_same_goal(
                new_goal, intent_class, current_domains,
                pii_session_hint=pii_session_hint,
                context_dependent=context_dependent,
            )
            if is_ongoing:
                self._goal_domains |= current_domains
                self._goal_status = "ONGOING"
            else:
                self._goal_status = "ABANDONED"
                self._set_active_goal(new_goal, intent_class, current_domains)
            return self._goal_status, self._active_goal, similarity

        # No goal detected — keep ONGOING if already active.
        if prior_status in ("NEW", "ONGOING"):
            self._goal_status = "ONGOING"
        return self._goal_status, self._active_goal, 1.0

    # ── State (serialized into ctx.metadata["id_state"] by the ID stage) ──────

    def to_dict(self) -> dict:
        return {
            "active_goal": self._active_goal,
            "goal_status": self._goal_status,
            "goal_intent": self._goal_intent,
            "goal_tokens": sorted(self._goal_tokens),
            "goal_domains": sorted(self._goal_domains),
            "goal_history": list(self._goal_history),
        }

    def from_dict(self, state: Optional[dict]) -> None:
        state = state or {}
        self._active_goal = state.get("active_goal")
        self._goal_status = state.get("goal_status", "NEW")
        self._goal_intent = state.get("goal_intent", "")
        self._goal_tokens = set(state.get("goal_tokens", []))
        self._goal_domains = set(state.get("goal_domains", []))
        self._goal_history = deque(state.get("goal_history", []), maxlen=self._history_size)

    def reset(self) -> None:
        self._active_goal = None
        self._goal_tokens = set()
        self._goal_domains = set()
        self._goal_intent = ""
        self._goal_status = "NEW"
        self._goal_history.clear()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _set_active_goal(self, goal: str, intent: str, domains: set[str]) -> None:
        """Set a new active goal and push it to history. Called only on a real change."""
        self._active_goal = goal
        self._goal_tokens = _tokenize(goal)
        self._goal_domains = domains
        self._goal_intent = intent
        self._push_to_history(goal)

    def _push_to_history(self, goal: str) -> None:
        if not self._goal_history or self._goal_history[0] != goal:
            self._goal_history.appendleft(goal)

    async def _is_same_goal(
        self,
        new_goal: str,
        intent_class: str,
        current_domains: set[str],
        pii_session_hint: bool = False,
        context_dependent: bool = False,
    ) -> tuple[bool, float]:
        """Return (is_ongoing, similarity). Fast-path decisions report similarity 1.0."""
        # Stage 0: CLARIFICATION is definitionally about prior context.
        if intent_class == "CLARIFICATION":
            return True, 1.0

        # Stage 1: domain intersection signal.
        if self._require_domain_match and current_domains and self._goal_domains:
            if (current_domains & self._goal_domains) and intent_class in _GOAL_INTENTS:
                return True, 1.0

        # Stage 1.5: anaphoric fast-path with pii_session_hint (no embedding).
        # domain=OTHER or empty almost certainly refers to prior context.
        # NOTE: GENERAL is a real domain for new questions — kept out of this path.
        if pii_session_hint and intent_class in _GOAL_INTENTS:
            if not current_domains or current_domains <= {"OTHER"}:
                return True, 1.0

        # Stage 1.6: NER-detected back-reference (pronoun/possessive) → continuation.
        if context_dependent and intent_class in _GOAL_INTENTS:
            return True, 1.0

        # Stage 2: contextual semantic similarity.
        threshold = self._pii_threshold if pii_session_hint else self._threshold
        sim = await self._compute_similarity(new_goal)
        return sim >= threshold, sim

    async def _compute_similarity(self, new_goal: str) -> float:
        """
        Contextually-enriched similarity between new_goal and the active goal.

        ONE-SIDED enrichment (anchor only):
          a = "<active_goal> | <goal_history>"   ← richer anchor for the current topic
          b = "<new_goal>"                        ← unchanged (no contamination)

        Symmetric enrichment is avoided: adding the same context to both strings
        creates false positives (unrelated goals sharing topic terms get boosted).
        Falls back to Jaccard on raw tokens if similarity_fn is None or raises.
        """
        if self._similarity_fn is not None:
            try:
                history = [g for g in self._goal_history if g != self._active_goal]
                context = " | ".join(history)
                a = f"{self._active_goal} | {context}" if context else (self._active_goal or "")
                b = new_goal  # unchanged — do NOT contaminate with topic terms
                return await self._similarity_fn(a, b)
            except Exception:  # noqa: BLE001 — any embedder failure degrades to Jaccard
                pass

        # Jaccard fallback (no embedding backend); raw tokens, no enrichment.
        return _jaccard(self._goal_tokens, _tokenize(new_goal))
