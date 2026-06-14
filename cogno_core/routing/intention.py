"""
cogno_core.routing.intention — BDI-style active intentions across turns.

An intention is an explicit commitment the user is working toward (e.g. "configure
Docker on Ubuntu"), distinct from the latent `goal` extracted by NER. Multiple
intentions can be active at once and are resolved independently.

Lifecycle:
  OPEN   → active, not yet resolved
  CLOSED → resolved (goal COMPLETED or ABANDONED)

Up to 5 concurrent open intentions (oldest evicted FIFO over the limit). Pure
state management — no LLM, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Intent classes that create new intentions.
_INTENTION_INTENTS = {"ACTION_REQUEST", "CREATIVE_TASK", "INFORMATION_REQUEST"}

# Max concurrent active intentions.
_MAX_INTENTIONS = 5


@dataclass
class Intention:
    """A single active intention entry."""
    text: str          # human-readable description (English)
    intent_class: str  # the intent_class that originated this intention
    status: str = "OPEN"  # OPEN | CLOSED


class IntentionTracker:
    """Maintains a list of active BDI intentions across turns."""

    def __init__(self) -> None:
        self._intentions: list[Intention] = []

    @property
    def active(self) -> list[str]:
        """Text of all OPEN intentions."""
        return [i.text for i in self._intentions if i.status == "OPEN"]

    def update(self, intent: "_IntentLike", goal_status: str) -> list[str]:
        """
        Process a turn; return the current active intention texts.

        Args:
            intent:      current NER output (needs intent_class, goal, and the
                         entities_concepts/entities_objects fallbacks).
            goal_status: GoalManager output for this turn.
        """
        # Close all when the goal is COMPLETED.
        if goal_status == "COMPLETED":
            for it in self._intentions:
                it.status = "CLOSED"
            self._intentions = [i for i in self._intentions if i.status == "OPEN"]
            return self.active

        # Close the oldest open intention when the goal is ABANDONED.
        if goal_status == "ABANDONED":
            for it in self._intentions:
                if it.status == "OPEN":
                    it.status = "CLOSED"
                    break
            self._intentions = [i for i in self._intentions if i.status == "OPEN"]

        # Add a new intention if this turn starts one.
        if goal_status in ("NEW", "ABANDONED") and intent.intent_class in _INTENTION_INTENTS:
            text = intent.goal or _infer_intention(intent)
            open_texts = [i.text for i in self._intentions if i.status == "OPEN"]
            if text and text not in open_texts:
                self._intentions.append(Intention(text=text, intent_class=intent.intent_class))
                open_intentions = [i for i in self._intentions if i.status == "OPEN"]
                if len(open_intentions) > _MAX_INTENTIONS:
                    open_intentions[0].status = "CLOSED"  # evict oldest (FIFO)

        # Keep the list tidy.
        self._intentions = [i for i in self._intentions if i.status == "OPEN"]
        return self.active

    # ── State (serialized into ctx.metadata["id_state"] by the ID stage) ──────

    def to_dict(self) -> dict:
        return {"intentions": [
            {"text": i.text, "intent_class": i.intent_class, "status": i.status}
            for i in self._intentions
        ]}

    def from_dict(self, state: Optional[dict]) -> None:
        state = state or {}
        self._intentions = [
            Intention(text=d["text"], intent_class=d.get("intent_class", ""),
                      status=d.get("status", "OPEN"))
            for d in state.get("intentions", [])
        ]

    def reset(self) -> None:
        self._intentions = []


class _IntentLike:
    """Structural hint for the subset of IntentResult that IntentionTracker reads."""
    intent_class: str
    goal: Optional[str]
    entities_concepts: list[str]
    entities_objects: list[str]


def _infer_intention(intent: "_IntentLike") -> Optional[str]:
    """Derive an intention text when `goal` is not set (intent_class + first entity)."""
    action_map = {
        "ACTION_REQUEST": "perform task",
        "INFORMATION_REQUEST": "get information",
        "CREATIVE_TASK": "create content",
    }
    base = action_map.get(intent.intent_class, "complete request")
    context = (intent.entities_concepts[:1] or intent.entities_objects[:1] or [])
    return f"{base}: {context[0]}" if context else base
