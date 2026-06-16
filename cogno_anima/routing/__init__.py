"""
cogno_anima.routing — pure continuity/attention helpers for the ID stage.

These are dependency-light, I/O-free building blocks consumed by the ID stage
(`cogno_anima.stages.id`). They hold no infrastructure or business concepts
(personas, MCP modules, skills) — the embedder is reached only through an
injected async `similarity_fn`.
"""

from cogno_anima.routing.goal import GoalManager
from cogno_anima.routing.attention import AttentionFilter
from cogno_anima.routing.intention import IntentionTracker, Intention

__all__ = [
    "GoalManager",
    "AttentionFilter",
    "IntentionTracker",
    "Intention",
]
