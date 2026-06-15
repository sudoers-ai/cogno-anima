"""
cogno_core.routing — pure continuity/attention helpers for the ID stage.

These are dependency-light, I/O-free building blocks consumed by the ID stage
(`cogno_core.stages.id`). They hold no infrastructure or business concepts
(personas, MCP modules, skills) — the embedder is reached only through an
injected async `similarity_fn`.
"""

from cogno_core.routing.goal import GoalManager
from cogno_core.routing.attention import AttentionFilter
from cogno_core.routing.intention import IntentionTracker, Intention

__all__ = [
    "GoalManager",
    "AttentionFilter",
    "IntentionTracker",
    "Intention",
]
