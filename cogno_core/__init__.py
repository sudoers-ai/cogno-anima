"""
cogno_core — The cognitive core intelligence pipeline library.
"""

from cogno_core.types import (
    StageMetrics,
    NoumenoResult,
    IntentResult,
    IdResult,
    DriftMetrics,
    PipelineContext,
)
from cogno_core.errors import CognoError, StageParseError
from cogno_core.llm import (
    LLMBackend,
    Embedder,
    OllamaBackend,
    OllamaEmbedder,
    CachingEmbedder,
    EmbeddingUsage,
)
from cogno_core.routing import GoalManager, AttentionFilter, IntentionTracker
from cogno_core.stages.base import BaseStage
from cogno_core.stages.noumeno import Noumeno
from cogno_core.stages.ner import IntentAnalyzer
from cogno_core.stages.drift import DriftCalculator, DriftThresholds

__all__ = [
    "StageMetrics",
    "NoumenoResult",
    "IntentResult",
    "IdResult",
    "DriftMetrics",
    "PipelineContext",
    "LLMBackend",
    "Embedder",
    "OllamaBackend",
    "OllamaEmbedder",
    "CachingEmbedder",
    "EmbeddingUsage",
    "GoalManager",
    "AttentionFilter",
    "IntentionTracker",
    "BaseStage",
    "Noumeno",
    "IntentAnalyzer",
    "DriftCalculator",
    "DriftThresholds",
    "CognoError",
    "StageParseError",
]
