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
from cogno_core.stages.base import BaseStage
from cogno_core.stages.noumeno import Noumeno
from cogno_core.stages.ner import IntentAnalyzer
from cogno_core.stages.drift import DriftCalculator

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
    "BaseStage",
    "Noumeno",
    "IntentAnalyzer",
    "DriftCalculator",
    "CognoError",
    "StageParseError",
]
