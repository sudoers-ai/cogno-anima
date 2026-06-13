"""
cogno_core — The cognitive core intelligence pipeline library.
"""

from cogno_core.types import (
    StageMetrics,
    NoumenoResult,
    IntentResult,
    DriftMetrics,
    PipelineContext,
)
from cogno_core.llm import LLMBackend, Embedder, OllamaBackend, OllamaEmbedder
from cogno_core.stages.base import BaseStage
from cogno_core.stages.noumeno import Noumeno
from cogno_core.stages.ner import IntentAnalyzer
from cogno_core.stages.drift import DriftCalculator

__all__ = [
    "StageMetrics",
    "NoumenoResult",
    "IntentResult",
    "DriftMetrics",
    "PipelineContext",
    "LLMBackend",
    "Embedder",
    "OllamaBackend",
    "OllamaEmbedder",
    "BaseStage",
    "Noumeno",
    "IntentAnalyzer",
    "DriftCalculator",
]
