"""
cogno_core — The cognitive core intelligence pipeline library.
"""

from cogno_core.types import (
    StageMetrics,
    NoumenoResult,
    IntentResult,
    IdResult,
    ToolResult,
    ToolExecution,
    EgoStep,
    EgoResult,
    DriftMetrics,
    PipelineContext,
)
from cogno_core.errors import (
    CognoError,
    StageParseError,
    ToolExecutionError,
    MCPDispatchError,
)
from cogno_core.llm import (
    LLMBackend,
    ToolCallingBackend,
    Embedder,
    OllamaBackend,
    OllamaEmbedder,
    CachingEmbedder,
    EmbeddingUsage,
    parse_tool_calls_from_text,
)
from cogno_core.tools import ToolDispatcher
from cogno_core.routing import GoalManager, AttentionFilter, IntentionTracker
from cogno_core.stages.base import BaseStage
from cogno_core.stages.noumeno import Noumeno
from cogno_core.stages.ner import IntentAnalyzer
from cogno_core.stages.id import IDStage
from cogno_core.stages.ego import EgoStage
from cogno_core.stages.drift import DriftCalculator, DriftThresholds

__all__ = [
    "StageMetrics",
    "NoumenoResult",
    "IntentResult",
    "IdResult",
    "ToolResult",
    "ToolExecution",
    "EgoStep",
    "EgoResult",
    "DriftMetrics",
    "PipelineContext",
    "LLMBackend",
    "ToolCallingBackend",
    "Embedder",
    "OllamaBackend",
    "OllamaEmbedder",
    "CachingEmbedder",
    "EmbeddingUsage",
    "parse_tool_calls_from_text",
    "ToolDispatcher",
    "GoalManager",
    "AttentionFilter",
    "IntentionTracker",
    "BaseStage",
    "Noumeno",
    "IntentAnalyzer",
    "IDStage",
    "EgoStage",
    "DriftCalculator",
    "DriftThresholds",
    "CognoError",
    "StageParseError",
    "ToolExecutionError",
    "MCPDispatchError",
]
