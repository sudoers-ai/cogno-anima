"""
cogno_anima — The cognitive core intelligence pipeline library.
"""

from cogno_anima.types import (
    StageMetrics,
    NoumenoResult,
    IntentResult,
    IdResult,
    ToolResult,
    ToolExecution,
    EgoStep,
    EgoResult,
    ScopeCheckResult,
    SuperegoResult,
    DriftMetrics,
    PipelineContext,
)
from cogno_anima.errors import (
    CognoError,
    StageParseError,
    ToolExecutionError,
    MCPDispatchError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
)
from cogno_synapse import (
    LLMBackend,
    ToolCallingBackend,
    Embedder,
    OllamaBackend,
    OllamaEmbedder,
    CachingEmbedder,
    EmbeddingUsage,
    parse_tool_calls_from_text,
    OpenAIBackend,
    AnthropicBackend,
    GroqBackend,
    GeminiBackend,
    BedrockBackend,
    FallbackBackend,
    create_backend,
)
from cogno_anima.tools import ToolDispatcher, CompositeDispatcher
from cogno_anima.routing import GoalManager, AttentionFilter, IntentionTracker
from cogno_anima.stages.base import BaseStage
from cogno_anima.stages.noumeno import Noumeno
from cogno_anima.stages.ner import IntentAnalyzer
from cogno_anima.stages.id import IDStage
from cogno_anima.stages.ego import EgoStage
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.stages.drift import DriftCalculator, DriftThresholds

__all__ = [
    "StageMetrics",
    "NoumenoResult",
    "IntentResult",
    "IdResult",
    "ToolResult",
    "ToolExecution",
    "EgoStep",
    "EgoResult",
    "ScopeCheckResult",
    "SuperegoResult",
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
    "OpenAIBackend",
    "AnthropicBackend",
    "GroqBackend",
    "GeminiBackend",
    "BedrockBackend",
    "FallbackBackend",
    "create_backend",
    "ToolDispatcher",
    "CompositeDispatcher",
    "GoalManager",
    "AttentionFilter",
    "IntentionTracker",
    "BaseStage",
    "Noumeno",
    "IntentAnalyzer",
    "IDStage",
    "EgoStage",
    "SuperegoStage",
    "DriftCalculator",
    "DriftThresholds",
    "CognoError",
    "StageParseError",
    "ToolExecutionError",
    "MCPDispatchError",
    "InvalidAPIKeyError",
    "MissingAPIKeyError",
]
