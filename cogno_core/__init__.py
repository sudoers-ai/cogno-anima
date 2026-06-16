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
    ScopeCheckResult,
    SuperegoResult,
    DriftMetrics,
    PipelineContext,
)
from cogno_core.errors import (
    CognoError,
    StageParseError,
    ToolExecutionError,
    MCPDispatchError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
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
    OpenAIBackend,
    AnthropicBackend,
    GroqBackend,
    GeminiBackend,
    BedrockBackend,
    FallbackBackend,
    create_backend,
)
from cogno_core.tools import ToolDispatcher
from cogno_core.routing import GoalManager, AttentionFilter, IntentionTracker
from cogno_core.stages.base import BaseStage
from cogno_core.stages.noumeno import Noumeno
from cogno_core.stages.ner import IntentAnalyzer
from cogno_core.stages.id import IDStage
from cogno_core.stages.ego import EgoStage
from cogno_core.stages.superego import SuperegoStage
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
