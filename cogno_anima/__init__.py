"""
cogno_anima — The cognitive core intelligence pipeline library.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

try:
    __version__ = _dist_version("cogno-anima")
except PackageNotFoundError:  # source tree without an installed dist (e.g. vendored checkout)
    __version__ = "0.0.0"


from cogno_anima.types import (
    committed_this_turn,
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
from cogno_anima.vocab import sanitize_voice_traits

__all__ = [
    "sanitize_voice_traits",
    "committed_this_turn",
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
