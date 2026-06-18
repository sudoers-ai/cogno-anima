"""Typed errors for the cognitive pipeline.

Stages propagate errors to the caller/orchestrator (no internal fallback). A
typed error makes that contract explicit and lets the host distinguish a
malformed-LLM-output failure from other exceptions when deciding to retry,
swap models, or abort.
"""

from __future__ import annotations

# The auth/transport errors moved to cogno-synapse (raised by the backends/
# factory). Re-exported here so `from cogno_anima.errors import InvalidAPIKeyError`
# keeps working. NOTE: they now subclass cogno_synapse.SynapseError, not CognoError.
from cogno_synapse.errors import (  # noqa: F401
    InvalidAPIKeyError as InvalidAPIKeyError,
    MissingAPIKeyError as MissingAPIKeyError,
)


class CognoError(Exception):
    """Base class for all cogno_anima errors."""


class StageParseError(CognoError, ValueError):
    """An LLM response could not be parsed into a stage's JSON contract.

    Subclasses ValueError so existing `except ValueError` handlers still catch it.
    The original decoding exception is chained as ``__cause__`` and the raw text
    is kept on ``raw`` for logging/diagnostics.
    """

    def __init__(self, stage: str, raw: str, original: Exception) -> None:
        self.stage = stage
        self.raw = raw
        self.original = original
        preview = (raw or "").strip().replace("\n", " ")[:120]
        super().__init__(
            f"{stage}: could not parse LLM JSON response ({original}); got: {preview!r}"
        )


class ToolExecutionError(CognoError):
    """A tool failed during EGO execution in a way the loop cannot recover from.

    Raised by the EGO when an *unexpected* exception escapes the host's
    ``ToolDispatcher.execute`` — i.e. one the host did not classify as a
    recoverable business/validation failure (those are returned as
    ``ToolResult(ok=False, error=...)`` and fed back to the model instead).
    The EGO never guesses recoverability: anything that escapes is wrapped here
    and propagated so the host decides (retry, swap, abort). Carries the tool
    name + arguments and chains the original exception as ``__cause__``.
    """

    def __init__(self, tool: str, arguments: dict, original: Exception) -> None:
        self.tool = tool
        self.arguments = arguments
        self.original = original
        super().__init__(f"tool {tool!r} failed: {original}")
        self.__cause__ = original


class MCPDispatchError(ToolExecutionError):
    """The dispatcher could not dispatch the tool — infrastructure failure
    (MCP server down, connection refused, auth rejected, timeout).

    A *deliberate fatal* signal the host raises from ``execute``: retrying with
    different arguments is pointless, so the EGO propagates it instead of
    feeding it back (a model cannot fix a dead connection, and pushing it to
    "complete the task" risks hallucinated success).
    """
