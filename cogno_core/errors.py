"""Typed errors for the cognitive pipeline.

Stages propagate errors to the caller/orchestrator (no internal fallback). A
typed error makes that contract explicit and lets the host distinguish a
malformed-LLM-output failure from other exceptions when deciding to retry,
swap models, or abort.
"""

from __future__ import annotations


class CognoError(Exception):
    """Base class for all cogno_core errors."""


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
