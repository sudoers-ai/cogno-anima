"""
cogno_anima.llm.fallback — try a chain of backends, in order, until one succeeds.

A composable resilience primitive (like ``CachingEmbedder``): wrap an ordered
list of backends; each call tries them in turn, returning the first success and
re-raising the last error if all fail. Backends that lack native FC are skipped
for ``chat_with_tools``.

Deliberately infra-agnostic — the parent's Redis-backed circuit breaker + health
probe threads are a HOST concern and are NOT ported. A host that wants
circuit-breaking can wrap this. Pairs with the adapted backends that **raise** on
failure (so this can catch and fail over).
"""

from __future__ import annotations

import logging

from cogno_anima.llm.base import LLMBackend

logger = logging.getLogger("cogno_anima.llm.fallback")


class FallbackBackend:
    """Ordered failover across backends; first success wins, last error propagates."""

    def __init__(self, backends: list[LLMBackend]) -> None:
        if not backends:
            raise ValueError("FallbackBackend requires at least one backend")
        self.backends = backends
        self._last_successful: LLMBackend | None = None

    async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
        last_exc: Exception | None = None
        for backend in self.backends:
            try:
                result = await backend.generate(system, prompt)
            except Exception as exc:  # noqa: BLE001 — try the next backend
                last_exc = exc
                logger.warning("%s.generate failed (%s) — failing over",
                               backend.__class__.__name__, exc)
                continue
            self._last_successful = backend
            return result
        assert last_exc is not None
        raise last_exc

    async def chat_with_tools(self, messages, tools, tool_choice=None):
        last_exc: Exception | None = None
        tried = False
        for backend in self.backends:
            if not getattr(backend, "supports_native_tools", lambda: False)():
                continue
            tried = True
            try:
                result = await backend.chat_with_tools(messages, tools, tool_choice)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning("%s.chat_with_tools failed (%s) — failing over",
                               backend.__class__.__name__, exc)
                continue
            self._last_successful = backend
            return result
        if not tried:
            raise RuntimeError("FallbackBackend: no backend in the chain supports native tools")
        assert last_exc is not None
        raise last_exc

    def supports_native_tools(self) -> bool:
        return any(getattr(b, "supports_native_tools", lambda: False)() for b in self.backends)

    @property
    def model(self) -> str:
        active = self._last_successful or self.backends[0]
        return getattr(active, "model", "unknown")
