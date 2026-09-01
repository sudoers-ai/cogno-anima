"""
cogno_anima.tools.composite — merge several ToolDispatchers into one.

The EGO knows only a single :class:`ToolDispatcher`, but a persona may draw tools
from several sources at once — in-process skills (``cogno-cortex``), an MCP client
(``cogno-mcp`` / host), the host's own native functions. The host resolves each of
the persona's modules to a source dispatcher and merges them here; the EGO still
sees one flat tool set.

``tools_schema()`` is the union of the sources' schemas (first source wins on a
name collision); ``execute(name, ...)`` routes the call to the dispatcher that owns
that name. The composite also satisfies :class:`ToolPolicyDispatcher`, delegating
``is_mutating``/``requires_confirmation`` to the owning source — and, for a source
that does *not* implement the policy protocol, falling back to the EGO's fail-safe
defaults (treat as mutating so read-only mode masks it; no confirmation gate, which
is opt-in). Build one per turn (or whenever the source set changes).
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from cogno_anima.tools.base import ToolDispatcher, ToolPolicyDispatcher
from cogno_anima.types import ToolResult

logger = logging.getLogger(__name__)


class CompositeDispatcher:
    """A :class:`ToolDispatcher` (+ :class:`ToolPolicyDispatcher`) over many sources."""

    def __init__(self, sources: Sequence[ToolDispatcher]) -> None:
        self._sources = list(sources)
        self._cache: Optional[tuple[list[dict], dict[str, ToolDispatcher]]] = None

    def _resolve(self) -> tuple[list[dict], dict[str, ToolDispatcher]]:
        """Build (and cache) the union schema list + the name→source index."""
        if self._cache is None:
            schemas: list[dict] = []
            index: dict[str, ToolDispatcher] = {}
            for src in self._sources:
                for schema in src.tools_schema():
                    name = (schema.get("function") or {}).get("name")
                    if not name:
                        continue  # un-callable without a name — drop it
                    if name in index:
                        logger.warning(
                            "event=tool_name_collision name=%s (first source wins)", name)
                        continue
                    index[name] = src
                    schemas.append(schema)
            self._cache = (schemas, index)
        return self._cache

    def tools_schema(self) -> list[dict]:
        return list(self._resolve()[0])

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        src = self._resolve()[1].get(name)
        if src is None:
            # unknown name → recoverable, so the EGO feeds it back and self-corrects
            return ToolResult(output="", ok=False, error=f"unknown tool: {name}")
        return await src.execute(name, arguments)

    # ── ToolPolicyDispatcher ──────────────────────────────────────────────
    def is_mutating(self, name: str) -> bool:
        src = self._resolve()[1].get(name)
        if isinstance(src, ToolPolicyDispatcher):
            return src.is_mutating(name)
        # conservative: an un-classified source is assumed to mutate, so the
        # read-only mask hides it (matches the EGO's "no policy → mask all").
        return True

    def requires_confirmation(self, name: str) -> bool:
        src = self._resolve()[1].get(name)
        if isinstance(src, ToolPolicyDispatcher):
            return src.requires_confirmation(name)
        # the confirmation gate is opt-in; an un-classified source did not opt in.
        return False

    def source_requires_confirmation(self, name: str) -> bool:
        """Did the SOURCE ITSELF declare this tool destructive — without any blanket rule a
        host gate may have layered on top?

        `requires_confirmation` flattens two different facts ("the source marked this
        destructive" and "this is a write and nobody exempted it"), and a caller above the
        router may need them apart. A host that wraps each source in a gate applying a blanket
        "every unexempted write confirms" rule exposes this finer predicate on the wrapper; the
        router has to FORWARD it, and until now it did not.

        **Why this method exists, measured 2026-09-01 end to end.** A contact who wrote
        «prefiro não falar com robô, me passa pra uma pessoa» was answered «Só pra confirmar:
        executo human handoff. Posso seguir?» — and no escalation happened. The host's
        anti-holding floor, which exists precisely to stop that, needs this predicate to tell a
        source's own verdict from the blanket rule; reaching it through the wrappers'
        `__getattr__`, it hit THIS class — which defines neither the method nor `__getattr__` —
        and fell back to its conservative answer, which HOLDS. The floor's own docstring called
        that fallback "unreachable in production: every source is wrapped". Every source IS
        wrapped; the wrapping just sits INSIDE this router, and the router was a wall. True in
        the letter, false in the effect.

        **A generic `__getattr__` is deliberately NOT the fix here.** That is the right answer
        for a WRAPPER — one layer over one source — and the wrong one for a ROUTER over many:
        it would not know which source to forward to and would have to pick one. Explicit and
        routed, like the two policy methods above it.

        The fallbacks, in order, and each one is a different question:
          * the source exposes the finer predicate → ask it (the production shape);
          * the source is a policy source WITHOUT it → its own `requires_confirmation` IS the
            source verdict, because there is no blanket rule layered in front of it. Answering
            ``False`` here would let a caller waive a tool the source really did mark
            destructive — the one thing the floor above must never do;
          * no policy at all → it declared nothing, so nothing was declared destructive. Such a
            source can never reach the gate anyway (``requires_confirmation`` is ``False`` for
            it), so this branch decides nothing it could get wrong.
        """
        src = self._resolve()[1].get(name)
        fn = getattr(src, "source_requires_confirmation", None)
        if fn is not None:
            return bool(fn(name))
        if isinstance(src, ToolPolicyDispatcher):
            return src.requires_confirmation(name)
        return False
