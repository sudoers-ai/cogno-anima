"""
cogno_anima.tools.id_provenance — a guarded write must name an id READ this turn.

The finding this exists for: an executor acted on record ids recalled from an OLD
session's injected memories instead of the ids the SAME turn's read had just returned.
Every write landed as an honest no-op ("was ALREADY CONFIRMED"), the reply said so, and
the user's REAL pending records stayed untouched. A docstring plea ("ids MUST come from a
read in THIS turn") does not beat a seductive memory; this guard makes the discipline
DETERMINISTIC:

    a guarded tool call is refused (recoverably) unless its id argument literally
    appeared in a successful tool result EARLIER IN THE SAME TURN.

The refusal text tells the model to read first, so the recoverable-error contract
(``ToolResult(ok=False)`` is fed back and the model self-corrects) turns the guard into
exactly the read→write flow we want, at the cost of one extra cheap read. Deliberately NOT
satisfied by ids in the injected context/memories — that is precisely the stale source
being fenced off.

**The guarded map is injected, never defaulted here.** Which tools take an id, what that
argument is called, and which read legitimises it are facts about the CALLER's tool
catalog; the core would be guessing. The read tool is part of the declaration because it
is part of the REFUSAL — naming a tool this persona does not have turns a self-correction
into a loop.

Scope notes (the caller's, and worth stating because both are load-bearing):

* **Per turn.** Build a FRESH instance each turn, so provenance never leaks across turns —
  that would recreate the staleness this fences off.
* **A confirmation turn should not be wrapped.** A call held by the confirmation gate was
  proposed from a read on the PROPOSE turn and approved by the user; demanding a re-read
  on the bare "yes" turn would break the confirmation UX.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from cogno_anima.types import ToolResult

logger = logging.getLogger(__name__)

__all__ = ["IdProvenanceDispatcher"]


class IdProvenanceDispatcher:
    """Refuse a guarded write whose id was not returned by a read earlier this turn.

    ``guarded`` maps ``tool name → (id argument, the READ that legitimises it)``. A bare
    string is accepted for the id argument alone; the read tool then has to be supplied by
    the caller in tuple form, so a map written as ``{"cancel_x": "x_id"}`` keeps working
    against a caller that never needed the hint.
    """

    #: What a bare-string entry means when the caller named no read tool. Deliberately
    #: neutral: the refusal then says "read the relevant list first" instead of naming a
    #: tool the persona may not have.
    DEFAULT_READ_HINT = "the corresponding list tool"

    def __init__(self, inner: Any, *,
                 guarded: "Mapping[str, str | tuple[str, str]]") -> None:
        self._inner = inner
        self._guarded: dict[str, tuple[str, str]] = {
            k: (v if isinstance(v, tuple) else (v, self.DEFAULT_READ_HINT))
            for k, v in dict(guarded).items()}
        self._seen: list[str] = []          # ok result strings, THIS turn only
        self.refusals = 0                   # observability: refusals absorbed this turn

    def tools_schema(self) -> list[dict]:
        return self._inner.tools_schema()

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        entry = self._guarded.get(name)
        if entry:
            arg, read_tool = entry
            wanted = str(arguments.get(arg, "") or "").strip()
            if wanted and not any(wanted in r for r in self._seen):
                self.refusals += 1
                logger.warning("event=id_provenance_refused tool=%s %s=%s", name, arg, wanted)
                return ToolResult(output="", ok=False, error=(
                    f"'{arg}={wanted}' was not returned by any read in THIS turn — the id may "
                    f"be stale (from memory or an earlier conversation). Call "
                    f"{read_tool} first and use ONLY ids from its result."))
        result = await self._inner.execute(name, arguments)
        if result.ok and result.output:
            self._seen.append(str(result.output))
        return result

    # ── ToolPolicyDispatcher passthrough (the EGO gates keep working) ─────────────────
    #
    # Declared unconditionally, and answering the EGO's own fail-safe defaults when the
    # inner has no policy: mutating (so gate A's read-only mask hides it) and not
    # destructive (gate B is opt-in). That is the same choice
    # :class:`~cogno_anima.tools.composite.CompositeDispatcher` makes, and the opposite of
    # the recording wrappers, which bind these conditionally — the difference is that those
    # add no verdict of their own, so a probe answering True over them would be a claim
    # about a source they merely observe.
    def is_mutating(self, name: str) -> bool:
        inner = getattr(self._inner, "is_mutating", None)
        return inner(name) if inner else True

    def requires_confirmation(self, name: str) -> bool:
        inner = getattr(self._inner, "requires_confirmation", None)
        return inner(name) if inner else False

    # ── delegation for anything this wrapper does not mediate ────────────────────────
    # A wrapper that lists its methods LOSES every method it did not think of, and does it
    # in silence. A caller reaching DOWN the chain for a finer policy predicate
    # (``source_requires_confirmation`` — did the SOURCE itself call this destructive, as
    # opposed to a blanket "every unexempted write confirms" rule a gate layered on top)
    # found three wrappers in a row swallowing it, and fell back to its conservative answer,
    # which HOLDS: a contact asking to be transferred to a person was answered "shall I
    # proceed?" and never transferred.
    #
    # Forwarding by name fixes the CLASS, not one method: the next policy predicate
    # traverses without anyone having to remember this file. What this wrapper DOES mediate
    # stays mediated — an explicitly defined method always wins over ``__getattr__``.
    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)
