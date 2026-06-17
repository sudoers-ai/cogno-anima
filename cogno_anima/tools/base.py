"""
cogno_anima.tools.base — the host-facing tool execution contract.

The EGO is a pure executor: it DECIDES which tool to call, but never touches the
DB/MCP/API itself ("EGO = brain, dispatcher = hands"). The host implements this
protocol and injects it into ``EgoStage.process`` — that is the only seam through
which side effects happen, which is what lets the host wrap them in a
transaction / write-behind buffer / outbox without the core knowing.

What stays OUT of this protocol (host concerns, by design):
  * which tools a persona/identity may use (RBAC, persona ceiling, MCP module);
  * narrowing a large catalog to the most relevant tools (host may use an
    embedding retriever before building ``tools_schema``);
  * atomicity / rollback / outbox (the host wraps ``execute`` however it likes).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from cogno_anima.types import ToolResult


@runtime_checkable
class ToolDispatcher(Protocol):
    """Host-injected tool registry + executor."""

    def tools_schema(self) -> list[dict]:
        """OpenAI-format schemas for the FINAL tool set the EGO may use.

        Already filtered (RBAC/persona) and, if the host wants, narrowed
        (top-K by relevance) — the EGO trusts this list as-is and offers it to
        the model.
        """
        ...

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        """Run one tool against the host's DB/MCP/API.

        Contract:
          * recoverable failure (bad args, business rejection) → return
            ``ToolResult(ok=False, error=...)`` so the EGO feeds it back and the
            model self-corrects;
          * fatal failure (infra: connection/auth/timeout) → ``raise
            MCPDispatchError`` so the EGO propagates to the host.
        """
        ...


@runtime_checkable
class ToolPolicyDispatcher(ToolDispatcher, Protocol):
    """Optional extension: a dispatcher that classifies its tools.

    Kept SEPARATE from ``ToolDispatcher`` (mirrors the ``ToolCallingBackend``
    pattern): a host that does not care about read-only / confirmation gating
    implements only ``ToolDispatcher`` and both gates degrade safely. The EGO
    probes ``isinstance(dispatcher, ToolPolicyDispatcher)``.

    The core NEVER hardcodes which tools mutate or are destructive — only the
    host knows. Two orthogonal axes:

      * ``is_mutating`` drives the **read-only mask** (when the host sets
        ``ctx.metadata["ego_readonly"]`` because the user was tentative, the EGO
        offers only non-mutating tools). Fail-safe: without this protocol the EGO
        masks ALL tools in read-only mode (proposes via draft, touches nothing).
      * ``requires_confirmation`` drives the **confirmation gate** (the EGO holds
        a destructive call and signals ``EgoResult.pending_confirmation`` until
        the host sets ``ctx.metadata["ego_confirmed"]``). Opt-in: without this
        protocol the core cannot know a tool is destructive, so no gate fires.
    """

    def is_mutating(self, name: str) -> bool:
        """True if the tool writes / causes a side effect (vs. a pure read)."""
        ...

    def requires_confirmation(self, name: str) -> bool:
        """True if the tool is destructive/aggressive and must be confirmed first."""
        ...
