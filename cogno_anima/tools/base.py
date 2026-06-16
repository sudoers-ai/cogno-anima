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
