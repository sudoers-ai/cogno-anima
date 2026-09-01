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

    **IF YOU WRAP A DISPATCHER, DECLARE THESE METHODS ON THE WRAPPER.** Delegating
    them through ``__getattr__`` compiles, passes every call at runtime, and makes
    the ``isinstance`` probe above answer **False** on Python 3.12+ — so the gates
    silently do not exist for your wrapper. Measured on both interpreters, a wrapper
    over a source that DOES implement the policy::

        py3.10   __getattr__ -> isinstance True      (the probe walks the fallback)
        py3.12   __getattr__ -> isinstance False     (static resolution; no fallback)

    3.12 verifies ``runtime_checkable`` protocols with ``inspect.getattr_static``,
    which reads ``type(obj).__mro__`` and ``obj.__dict__`` and **never calls**
    ``__getattr__``. The failure is silent in the worst direction: the read-only mask
    over-masks (it fails safe), but the confirmation gate simply **never fires** — a
    destructive tool executes without the user's confirmation, on every turn, with no
    error anywhere.

    And a wrapper must not simply DECLARE the methods either: answering on behalf of a
    source that has no policy makes the probe true for a source that declared nothing,
    which arms a gate over guesses. **Bind them conditionally, as instance attributes**
    — ``getattr_static`` reads ``obj.__dict__``, so this satisfies both interpreters and
    keeps the probe honest about the source underneath::

        class MyWrapper:
            def __init__(self, inner):
                self._inner = inner
                for name in ("is_mutating", "requires_confirmation"):
                    if hasattr(inner, name):
                        setattr(self, name, getattr(inner, name))

    The same applies to every ``runtime_checkable`` probe in this codebase — notably
    ``ToolCallingBackend`` (``cogno_synapse``), where the same mistake degrades a
    native function-calling backend to the text path with nothing said. ``pytest
    tests/unit/test_protocol_probe_contract.py`` pins the rule with the mechanism 3.12
    uses, so it holds on any interpreter.
    """

    def is_mutating(self, name: str) -> bool:
        """True if the tool writes / causes a side effect (vs. a pure read)."""
        ...

    def requires_confirmation(self, name: str) -> bool:
        """True if the tool is destructive/aggressive and must be confirmed first."""
        ...
