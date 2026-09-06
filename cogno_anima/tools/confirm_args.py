"""
cogno_anima.tools.confirm_args — the RETURN half of the EGO's third confirmation gate.

Gate C is a tool saying, about THIS call and what it just READ, *"I did not commit — ask
first"* (:class:`~cogno_anima.types.ToolResult` with ``needs_confirmation=True``). The host
then asks, the user says yes, and the turn is replayed with ``ego_confirmed``. The replay
is where the mechanism was open at both ends: it re-sent the SAME arguments, so from the
tool's side nothing had changed and it asked again — and ``EgoStage._refuse_if_still_asking``
turned that into a loud failure, correctly. The confirmation never reached the tool.

**What travels is the tool's own argument, never one this layer invented.** A yes/no is not
always what a tool needs: a ledger removal may confirm with ``confirm_tx_id=<row id>``
precisely so that a row landing between the proposal and the go-ahead cannot move the
target. Only the tool knows that. So the tool NAMES what it needs, and the caller decides
only **whether**: the user confirmed this call, or did not.

**The named arguments are read DUCK-TYPED, off a ``confirm_arguments`` attribute that
:class:`~cogno_anima.types.ToolResult` deliberately does not declare.** What a skill needs
in order to commit is the skill's business, not the pipeline's, so the carrier belongs to
whichever bridge speaks that skill's protocol — ``cogno_mcp.MCPToolResult`` is one such
subclass. A result without the attribute simply records nothing, which is the same answer
as a tool that asked for nothing.

**Why a pre-placed mutable and not a key stamped from inside.** Identical to
:mod:`cogno_anima.tools.commit_sink`, for the identical reason: a turn's metadata is
copied on the way in, shallowly, so a pre-placed object is the same object all the way
down while a new key added from inside never reaches the caller — and fails silently.

**Why it is captured at the dispatcher and not read off the EGO's trace.**
:class:`~cogno_anima.types.ToolExecution` carries tool / arguments / result and nothing
else, so what the tool asked for is dropped at the stage boundary. The last place the
result is still whole is the dispatcher chain, and the capture has to happen there or not
at all.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from cogno_anima.tools.binding import bind_delegated

log = logging.getLogger(__name__)

__all__ = [
    "MAX_CONFIRM_ARGS",
    "new_confirm_args",
    "call_key",
    "held_calls",
    "ConfirmArgumentRecordingDispatcher",
]

# A tool asking for an unbounded blob would put it in the caller's session state and then
# back into a live call. Bounded here, at the only door, rather than trusted per source.
MAX_CONFIRM_ARGS = 8


def new_confirm_args() -> "dict[str, dict]":
    """A fresh recorder for ONE turn. Pre-place it in the caller's metadata *before* the
    turn starts (see :func:`~cogno_anima.tools.commit_sink.new_sink` for why)."""
    return {}


def call_key(tool: str, arguments: Any) -> str:
    """The identity of ONE call, so a step with two calls to the same tool keeps them apart.

    The recorder sits outermost and the EGO records the same ``arguments`` it passed in, so
    the two sides agree by construction. ``default=str`` because a key must never raise: an
    unserialisable argument should make this call unmatchable, not break the turn.
    """
    try:
        return json.dumps([str(tool), arguments], sort_keys=True, default=str)
    except (TypeError, ValueError):    # pragma: no cover - default=str covers ~everything
        return json.dumps([str(tool), repr(arguments)])


class ConfirmArgumentRecordingDispatcher:
    """Records what a TOOL asked to be given back, at the moment it asks.

    Wraps, never replaces: everything else delegates, so a read-only mask, both
    confirmation gates and any narrowing underneath keep working exactly as before. It
    observes; it changes no argument, no surface and no verdict.

    Only a result that is actually ASKING is recorded. Arguments without a question are not
    a pending confirmation, and treating them as one would let a source pre-load a replay
    for a call nobody held.
    """

    # No ``__slots__``, and that is load-bearing rather than an oversight — the same reason
    # :class:`~cogno_anima.tools.commit_sink.CommitRecordingDispatcher` states at length:
    # the policy members are bound onto the INSTANCE, and naming them in ``__slots__`` puts
    # a descriptor on the CLASS that satisfies ``isinstance(x, ToolPolicyDispatcher)`` even
    # when the inner has no policy at all. A safety wrapper must not be able to disarm the
    # gate-A fail-safe by existing.

    def __init__(self, inner: Any, sink: "dict[str, dict]") -> None:
        self._inner = inner
        self._sink = sink
        bind_delegated(self, inner, "is_mutating", "requires_confirmation")

    def tools_schema(self) -> "list[dict]":
        return self._inner.tools_schema()

    async def execute(self, name: str, arguments: dict) -> Any:
        result = await self._inner.execute(name, arguments)
        if getattr(result, "needs_confirmation", False):
            asked = getattr(result, "confirm_arguments", None)
            if isinstance(asked, dict) and asked:
                if len(asked) > MAX_CONFIRM_ARGS:
                    log.warning("event=confirm_args_too_many tool=%s n=%d — dropped",
                                name, len(asked))
                else:
                    self._sink[call_key(name, arguments)] = dict(asked)
        return result

    def __getattr__(self, item: str) -> Any:
        """Everything else IS the inner's — including what nobody thought to forward.

        A wrapper that lists its methods loses every method it did not name, in silence: a
        counter on the guard beneath it reads 0, and a caller reaching down for a finer
        policy predicate (``source_requires_confirmation``) gets the flattened answer
        instead. Both have already been swallowed once by wrappers that forwarded by hand.
        The policy members travel the other way, through
        :func:`~cogno_anima.tools.binding.bind_delegated`.
        """
        return getattr(self._inner, item)


def held_calls(pending: Any, *, confirm_arguments: Any) -> "list[dict]":
    """The turn's held calls, in the shape a confirmation turn replays.

    ``pending`` is what the EGO reports as
    :attr:`~cogno_anima.types.EgoResult.pending_confirmation`; each entry contributes
    ``{"tool": ..., "arguments": ...}`` with whatever the matching tool asked for merged in.

    ``confirm_arguments`` is keyword-only and has NO DEFAULT, deliberately. This is the one
    place where forgetting the return trip is invisible: the replay still runs, the tool
    still answers, and the only symptom is a turn that reports a failure the user cannot act
    on — weeks later, in production, on somebody else's vertical. A parameter without a
    default is the cheapest net there is against that: a call site that has nothing to pass
    has to SAY so, in code, where the next reader can see it.

    A recorded argument is only ever ADDED. Overwriting one the model chose would let a
    source change what a confirmed call does to something other than what the user was
    shown — the proposal named a row, and the commit must be that row.
    """
    lookup = confirm_arguments if isinstance(confirm_arguments, dict) else {}
    out: list[dict] = []
    for call in (pending or []):
        tool = str(getattr(call, "tool", "") or "")
        args = dict(getattr(call, "arguments", None) or {})
        for key, value in (lookup.get(call_key(tool, args)) or {}).items():
            if key not in args:
                args[key] = value
        out.append({"tool": tool, "arguments": args})
    return out
