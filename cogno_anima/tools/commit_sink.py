"""
cogno_anima.tools.commit_sink — record a turn's COMMIT so the fact can outlive the turn.

``committed_this_turn`` (``cogno_anima.types``) answers *"did this turn run a mutating
tool successfully?"* from the trace that rides on the :class:`PipelineContext`. There is
one path where that trace dies mid-turn: a host that retries a turn on another model (or
any orchestrator that catches an exception and starts over). Attempt 1 commits an ordinary
write, a LATER stage raises, and the exception takes the context — and its execution
record — with it. The retry is a fresh turn in which the write is nowhere, and every
consumer that asks "did this turn write?" answers no about a turn that wrote.

This module is the half that survives that: a recorder placed IN THE DISPATCHER CHAIN,
writing into a container the CALLER owns.

**The channel is a mutable the CALLER pre-places, never a key stamped from inside.**
A turn's metadata is typically copied on the way in (``dict(meta)``, then
``ctx.metadata.update(...)``). Those copies are SHALLOW, so:

* a pre-placed object is the SAME object all the way down — a mutation from inside the
  turn is visible to the caller afterwards, exception or not;
* a NEW key added from inside never reaches the caller, and fails **silently**.

**Why the dispatcher and not a post-execution hook.** A hook that fires once the turn's
execution is judged runs AFTER the executor loop returns — and the branch this mechanism
exists for (attempt 1 writes, then something later raises) is precisely the branch where
such a hook never fires: the exception leaves the loop and the container stays empty. The
executor is also the stage with N calls, so it is the MORE likely of the two, not the
corner. A write, by contrast, is over the moment ``execute`` returns. Recording there
survives any exception raised afterwards, by anything, at any depth. A hook may stay wired
as well and the two compose without care: the predicate is "did anything land in the
container".

Usage (the host owns the metadata key — the core reads none of this)::

    sink = new_sink()
    meta[MY_OWN_KEY] = sink                       # pre-placed by the CALLER
    dispatcher = CommitRecordingDispatcher(dispatcher, sink)
    ...
    if committed(meta[MY_OWN_KEY]):
        ctx.metadata[mk.PRIOR_ATTEMPT_COMMITTED] = True   # declare it to the retry
"""

from __future__ import annotations

from typing import Any

from cogno_anima.tools.binding import bind_delegated

__all__ = ["new_sink", "committed", "CommitRecordingDispatcher"]


def new_sink() -> "list[Any]":
    """A fresh recorder for ONE turn. Pre-place it in the caller's metadata *before* the
    turn starts — see the module docstring for why a key stamped from inside never
    comes back."""
    return []


def committed(sink: Any) -> bool:
    """Did the turn this sink watched commit?

    Tolerant by design: the container travels through dicts a caller may rebuild or
    serialise, and a predicate on the failure path must never raise. Anything that is not
    a populated list answers ``False``.

    A note that is theoretical today and written because the cost changed: what the
    recorder sees is the RAW :class:`~cogno_anima.types.ToolResult`, and the EGO rewrites a
    ``needs_confirmation=True`` into ``ok=False, side_effect=False`` precisely so a
    PROPOSAL can never count as a write. A source that returned both as ``True`` on a
    proposal would make this container say "it wrote".
    """
    return bool(sink) if isinstance(sink, list) else False


class CommitRecordingDispatcher:
    """Records a commit AT THE MOMENT IT HAPPENS, into the caller's pre-placed sink.

    Wraps, never replaces: every other attribute delegates, so a read-only mask, a
    confirmation gate and any narrowing underneath keep working exactly as before.
    """

    # No ``__slots__`` here, and that is load-bearing — not an oversight. The policy
    # members are bound onto the INSTANCE (see :mod:`cogno_anima.tools.binding`), which a
    # slotted class cannot hold; and naming them in ``__slots__`` instead is WORSE than
    # binding nothing, because the slot DESCRIPTOR lives on the class and satisfies the
    # protocol even when the slot was never set — a wrapper claiming a policy its source
    # does not have. Two instances per turn; the memory was never the point.

    def __init__(self, inner: Any, sink: "list[Any]") -> None:
        self._inner = inner
        self._sink = sink
        bind_delegated(self, inner, "is_mutating", "requires_confirmation")

    def tools_schema(self) -> "list[dict]":
        return self._inner.tools_schema()

    async def execute(self, name: str, arguments: dict) -> Any:
        result = await self._inner.execute(name, arguments)
        # ``ok`` AND ``side_effect``, the same pair ``committed_this_turn`` requires: a
        # mutation that FAILED changed nothing, and a read is not a commit. Recorded before
        # returning, so an exception raised by anything downstream cannot un-record it.
        if getattr(result, "ok", False) and getattr(result, "side_effect", False):
            self._sink.append(True)
        return result

    def __getattr__(self, name: str) -> Any:
        """Everything else IS the inner's — including what nobody thought to forward.

        The first version of this wrapper listed the two policy methods by hand and stopped
        there, and that hand-written list was wrong twice over:

        * It DROPPED what it did not name. A caller reading a counter off the guard beneath
          it (how many calls a provenance guard refused this turn, say) gets 0 through an
          opaque wrapper, and the refusals vanish from observability — silently, and a
          counter reading zero looks like a clean turn, which is the worst failure mode a
          counter has.
        * It LIED about what it did name. Declaring ``is_mutating`` unconditionally makes
          ``isinstance(self, ToolPolicyDispatcher)`` true even when the inner has no policy
          at all — and that probe is exactly how the EGO decides whether gate A's fail-safe
          applies. A wrapper answering "yes, there is a policy" on behalf of a source that
          has none turns "mask every tool" into "mask nothing", then raises
          ``AttributeError`` on the first call. Adding a safety wrapper must not be able to
          disarm a safety gate.

        Delegation fixes both at once: an attribute the inner lacks raises ``AttributeError``
        from here too, so ``hasattr`` — and therefore the protocol probe — keeps telling the
        truth about the object underneath. The policy members travel the other way, through
        :func:`~cogno_anima.tools.binding.bind_delegated`, because ``__getattr__`` alone is
        invisible to the static resolution Python 3.12 uses.

        Deliberately the OPPOSITE choice from :class:`CompositeDispatcher`, and the asymmetry
        is the point: that one is a ROUTER over MANY sources and could not know which to
        forward to, so it names each method explicitly. This is a WRAPPER over ONE inner,
        where forwarding is unambiguous.
        """
        return getattr(self._inner, name)
