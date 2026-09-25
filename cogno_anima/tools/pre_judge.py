"""
cogno_anima.tools.pre_judge — the judge reads a WRITE before it goes out, in SHADOW.

The SUPEREGO judge runs AFTER the executor: by the time it reads a write, the write has already
left — the message was sent, the entry recorded, the slot booked — and its critique can only
steer what the contact is TOLD about it. Moving that judgement in front of the call is the
eventual design; this module is the INSTRUMENT that says whether doing so would be right, and it
changes nothing:

* for every call the dispatcher underneath classifies as a write (``is_mutating``), an injected
  ``judge`` callback is launched over the PROPOSAL — the tool's name and the arguments the
  executor chose — **concurrently with the call, never in front of it**. ``execute`` does not
  await the judge, does not read its verdict and returns exactly what the inner dispatcher
  returned, exception included;
* the verdict lands in a sink the CALLER pre-placed, from a closed alphabet (:data:`PRE_VERDICTS`:
  ``approved | critique | error | timeout``) and **never with text**: the critique a judge writes
  quotes the arguments, and the arguments are contact data. What the sink keeps is the tool name,
  the verdict, the wall time and whether the call committed (``ok`` AND ``side_effect``, the pair
  ``committed_this_turn`` reads) — the four facts the activation decision needs: how often the
  pre-judge agrees with the judge that runs after, and how many writes it would have stopped that
  went out anyway;
* the callback's cost travels as a :class:`~cogno_anima.types.StageMetrics` whose ``stage`` is
  ALWAYS :data:`JUDGE_PRE_STAGE`, whatever the callback labelled it — so a ledger that groups by
  stage books it on a line of its own and can never fold it into the judge's.

**Why concurrently and not in front.** A write that waits for a second model call pays its
latency on every write, and a shadow exists to measure, not to cost. The judge starts the moment
the call does; the executor's next step, the post-execution judge and the voice all run while it
thinks, so on an ordinary turn the verdict is in long before the turn ends.

**The ceiling, and what happens to a verdict that is not in.** Each judgement has its own
``timeout_s``, counted from launch. At the end of the turn the caller calls
:meth:`PreJudgeSink.settle`, which waits at most ``grace_s`` (default 0: the shadow never delays a
reply) and then CANCELS whatever is still running, recording it as ``timeout``. Cancelling rather
than abandoning is deliberate: a task that outlives its turn is a model call nobody accounts for.
A cancelled call reports no tokens, so its metrics carry 0 — "unknown", which the ``timeout``
verdict beside it is what makes readable; a call that ERRORED records whatever the callback
measured before it failed.

**What this module does NOT decide** — each is the caller's:

* the judge itself — the callback, which is given only the :class:`Proposal`; what the request
  was, which rules apply and which model answers are closed over by whoever builds it
  (:class:`cogno_anima.stages.proposal_judge.ProposalJudge` is the one this package ships);
* WHICH calls are writes — read from the inner dispatcher's own ``is_mutating``. A source with no
  policy is never judged: the core does not guess which tools mutate, and judging every call to be
  safe would spend a model call on every read, which is the one cost this shadow promised not to
  have.

Usage (the caller owns where the records go — the core reads none of this)::

    sink = PreJudgeSink()
    dispatcher = PreJudgeDispatcher(dispatcher, judge=my_judge, sink=sink)
    try:
        ctx = await run_the_turn(dispatcher)
    finally:
        await sink.settle()                      # cancels the stragglers, never raises
    ctx.retry_metrics.extend(sink.metrics)       # the cost, on its own ledger line
    trace["pre"] = sink.records                  # the verdicts, closed alphabet
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from cogno_anima.tools.binding import bind_delegated
from cogno_anima.types import StageMetrics

logger = logging.getLogger(__name__)

__all__ = [
    "JUDGE_PRE_STAGE",
    "PRE_VERDICTS",
    "PRE_APPROVED",
    "PRE_CRITIQUE",
    "PRE_ERROR",
    "PRE_TIMEOUT",
    "DEFAULT_PRE_JUDGE_TIMEOUT_S",
    "Proposal",
    "PreJudgment",
    "PreJudge",
    "PreJudgeSink",
    "PreJudgeDispatcher",
]

#: The ledger label of every pre-judgement's cost — its own line, NEVER the judge's.
JUDGE_PRE_STAGE = "judge_pre"

PRE_APPROVED = "approved"
PRE_CRITIQUE = "critique"
PRE_ERROR = "error"
PRE_TIMEOUT = "timeout"
#: The closed alphabet of a pre-verdict. ``error`` = the judge could not produce a verdict (the
#: backend raised, the answer was not a JSON boolean); ``timeout`` = no verdict inside the ceiling,
#: or none by the time the turn settled. Neither is a rejection, and neither affects the turn.
PRE_VERDICTS = (PRE_APPROVED, PRE_CRITIQUE, PRE_ERROR, PRE_TIMEOUT)
# What a CALLBACK may answer. `timeout` is the wrapper's to say: a callback that claims it is
# reporting on its own lateness, which only the clock outside it can know.
_CALLBACK_VERDICTS = frozenset({PRE_APPROVED, PRE_CRITIQUE, PRE_ERROR})

#: The judgement's own ceiling, from launch. Well above the post-execution judge's measured p90
#: (6.6 s): the point of the ceiling is to bound a HUNG call, not to race an ordinary one.
DEFAULT_PRE_JUDGE_TIMEOUT_S = 20.0


@dataclass(frozen=True)
class Proposal:
    """What the executor is about to run: the tool and a COPY of the arguments it chose.

    A copy, because the call and the judge run at the same time and a dispatcher underneath may
    enrich the arguments in place (an RBAC layer injecting the requester's id, say): the judge must
    read what the EXECUTOR proposed, and must not race the layer that edits it.
    """

    tool: str
    arguments: dict


@dataclass(frozen=True)
class PreJudgment:
    """A callback's answer: a verdict from ``approved | critique | error`` and what it cost.

    ``metrics`` is optional so a callback that failed before it could measure anything still has a
    well-formed answer; the wrapper then records an empty cost under :data:`JUDGE_PRE_STAGE`.
    """

    verdict: str
    metrics: Optional[StageMetrics] = None


#: The injected judge. Receives the proposal only; everything else it needs is the caller's.
PreJudge = Callable[[Proposal], Awaitable[PreJudgment]]


@dataclass
class _Entry:
    tool: str
    model: str
    started: float
    task: "Optional[asyncio.Task[None]]" = None
    timer: "Optional[asyncio.TimerHandle]" = None
    verdict: Optional[str] = None
    elapsed_ms: float = 0.0
    metrics: Optional[StageMetrics] = None
    committed: Optional[bool] = None       # None = the call raised (no result to read)

    def finish(self, verdict: str, metrics: Optional[StageMetrics]) -> None:
        if self.verdict is not None:         # first word wins; settle never overwrites a verdict
            return
        self.elapsed_ms = (time.perf_counter() - self.started) * 1000
        self.verdict = verdict if isinstance(verdict, str) and verdict in PRE_VERDICTS \
            else PRE_ERROR
        base = metrics if isinstance(metrics, StageMetrics) else StageMetrics(
            stage=JUDGE_PRE_STAGE, elapsed_ms=self.elapsed_ms, tokens_in=0, tokens_out=0,
            model=self.model)
        # The label is the WRAPPER's guarantee, not the callback's courtesy: a host that builds
        # its judge on the SUPEREGO's metrics would otherwise hand in `superego_judge` and sum
        # the shadow into the very judge it is being compared against.
        self.metrics = base.model_copy(update={"stage": JUDGE_PRE_STAGE})


@dataclass
class PreJudgeSink:
    """One turn's pre-verdicts. Pre-place it; read it only after :meth:`settle`.

    Holds the running judgements too, which is why it is an object and not a bare list: the
    caller that settles the turn is rarely the code that built the dispatcher chain, and the
    stragglers must be cancellable from where the turn ends.
    """

    _entries: "list[_Entry]" = field(default_factory=list)

    def _add(self, entry: _Entry) -> None:
        self._entries.append(entry)

    async def settle(self, *, grace_s: float = 0.0) -> None:
        """Close the turn: wait at most ``grace_s`` for judgements still running, then cancel the
        rest and record them as ``timeout``. Idempotent, and never raises — it runs on the path
        where a turn is ending, possibly because something else already failed."""
        try:
            pending = [e.task for e in self._entries if e.task is not None and not e.task.done()]
            if pending:
                # ONE loop turn, even at grace 0: a verdict whose answer has already arrived is
                # queued to resume, and cancelling it there would file a verdict that exists as a
                # `timeout`. One iteration of the loop is not a delay anybody can measure.
                await asyncio.sleep(0)
            still = [t for t in pending if not t.done()]
            if still and grace_s > 0:
                await asyncio.wait(still, timeout=grace_s)
            late = [t for t in pending if not t.done()]
            for task in late:
                task.cancel()
            if late:
                await asyncio.gather(*late, return_exceptions=True)
        except asyncio.CancelledError:
            # The turn itself is being cancelled: still leave nothing running behind it.
            for e in self._entries:
                if e.task is not None and not e.task.done():
                    e.task.cancel()
            raise
        except Exception:        # noqa: BLE001 — a recorder must never cost the turn
            logger.warning("event=judge_pre_settle_failed", exc_info=True)
        for e in self._entries:
            if e.timer is not None:
                e.timer.cancel()     # a judgement cancelled before its first step never ran its own
            if e.verdict is None:
                e.finish(PRE_TIMEOUT, None)

    @property
    def records(self) -> "list[dict[str, Any]]":
        """The settled verdicts, in launch order — closed alphabet, no text.

        ``committed`` is ``True``/``False`` from the call's own result, ``None`` when the call
        raised. A judgement not yet settled is not listed: an unsettled record would be a verdict
        nobody gave.
        """
        return [{"tool": e.tool, "verdict": e.verdict, "ms": int(round(e.elapsed_ms)),
                 "committed": e.committed}
                for e in self._entries if e.verdict is not None]

    @property
    def metrics(self) -> "list[StageMetrics]":
        """What the settled judgements cost, one :data:`JUDGE_PRE_STAGE` row each."""
        return [e.metrics for e in self._entries
                if e.verdict is not None and e.metrics is not None]


class PreJudgeDispatcher:
    """Launches the injected judge over every WRITE the executor issues, and never waits for it.

    Wraps, never replaces: the tool surface, both policy predicates and ``execute``'s result are
    the inner dispatcher's, untouched. The only thing added is a task started beside the call.
    Build a FRESH instance (and sink) per turn.
    """

    # No ``__slots__``, and that is load-bearing — the reason
    # :class:`~cogno_anima.tools.commit_sink.CommitRecordingDispatcher` states: the policy members
    # are bound onto the INSTANCE, and a slot descriptor on the class would satisfy the protocol
    # probe for a source that declared no policy at all.

    def __init__(self, inner: Any, *, judge: "PreJudge", sink: PreJudgeSink,
                 timeout_s: float = DEFAULT_PRE_JUDGE_TIMEOUT_S) -> None:
        self._inner = inner
        self._judge = judge
        self._sink = sink
        self._timeout_s = float(timeout_s)
        #: Observability, per turn: how many judgements this wrapper launched.
        self.pre_judged = 0
        bind_delegated(self, inner, "is_mutating", "requires_confirmation")

    def tools_schema(self) -> "list[dict]":
        return self._inner.tools_schema()

    async def execute(self, name: str, arguments: dict) -> Any:
        entry = self._launch(name, arguments)
        # THE CALL, exactly as it would run without this wrapper — not after the judge, not
        # instead of it, and with no `try`: an exception is the inner's to raise and the
        # executor's to handle, as it always was.
        result = await self._inner.execute(name, arguments)
        if entry is not None:
            entry.committed = bool(getattr(result, "ok", False)
                                   and getattr(result, "side_effect", False))
        return result

    def _is_write(self, name: str) -> bool:
        probe = getattr(self._inner, "is_mutating", None)
        if probe is None:
            return False                      # no policy → no judgement (module docstring)
        try:
            return bool(probe(name))
        except Exception:                     # noqa: BLE001 — an unanswerable probe judges nothing
            return False

    def _launch(self, name: str, arguments: Any) -> Optional[_Entry]:
        """Start the judgement and return at once. Never raises: a shadow that cannot start is a
        call without a shadow, not a call that fails."""
        try:
            if not self._is_write(name):
                return None
            try:
                args = copy.deepcopy(arguments) if isinstance(arguments, dict) else {}
            except Exception:                 # noqa: BLE001 — an uncopyable value: shallow is enough
                args = dict(arguments) if isinstance(arguments, dict) else {}
            entry = _Entry(tool=str(name), model=str(getattr(self._judge, "model", "") or
                                                      "unknown"),
                           started=time.perf_counter())
            entry.task = asyncio.ensure_future(self._judge_one(entry, Proposal(str(name), args)))
            # The ceiling is a TIMER beside the task, not `wait_for` around the judge: before
            # 3.12 `wait_for` runs the awaitable as a SECOND task, so an answer that is already
            # in needs several loop turns to register — and `settle` at grace 0 would file it as
            # a timeout. Here the judge runs inside the task itself and finishes in its own step.
            entry.timer = asyncio.get_running_loop().call_later(
                self._timeout_s, self._expire, entry)
            self._sink._add(entry)
            self.pre_judged += 1
            return entry
        except Exception:                     # noqa: BLE001 — see the docstring
            logger.warning("event=judge_pre_launch_failed tool=%s", name, exc_info=True)
            return None

    @staticmethod
    def _expire(entry: _Entry) -> None:
        """The ceiling: record the timeout FIRST, then cancel — the verdict is the clock's."""
        if entry.task is not None and not entry.task.done():
            entry.finish(PRE_TIMEOUT, None)
            entry.task.cancel()

    async def _judge_one(self, entry: _Entry, proposal: Proposal) -> None:
        try:
            judged = await self._judge(proposal)
        except asyncio.CancelledError:
            raise                             # the ceiling or `settle` owns a cancelled judgement
        except Exception as exc:              # noqa: BLE001 — a judge failure is a verdict
            logger.warning("event=judge_pre_failed tool=%s error=%s", entry.tool,
                           type(exc).__name__)
            entry.finish(PRE_ERROR, None)
            return
        finally:
            if entry.timer is not None:
                entry.timer.cancel()
        verdict = getattr(judged, "verdict", None)
        # `isinstance` FIRST: the set is hashed, and an unhashable answer (a list) raising here
        # would kill the task after the judge had answered — filed as a `timeout` it never was.
        entry.finish(verdict if isinstance(verdict, str) and verdict in _CALLBACK_VERDICTS
                     else PRE_ERROR, getattr(judged, "metrics", None))
        logger.info("event=judge_pre tool=%s verdict=%s ms=%d", entry.tool, entry.verdict,
                    int(entry.elapsed_ms))

    def __getattr__(self, item: str) -> Any:
        """Everything else IS the inner's — a counter on a guard beneath, a finer policy
        predicate a caller reaches down for. The policy members travel the other way, through
        :func:`~cogno_anima.tools.binding.bind_delegated`, because ``__getattr__`` alone is
        invisible to the static resolution Python 3.12 uses for the protocol probe."""
        return getattr(self._inner, item)
