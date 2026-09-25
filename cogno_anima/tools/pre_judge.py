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
A cancelled call reports no tokens — but the provider bills a request it RECEIVED, so a callback
that calls ``Proposal.note_prompt(n)`` before its await gets that estimate charged on a line that
says so (:data:`JUDGE_PRE_ESTIMATED_STAGE`). A callback that never calls it, or a judgement cut
before it ever ran, carries 0 — "unknown", which the ``timeout`` verdict beside it is what makes
readable; a call that ERRORED records whatever the callback measured before it failed.

**ENFORCEMENT, opt-in and per tool (F2.3a-on).** A caller that has MEASURED the pre-verdict on a
tool may make it count there: ``enforce`` is a predicate ``(tool) -> bool`` the caller injects,
and for a WRITE it answers ``True`` to, the call WAITS for its judgement (ceiling
``enforce_timeout_s``) and

* ``approved`` → the call runs, as in the shadow;
* ``critique`` → the call does NOT run. The wrapper returns a PROPOSAL instead — the
  ``ToolResult(needs_confirmation=True)`` of gate C, built by the caller's ``confirm`` callback
  (the sentence the contact reads is the caller's, never this module's) — so the executor stops
  and the contact is asked. A proposal commits nothing, and the wrapper holds it to that: a
  ``confirm`` answer that is not a proposal (``needs_confirmation`` false, ``ok`` or
  ``side_effect`` true) is replaced by a neutral one;
* ``error`` / ``timeout`` → the call runs (FAIL OPEN: with no verdict the behaviour is the
  shadow's), and the record SAYS so — a fail-open nobody counts becomes the mechanism.

``confirmed`` is the caller's predicate ``(tool, arguments) -> bool`` for "the contact already
confirmed THIS call": such a call runs without being judged again, because the contact's yes to a
re-made proposal IS the verdict (``docs/ACT_CONFIRM_READONLY.md``: nothing is executed that was
not re-proposed to the contact). It is asked with the call's own arguments, so a yes to a proposal
about one object never waves through a call on another.

An enforced call is judged ONCE — the enforcement judgement is the shadow's record for that call,
in the same sink, with two more keys (``enforced`` and ``outcome``, from :data:`PRE_OUTCOMES`).
Every tool ``enforce`` does not name is judged exactly as before; without ``enforce`` this
wrapper is the shadow, byte for byte.

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
from cogno_anima.types import StageMetrics, ToolResult

logger = logging.getLogger(__name__)

__all__ = [
    "JUDGE_PRE_STAGE",
    "JUDGE_PRE_ESTIMATED_STAGE",
    "PRE_VERDICTS",
    "PRE_APPROVED",
    "PRE_CRITIQUE",
    "PRE_ERROR",
    "PRE_TIMEOUT",
    "DEFAULT_PRE_JUDGE_TIMEOUT_S",
    "DEFAULT_ENFORCE_TIMEOUT_S",
    "PRE_OUTCOME_EXECUTED",
    "PRE_OUTCOME_HELD",
    "PRE_OUTCOME_FAIL_OPEN",
    "PRE_OUTCOMES",
    "Proposal",
    "PreJudgment",
    "PreJudge",
    "PreJudgeSink",
    "PreJudgeDispatcher",
]

#: The ledger label of every pre-judgement's cost — its own line, NEVER the judge's.
JUDGE_PRE_STAGE = "judge_pre"
#: The label of a CUT judgement whose prompt had already been handed to the backend: its tokens
#: are the callback's ESTIMATE, not a count the provider reported. The suffix is the host's own
#: convention for an estimated charge (``kb_ingest:estimated``) — one spelling for both.
JUDGE_PRE_ESTIMATED_STAGE = f"{JUDGE_PRE_STAGE}:estimated"

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
#: The ceiling of an ENFORCED judgement — shorter, because here the call (and the reply) waits.
DEFAULT_ENFORCE_TIMEOUT_S = 8.0

#: What an ENFORCED call did, in the record (closed): it ran after an ``approved``, it was HELD
#: for the contact's confirmation after a ``critique``, or it ran with no verdict (``error`` /
#: ``timeout``) — the fail-open, counted.
PRE_OUTCOME_EXECUTED = "executed"
PRE_OUTCOME_HELD = "held"
PRE_OUTCOME_FAIL_OPEN = "executed_fail_open"
PRE_OUTCOMES = (PRE_OUTCOME_EXECUTED, PRE_OUTCOME_HELD, PRE_OUTCOME_FAIL_OPEN)

#: The proposal returned when the caller's ``confirm`` gives none, or gives something that is not
#: a proposal. Neutral on purpose: which words a contact reads is the caller's.
_NEUTRAL_PROPOSAL = "Held for confirmation before this action runs."


@dataclass(frozen=True)
class Proposal:
    """What the executor is about to run: the tool and a COPY of the arguments it chose.

    A copy, because the call and the judge run at the same time and a dispatcher underneath may
    enrich the arguments in place (an RBAC layer injecting the requester's id, say): the judge must
    read what the EXECUTOR proposed, and must not race the layer that edits it.
    """

    tool: str
    arguments: dict
    #: The hook a callback calls with its prompt's ESTIMATED input tokens, BEFORE it awaits the
    #: backend (F2.3a-bis). A judgement cut by the ceiling or by ``settle`` after that point was
    #: in all likelihood sent, and the provider bills a request it received whether or not the
    #: answer was read — recording 0 there makes the activation's cost per turn come out LOW. A
    #: callback that never calls it (every callback written before the hook) records 0 as
    #: before. Not part of equality: two proposals are the same call whatever hook rides along.
    note_prompt: "Optional[Callable[[int], None]]" = field(default=None, compare=False,
                                                           repr=False)


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
    #: Set only on an ENFORCED call: the verdict counted, and what the call did.
    enforced: bool = False
    outcome: Optional[str] = None
    #: What the callback said its prompt would cost, BEFORE it awaited (``Proposal.note_prompt``).
    prompt_tokens_estimated: Optional[int] = None

    def note_prompt(self, tokens: int) -> None:
        """The callback's estimate, taken only while the record is open and only as a
        non-negative integer — a hook handed to foreign code must not be a way to write junk
        into a ledger, nor to rewrite a closed record."""
        if self.verdict is None and isinstance(tokens, int) and not isinstance(tokens, bool) \
                and tokens >= 0:
            self.prompt_tokens_estimated = tokens

    def finish(self, verdict: str, metrics: Optional[StageMetrics]) -> None:
        # FIRST WORD WINS, and it is load-bearing, not tidiness: the ceiling and `settle` write
        # `timeout` and THEN cancel, and a judge that swallows the cancel (or turns it into an
        # exception) still reaches `_judge_one`'s own `finish` afterwards. Without this line that
        # late answer would overwrite the clock's verdict.
        if self.verdict is not None:
            return
        self.elapsed_ms = (time.perf_counter() - self.started) * 1000
        self.verdict = verdict if isinstance(verdict, str) and verdict in PRE_VERDICTS \
            else PRE_ERROR
        if isinstance(metrics, StageMetrics):
            base, stage = metrics, JUDGE_PRE_STAGE
        elif self.verdict == PRE_TIMEOUT and self.prompt_tokens_estimated:
            # CUT after the prompt was handed over: charge the estimate, and SAY it is one. Only
            # the clock's verdict takes this path — a judgement that answered reports its own
            # tokens, and an `error` keeps what the callback measured (a backend that raised
            # before sending spent nothing a caller can see).
            base = StageMetrics(stage=JUDGE_PRE_ESTIMATED_STAGE, elapsed_ms=self.elapsed_ms,
                                tokens_in=self.prompt_tokens_estimated, tokens_out=0,
                                model=self.model)
            stage = JUDGE_PRE_ESTIMATED_STAGE
        else:
            base = StageMetrics(stage=JUDGE_PRE_STAGE, elapsed_ms=self.elapsed_ms, tokens_in=0,
                                tokens_out=0, model=self.model)
            stage = JUDGE_PRE_STAGE
        # The label is the WRAPPER's guarantee, not the callback's courtesy: a host that builds
        # its judge on the SUPEREGO's metrics would otherwise hand in `superego_judge` and sum
        # the shadow into the very judge it is being compared against.
        self.metrics = base.model_copy(update={"stage": stage})


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
            late = [(e, e.task) for e in self._entries
                    if e.task is not None and not e.task.done()]
            for e, task in late:
                # The verdict FIRST, then the cancel — the same order as the ceiling's
                # `_expire`: a judge that swallows the cancel and answers anyway must find the
                # record already closed, or an answer that arrived after the turn ended would be
                # filed as if it had been in time.
                e.finish(PRE_TIMEOUT, None)
                task.cancel()
            if late:
                await asyncio.gather(*(task for _, task in late), return_exceptions=True)
        except asyncio.CancelledError:
            # The turn itself is being cancelled: still leave nothing running behind it.
            for e in self._entries:
                if e.task is not None and not e.task.done():
                    e.finish(PRE_TIMEOUT, None)     # the same order: the verdict, then the cancel
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
        out: "list[dict[str, Any]]" = []
        for e in self._entries:
            if e.verdict is None:
                continue
            row: "dict[str, Any]" = {"tool": e.tool, "verdict": e.verdict,
                                     "ms": int(round(e.elapsed_ms)), "committed": e.committed}
            if e.enforced:
                # Only on an enforced call — a shadow record keeps its four keys, byte for byte.
                row["enforced"] = True
                row["outcome"] = e.outcome if e.outcome in PRE_OUTCOMES else None
            out.append(row)
        return out

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
                 timeout_s: float = DEFAULT_PRE_JUDGE_TIMEOUT_S,
                 enforce: "Optional[Callable[[str], bool]]" = None,
                 confirm: "Optional[Callable[[Proposal], Any]]" = None,
                 confirmed: "Optional[Callable[[str, dict], bool]]" = None,
                 enforce_timeout_s: float = DEFAULT_ENFORCE_TIMEOUT_S) -> None:
        self._inner = inner
        self._judge = judge
        self._sink = sink
        self._timeout_s = float(timeout_s)
        self._enforce = enforce
        self._confirm = confirm
        self._confirmed = confirmed
        self._enforce_timeout_s = float(enforce_timeout_s)
        #: Observability, per turn: how many judgements this wrapper launched.
        self.pre_judged = 0
        bind_delegated(self, inner, "is_mutating", "requires_confirmation")

    def tools_schema(self) -> "list[dict]":
        return self._inner.tools_schema()

    async def execute(self, name: str, arguments: dict) -> Any:
        if self._enforces(name):
            return await self._execute_enforced(name, arguments)
        entry = self._launch(name, arguments)
        # THE CALL, exactly as it would run without this wrapper — not after the judge, not
        # instead of it, and with no `try`: an exception is the inner's to raise and the
        # executor's to handle, as it always was.
        result = await self._inner.execute(name, arguments)
        if entry is not None:
            entry.committed = bool(getattr(result, "ok", False)
                                   and getattr(result, "side_effect", False))
        return result

    def _enforces(self, name: str) -> bool:
        """Does the verdict COUNT for this call? Only for a WRITE the caller's ``enforce`` names —
        the policy underneath still decides what a write is — and never on a raise."""
        if self._enforce is None or not self._is_write(name):
            return False
        try:
            return self._enforce(name) is True
        except Exception:                     # noqa: BLE001 — an unanswerable policy enforces nothing
            logger.warning("event=judge_pre_enforce_policy_failed tool=%s", name, exc_info=True)
            return False

    def _already_confirmed(self, name: str, arguments: Any) -> bool:
        if self._confirmed is None:
            return False
        try:
            return self._confirmed(name, dict(arguments) if isinstance(arguments, dict)
                                   else {}) is True
        except Exception:                     # noqa: BLE001 — unknown is NOT confirmed
            logger.warning("event=judge_pre_confirmed_failed tool=%s", name, exc_info=True)
            return False

    async def _execute_enforced(self, name: str, arguments: dict) -> Any:
        """The ENFORCED path (module docstring): wait for the judgement, then run, hold, or run
        with the fail-open counted. A call the contact already confirmed runs unjudged."""
        if self._already_confirmed(name, arguments):
            return await self._inner.execute(name, arguments)
        entry = self._launch(name, arguments, ceiling_s=self._enforce_timeout_s)
        if entry is None:                     # the judgement could not even start: fail open
            return await self._inner.execute(name, arguments)
        entry.enforced = True
        if entry.task is not None:
            # `wait`, not `await task`: the ceiling CANCELS the task, and that must read as a
            # timeout here, never as this coroutine being cancelled. A cancel of THIS coroutine
            # (the turn ending) still propagates; `settle` then closes the judgement.
            await asyncio.wait({entry.task})
        if entry.verdict == PRE_CRITIQUE:
            entry.outcome = PRE_OUTCOME_HELD
            entry.committed = False
            logger.info("event=judge_pre_enforced tool=%s verdict=%s outcome=%s", name,
                        entry.verdict, entry.outcome)
            return self._proposal(name, arguments)
        entry.outcome = (PRE_OUTCOME_EXECUTED if entry.verdict == PRE_APPROVED
                         else PRE_OUTCOME_FAIL_OPEN)
        logger.info("event=judge_pre_enforced tool=%s verdict=%s outcome=%s", name,
                    entry.verdict, entry.outcome)
        result = await self._inner.execute(name, arguments)
        entry.committed = bool(getattr(result, "ok", False)
                               and getattr(result, "side_effect", False))
        return result

    def _proposal(self, name: str, arguments: Any) -> Any:
        """The caller's proposal for a held call — and only ever a PROPOSAL: something that says
        it needs confirmation, did not succeed and wrote nothing. Anything else is replaced."""
        neutral = ToolResult(output=_NEUTRAL_PROPOSAL, ok=False, error="needs_confirmation",
                             side_effect=False, needs_confirmation=True)
        if self._confirm is None:
            return neutral
        try:
            made = self._confirm(Proposal(str(name), dict(arguments)
                                          if isinstance(arguments, dict) else {}))
        except Exception:                     # noqa: BLE001 — a proposal must still be made
            logger.warning("event=judge_pre_confirm_failed tool=%s", name, exc_info=True)
            return neutral
        if (getattr(made, "needs_confirmation", False) is True
                and getattr(made, "ok", True) is False
                and getattr(made, "side_effect", True) is False):
            return made
        logger.warning("event=judge_pre_confirm_not_a_proposal tool=%s", name)
        return neutral

    def _is_write(self, name: str) -> bool:
        probe = getattr(self._inner, "is_mutating", None)
        if probe is None:
            return False                      # no policy → no judgement (module docstring)
        try:
            return bool(probe(name))
        except Exception:                     # noqa: BLE001 — an unanswerable probe judges nothing
            return False

    def _launch(self, name: str, arguments: Any, *,
                ceiling_s: Optional[float] = None) -> Optional[_Entry]:
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
            entry.task = asyncio.ensure_future(self._judge_one(
                entry, Proposal(str(name), args, note_prompt=entry.note_prompt)))
            # The ceiling is a TIMER beside the task, not `wait_for` around the judge: before
            # 3.12 `wait_for` runs the awaitable as a SECOND task, so an answer that is already
            # in needs several loop turns to register — and `settle` at grace 0 would file it as
            # a timeout. Here the judge runs inside the task itself and finishes in its own step.
            entry.timer = asyncio.get_running_loop().call_later(
                self._timeout_s if ceiling_s is None else ceiling_s, self._expire, entry)
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
