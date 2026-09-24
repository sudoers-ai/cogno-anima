"""
cogno_anima.tools.idempotency — the same effect never goes out twice without saying so.

Nothing in the pipeline stopped the SAME write from being executed twice. The paths that do it
are all ordinary: a correction loop that re-runs the executor after the judge rejected a turn
whose first attempt already wrote; a grounding repair that re-executes with the tool forced; a
"yes" answered twice to the same proposal; a model that simply issues the call again. Each one
is rare, and the damage is asymmetric — tokens spent twice recover on their own, a person who
received the same message twice or a ledger that carries the same income twice does not. The
owner's words: *sending the message twice and saying it was not sent is dangerous*.

**The mechanism: a key per effectful call, looked up BEFORE the call runs.** The key is a
digest of ``(scope, tool, normalised arguments)``; ``scope`` is the caller's (who is asking, in
which tenant) and opaque here. When the same key already SUCCEEDED inside the tool's declared
window, the call is **not executed** and the executor receives ``ok=True, side_effect=False``
with a note saying when it was done — so nothing is blocked in silence: the contact is always
told it was already done. ``side_effect=False`` is the truth about THIS call (it wrote nothing),
which is exactly what :func:`~cogno_anima.types.committed_this_turn` and the rest of that family
need to read: they conjoin ``ok`` and ``side_effect`` per call, and this module records a
success under the same pair.

**An exact key, never a similarity.** Two messages that differ by one word are two messages; the
normalisation only removes what carries no meaning (key order, surrounding and repeated
whitespace, Unicode composition, ``None`` for an absent optional, ``150.0`` for ``150``). Case
is kept.

**Claim BEFORE, complete on success, release on a failure that committed nothing.** The claim
is atomic in the store (the second of two concurrent identical calls finds the first one's
claim), so "the same call twice at once" is covered by the same rule as "the same call twice in
a row". What a single transaction cannot give — the effect and its record committed together —
is impossible here in general: a message leaves through a channel API and a vertical writes
through its own connection. So the order is chosen for the owner's rule instead, **at most
once**:

  * ``ok`` AND ``side_effect`` → the claim becomes a SUCCESS, and the window runs from now;
  * ``ok=False``, or nothing written, or a proposal (``needs_confirmation``) → the claim is
    RELEASED: nothing was committed, so an identical call may run again;
  * the call RAISED → the claim is KEPT, as PENDING: the outcome is unknown (a timeout on a send
    is precisely "it may have gone out"), and an identical call inside the window is not run but
    told exactly that. The pending claim expires with the window — it never blocks forever.

**Fail-OPEN on the store.** A store that cannot be reached must not switch off every write of
the deployment to protect against a rare repeat, so the call executes, a WARNING
``event=idempotency_store_unavailable`` is logged, and ``on_unavailable`` (the caller's hook, for
its own per-turn record) is called. The direction is written here because it is a decision.

**What this module does NOT decide** — every one of them is the caller's catalog:

  * WHICH tools are guarded and their window (``rules``, injected with no default: a tool that is
    not declared is never looked up, which is also why a read never touches the store);
  * WHO the scope is (an empty scope disables the guard rather than merging every contact into
    one key space — a false "already done" to a stranger is a leak, not a safety);
  * the SENTENCE the contact reads (``notice``, in the tenant's language — the default is
    English and generic).

**What it deliberately does not offer: a way to repeat inside the window.** Asking "shall I send
it again?" would need the confirmation to reach this guard (the gate-C return trip), and a
question whose "yes" lands on the same key would answer "already done" again — a promise the
system cannot keep. The note therefore names the exit that works: change what is sent, or let
the window pass.
"""

from __future__ import annotations

import hashlib
import json
import logging
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any, Callable, Mapping, Optional, Protocol, Union, runtime_checkable

from cogno_anima.tools.binding import bind_delegated
from cogno_anima.types import ToolResult

logger = logging.getLogger(__name__)

__all__ = [
    "CLAIMED",
    "DONE",
    "PENDING",
    "Claim",
    "IdempotencyRule",
    "IdempotencyStore",
    "InMemoryIdempotencyStore",
    "IdempotentDispatcher",
    "default_notice",
    "idempotency_key",
    "normalize_arguments",
]

#: The claim is this caller's: the call may run.
CLAIMED = "claimed"
#: An identical call SUCCEEDED inside the window: the call is not run.
DONE = "done"
#: An identical call was claimed inside the window and never recorded an outcome (it is still
#: running, or it raised mid-flight): the call is not run, and the note says it MAY have gone out.
PENDING = "pending"

# Versioned, so a change to the normalisation below starts a fresh key space instead of
# comparing new keys against old ones computed by different rules.
_KEY_VERSION = "cogno-idempotency/1"


@dataclass(frozen=True)
class Claim:
    """What the store answered for one key. ``at`` is when the prior identical call succeeded
    (``DONE``) or was claimed (``PENDING``); ``None`` on ``CLAIMED``."""

    state: str
    at: Optional[datetime] = None


@dataclass(frozen=True)
class IdempotencyRule:
    """One guarded tool: its window, and which arguments make two calls "the same".

    ``key_arguments=None`` means EVERY argument — the default, and the exact reading. A subset
    is a caller's declaration that the other arguments do not distinguish two calls; it exists
    so that choice is a declaration and not a rewrite, not because any tool needs it today.
    """

    window_s: float
    key_arguments: Optional[tuple[str, ...]] = None


@runtime_checkable
class IdempotencyStore(Protocol):
    """The persistent record the guard consults. All three calls are keyed by the digest only;
    the store never sees an argument."""

    async def claim(self, key: str, *, now: datetime, window_s: float) -> Claim:
        """ATOMIC test-and-set. A live row for ``key`` (one whose window has not passed) is
        answered as it is (``DONE``/``PENDING`` with its instant) and left untouched; otherwise a
        ``PENDING`` row is written for this caller and ``CLAIMED`` is returned. A read followed
        by a write would let two concurrent identical calls both read "free" and both run."""
        ...

    async def complete(self, key: str, *, now: datetime, window_s: float) -> None:
        """The claimed call succeeded: record ``DONE`` at ``now``; the window runs from here."""
        ...

    async def release(self, key: str) -> None:
        """The claimed call committed nothing: drop the PENDING claim. Must never remove a
        ``DONE`` row — releasing is not forgetting a success."""
        ...


class InMemoryIdempotencyStore:
    """Process-local record — the zero-infra implementation, and the double every test uses.
    Correct for ONE process; a deployment with several workers needs a shared store."""

    def __init__(self) -> None:
        self._rows: dict[str, tuple[str, datetime, datetime]] = {}   # key → (state, at, expires)

    async def claim(self, key: str, *, now: datetime, window_s: float) -> Claim:
        # No ``await`` between the read and the write: in single-threaded asyncio that is what
        # makes this atomic with respect to other tasks.
        for k in [k for k, row in self._rows.items() if row[2] <= now]:
            del self._rows[k]
        row = self._rows.get(key)
        if row is not None:
            return Claim(state=row[0], at=row[1])
        self._rows[key] = (PENDING, now, now + timedelta(seconds=window_s))
        return Claim(state=CLAIMED)

    async def complete(self, key: str, *, now: datetime, window_s: float) -> None:
        self._rows[key] = (DONE, now, now + timedelta(seconds=window_s))

    async def release(self, key: str) -> None:
        row = self._rows.get(key)
        if row is not None and row[0] == PENDING:
            del self._rows[key]


def normalize_arguments(value: Any) -> Any:
    """The arguments as they MEAN, not as they were typed — and nothing looser than that.

    Removed: key order (the digest sorts), whitespace at the ends and runs of it inside a string,
    Unicode composition (NFC), ``None`` values (an optional left out and one passed as null are
    the same call), and the ``.0`` of an integral float. Kept: case, punctuation, every word.
    A key that treated "similar" text as the same would withhold a DIFFERENT message and tell the
    contact it was already sent — the wrong answer in the worst direction.
    """
    if isinstance(value, Mapping):
        return {str(k): normalize_arguments(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [normalize_arguments(v) for v in value]
    if isinstance(value, str):
        return " ".join(unicodedata.normalize("NFC", value).split())
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def idempotency_key(scope: str, tool: str, arguments: Any, *,
                    key_arguments: Optional[tuple[str, ...]] = None) -> str:
    """The digest that names ONE effect: ``sha256`` over the scope, the tool and the normalised
    arguments (or only ``key_arguments`` of them). A digest and not the text, so the store holds
    no argument — only a pseudonymous name for the call. ``default=str`` because a key must never
    raise: an unserialisable argument makes the call harder to match, never the turn fail."""
    args = arguments if isinstance(arguments, Mapping) else {}
    if key_arguments is not None:
        args = {k: args[k] for k in key_arguments if k in args}
    payload = json.dumps([_KEY_VERSION, str(scope), str(tool), normalize_arguments(args)],
                         sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


#: ``(tool, state, at_local) -> sentence`` — what the CONTACT is told, in the tenant's language.
Notice = Callable[[str, str, datetime], str]


def default_notice(tool: str, state: str, at_local: datetime) -> str:
    """The English, tool-agnostic fallback. A caller with a language of its own passes its own.

    It names the exit that WORKS: a different request is a different key and runs. It does not
    offer to repeat — see the module docstring for why a "yes" could not be honoured.
    """
    hhmm = at_local.strftime("%H:%M")
    if state == PENDING:
        return (f"An identical request started at {hhmm} and its result was not recorded — it may "
                f"have gone through, so I did not repeat it. If you want another one, tell me "
                f"what changes.")
    return (f"I already did exactly this at {hhmm} — I did not repeat it. If you want another "
            f"one, tell me what changes.")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: datetime) -> datetime:
    """A naive instant is read as UTC — the store and the clock speak UTC by contract."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


class IdempotentDispatcher:
    """Withholds an effectful call whose exact twin already succeeded inside its window.

    Wraps, never replaces: the tool surface, the read-only mask and both confirmation gates
    underneath are untouched — it only decides whether ``execute`` reaches the inner dispatcher.
    A call held by gate B never reaches ``execute`` at all, so the guard acts on the confirmed
    replay, which is exactly where "yes" answered twice would otherwise send twice.

    ``rules`` maps a tool name to an :class:`IdempotencyRule` (a bare number is read as the
    window in seconds). Build a FRESH instance per turn: ``scope`` is the turn's requester, and
    the two counters below are per turn.
    """

    # No ``__slots__``, and that is load-bearing — the reason
    # :class:`~cogno_anima.tools.commit_sink.CommitRecordingDispatcher` states: the policy members
    # are bound onto the INSTANCE, and a slot descriptor on the class would satisfy the protocol
    # probe for a source that declared no policy at all.

    def __init__(self, inner: Any, store: Any, *, scope: str,
                 rules: "Mapping[str, Union[IdempotencyRule, float]]",
                 clock: Callable[[], datetime] = _utcnow,
                 tz: Optional[tzinfo] = None,
                 notice: Optional[Notice] = None,
                 on_unavailable: Optional[Callable[[str], None]] = None) -> None:
        self._inner = inner
        self._store = store
        self._scope = str(scope or "")
        self._rules: dict[str, IdempotencyRule] = {
            str(k): (v if isinstance(v, IdempotencyRule) else IdempotencyRule(window_s=float(v)))
            for k, v in dict(rules or {}).items()}
        self._clock = clock
        self._tz = tz or timezone.utc
        self._notice = notice or default_notice
        self._on_unavailable = on_unavailable
        #: Observability, per turn: calls this guard withheld, and store failures it rode over.
        self.repeats_withheld = 0
        self.store_unavailable = 0
        bind_delegated(self, inner, "is_mutating", "requires_confirmation")

    def tools_schema(self) -> "list[dict]":
        return self._inner.tools_schema()

    async def execute(self, name: str, arguments: dict) -> Any:
        rule = self._rules.get(name)
        if rule is None or rule.window_s <= 0 or not self._scope:
            return await self._inner.execute(name, arguments)
        key = idempotency_key(self._scope, name, arguments, key_arguments=rule.key_arguments)
        try:
            claim = await self._store.claim(key, now=_aware(self._clock()), window_s=rule.window_s)
        except Exception as exc:        # noqa: BLE001 — fail-OPEN, see the module docstring
            self._unavailable(name, "claim", exc)
            return await self._inner.execute(name, arguments)
        state = getattr(claim, "state", None)
        if state in (DONE, PENDING):
            return self._withheld(name, claim)
        if state != CLAIMED:            # a store answering outside its alphabet claimed nothing
            self._unavailable(name, "claim", ValueError(f"claim state {state!r}"))
            return await self._inner.execute(name, arguments)

        # No ``try`` here, deliberately: a call that RAISES leaves its claim PENDING, because the
        # outcome is unknown (a send that timed out may have gone out). It expires with the window.
        result = await self._inner.execute(name, arguments)
        if (getattr(result, "ok", False) and getattr(result, "side_effect", False)
                and not getattr(result, "needs_confirmation", False)):
            try:
                await self._store.complete(key, now=_aware(self._clock()),
                                           window_s=rule.window_s)
            except Exception as exc:    # noqa: BLE001 — the effect happened; say so, do not undo
                self._unavailable(name, "complete", exc)
        else:
            try:
                await self._store.release(key)
            except Exception as exc:    # noqa: BLE001 — a stuck claim only errs towards "not again"
                self._unavailable(name, "release", exc)
        return result

    def _withheld(self, name: str, claim: Claim) -> ToolResult:
        self.repeats_withheld += 1
        at_local = _aware(claim.at or self._clock()).astimezone(self._tz)
        hhmm = at_local.strftime("%H:%M")
        sentence = self._notice(name, claim.state, at_local)
        logger.info("event=idempotency_withheld tool=%s state=%s at=%s", name, claim.state, hhmm)
        if claim.state == PENDING:
            head = (f"NOT REPEATED. An identical `{name}` call was STARTED at {hhmm} and its "
                    f"outcome was never recorded — it MAY have gone through. It was not executed "
                    f"again now: do not say it failed, and do not say it was done now.")
        else:
            head = (f"NOT REPEATED. This exact `{name}` call already SUCCEEDED at {hhmm}. It was "
                    f"not executed again, so nothing new was sent or written now; do not call it "
                    f"again with the same arguments.")
        return ToolResult(output=f"{head} Tell the contact exactly this: «{sentence}»",
                          ok=True, side_effect=False)

    def _unavailable(self, name: str, step: str, exc: BaseException) -> None:
        self.store_unavailable += 1
        logger.warning("event=idempotency_store_unavailable tool=%s step=%s error=%s — "
                       "proceeding without the repeat check", name, step, type(exc).__name__)
        if self._on_unavailable is not None:
            try:
                self._on_unavailable(step)
            except Exception:           # noqa: BLE001 — a recorder must never cost the call
                logger.warning("event=idempotency_unavailable_hook_failed tool=%s", name)

    def __getattr__(self, item: str) -> Any:
        """Everything else IS the inner's — a counter on a guard beneath, a finer policy
        predicate a caller reaches down for. The policy members travel the other way, through
        :func:`~cogno_anima.tools.binding.bind_delegated`, because ``__getattr__`` alone is
        invisible to the static resolution Python 3.12 uses for the protocol probe."""
        return getattr(self._inner, item)
