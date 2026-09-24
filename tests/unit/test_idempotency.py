"""``IdempotentDispatcher`` — the same effect never goes out twice without saying so.

Every twin below is run in BOTH worlds by construction: the "first" call of each pair is the
world without a prior (it must execute), the "second" is the world with one (it must not), and
the inner dispatcher's call list is the number that tells them apart. A twin that asserted only
the second half would pass over a guard that withheld EVERYTHING.

Names, recipients and messages are invented.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from cogno_anima.tools import (IdempotencyRule, IdempotencyStore, IdempotentDispatcher,
                               InMemoryIdempotencyStore, idempotency_key)
from cogno_anima.tools.base import ToolPolicyDispatcher
from cogno_anima.tools.idempotency import (CLAIMED, DONE, PENDING, Claim, default_notice,
                                           normalize_arguments)
from cogno_anima.types import PipelineContext, ToolExecution, ToolResult, committed_this_turn

T0 = datetime(2026, 3, 10, 17, 5, tzinfo=timezone.utc)     # 14:05 in São Paulo
SCOPE = "tenant-x:user-7"
MSG = {"target": "Profa. Irene Valdez", "message": "A aula de quinta passa para as 19h."}
RULES = {"notify_user": 1800}


class _Clock:
    """An injected clock — the window is measured against THIS, never against the wall."""

    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


class _Source:
    """Sends, reads, fails, raises — and counts every call that reached it."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.refusals = 3                         # an attribute nobody thought to forward

    def tools_schema(self):
        return [{"function": {"name": "notify_user"}}]

    def is_mutating(self, name):
        return name != "list_classes"

    def requires_confirmation(self, name):
        return name == "notify_user"

    async def execute(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        if name == "notify_user":
            return ToolResult(output=f"Message delivered to {arguments.get('target')}",
                              ok=True, side_effect=True)
        if name == "list_classes":
            return ToolResult(output="3 classes", ok=True, side_effect=False)
        if name == "rejected_write":
            return ToolResult(output="", ok=False, error="slot taken", side_effect=False)
        if name == "noop_write":
            return ToolResult(output="already confirmed — no change", ok=True, side_effect=False)
        if name == "proposing_write":
            return ToolResult(output="shall I?", ok=True, side_effect=True,
                              needs_confirmation=True)
        if name == "exploding_write":
            raise TimeoutError("channel did not answer")
        raise AssertionError(name)


class _SpyStore(InMemoryIdempotencyStore):
    def __init__(self) -> None:
        super().__init__()
        self.touched = 0

    async def claim(self, key, *, now, window_s):
        self.touched += 1
        return await super().claim(key, now=now, window_s=window_s)


def _guard(src, store, clock, *, rules=None, scope=SCOPE, **kw):
    """ONE turn's guard. A fresh instance per turn, as the host builds it."""
    return IdempotentDispatcher(src, store, scope=scope, rules=RULES if rules is None else rules,
                                clock=clock, tz=ZoneInfo("America/Sao_Paulo"), **kw)


# ── the main twin: the same message, two turns ────────────────────────────────────────────

async def test_the_same_message_in_the_next_turn_is_not_sent_again_and_says_when():
    src, store, clock = _Source(), InMemoryIdempotencyStore(), _Clock()

    first = await _guard(src, store, clock).execute("notify_user", MSG)
    clock.advance(4 * 60)                                    # four minutes later, a new turn
    turn2 = _guard(src, store, clock)
    second = await turn2.execute("notify_user", dict(MSG))

    assert first.ok and first.side_effect and first.output.startswith("Message delivered")
    assert len(src.calls) == 1, "the repeat reached the channel — it would be sent twice"
    assert second.ok is True and second.side_effect is False
    assert "NOT REPEATED" in second.output and "14:05" in second.output
    assert default_notice("notify_user", DONE, T0.astimezone(ZoneInfo("America/Sao_Paulo"))) \
        in second.output
    assert turn2.repeats_withheld == 1


async def test_a_retry_inside_the_same_turn_is_withheld_too():
    """The correction loop and the grounding repair re-run the executor INSIDE one turn."""
    src, store, clock = _Source(), InMemoryIdempotencyStore(), _Clock()
    guard = _guard(src, store, clock)
    await guard.execute("notify_user", MSG)
    again = await guard.execute("notify_user", MSG)
    assert len(src.calls) == 1 and "NOT REPEATED" in again.output


async def test_the_withheld_call_is_not_counted_as_a_commit_by_the_family():
    """`committed_this_turn` conjoins ok AND side_effect per call — the withheld call wrote
    nothing THIS turn, and the family must read it that way (while the real send is a commit)."""
    src, store, clock = _Source(), InMemoryIdempotencyStore(), _Clock()
    sent = await _guard(src, store, clock).execute("notify_user", MSG)
    withheld = await _guard(src, store, clock).execute("notify_user", MSG)

    def _turn(r: ToolResult) -> PipelineContext:
        call = ToolExecution(tool="notify_user", arguments=MSG, result=r.output, ok=r.ok,
                             side_effect=r.side_effect, tool_mutating=True)
        ctx = PipelineContext(user_input="avisa a professora")
        ctx.turn_executions = [call]
        return ctx

    assert committed_this_turn(_turn(sent)) is True
    assert committed_this_turn(_turn(withheld)) is False


# ── the window ────────────────────────────────────────────────────────────────────────────

async def test_the_same_message_AFTER_the_window_is_sent():
    src, store, clock = _Source(), InMemoryIdempotencyStore(), _Clock()
    await _guard(src, store, clock).execute("notify_user", MSG)
    clock.advance(1800 - 1)
    inside = await _guard(src, store, clock).execute("notify_user", MSG)
    clock.advance(2)
    after = await _guard(src, store, clock).execute("notify_user", MSG)
    assert "NOT REPEATED" in inside.output
    assert after.side_effect is True and len(src.calls) == 2


async def test_the_window_runs_from_the_SUCCESS_not_from_the_claim():
    class _SlowSource(_Source):
        async def execute(self, name, arguments):
            clock.advance(600)                               # the send took ten minutes
            return await super().execute(name, arguments)

    clock = _Clock()
    src, store = _SlowSource(), InMemoryIdempotencyStore()
    await _guard(src, store, clock).execute("notify_user", MSG)
    clock.advance(1500)                  # 2100 s after the claim, 1500 s after the success
    again = await _guard(src, store, clock).execute("notify_user", MSG)
    assert "NOT REPEATED" in again.output and len(src.calls) == 1


# ── different effects are different effects ───────────────────────────────────────────────

async def test_one_word_changed_is_a_different_message_and_is_sent():
    src, store, clock = _Source(), InMemoryIdempotencyStore(), _Clock()
    await _guard(src, store, clock).execute("notify_user", MSG)
    changed = dict(MSG, message="A aula de sexta passa para as 19h.")
    r = await _guard(src, store, clock).execute("notify_user", changed)
    assert r.side_effect is True and len(src.calls) == 2


async def test_another_recipient_is_a_different_message_and_is_sent():
    src, store, clock = _Source(), InMemoryIdempotencyStore(), _Clock()
    await _guard(src, store, clock).execute("notify_user", MSG)
    r = await _guard(src, store, clock).execute("notify_user",
                                                dict(MSG, target="Prof. Otávio Lemke"))
    assert r.side_effect is True and len(src.calls) == 2


async def test_another_requester_is_another_scope():
    src, store, clock = _Source(), InMemoryIdempotencyStore(), _Clock()
    await _guard(src, store, clock).execute("notify_user", MSG)
    r = await _guard(src, store, clock, scope="tenant-x:user-8").execute("notify_user", MSG)
    assert r.side_effect is True and len(src.calls) == 2


async def test_whitespace_and_key_order_do_not_make_a_new_message():
    src, store, clock = _Source(), InMemoryIdempotencyStore(), _Clock()
    await _guard(src, store, clock).execute("notify_user", MSG)
    retyped = {"message": "  A aula de quinta   passa para as 19h. ", "target": MSG["target"]}
    r = await _guard(src, store, clock).execute("notify_user", retyped)
    assert "NOT REPEATED" in r.output and len(src.calls) == 1


async def test_case_is_kept_the_key_is_exact_not_similar():
    src, store, clock = _Source(), InMemoryIdempotencyStore(), _Clock()
    await _guard(src, store, clock).execute("notify_user", MSG)
    r = await _guard(src, store, clock).execute(
        "notify_user", dict(MSG, message=MSG["message"].upper()))
    assert r.side_effect is True and len(src.calls) == 2


async def test_a_declared_key_subset_ignores_the_other_arguments():
    rules = {"notify_user": IdempotencyRule(window_s=1800, key_arguments=("target",))}
    src, store, clock = _Source(), InMemoryIdempotencyStore(), _Clock()
    await _guard(src, store, clock, rules=rules).execute("notify_user", MSG)
    r = await _guard(src, store, clock, rules=rules).execute(
        "notify_user", dict(MSG, message="outra coisa"))
    assert "NOT REPEATED" in r.output and len(src.calls) == 1


# ── what never touches the record ─────────────────────────────────────────────────────────

async def test_a_read_never_passes_through_the_record():
    src, store, clock = _Source(), _SpyStore(), _Clock()
    guard = _guard(src, store, clock)
    for _ in range(3):
        r = await guard.execute("list_classes", {})
        assert r.output == "3 classes"
    assert store.touched == 0 and len(src.calls) == 3


async def test_control_a_new_ordinary_call_is_passed_through_unchanged():
    src, store, clock = _Source(), _SpyStore(), _Clock()
    r = await _guard(src, store, clock).execute("notify_user", MSG)
    assert r == ToolResult(output="Message delivered to Profa. Irene Valdez", ok=True,
                           side_effect=True)
    assert store.touched == 1 and src.calls == [("notify_user", MSG)]


async def test_an_empty_scope_disables_the_guard_instead_of_merging_contacts():
    src, store, clock = _Source(), _SpyStore(), _Clock()
    for _ in range(2):
        await _guard(src, store, clock, scope="").execute("notify_user", MSG)
    assert store.touched == 0 and len(src.calls) == 2


async def test_a_zero_window_disables_the_rule():
    src, store, clock = _Source(), _SpyStore(), _Clock()
    for _ in range(2):
        await _guard(src, store, clock, rules={"notify_user": 0}).execute("notify_user", MSG)
    assert store.touched == 0 and len(src.calls) == 2


# ── what a call that committed nothing leaves behind ──────────────────────────────────────

@pytest.mark.parametrize("tool", ["rejected_write", "noop_write", "proposing_write"])
async def test_a_call_that_committed_nothing_releases_its_claim(tool):
    rules = {tool: 1800}
    src, store, clock = _Source(), InMemoryIdempotencyStore(), _Clock()
    await _guard(src, store, clock, rules=rules).execute(tool, {"x": 1})
    await _guard(src, store, clock, rules=rules).execute(tool, {"x": 1})
    assert len(src.calls) == 2, f"{tool}: a call that wrote nothing blocked its own retry"


async def test_a_call_that_RAISED_stays_pending_and_the_twin_is_told_it_may_have_gone_out():
    rules = {"exploding_write": 1800}
    src, store, clock = _Source(), InMemoryIdempotencyStore(), _Clock()
    with pytest.raises(TimeoutError):
        await _guard(src, store, clock, rules=rules).execute("exploding_write", {"x": 1})
    clock.advance(60)
    r = await _guard(src, store, clock, rules=rules).execute("exploding_write", {"x": 1})
    assert len(src.calls) == 1
    assert r.ok is True and r.side_effect is False
    assert "MAY have gone through" in r.output and "14:05" in r.output
    assert "did not repeat it" in r.output
    clock.advance(1800)                                     # the pending claim expires
    with pytest.raises(TimeoutError):
        await _guard(src, store, clock, rules=rules).execute("exploding_write", {"x": 1})
    assert len(src.calls) == 2


async def test_two_identical_calls_at_once_run_ONE():
    gate = asyncio.Event()

    class _InFlight(_Source):
        async def execute(self, name, arguments):
            await gate.wait()
            return await super().execute(name, arguments)

    src, store, clock = _InFlight(), InMemoryIdempotencyStore(), _Clock()
    first = asyncio.create_task(_guard(src, store, clock).execute("notify_user", MSG))
    await asyncio.sleep(0)                                   # let the first one claim
    second = await _guard(src, store, clock).execute("notify_user", MSG)
    gate.set()
    await first
    assert len(src.calls) == 1
    assert "MAY have gone through" in second.output


# ── the store is consulted, not trusted with the turn ─────────────────────────────────────

class _DeadStore:
    def __init__(self, fail: str) -> None:
        self.fail = fail
        self._ok = InMemoryIdempotencyStore()

    async def claim(self, key, *, now, window_s):
        if self.fail == "claim":
            raise ConnectionError("down")
        return await self._ok.claim(key, now=now, window_s=window_s)

    async def complete(self, key, *, now, window_s):
        if self.fail == "complete":
            raise ConnectionError("down")

    async def release(self, key):
        if self.fail == "release":
            raise ConnectionError("down")


async def test_a_dead_store_fails_OPEN_loudly_and_tells_the_caller(caplog):
    marks: list[str] = []
    src, clock = _Source(), _Clock()
    guard = _guard(src, _DeadStore("claim"), clock, on_unavailable=marks.append)
    with caplog.at_level(logging.WARNING, logger="cogno_anima.tools.idempotency"):
        r = await guard.execute("notify_user", MSG)
    assert r.side_effect is True and len(src.calls) == 1
    assert marks == ["claim"] and guard.store_unavailable == 1
    assert "event=idempotency_store_unavailable" in caplog.text


@pytest.mark.parametrize("step,tool", [("complete", "notify_user"), ("release", "noop_write")])
async def test_a_store_that_fails_AFTER_the_call_returns_the_result_anyway(step, tool):
    marks: list[str] = []
    src, clock = _Source(), _Clock()
    guard = _guard(src, _DeadStore(step), clock, rules={tool: 1800}, on_unavailable=marks.append)
    r = await guard.execute(tool, MSG)
    assert r.ok is True and marks == [step]


async def test_a_hook_that_raises_does_not_cost_the_call():
    def _boom(_step):
        raise RuntimeError("recorder broke")

    src = _Source()
    r = await _guard(src, _DeadStore("claim"), _Clock(), on_unavailable=_boom).execute(
        "notify_user", MSG)
    assert r.side_effect is True


async def test_a_store_answering_outside_its_alphabet_claims_nothing():
    class _Garbled(InMemoryIdempotencyStore):
        async def claim(self, key, *, now, window_s):
            return Claim(state="maybe")

    src = _Source()
    for _ in range(2):
        await _guard(src, _Garbled(), _Clock()).execute("notify_user", MSG)
    assert len(src.calls) == 2


# ── the sentence ──────────────────────────────────────────────────────────────────────────

async def test_the_callers_notice_is_what_the_contact_is_told():
    seen: list[tuple] = []

    def _pt(tool, state, at_local):
        seen.append((tool, state, at_local.strftime("%H:%M")))
        return f"Já enviei esta mesma mensagem às {at_local:%H:%M} — não repeti."

    src, store, clock = _Source(), InMemoryIdempotencyStore(), _Clock()
    await _guard(src, store, clock, notice=_pt).execute("notify_user", MSG)
    r = await _guard(src, store, clock, notice=_pt).execute("notify_user", MSG)
    assert "«Já enviei esta mesma mensagem às 14:05 — não repeti.»" in r.output
    assert seen == [("notify_user", DONE, "14:05")]


def test_the_default_notice_offers_only_the_exit_that_works():
    at = datetime(2026, 3, 10, 14, 5)
    for state in (DONE, PENDING):
        text = default_notice("notify_user", state, at)
        assert "14:05" in text and "tell me what changes" in text
        assert "again?" not in text, "a 'yes' to 'again?' would land on the same key"


# ── the key and the store ─────────────────────────────────────────────────────────────────

def test_the_key_is_a_digest_that_carries_no_argument():
    k = idempotency_key(SCOPE, "notify_user", MSG)
    assert len(k) == 64 and all(c in "0123456789abcdef" for c in k)
    assert "Irene" not in k and k == idempotency_key(SCOPE, "notify_user", dict(MSG))
    assert k != idempotency_key(SCOPE, "remind_me", MSG)


def test_normalisation_is_meaning_preserving_and_nothing_looser():
    assert normalize_arguments({"b": " x  y ", "a": None, "n": 150.0, "f": False}) == \
        {"b": "x y", "n": 150, "f": False}
    assert normalize_arguments({"t": "Café"}) == {"t": "Café"}
    assert normalize_arguments(["A", "a"]) == ["A", "a"]
    assert idempotency_key(SCOPE, "t", {"x": object()})       # never raises


async def test_the_memory_store_never_releases_a_success():
    store = InMemoryIdempotencyStore()
    assert (await store.claim("k", now=T0, window_s=60)).state == CLAIMED
    await store.complete("k", now=T0, window_s=60)
    await store.release("k")
    assert (await store.claim("k", now=T0, window_s=60)) == Claim(state=DONE, at=T0)
    assert isinstance(store, IdempotencyStore)


# ── the wrapper is transparent ────────────────────────────────────────────────────────────

def test_the_wrapper_keeps_the_policy_probe_honest_and_forwards_the_rest():
    guard = _guard(_Source(), InMemoryIdempotencyStore(), _Clock())
    assert isinstance(guard, ToolPolicyDispatcher)
    assert guard.requires_confirmation("notify_user") is True
    assert guard.refusals == 3
    assert guard.tools_schema() == [{"function": {"name": "notify_user"}}]
