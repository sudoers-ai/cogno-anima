"""``CommitRecordingDispatcher`` — the turn's write, recorded where it cannot be undone.

The properties, one per failure this exists for:

  * a successful MUTATION lands in the caller's container;
  * a failed one, and a READ, do not (``ok`` AND ``side_effect``, the pair
    ``committed_this_turn`` requires);
  * the record survives an exception raised AFTER the write — the whole point of recording
    at ``execute`` instead of at a post-turn hook;
  * the wrapper stays transparent: schema passes through, unmediated attributes forward,
    and the policy probe keeps telling the truth about the source underneath.
"""

from __future__ import annotations

import inspect

import pytest

from cogno_anima.tools import CommitRecordingDispatcher, committed, new_sink
from cogno_anima.tools.base import ToolPolicyDispatcher
from cogno_anima.types import ToolResult


class _Source:
    """A source that writes, reads, and fails — plus a counter nobody thought to forward."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.refusals = 7                    # the attribute an opaque wrapper would swallow

    def tools_schema(self): return [{"function": {"name": "write_it"}}]

    async def execute(self, name, arguments):
        self.calls.append(name)
        if name == "write_it":
            return ToolResult(output="done", ok=True, side_effect=True)
        if name == "read_it":
            return ToolResult(output="rows", ok=True, side_effect=False)
        return ToolResult(output="", ok=False, error="nope", side_effect=True)

    def is_mutating(self, name): return name != "read_it"

    def requires_confirmation(self, name): return False


class _NoPolicy:
    def tools_schema(self): return []

    async def execute(self, name, arguments): return ToolResult(output="", ok=True)


def test_an_empty_or_broken_container_is_not_a_commit():
    """The predicate travels through dicts a caller may rebuild or serialise, so it must
    answer for anything — a failure-path predicate that raises is worse than a wrong one."""
    assert committed(new_sink()) is False
    assert committed(None) is False
    assert committed("[]") is False
    assert committed({"a": 1}) is False
    assert committed([True]) is True


async def test_a_successful_mutation_is_recorded():
    sink = new_sink()
    d = CommitRecordingDispatcher(_Source(), sink)
    await d.execute("write_it", {})
    assert committed(sink) is True


async def test_a_read_and_a_FAILED_mutation_are_not():
    """MUTATION: record on ``ok`` alone, or on ``side_effect`` alone -> one of these dies.

    Both halves matter and for different reasons: a read is not a commit, and a mutation
    that FAILED changed nothing. A container that says "it wrote" over either one sends the
    next turn's guard the wrong answer about a turn that wrote nothing.
    """
    for tool in ("read_it", "fail_it"):
        sink = new_sink()
        await CommitRecordingDispatcher(_Source(), sink).execute(tool, {})
        assert committed(sink) is False, f"{tool} counted as a commit"


async def test_the_record_SURVIVES_an_exception_raised_after_the_write():
    """The branch this whole module exists for. A post-turn hook never fires here — the
    exception leaves the loop first — so the fact would be lost with the context.

    MUTATION: record after the loop instead of inside ``execute`` -> this dies.
    """
    class _RaisesOnTheSecondCall:
        def __init__(self) -> None: self.n = 0

        def tools_schema(self): return []

        async def execute(self, name, arguments):
            self.n += 1
            if self.n == 1:
                return ToolResult(output="written", ok=True, side_effect=True)
            raise RuntimeError("a later stage blew up")

    sink = new_sink()
    d = CommitRecordingDispatcher(_RaisesOnTheSecondCall(), sink)
    await d.execute("write_it", {})
    with pytest.raises(RuntimeError):
        await d.execute("anything", {})
    assert committed(sink) is True, "the write vanished with the exception"


async def test_the_wrapper_is_transparent():
    src = _Source()
    d = CommitRecordingDispatcher(src, new_sink())
    assert d.tools_schema() == src.tools_schema()
    result = await d.execute("write_it", {"x": 1})
    assert result.ok and result.output == "done"
    assert src.calls == ["write_it"]


def test_an_unmediated_attribute_is_FORWARDED_not_swallowed():
    """A wrapper that lists its methods loses every method it did not name, in silence — a
    counter read through it returns 0, and a zero looks like a clean turn.

    MUTATION: delete ``__getattr__`` -> this dies.
    """
    d = CommitRecordingDispatcher(_Source(), new_sink())
    assert d.refusals == 7
    with pytest.raises(AttributeError):
        d.no_such_attribute_anywhere


def test_the_policy_probe_keeps_telling_the_truth_about_the_source():
    """Both directions, because only the pair discriminates: a wrapper that declared the
    policy unconditionally would pass the first assertion and fail the second, and that is
    a safety wrapper disarming gate A's fail-safe.

    MUTATION: declare ``is_mutating``/``requires_confirmation`` on the class, or name them
    in ``__slots__`` -> the second half dies.
    """
    with_policy = CommitRecordingDispatcher(_Source(), new_sink())
    without = CommitRecordingDispatcher(_NoPolicy(), new_sink())
    assert isinstance(with_policy, ToolPolicyDispatcher)
    assert not isinstance(without, ToolPolicyDispatcher)
    for m in ("is_mutating", "requires_confirmation"):
        assert inspect.getattr_static(with_policy, m, None) is not None
        assert inspect.getattr_static(without, m, None) is None
    assert with_policy.is_mutating("write_it") is True
