"""``ConfirmArgumentRecordingDispatcher`` + ``held_calls`` — the return half of gate C.

A tool that answers ``needs_confirmation=True`` is saying *"about THIS call, on what I just
read: ask first"*. The confirmation turn replays the held call, and unless the tool's own
named argument travels with it the tool asks again — which ``EgoStage._refuse_if_still_asking``
correctly turns into a loud failure. These tests pin the capture and the replay shape.
"""

from __future__ import annotations

import inspect

import pytest

from cogno_anima.tools import (
    MAX_CONFIRM_ARGS,
    ConfirmArgumentRecordingDispatcher,
    call_key,
    held_calls,
    new_confirm_args,
)
from cogno_anima.tools.base import ToolPolicyDispatcher
from cogno_anima.types import ToolResult


class _AskingResult(ToolResult):
    """What a bridge returns when the TOOL asked for something back.

    ``cogno_anima.ToolResult`` deliberately declares no ``confirm_arguments`` field — what a
    skill needs in order to commit is the skill's business — so the carrier is a subclass
    owned by the bridge that speaks that protocol (``cogno_mcp.MCPToolResult`` is the real
    one). The recorder reads the attribute duck-typed, which is exactly what this mirrors.
    """

    confirm_arguments: dict = {}


class _Asks:
    """Asks about every call, naming a per-call argument (the row it actually read)."""

    def __init__(self) -> None:
        self.refusals = 3                    # the attribute an opaque wrapper would swallow

    def tools_schema(self): return [{"function": {"name": "remove_entry"}}]

    async def execute(self, name, arguments):
        return _AskingResult(output="about to remove X — confirm?", ok=False,
                             needs_confirmation=True,
                             confirm_arguments={"confirm_tx_id": f"tx-{arguments.get('query')}"})

    def is_mutating(self, name): return True

    def requires_confirmation(self, name): return False


class _Quiet:
    """Runs and commits — no question, therefore nothing to record."""

    def tools_schema(self): return []

    async def execute(self, name, arguments):
        return _AskingResult(output="done", ok=True, side_effect=True,
                             confirm_arguments={"confirm_tx_id": "tx-9"})


class _NoPolicy:
    def tools_schema(self): return []

    async def execute(self, name, arguments): return ToolResult(output="", ok=True)


class _Held:
    def __init__(self, tool, arguments):
        self.tool, self.arguments = tool, arguments


async def test_what_the_tool_asked_for_is_captured_under_the_call_it_asked_about():
    sink = new_confirm_args()
    rec = ConfirmArgumentRecordingDispatcher(_Asks(), sink)
    await rec.execute("remove_entry", {"query": "internet"})
    assert sink == {call_key("remove_entry", {"query": "internet"}): {"confirm_tx_id": "tx-internet"}}


async def test_two_calls_to_the_SAME_tool_keep_their_own_answers():
    """MUTATION: key the container by tool name alone -> this dies, and a step with two
    proposals replays both with the second one's row."""
    sink = new_confirm_args()
    rec = ConfirmArgumentRecordingDispatcher(_Asks(), sink)
    await rec.execute("remove_entry", {"query": "a"})
    await rec.execute("remove_entry", {"query": "b"})
    assert sink[call_key("remove_entry", {"query": "a"})] == {"confirm_tx_id": "tx-a"}
    assert sink[call_key("remove_entry", {"query": "b"})] == {"confirm_tx_id": "tx-b"}


async def test_arguments_WITHOUT_a_question_are_not_recorded():
    """Arguments without a question are not a pending confirmation, and treating them as one
    would let a source pre-load a replay for a call nobody held.

    MUTATION: record whenever ``confirm_arguments`` is present -> this dies.
    """
    sink = new_confirm_args()
    await ConfirmArgumentRecordingDispatcher(_Quiet(), sink).execute("remove_entry", {})
    assert sink == {}


async def test_a_bare_ToolResult_asking_with_NO_named_arguments_records_nothing():
    """The core's own ``ToolResult`` carries no ``confirm_arguments``, and that is by design:
    a tool asking a plain yes/no needs nothing back. The recorder must read the attribute
    duck-typed and stay silent when it is absent — never invent a key.

    MUTATION: ``result.confirm_arguments`` instead of ``getattr(..., None)`` -> this raises.
    """
    class _PlainAsk:
        def tools_schema(self): return []

        async def execute(self, name, arguments):
            return ToolResult(output="are you sure?", ok=False, needs_confirmation=True)

    sink = new_confirm_args()
    await ConfirmArgumentRecordingDispatcher(_PlainAsk(), sink).execute("remove_entry", {})
    assert sink == {}


async def test_an_unbounded_blob_is_DROPPED_at_the_door():
    """A recorded blob goes into the caller's session state and then back into a live call.
    Bounded here, at the only door, rather than trusted per source."""
    class _Greedy(_Asks):
        async def execute(self, name, arguments):
            return _AskingResult(output="?", ok=False, needs_confirmation=True,
                                 confirm_arguments={f"k{i}": "v"
                                                    for i in range(MAX_CONFIRM_ARGS + 1)})

    sink = new_confirm_args()
    await ConfirmArgumentRecordingDispatcher(_Greedy(), sink).execute("remove_entry", {})
    assert sink == {}


def test_call_key_never_raises_on_an_unserialisable_argument():
    """A key must never break the turn; an unmatchable call is the acceptable degradation."""
    class _Opaque:
        def __repr__(self): return "<opaque>"

    key = call_key("t", {"x": _Opaque()})
    assert isinstance(key, str) and "t" in key


def test_held_calls_merges_what_the_tool_asked_for():
    held = held_calls([_Held("remove_entry", {"query": "internet"})],
                      confirm_arguments={call_key("remove_entry", {"query": "internet"}):
                                         {"confirm_tx_id": "tx-1"}})
    assert held == [{"tool": "remove_entry",
                     "arguments": {"query": "internet", "confirm_tx_id": "tx-1"}}]


def test_held_calls_never_OVERWRITES_an_argument_the_model_chose():
    """The proposal named a row and the commit must be that row: letting a source rewrite a
    chosen argument would make the confirmed call do something the user was never shown.

    MUTATION: ``args[key] = value`` unconditionally -> this dies.
    """
    held = held_calls([_Held("remove_entry", {"query": "internet"})],
                      confirm_arguments={call_key("remove_entry", {"query": "internet"}):
                                         {"query": "everything", "confirm_tx_id": "tx-1"}})
    assert held[0]["arguments"]["query"] == "internet"
    assert held[0]["arguments"]["confirm_tx_id"] == "tx-1"


def test_held_calls_degrades_on_junk_and_has_NO_default_for_the_return_trip():
    """The keyword is mandatory on purpose: a call site with nothing to pass has to SAY so,
    in code. Forgetting it is otherwise invisible — the replay runs, the tool answers, and
    the only symptom is a turn reporting a failure the user cannot act on."""
    assert held_calls([], confirm_arguments=None) == []
    assert held_calls(None, confirm_arguments={}) == []
    assert held_calls([_Held("book", {"day": "friday"})], confirm_arguments="junk") == [
        {"tool": "book", "arguments": {"day": "friday"}}]
    assert "confirm_arguments" in inspect.signature(held_calls).parameters
    assert inspect.signature(held_calls).parameters["confirm_arguments"].default \
        is inspect.Parameter.empty
    with pytest.raises(TypeError):
        held_calls([])                       # type: ignore[call-arg]


def test_the_wrapper_is_transparent_and_forwards_what_it_does_not_mediate():
    rec = ConfirmArgumentRecordingDispatcher(_Asks(), new_confirm_args())
    assert rec.tools_schema() == [{"function": {"name": "remove_entry"}}]
    assert rec.refusals == 3
    with pytest.raises(AttributeError):
        rec.no_such_attribute_anywhere


def test_the_policy_probe_keeps_telling_the_truth_about_the_source():
    """MUTATION: declare the policy methods on the class -> the second half dies, and a
    wrapper starts claiming a policy for a source that declared none."""
    with_policy = ConfirmArgumentRecordingDispatcher(_Asks(), new_confirm_args())
    without = ConfirmArgumentRecordingDispatcher(_NoPolicy(), new_confirm_args())
    assert isinstance(with_policy, ToolPolicyDispatcher)
    assert not isinstance(without, ToolPolicyDispatcher)
