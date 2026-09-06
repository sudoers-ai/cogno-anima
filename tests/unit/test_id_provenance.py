"""``IdProvenanceDispatcher`` — a guarded write must name an id READ this turn.

The finding: an executor acted on record ids recalled from an OLD session's memories
instead of the ids the SAME turn's read had returned. Every write landed as an honest
no-op and the user's real records stayed untouched. The refusal is RECOVERABLE, so the
model self-corrects into reading first — which is the flow we wanted all along.
"""

from __future__ import annotations

import pytest

from cogno_anima.tools import IdProvenanceDispatcher
from cogno_anima.tools.base import ToolPolicyDispatcher
from cogno_anima.types import ToolResult

# A caller's map: ``tool -> (id argument, the READ that legitimises it)``. Two families, so
# the refusal can be shown naming the read THIS caller has rather than another one's.
GUARDED = {
    "close_ticket": ("ticket_id", "list_tickets"),
    "drop_note": ("note_id", "list_notes"),
}


class _Source:
    def __init__(self) -> None:
        self.executed: list[tuple[str, dict]] = []

    def tools_schema(self) -> list[dict]:
        return [{"function": {"name": "list_tickets"}}, {"function": {"name": "close_ticket"}}]

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        self.executed.append((name, arguments))
        if name == "list_tickets":
            return ToolResult(output="9858fb82: open\nd973ed23: open", ok=True)
        if name == "list_notes":
            return ToolResult(output="4f2a9c1d: a note", ok=True)
        return ToolResult(output=f"{arguments.get('ticket_id') or arguments.get('note_id')} closed",
                          ok=True, side_effect=True)

    def is_mutating(self, name: str) -> bool:
        return not name.startswith("list_")

    def requires_confirmation(self, name: str) -> bool:
        return False


class _NoPolicy:
    def tools_schema(self): return []

    async def execute(self, name, arguments): return ToolResult(output="x", ok=True)


def _guard(inner=None):
    return IdProvenanceDispatcher(inner or _Source(), guarded=GUARDED)


async def test_a_stale_id_is_refused_recoverably_and_never_reaches_the_source():
    """THE LIVE DEFECT: an id from an old session's memory, with no read this turn."""
    inner = _Source()
    r = await IdProvenanceDispatcher(inner, guarded=GUARDED).execute(
        "close_ticket", {"ticket_id": "d87ed730"})
    assert r.ok is False and r.error
    assert "THIS turn" in r.error and "list_tickets first" in r.error
    assert inner.executed == [], "the call reached the source — the guard did not hold"


async def test_an_id_from_a_SAME_TURN_read_is_allowed():
    """The twin. Without it, a guard that refused ALWAYS would pass the test above."""
    inner = _Source()
    g = IdProvenanceDispatcher(inner, guarded=GUARDED)
    await g.execute("list_tickets", {})
    r = await g.execute("close_ticket", {"ticket_id": "9858fb82"})
    assert r.ok and "9858fb82" in r.output
    assert [n for n, _ in inner.executed] == ["list_tickets", "close_ticket"]


async def test_the_refusal_names_the_READ_THIS_caller_declared():
    """A recoverable error is a self-correction ONLY if the tool it names exists here.
    Naming another family's read turns the contract into a loop, which is strictly worse
    than the unguarded state it replaces.

    MUTATION: hardcode one read tool for every entry -> this dies.
    """
    r = await _guard().execute("drop_note", {"note_id": "aa11bb22"})
    assert "list_notes" in r.error and "list_tickets" not in r.error


async def test_a_bare_string_declaration_keeps_working_with_a_NEUTRAL_hint():
    """Backwards compatibility for a caller that passes the old ``tool -> id_arg`` shape.
    The default hint names no tool, because the core does not know the caller's catalog —
    a caller that wants its own wording sets ``DEFAULT_READ_HINT`` or passes the tuple."""
    r = await IdProvenanceDispatcher(_Source(), guarded={"close_ticket": "ticket_id"}).execute(
        "close_ticket", {"ticket_id": "d87ed730"})
    assert r.ok is False
    assert IdProvenanceDispatcher.DEFAULT_READ_HINT in r.error


async def test_a_caller_may_override_the_bare_string_hint_with_its_own_read_tool():
    """The seam the core leaves for a catalog it cannot know. MUTATION: read the constant
    off the module instead of ``self`` -> this dies."""
    class _Mine(IdProvenanceDispatcher):
        DEFAULT_READ_HINT = "list_my_things"

    r = await _Mine(_Source(), guarded={"close_ticket": "ticket_id"}).execute(
        "close_ticket", {"ticket_id": "d87ed730"})
    assert "list_my_things first" in r.error


async def test_provenance_does_NOT_leak_across_turns():
    """A FRESH wrapper per turn: the previous turn's read grants nothing to the next — the
    staleness this exists to fence off is exactly a read from another turn."""
    inner = _Source()
    turn1 = IdProvenanceDispatcher(inner, guarded=GUARDED)
    await turn1.execute("list_tickets", {})
    turn2 = IdProvenanceDispatcher(inner, guarded=GUARDED)
    assert (await turn2.execute("close_ticket", {"ticket_id": "9858fb82"})).ok is False


async def test_an_UNGUARDED_tool_and_a_guarded_call_with_no_id_pass_through():
    """Over-tightening check: the guard must not touch what it was not asked to guard, and
    must not refuse a guarded tool called without the id argument at all."""
    inner = _Source()
    g = IdProvenanceDispatcher(inner, guarded=GUARDED)
    assert (await g.execute("list_tickets", {})).ok
    assert (await g.execute("close_ticket", {})).ok                 # no id -> nothing to check
    assert (await g.execute("close_ticket", {"ticket_id": ""})).ok   # blank -> nothing to check
    assert [n for n, _ in inner.executed] == ["list_tickets", "close_ticket", "close_ticket"]


async def test_a_FAILED_read_grants_nothing():
    """MUTATION: record the output regardless of ``ok`` -> this dies. An error message that
    happens to quote the id would otherwise legitimise the write it just refused."""
    class _FailingRead(_Source):
        async def execute(self, name, arguments):
            if name == "list_tickets":
                return ToolResult(output="9858fb82 could not be loaded", ok=False, error="boom")
            return await super().execute(name, arguments)

    g = IdProvenanceDispatcher(_FailingRead(), guarded=GUARDED)
    await g.execute("list_tickets", {})
    assert (await g.execute("close_ticket", {"ticket_id": "9858fb82"})).ok is False


async def test_the_refusals_counter_is_visible_for_observability():
    g = _guard()
    assert g.refusals == 0
    await g.execute("close_ticket", {"ticket_id": "nope"})
    await g.execute("drop_note", {"note_id": "nope"})
    assert g.refusals == 2


def test_schema_passes_through_and_policy_answers_the_fail_safe_defaults():
    g = _guard()
    assert [t["function"]["name"] for t in g.tools_schema()] == ["list_tickets", "close_ticket"]
    assert g.is_mutating("close_ticket") is True
    assert g.is_mutating("list_tickets") is False
    assert g.requires_confirmation("close_ticket") is False
    # A source with NO policy: mutating (gate A masks it) and not destructive (gate B is
    # opt-in) — the EGO's own fail-safe defaults, same as the router's.
    bare = IdProvenanceDispatcher(_NoPolicy(), guarded={})
    assert isinstance(bare, ToolPolicyDispatcher)
    assert bare.is_mutating("anything") is True
    assert bare.requires_confirmation("anything") is False


def test_an_unmediated_attribute_is_FORWARDED_not_swallowed():
    """A wrapper that lists its methods loses every method it did not name, in silence — a
    caller reaching down for a finer policy predicate gets the conservative answer instead.

    MUTATION: delete ``__getattr__`` -> this dies.
    """
    class _WithExtra(_Source):
        def source_requires_confirmation(self, name): return name == "close_ticket"

    g = IdProvenanceDispatcher(_WithExtra(), guarded=GUARDED)
    assert g.source_requires_confirmation("close_ticket") is True
    assert g.source_requires_confirmation("list_tickets") is False
    with pytest.raises(AttributeError):
        g.no_such_attribute_anywhere


def test_the_default_read_hint_NAMES_NO_TOOL():
    """The boundary property of this module, and the reason the map is injected at all.

    ``DEFAULT_READ_HINT`` is rendered into a refusal a model reads and acts on. If it named
    a real tool, it would name ONE caller's tool — and this lib is public. A caller without
    that tool gets a refusal telling it to call something it does not have, which turns the
    self-correction the guard exists to produce into a loop.

    **This test exists because the mutation survived without it.** Its neighbour asserts
    ``DEFAULT_READ_HINT in r.error``, which is true of ANY hint — including a product tool
    name — so it pins that the hint is RENDERED and says nothing about what it may be.

    MUTATION: set ``DEFAULT_READ_HINT`` to a tool identifier (any ``snake_case`` token a
    catalog might really contain) -> this dies, and nothing else in the suite moves.
    """
    hint = IdProvenanceDispatcher.DEFAULT_READ_HINT
    assert not hint.isidentifier(), (
        f"the core's default read hint reads as a tool name ({hint!r}) — it must be prose "
        f"that names no tool, because the catalog belongs to the caller")
    assert " " in hint.strip(), f"a hint with no space is a token, not prose: {hint!r}"
    # POSITIVE CONTROL: without it, a check that accepted everything would pass over the
    # same constant and this test would stop discriminating.
    assert "list_records".isidentifier(), "the check no longer recognises a tool name"
