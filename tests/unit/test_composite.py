"""Unit tests for cogno_anima.tools.CompositeDispatcher (merge many sources)."""


from cogno_anima.tools import CompositeDispatcher, ToolDispatcher, ToolPolicyDispatcher
from cogno_anima.types import ToolResult


class Stub:
    """A plain ToolDispatcher over a fixed set of tool names."""

    def __init__(self, *names):
        self._names = list(names)
        self.executed: list[str] = []

    def tools_schema(self):
        return [{"type": "function", "function": {"name": n, "description": n, "parameters": {}}}
                for n in self._names]

    async def execute(self, name, arguments):
        self.executed.append(name)
        return ToolResult(output=f"{name}->{self!r}", ok=True)


class PolicyStub(Stub):
    """A ToolPolicyDispatcher: classifies tools as mutating / destructive."""

    def __init__(self, *names, mutating=(), destructive=()):
        super().__init__(*names)
        self._mutating = set(mutating)
        self._destructive = set(destructive)

    def is_mutating(self, name):
        return name in self._mutating

    def requires_confirmation(self, name):
        return name in self._destructive


def test_satisfies_both_protocols():
    comp = CompositeDispatcher([Stub("a")])
    assert isinstance(comp, ToolDispatcher)
    assert isinstance(comp, ToolPolicyDispatcher)


def test_tools_schema_is_the_union():
    comp = CompositeDispatcher([Stub("a", "b"), Stub("c")])
    names = [s["function"]["name"] for s in comp.tools_schema()]
    assert names == ["a", "b", "c"]


async def test_execute_routes_to_owning_source():
    s1, s2 = Stub("a", "b"), Stub("c")
    comp = CompositeDispatcher([s1, s2])
    await comp.execute("b", {})
    await comp.execute("c", {})
    assert s1.executed == ["b"]
    assert s2.executed == ["c"]


async def test_unknown_tool_is_recoverable():
    comp = CompositeDispatcher([Stub("a")])
    res = await comp.execute("nope", {})
    assert res.ok is False
    assert "unknown tool" in (res.error or "")


def test_name_collision_first_source_wins():
    s1, s2 = Stub("dup"), Stub("dup")
    comp = CompositeDispatcher([s1, s2])
    # only one "dup" in the union
    assert [s["function"]["name"] for s in comp.tools_schema()] == ["dup"]


async def test_name_collision_routes_to_first():
    s1, s2 = Stub("dup"), Stub("dup")
    comp = CompositeDispatcher([s1, s2])
    await comp.execute("dup", {})
    assert s1.executed == ["dup"] and s2.executed == []


def test_schema_without_name_is_dropped():
    class Bad:
        def tools_schema(self):
            return [{"type": "function", "function": {"description": "no name"}}]

        async def execute(self, name, arguments):
            return ToolResult(output="", ok=True)

    comp = CompositeDispatcher([Bad(), Stub("ok")])
    assert [s["function"]["name"] for s in comp.tools_schema()] == ["ok"]


def test_policy_delegates_to_policy_source():
    pol = PolicyStub("read", "write", "drop", mutating=("write", "drop"), destructive=("drop",))
    comp = CompositeDispatcher([pol])
    assert comp.is_mutating("write") is True
    assert comp.is_mutating("read") is False
    assert comp.requires_confirmation("drop") is True
    assert comp.requires_confirmation("write") is False


def test_non_policy_source_is_conservative():
    """A plain source is unclassified → assume mutating (masked in read-only),
    but the opt-in confirmation gate does not fire."""
    comp = CompositeDispatcher([Stub("x")])
    assert comp.is_mutating("x") is True
    assert comp.requires_confirmation("x") is False


def test_mixed_sources_policy_per_owner():
    pol = PolicyStub("read", mutating=())          # read is non-mutating
    plain = Stub("legacy")                          # unclassified
    comp = CompositeDispatcher([pol, plain])
    assert comp.is_mutating("read") is False        # delegated to the policy source
    assert comp.is_mutating("legacy") is True       # conservative default


def test_empty_composite():
    comp = CompositeDispatcher([])
    assert comp.tools_schema() == []


async def test_empty_composite_execute_is_recoverable():
    comp = CompositeDispatcher([])
    res = await comp.execute("anything", {})
    assert res.ok is False


# ── source_requires_confirmation: the router must FORWARD the finer verdict ───────────
# Added 2026-09-01 after an end-to-end measurement: a contact who asked for a human was
# answered "Só pra confirmar: executo human handoff. Posso seguir?" and no escalation
# happened. A host floor that exists to prevent exactly that needs to tell a SOURCE's own
# destructive verdict from a blanket "every write confirms" rule; it reaches for the finer
# predicate through the wrappers and hit this router, which defined neither the method nor
# `__getattr__`, and fell back to holding.

class _Finer(PolicyStub):
    """A source that exposes the finer predicate, as a host gate wrapper does."""

    def __init__(self, *names, mutating=(), destructive=(), source_says=()):
        super().__init__(*names, mutating=mutating, destructive=destructive)
        self._source_says = set(source_says)

    def source_requires_confirmation(self, name):
        return name in self._source_says


def test_the_router_forwards_the_finer_predicate_to_the_owning_source():
    """The production shape: ask the source that owns the tool, not the first one.

    Mutation that kills it: drop the method — `getattr` on the router raises/misses and the
    caller above falls back to its conservative answer, which is what shipped the defect.
    """
    comp = CompositeDispatcher([
        _Finer("book", mutating=("book",), destructive=("book",), source_says=()),
        _Finer("wipe", mutating=("wipe",), destructive=("wipe",), source_says=("wipe",)),
    ])
    assert comp.source_requires_confirmation("book") is False   # blanket rule only
    assert comp.source_requires_confirmation("wipe") is True    # the source itself said so


def test_a_policy_source_without_the_finer_predicate_answers_with_its_own_verdict():
    """The branch that keeps the floor above SAFE, and the reason it is not simply `False`.

    A policy source with no blanket rule layered in front of it has one verdict, and that
    verdict IS the source's. Answering `False` here would let the caller waive a tool the
    source really did mark destructive — the single thing the anti-holding floor must never do.

    Mutation that kills it: `return False` for a source lacking the finer predicate.
    """
    comp = CompositeDispatcher([PolicyStub("cancel", mutating=("cancel",), destructive=("cancel",))])
    assert comp.source_requires_confirmation("cancel") is True


def test_a_source_with_no_policy_declared_nothing():
    """Symmetric with `requires_confirmation` above: an un-classified source did not opt in,
    so it declared nothing destructive — and it can never reach the gate anyway.

    Mutation that kills it: `return True` as a blanket conservative default — every composite
    with a plain source starts holding tools nobody classified.
    """
    comp = CompositeDispatcher([Stub("plain")])
    assert comp.source_requires_confirmation("plain") is False


def test_an_unknown_tool_is_not_claimed_by_anyone():
    """Mutation that kills it: index `_resolve()[1][name]` instead of `.get` — a hallucinated
    tool name raises inside a POLICY read, where the EGO expects a boolean."""
    comp = CompositeDispatcher([PolicyStub("a", mutating=("a",))])
    assert comp.source_requires_confirmation("does_not_exist") is False
