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
