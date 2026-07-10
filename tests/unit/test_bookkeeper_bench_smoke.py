"""Smoke test for the BOOKKEEPER cognobench cases — guards the plumbing without Ollama.

Asserts the toolset + cases are well-formed (every expectation references a real tool) and the
dispatcher executes each tool, so CI catches a broken port even though the scored run needs a model.
"""

import pytest

from cognobench.bookkeeper_cases import (
    BOOKKEEPER_CASES,
    BOOKKEEPER_TOOLS,
    DESTRUCTIVE_TOOLS,
    SIDE_EFFECT_TOOLS,
    VALID_TOOLS,
    BookkeeperDispatcher,
)


def test_toolset_matches_the_vertical():
    names = {t["function"]["name"] for t in BOOKKEEPER_TOOLS}
    assert names == {"add_income", "add_outcome", "get_summary", "list_clients",
                     "search", "remove_by_search", "get_usage", "help"}
    assert SIDE_EFFECT_TOOLS <= names and DESTRUCTIVE_TOOLS <= SIDE_EFFECT_TOOLS
    assert "remove_by_search" in DESTRUCTIVE_TOOLS       # the held-for-confirmation tool


def test_every_case_expectation_references_a_real_tool():
    assert BOOKKEEPER_CASES
    for c in BOOKKEEPER_CASES:
        for tool in (c.expect_tool, c.expect_pending):
            assert not tool or tool in VALID_TOOLS, f"{c.id}: {tool!r} not a real tool"


@pytest.mark.asyncio
async def test_dispatcher_executes_every_tool_and_flags_side_effects():
    disp = BookkeeperDispatcher()
    for name in VALID_TOOLS:
        res = await disp.execute(name, {})
        assert res.ok, f"{name} failed"
        assert res.side_effect is (name in SIDE_EFFECT_TOOLS)
    assert disp.requires_confirmation("remove_by_search") is True
    assert disp.is_mutating("get_summary") is False
