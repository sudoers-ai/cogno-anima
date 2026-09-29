"""A source read that was ON THE TABLE and never called — the fact the orchestrator needs.

The defect this exists for, measured on a rehearsal tenant (the shape only, never its data):

    an INFORMATION_REQUEST to a bookkeeping persona, with a document-reading tool offered
    the executor called only the ledger summary, which came back empty
    the draft asserted ABSENCE: "there are no rent entries for that property"
    the judge rejected it — rightly — and with a budget of one attempt the voice shipped the
    same negative, which was false: the documents held the four values

The voice cannot read, so re-voicing cannot repair that turn. The orchestrator can (cogno-soma
`_owes_a_read`), but only if it can tell *"the read was never made"* apart from *"the read was
made and found nothing"*. That second reading is a TRUE negative and must pass as today.

`source_reads_not_called` answers that one question. Three inputs, each from the layer that owns
it: WHICH tools are source reads (the host's `mk.SOURCE_READS`), WHAT was on the table
(`ego_result.tools_offered`) and WHAT was called (the shared walk, `_any_execution`). The consult
half of the walk is pinned in `test_consult_is_the_second_source.py`, beside the rest of the
family. This file pins the other halves.

MUTATIONS (one per property, each named in the PR with what died):

  * drop the `_any_execution` clause (every declared read is "not called")
        -> the made-and-empty control and the failed-read control die;
  * `unreadable=True` -> `unreadable=False` on that walk
        -> the unreadable-carrier test dies;
  * drop the `on_table` filter (owed even when not offered)
        -> the not-offered control dies;
  * drop the bare-`str` branch
        -> the one-name test dies.
"""

from __future__ import annotations

import pytest

from cogno_anima import metakeys as mk
from cogno_anima import source_reads_not_called as exported
from cogno_anima.types import (
    EgoResult,
    EgoStep,
    PipelineContext,
    StageMetrics,
    ToolExecution,
    source_reads_not_called,
)

DOCS = "consult_documents"
SUMMARY = "get_summary"
OFFERED = [SUMMARY, DOCS, "resolve_date"]


def _m() -> StageMetrics:
    return StageMetrics(stage="ego", elapsed_ms=1.0, tokens_in=1, tokens_out=1, model="t")


def _read(tool: str, *, ok: bool = True, result: str = "3 rows") -> ToolExecution:
    return ToolExecution(tool=tool, arguments={"query": "rent"}, result=result, ok=ok,
                         side_effect=False, tool_mutating=False)


def _trace(*calls: ToolExecution, offered=OFFERED,
           draft: str = "There are no rent entries for that property.") -> EgoResult:
    return EgoResult(
        steps=[EgoStep(index=0, path="native", tool_calls=list(calls)),
               EgoStep(index=1, path="native", assistant_text=draft)],
        tools_offered=list(offered), metrics=_m())


def _turn(*calls: ToolExecution, offered=OFFERED, declared=(DOCS,)) -> PipelineContext:
    ctx = PipelineContext(user_input="quanto recebi de aluguel da casa da praia?")
    if declared is not None:
        ctx.metadata[mk.SOURCE_READS] = list(declared)
    ctx.ego_result = _trace(*calls, offered=offered)
    return ctx


def test_the_inlined_constant_matches_the_metakey():
    """`types.py` spells the key instead of importing `metakeys` (it is the bottom of the
    package). A typo in either would silently switch the feature off, and nothing else fails."""
    from cogno_anima.types import _SOURCE_READS

    assert _SOURCE_READS == mk.SOURCE_READS == "source_reads"


def test_it_is_exported_at_the_package_root():
    assert exported is source_reads_not_called


# ── the twin ────────────────────────────────────────────────────────────────────────────────

def test_the_measured_shape_owes_the_read_it_never_made():
    """THE PROPERTY. The documents were offered, only the ledger summary was called, and it
    came back empty. The answer names the tool, which is what the orchestrator's closed
    sentence will say to the executor on the extra pass."""
    ctx = _turn(_read(SUMMARY, result="Income R$0,00 (0 entries)"))
    assert source_reads_not_called(ctx) == [DOCS]


# ── the controls: each is a true negative, or a turn with nothing to owe ────────────────────

def test_a_read_that_was_MADE_and_found_nothing_is_not_owed():
    """Control (a). "Nothing relevant" out of the documents is a TRUE negative. A re-run would
    repeat a call the trace already shows, and the reply that says so must pass as today."""
    ctx = _turn(_read(SUMMARY, result="0 entries"), _read(DOCS, result="nothing relevant"))
    assert source_reads_not_called(ctx) == []


def test_a_read_that_FAILED_was_still_made():
    """`ok` is deliberately absent. The executor reached for the source and the call failed;
    the critique it drew is about that attempt, not about the absence of one."""
    ctx = _turn(_read(DOCS, ok=False, result=""))
    assert source_reads_not_called(ctx) == []


def test_a_source_that_was_NOT_on_the_table_is_never_owed():
    """Control (b) and (c). A persona without the documents cannot be told to read them. The
    table here is the one the production matches had: other reads, no document read."""
    ctx = _turn(_read(SUMMARY), offered=[SUMMARY, "knowledge_search", "search_history"])
    assert source_reads_not_called(ctx) == []


@pytest.mark.parametrize("declared", [None, [], (), "", "   ", {DOCS: True}, 7, [7, None]],
                         ids=["absent", "empty_list", "empty_tuple", "empty_str", "blank_str",
                              "mapping", "int", "no_strings"])
def test_nothing_declared_is_OFF(declared):
    """Control (f). No usable declaration means nothing is a source read. A host reverts the
    feature by not stamping the key, and a garbled key must read as that, never as "owed"."""
    ctx = _turn(_read(SUMMARY), declared=None)
    if declared is not None:
        ctx.metadata[mk.SOURCE_READS] = declared
    assert source_reads_not_called(ctx) == []


def test_a_bare_string_is_ONE_name():
    """Iterating ``"consult_documents"`` would yield single letters, none of them offered, and
    the feature would be off without anybody deciding it."""
    ctx = _turn(_read(SUMMARY))
    ctx.metadata[mk.SOURCE_READS] = DOCS
    assert source_reads_not_called(ctx) == [DOCS]


# ── the walk: the whole turn, not the surviving pass ────────────────────────────────────────

def test_a_read_made_on_an_EARLIER_pass_is_not_owed():
    """The correction loop REPLACES `ego_result` on every pass. The read lives only in
    `turn_executions`, the accumulator, and it still counts: the turn made it."""
    ctx = _turn(_read(SUMMARY))
    ctx.turn_executions = [_read(DOCS, result="nothing relevant")]
    assert source_reads_not_called(ctx) == []


def test_it_is_ALL_OR_NOTHING_on_the_called_side():
    """Two declared sources, one called. The executor looked in a source, so what it drafted is
    a reading of that source, which is not the defect. The pair proves the fixture can owe."""
    ctx = _turn(_read("knowledge_search"), offered=OFFERED + ["knowledge_search"],
                declared=(DOCS, "knowledge_search"))
    assert source_reads_not_called(ctx) == []

    none_called = _turn(_read(SUMMARY), offered=OFFERED + ["knowledge_search"],
                        declared=("knowledge_search", DOCS))
    assert source_reads_not_called(none_called) == [DOCS, "knowledge_search"], (
        "sorted, so every worker names the tools in the same order")


def test_only_what_was_OFFERED_is_returned():
    ctx = _turn(_read(SUMMARY), declared=(DOCS, "knowledge_search"))
    assert source_reads_not_called(ctx) == [DOCS]


# ── never costs the turn, and leans towards "called" ────────────────────────────────────────

class _Unreadable:
    """A duck-typed carrier whose own execution lists RAISE, with the source offered."""

    user_input = "quanto recebi?"
    metadata = {mk.SOURCE_READS: [DOCS]}
    consult_result = None

    @property
    def turn_executions(self):
        raise RuntimeError("the accumulator cannot be read")

    @property
    def ego_result(self):
        return _BrokenEgo()


class _BrokenEgo:
    tools_offered = OFFERED

    @property
    def tools_executed(self):
        raise RuntimeError("derived, and derived can raise")


def test_an_unreadable_turn_owes_nothing():
    """The consumer grants an EXTRA executor pass on a non-empty answer. A turn whose record
    cannot be read is not evidence that the read was skipped, so it gets one pass, as today."""
    assert source_reads_not_called(_Unreadable()) == []


def test_a_carrier_with_no_metadata_owes_nothing():
    class _Bare:
        pass

    assert source_reads_not_called(_Bare()) == []
