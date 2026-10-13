"""CognoBench reports what each stage had to REPAIR — next to the score, never in it.

The side metric is ``DimensionResult.meta["repairs"]`` (mark → count) with its denominator
``repair_calls`` and ``repaired_calls``. Stub mode, no model: this guards the plumbing and the
arithmetic; the numbers that mean something come from a real run.
"""

import asyncio

from cogno_anima import repairs as rp
from cognobench.dimensions import _tally_repairs
from cognobench.report import render
from cognobench.runner import run_bench
from cognobench.types import DimensionResult

_STAGE_OF = {"noumeno": "noumeno", "ner": "ner", "superego_scope": "superego_scope",
             "superego_judge": "superego_judge"}


def _run_stub(only, limit=0):
    return asyncio.run(run_bench(
        model="stub", embed_model="stub", base_url="http://localhost:11434",
        only=only, stub=True, limit=limit, calibrate=False))


class _Result:
    def __init__(self, repairs):
        self.repairs = repairs


def test_the_tally_counts_marks_and_carries_its_denominator():
    dim = DimensionResult(name="ner")
    _tally_repairs(dim, _Result(["default:goal", "cut:goal"]))
    _tally_repairs(dim, _Result(["default:goal"]))
    _tally_repairs(dim, _Result([]))
    _tally_repairs(dim, object())                 # a stand-in with no list: a clean call
    assert dim.meta["repairs"] == {"cut:goal": 1, "default:goal": 2}
    assert (dim.meta["repair_calls"], dim.meta["repaired_calls"]) == (4, 2)
    assert dim.total == 0, "a side metric adds no check to the score"


def test_the_four_json_dimensions_carry_the_metric_and_the_others_do_not():
    report = _run_stub(["noumeno", "ner", "superego", "drift"], limit=0)
    by_name = {d.name: d for d in report.dimensions}
    for name, stage in _STAGE_OF.items():
        meta = by_name[name].meta
        assert meta["repair_calls"] > 0, name
        assert 0 <= meta["repaired_calls"] <= meta["repair_calls"], name
        assert set(meta["repairs"]) <= rp.REPAIRS_BY_STAGE[stage], name
        assert sum(meta["repairs"].values()) >= meta["repaired_calls"], name
    for name in ("superego_voice", "drift"):
        assert "repair_calls" not in by_name[name].meta, name


def test_a_call_is_a_reply_that_was_read_not_a_case_that_ran():
    """The denominator counts REPLIES. A scope case the guard bypasses asked the classifier
    nothing, so it is a case and not a call — counting it would dilute the rate with turns that
    could never have carried a mark."""
    report = _run_stub(["superego"], limit=0)
    scope = next(d for d in report.dimensions if d.name == "superego_scope")
    cases = {c.case_id for c in scope.checks}
    assert scope.meta["repair_calls"] < len(cases), "the premise: the suite has a bypass case"
    judge = next(d for d in report.dimensions if d.name == "superego_judge")
    assert judge.meta["repair_calls"] == len({c.case_id for c in judge.checks})


def test_the_score_is_what_it_was_without_the_metric(monkeypatch):
    """The control: with the tally replaced by nothing, every dimension scores the same."""
    with_metric = _run_stub(["noumeno", "ner", "superego"], limit=0)
    import cognobench.dimensions as dims
    monkeypatch.setattr(dims, "_tally_repairs", lambda dim, result: None)
    without = _run_stub(["noumeno", "ner", "superego"], limit=0)
    assert [(d.name, d.total, d.correct_count) for d in with_metric.dimensions] == [
        (d.name, d.total, d.correct_count) for d in without.dimensions]
    assert any("repair_calls" in d.meta for d in with_metric.dimensions)
    assert not any("repair_calls" in d.meta for d in without.dimensions)


def test_the_report_prints_the_metric_with_its_denominator():
    report = _run_stub(["ner"], limit=3)
    text = render(report, show_failures=False)
    ner = next(d for d in report.dimensions if d.name == "ner")
    line = next(ln for ln in text.splitlines() if ln.strip().startswith("repairs"))
    assert f"{ner.meta['repaired_calls']}/{ner.meta['repair_calls']} replies repaired" in line
    assert "[ner]" in line
