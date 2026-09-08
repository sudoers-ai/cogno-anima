"""The interval travels with the number, always.

`temperature=0` asks a hosted provider for greedy decoding and nothing obliges it to
deliver — measured on this bench, `--only safety` gave 39.6 / 45.1 / 46.3 % on ONE tree,
one model, one afternoon. So a bare headline is one draw, and the instrument has to say
so instead of leaving a reader to assume the number is repeatable.
"""

from __future__ import annotations

from cognobench.report import render, render_aggregate
from cognobench.types import BenchReport, CheckResult, DimensionResult, aggregate_runs


def _report(correct: list[bool], *, repeat_total: int = 1) -> BenchReport:
    dim = DimensionResult(name="d")
    for i, ok in enumerate(correct):
        dim.checks.append(CheckResult("c", f"f{i}", "X", "X", ok))
    rep = BenchReport(dimensions=[dim], model="m")
    rep.config["repeat_total"] = repeat_total
    return rep


def test_the_median_is_reported_and_is_not_the_mean_or_an_extreme():
    """A mean is dragged by the one bad draw a hosted provider hands you; min and max are
    the interval, not the estimate. [10, 2, 9] separates all four: median 9, mean 7."""
    runs = [_report([True] * n + [False] * (10 - n)) for n in (10, 2, 9)]
    agg = aggregate_runs(runs)["d"]
    assert agg["correct_median"] == 9
    assert agg["correct_median"] != sum((10, 2, 9)) // 3      # not the mean (7)
    assert agg["correct_min"] == 2 and agg["correct_max"] == 10


def test_the_aggregate_line_carries_median_interval_and_n():
    runs = [_report([True] * n + [False] * (10 - n)) for n in (10, 2, 9)]
    out = render_aggregate(aggregate_runs(runs))
    assert "9/10 [2–10]" in out, out
    assert "median" in out and "floor" in out
    # And the reason, in the instrument and not only in a doc nobody re-reads.
    assert "does not oblige" in out


def test_a_single_run_says_the_interval_is_MISSING():
    """The one that matters: a lone number must not read as a measurement."""
    out = render(_report([True, False]), show_failures=False)
    assert "n=1 run" in out and "NO interval" in out and "--repeat 3" in out


def test_a_repeated_run_does_not_claim_to_be_a_single_draw():
    """The twin — the note must be conditional, or it becomes wallpaper."""
    out = render(_report([True, False], repeat_total=3), show_failures=False)
    assert "NO interval" not in out


def test_an_empty_dimension_does_not_get_a_dispersion_note():
    """Nothing was measured, so there is nothing to say an interval about."""
    assert "NO interval" not in render(BenchReport(model="m"), show_failures=False)
