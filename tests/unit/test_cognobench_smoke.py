"""
Smoke test for cognobench plumbing — runs in stub mode (no Ollama, no network).

Scores are meaningless in stub mode (fixed LLM output); this only guarantees
the harness wires NOUMENO → NER → Drift, dimensions execute without error, and
the report aggregates correctly.
"""

import asyncio

from cognobench.runner import run_bench, ALL_DIMENSIONS


def _run_stub(**kw):
    return asyncio.run(run_bench(
        model="stub", embed_model="stub", base_url="http://localhost:11434",
        only=kw.get("only", []), stub=True, limit=kw.get("limit", 2),
        calibrate=kw.get("calibrate", False),
    ))


def test_stub_bench_runs_all_dimensions():
    report = _run_stub(limit=2)
    names = [d.name for d in report.dimensions]
    assert names == list(ALL_DIMENSIONS)
    assert report.total > 0


def test_stub_bench_has_no_pipeline_errors():
    """Plumbing must not raise — every case should produce checks, not errors."""
    report = _run_stub(limit=3)
    for dim in report.dimensions:
        assert dim.errors == [], f"{dim.name} raised: {dim.errors}"
        assert dim.total > 0


def test_stub_drift_hard_invariants_hold():
    """Drift hard invariants (valid action, cumulative in [0,1]) must pass in stub mode."""
    report = _run_stub(only=["drift"], limit=5)
    drift = next(d for d in report.dimensions if d.name == "drift")
    hard = [c for c in drift.checks if c.field in ("action_valid", "cumulative_range")]
    assert hard, "expected hard-invariant checks"
    assert all(c.correct for c in hard)


def test_stub_id_multi_turn_runs_and_hard_invariants_hold():
    """ID dimension runs multi-turn in stub mode; valid goal_status/route always hold."""
    report = _run_stub(only=["id"], limit=3)
    idd = next(d for d in report.dimensions if d.name == "id")
    assert idd.errors == [], f"id raised: {idd.errors}"
    hard = [c for c in idd.checks if c.field.endswith("_goal_status_valid")
            or c.field.endswith("_route_valid")]
    assert hard, "expected per-turn hard-invariant checks"
    assert all(c.correct for c in hard)
    # multi-turn cases produce checks for more than the first turn
    assert any(c.field.startswith("t2_") for c in idd.checks)


def test_report_to_dict_shape():
    report = _run_stub(only=["ner"], limit=1)
    d = report.to_dict()
    assert set(d) >= {"model", "overall_accuracy", "total", "correct", "dimensions"}
