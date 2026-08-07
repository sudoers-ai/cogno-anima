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
    # "superego" is a selector: it scores as THREE dimensions (plan 0.5 — model
    # choice is per op, and one blended score hid a per-op split).
    expected = []
    for d in ALL_DIMENSIONS:
        expected.extend(["superego_scope", "superego_judge", "superego_voice"]
                        if d == "superego" else [d])
    assert names == expected
    assert report.total > 0


def test_stub_bench_has_no_pipeline_errors():
    """Plumbing must not raise — every case should produce checks, not errors.

    A small --limit can slice a per-op superego sub-dimension down to zero cases
    (the cap is positional); an empty sub-dim is fine, an errored one is not."""
    report = _run_stub(limit=3)
    assert report.total > 0
    for dim in report.dimensions:
        assert dim.errors == [], f"{dim.name} raised: {dim.errors}"
        assert dim.valid, f"{dim.name} invalid: {dim.invalid_reason}"


def test_stub_safety_hard_invariants_hold():
    """The safety dimension's ``*_deterministic`` checks are decided by the regex PII
    detector on the ORIGINAL text — no LLM involved — so every one of them must pass
    even under the stub backend. This is what makes the PII tier/validator/blocked-gate
    contract CI-enforceable (``*_llm`` checks are real-model only and may fail here)."""
    report = _run_stub(only=["safety"], limit=0)
    saf = next(d for d in report.dimensions if d.name == "safety")
    assert saf.errors == [], f"safety raised: {saf.errors}"
    hard = [c for c in saf.checks if c.field.endswith("_deterministic")]
    assert len(hard) >= 15, "expected the deterministic check set"
    bad = [(c.case_id, c.field, c.actual) for c in hard if not c.correct]
    assert not bad, f"deterministic safety checks failed: {bad}"
    # the CRITICAL credential case must trip the ID block gate even in stub mode
    assert any(c.case_id == "safety_credential_kv_blocks" and c.field == "blocked_deterministic"
               and c.correct for c in saf.checks)


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


def test_stub_ego_runs_and_hard_invariants_hold():
    """EGO dimension runs in stub mode; valid-steps/valid-tools invariants hold."""
    report = _run_stub(only=["ego"], limit=3)
    ego = next(d for d in report.dimensions if d.name == "ego")
    assert ego.errors == [], f"ego raised: {ego.errors}"
    hard = [c for c in ego.checks if c.field in ("steps_present", "dispatched_tools_valid")]
    assert hard, "expected EGO hard-invariant checks"
    assert all(c.correct for c in hard)


def test_stub_superego_runs_and_hard_invariants_hold():
    """SUPEREGO runs in stub mode as three per-op dimensions; bool/non-empty
    invariants hold in each (limit=0 → full suite, so every kind has cases)."""
    report = _run_stub(only=["superego"], limit=0)
    by_name = {d.name: d for d in report.dimensions}
    assert set(by_name) == {"superego_scope", "superego_judge", "superego_voice"}
    for name, se in by_name.items():
        assert se.errors == [], f"{name} raised: {se.errors}"
        assert se.total > 0, f"{name} scored no checks"
        hard = [c for c in se.checks if c.field in
                ("blocked_is_bool", "approved_is_bool", "response_nonempty")]
        assert hard, f"expected {name} hard-invariant checks"
        assert all(c.correct for c in hard)


def test_stub_conversations_runs_and_hard_invariants_hold():
    """Full-pipeline conversation simulation runs multi-turn in stub mode."""
    report = _run_stub(only=["conversations"], limit=3)
    conv = next(d for d in report.dimensions if d.name == "conversations")
    assert conv.errors == [], f"conversations raised: {conv.errors}"
    hard = [c for c in conv.checks if c.field in ("route_valid", "reached_terminal",
                                                  "dispatched_tools_valid")]
    assert hard, "expected conversation hard-invariant checks"
    assert all(c.correct for c in hard)
    # multi-turn: at least one case produced a 2nd-turn check
    assert any(".t2" in c.case_id for c in conv.checks)


def test_report_to_dict_shape():
    report = _run_stub(only=["ner"], limit=1)
    d = report.to_dict()
    assert set(d) >= {"model", "overall_accuracy", "total", "correct", "dimensions",
                      "run_id", "config", "suites", "invalid_dimensions"}


def test_report_to_dict_serializes_every_check():
    """Plan 0.1: an aggregate-only artifact cannot feed the noise floor, the
    stable score or the discrimination report — include_checks must carry every
    CheckResult with its join key."""
    report = _run_stub(only=["ner"], limit=2)
    d = report.to_dict(include_checks=True)
    dim = next(x for x in d["dimensions"] if x["dimension"] == "ner")
    assert len(dim["checks"]) == dim["total"] > 0
    assert set(dim["checks"][0]) == {"case_id", "field", "expected", "actual", "correct"}


def test_invalid_dimension_never_scores():
    """Plan 0.3: a transport failure marks the dimension INVALID and excludes it
    from every total — never a silently smaller denominator."""
    from cognobench.types import BenchReport, DimensionResult, CheckResult
    ok = DimensionResult(name="a", checks=[CheckResult("c", "f", "e", "e", True)])
    bad = DimensionResult(name="b", checks=[CheckResult("c", "f", "e", "x", False)],
                          invalid_reason="transport: boom")
    report = BenchReport(dimensions=[ok, bad])
    assert report.total == 1 and report.correct_count == 1
    assert [d.name for d in report.invalid_dimensions] == ["b"]
    assert report.to_dict()["invalid_dimensions"] == ["b"]


def test_transport_error_invalidates_model_error_scores():
    """Plan 0.3 classification: StageParseError = the model's fault → one failed
    check in the full denominator; transport/unknown = instrument → invalid."""
    import httpx
    from cognobench.dimensions import _note_error
    from cognobench.types import DimensionResult
    from cogno_anima.errors import StageParseError

    dim = DimensionResult(name="x")
    stop = _note_error(dim, "case1",
                       StageParseError("ner", "not json", ValueError("bad")))
    assert stop is False and dim.valid
    assert dim.total == 1 and dim.correct_count == 0     # scored as a failure

    stop = _note_error(dim, "case2", httpx.ConnectError("down"))
    assert stop is True and not dim.valid
    assert "transport" in dim.invalid_reason


def test_aggregate_runs_stable_score_and_noise_floor():
    """Plan 0.2: unstable = diverges across runs; stable = STRICT majority
    (a tie is fail — conservative and order-independent); floor =
    max(unstable, 2)."""
    from cognobench.types import BenchReport, DimensionResult, CheckResult, aggregate_runs

    def run(flip: bool):
        return BenchReport(dimensions=[DimensionResult(name="d", checks=[
            CheckResult("c1", "f", "e", "e", True),
            CheckResult("c2", "f", "e", "x", flip),      # diverges across runs
            CheckResult("c3", "f", "e", "x", False),
        ])])

    agg = aggregate_runs([run(True), run(True), run(False)])
    d = agg["d"]
    assert d["checks"] == 3 and d["unstable_checks"] == 1
    assert d["stable_correct"] == 2          # c1 always; c2 passes 2/3 (majority)
    assert d["noise_floor"] == 2             # max(1 unstable, conservative 2)
    assert d["correct_min"] == 1 and d["correct_max"] == 2


def test_aggregate_tie_is_fail_and_order_independent():
    """Review finding: Counter-based tie-break let run ORDER decide a 1-of-2
    split (and diverged from compare.py). Strict majority: [T,F] == [F,T] == fail."""
    from cognobench.types import BenchReport, DimensionResult, CheckResult, aggregate_runs

    def run(ok: bool):
        return BenchReport(dimensions=[DimensionResult(name="d", checks=[
            CheckResult("c1", "f", "e", "x", ok)])])

    for order in ([True, False], [False, True]):
        agg = aggregate_runs([run(order[0]), run(order[1])])
        assert agg["d"]["stable_correct"] == 0, f"tie must fail (order {order})"
        assert agg["d"]["unstable_checks"] == 1


def test_compare_roundtrip_excludes_calibrate_and_expands_case_errors(tmp_path):
    """--out → compare round trip (review finding: zero tests). Two things must
    hold: a --calibrate artifact never joins, and a run where a case garbled
    (case_error) votes FAIL on that case's keys from the sibling run."""
    import json as _json
    from cognobench import compare

    def write(name, model, checks, calibrate=False):
        (tmp_path / name).write_text(_json.dumps({
            "model": model, "config": {"calibrate": calibrate, "limit": None},
            "suites": {"ner": {"suite_id": "ner-v1", "suite_hash": "abc"}},
            "dimensions": [{"dimension": "ner", "valid": True, "checks": checks}],
        }))

    ok = {"case_id": "c1", "field": "intent_class", "expected": "X", "actual": "X",
          "correct": True}
    err = {"case_id": "c1", "field": "case_error", "expected": "no exception",
           "actual": "StageParseError", "correct": False}

    write("r1.json", "m", [ok])                       # run 1: c1 passes
    write("r2.json", "m", [err])                      # run 2: c1 garbles
    write("r3.json", "m", [ok], calibrate=True)       # calibrate: must not join
    by_model = compare._index(compare.load_runs([tmp_path]))
    votes = by_model["m"]["dims"]["ner"][("c1", "intent_class")]
    assert votes == [True, False], (
        f"expected the garbled run to vote FAIL on the sibling's key and the "
        f"calibrate run to be excluded; got {votes}"
    )
    assert compare._stable(votes) is False            # tie → fail, both tools agree
