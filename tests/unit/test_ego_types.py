"""Unit tests for the EGO data contracts and PipelineContext wiring."""

from cogno_core.types import (
    StageMetrics, ToolResult, ToolExecution, EgoStep, EgoResult, PipelineContext,
)


def _m(stage="ego", ti=0, to=0, et=0):
    return StageMetrics(stage=stage, elapsed_ms=1.0, tokens_in=ti, tokens_out=to,
                        embedding_tokens=et, model="stub")


def test_toolresult_defaults():
    r = ToolResult(output="ok")
    assert r.ok is True and r.error is None and r.side_effect is False
    # the dead fields really are gone
    assert not hasattr(r, "returns_raw_json")
    assert not hasattr(r, "compensating_tool")


def test_tools_executed_flattens_steps():
    steps = [
        EgoStep(index=0, path="native", assistant_text="",
                tool_calls=[ToolExecution(tool="a"), ToolExecution(tool="b")]),
        EgoStep(index=1, path="native", assistant_text="done"),
    ]
    res = EgoResult(steps=steps, metrics=_m())
    assert [t.tool for t in res.tools_executed] == ["a", "b"]


def test_draft_is_last_assistant_text():
    res = EgoResult(steps=[
        EgoStep(index=0, path="native", assistant_text="thinking",
                tool_calls=[ToolExecution(tool="a")]),
        EgoStep(index=1, path="native", assistant_text="final draft"),
    ], metrics=_m())
    assert res.draft == "final draft"


def test_draft_empty_when_no_steps():
    assert EgoResult(steps=[], metrics=_m()).draft == ""


def test_has_side_effects():
    res = EgoResult(steps=[EgoStep(index=0, path="native", tool_calls=[
        ToolExecution(tool="get", side_effect=False),
        ToolExecution(tool="write", side_effect=True),
    ])], metrics=_m())
    assert res.has_side_effects is True

    clean = EgoResult(steps=[EgoStep(index=0, path="native", tool_calls=[
        ToolExecution(tool="get", side_effect=False)])], metrics=_m())
    assert clean.has_side_effects is False


def test_no_user_facing_response_field():
    res = EgoResult(steps=[], metrics=_m())
    assert not hasattr(res, "response")
    assert not hasattr(res, "response_source")


def test_pipeline_context_ego_wiring():
    ctx = PipelineContext(user_input="hi")
    assert ctx.ego_result is None and ctx.ego_metrics is None

    ctx.ego_result = EgoResult(steps=[], metrics=_m(ti=14, to=6))
    assert ctx.ego_metrics is not None
    assert ctx.ego_metrics.tokens_total == 20
    # folded into the totals like the other stages
    assert ctx.ego_metrics in ctx.stage_metrics
    assert ctx.total_tokens == 20
