

# ── committed_this_turn: uma definição, três repos ────────────────────────────────────
def test_committed_reads_EVERY_attempt_not_just_the_survivor():
    """`ego_result` guarda só a tentativa que SOBREVIVEU — o orquestrador a substitui a cada
    retry. Seis consumidores liam o sobrevivente como se fosse o turno; a definição agora mora
    num lugar só, ao lado do tipo que ela interpreta."""
    from cogno_anima.types import (
        EgoResult,
        EgoStep,
        PipelineContext,
        StageMetrics,
        ToolExecution,
        committed_this_turn,
    )

    def _ego(calls):
        return EgoResult(steps=[EgoStep(index=0, path="native", assistant_text="",
                                        tool_calls=calls)],
                         metrics=StageMetrics(stage="ego", elapsed_ms=0.0, tokens_in=0,
                                              tokens_out=0, model="x"))

    wrote = ToolExecution(tool="confirm_appointment", arguments={}, result="now CONFIRMED",
                          ok=True, side_effect=True)
    read = ToolExecution(tool="list_appointments", arguments={}, result="3 rows", ok=True,
                         side_effect=False)
    failed = ToolExecution(tool="book_appointment", arguments={}, result="", error="slot taken",
                           ok=False, side_effect=True)

    ctx = PipelineContext(user_input="x")
    ctx.turn_executions = [wrote, read]      # a escrita ficou numa tentativa descartada…
    ctx.ego_result = _ego([read])            # …e a sobrevivente só leu
    assert committed_this_turn(ctx) is True

    # braço-controle: turno que só leu não é commit (sem ele, "sempre True" passaria)
    only_read = PipelineContext(user_input="x")
    only_read.turn_executions = [read, read]
    only_read.ego_result = _ego([read])
    assert committed_this_turn(only_read) is False

    # mutação que FALHOU não mudou nada
    broke = PipelineContext(user_input="x")
    broke.turn_executions = [failed]
    assert committed_this_turn(broke) is False

    # piso: sem acumulação (pipeline de um tiro só), cai no ego_result
    floor = PipelineContext(user_input="x")
    floor.ego_result = _ego([wrote])
    assert committed_this_turn(floor) is True
