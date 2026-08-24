

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


def test_a_commit_does_not_un_happen_when_its_context_dies():
    """A terceira fonte, e ela existe porque as duas primeiras podem sumir juntas.

    `turn_executions` e `ego_result` moram AMBOS no contexto, então morrem com ele — e há um
    caminho em que ele morre no meio do turno: o fallback de roteamento de modelo do host. A
    tentativa 1 comita uma escrita ordinária, um estágio POSTERIOR levanta, e a exceção leva o
    contexto e o registro de execução junto. A retentativa é um turno novo, com contexto novo,
    em que a escrita não está em lugar nenhum.

    Ali só o host sabe — então ele declara, e este predicado acredita. É por isso que a fixture
    tem de ser a da RETENTATIVA (contexto vazio + declaração), não a do caminho normal: um
    teste que exercite só o caminho normal passa com o defeito.
    """
    from cogno_anima import metakeys as mk
    from cogno_anima.types import PipelineContext, committed_this_turn

    # a retentativa: nada no contexto — a evidência morreu com a tentativa 1
    retry = PipelineContext(user_input="confirma pra mim")
    assert committed_this_turn(retry) is False           # sem a declaração, invisível
    retry.metadata[mk.PRIOR_ATTEMPT_COMMITTED] = True
    assert committed_this_turn(retry) is True            # o host disse, e isto acredita

    # TRUE-only: ausente significa "nada a acrescentar", nunca "nada foi comitado"
    absent = PipelineContext(user_input="x")
    assert committed_this_turn(absent) is False
    absent.metadata[mk.PRIOR_ATTEMPT_COMMITTED] = False
    assert committed_this_turn(absent) is False

    # e a declaração NÃO apaga o que o contexto mostra: um turno que escreveu de verdade
    # continua True mesmo sem ela (a terceira fonte soma, não substitui)
    from cogno_anima.types import EgoResult, EgoStep, StageMetrics, ToolExecution
    wrote = ToolExecution(tool="confirm_appointment", arguments={}, result="now CONFIRMED",
                          ok=True, side_effect=True)
    real = PipelineContext(user_input="x")
    real.ego_result = EgoResult(
        steps=[EgoStep(index=0, path="native", assistant_text="", tool_calls=[wrote])],
        metrics=StageMetrics(stage="ego", elapsed_ms=0.0, tokens_in=0, tokens_out=0, model="x"))
    assert committed_this_turn(real) is True


def test_the_predicate_never_raises_however_broken_the_carrier():
    """Seis camadas dependem dele — o portão do soma, o cache semântico, o medidor de escrita
    não aprovada, dois guards de re-step e o backstop de grounding. Se ele levantar, cai um
    turno cuja resposta já existe. A fonte nova lê `metadata`, que num carrier pato pode ser
    qualquer coisa."""
    from types import SimpleNamespace

    from cogno_anima.types import committed_this_turn

    class _Exploding:
        @property
        def metadata(self):
            raise RuntimeError("boom")

    assert committed_this_turn(_Exploding()) is False
    assert committed_this_turn(SimpleNamespace()) is False                    # sem metadata
    assert committed_this_turn(SimpleNamespace(metadata="não é mapa")) is False
    assert committed_this_turn(SimpleNamespace(metadata=None)) is False


def test_the_inlined_constant_matches_the_metakey():
    """`types.py` repete a string em vez de importar `metakeys` (é o fundo do pacote, e
    importar de lado por uma string é como um ciclo começa). Repetição só é segura pinada."""
    from cogno_anima import metakeys as mk
    from cogno_anima.types import _PRIOR_ATTEMPT_COMMITTED

    assert _PRIOR_ATTEMPT_COMMITTED == mk.PRIOR_ATTEMPT_COMMITTED

    # And the VALUE, not only the two copies to each other: a `sed -i` over the repo moves both
    # together and ships green, while `cogno_host/commit_sink.py` repeats the literal on purpose
    # (to stay importable against an older pin). Renaming the value is a data migration, which
    # is precisely what `metakeys.py` warns about — "a typo in a string does not fail, it
    # silently no-ops the feature". Without this line the two-copy assert only LOOKS like drift
    # protection.
    assert mk.PRIOR_ATTEMPT_COMMITTED == "prior_attempt_committed"
