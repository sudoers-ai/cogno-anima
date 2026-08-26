

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


def test_a_PARTIAL_turn_record_does_not_mask_the_survivors_write():
    """O piso não pode ser saltável.

    A forma anterior lia `turn_executions` e só consultava o `ego_result` quando ela estava
    VAZIA — o que assenta numa invariante que ninguém escreve e nada pina ("se não está vazia,
    está completa"). Com uma LEITURA em `turn_executions` e uma ESCRITA no `ego_result`, o
    predicado respondia False sobre um turno que escreveu. E False é a resposta LIBERADORA para
    quase todos os consumidores — o número vive num sítio só, o docstring do
    `committed_this_turn`, e repeti-lo aqui era a quarta cópia —, num predicado cujo primeiro
    parágrafo diz que o viés é fail-CLOSED. Achado pelo teste de paridade do cogno-host (#438)."""
    from types import SimpleNamespace as NS

    from cogno_anima.types import committed_this_turn

    escrita = NS(tool="book", ok=True, side_effect=True)
    leitura = NS(tool="list", ok=True, side_effect=False)
    ego = NS(tools_executed=[escrita])

    assert committed_this_turn(NS(metadata={}, turn_executions=[], ego_result=ego)) is True, \
        "pré-condição: com a lista vazia o piso é consultado"
    assert committed_this_turn(NS(metadata={}, turn_executions=[leitura], ego_result=ego)) is True, (
        "uma lista PARCIAL escondeu a escrita da sobrevivente — o piso voltou a ser saltável"
    )


def test_the_union_does_not_invent_a_commit():
    """A gémea: somar fontes só pode virar False→True, então a rede tem de provar que ela NÃO
    vira True sozinha. Nenhuma fonte com escrita ⇒ False, por mais listas que existam."""
    from types import SimpleNamespace as NS

    from cogno_anima.types import committed_this_turn

    leitura = NS(tool="list", ok=True, side_effect=False)
    falhou = NS(tool="book", ok=False, side_effect=True)
    ego = NS(tools_executed=[falhou])
    assert committed_this_turn(NS(metadata={}, turn_executions=[leitura], ego_result=ego)) is False


def test_a_BROKEN_source_costs_the_source_not_the_turn():
    """Irmão do `test_the_predicate_never_raises_however_broken_the_carrier`, que cobre a
    `metadata` a levantar e NÃO cobria a fonte de execuções.

    `EgoResult.tools_executed` é DERIVADO, e derivado pode levantar (um traço reproduzido sem os
    `steps`, um carregador magro). A primeira versão da união lia as duas fontes com avidez e
    passou a LEVANTAR num turno que a forma anterior respondia True — fail-open trocado por
    fail-FATAL, no turno pior possível: um que COMITOU. Achado em revisão, não por mim."""
    from types import SimpleNamespace as NS

    from cogno_anima.types import committed_this_turn

    class _EgoPartido:
        @property
        def tools_executed(self):
            raise RuntimeError("traço reproduzido sem os steps")

    escrita = NS(tool="book", ok=True, side_effect=True)

    ctx = NS(metadata={}, turn_executions=[escrita], ego_result=_EgoPartido())
    assert committed_this_turn(ctx) is True, (
        "a fonte partida custou o turno — e a OUTRA fonte tinha a resposta"
    )

    # E agora a ORDEM INVERSA, que é o caso que discrimina o `continue` do `return False`.
    # Acima, a fonte partida vem em SEGUNDO e a primeira já respondeu antes de lhe tocar — então
    # aquele caso passa com qualquer das duas degradações. Aqui a fonte partida vem PRIMEIRO e a
    # resposta está na SEGUNDA: só o `continue` a alcança. Um `except: return False` transformaria
    # "esta fonte não diz nada" em "nada foi comitado", trocando levantar por LIBERAR — que é o
    # erro exato que este PR existe para remover.
    class _CarrierComPrimeiraFontePartida:
        metadata: dict = {}

        @property
        def turn_executions(self):
            raise RuntimeError("carregador magro: a lista do turno não existe")

        ego_result = NS(tools_executed=[escrita])

    assert committed_this_turn(_CarrierComPrimeiraFontePartida()) is True, (
        "a primeira fonte partiu-se e a resposta estava na segunda — a degradação virou "
        "'nada foi comitado' em vez de 'esta fonte não diz nada'"
    )

    # E o nível ABAIXO: a lista lê-se, o ITEM é que explode. Distingue proteger a LEITURA de
    # proteger a VARREDURA — com o `any()` fora do `try`, a lista chega inteira e o predicado
    # levanta na iteração. Realista: `getattr(x, "attr", default)` engole só `AttributeError`, e
    # um `turn_executions` que seja gerador (um cursor duck-typed) falha ao iterar, não ao ler.
    # Item partido na PRIMEIRA fonte, pela mesma regra de ordem descoberta acima.
    class _ExecQueExplode:
        ok = True
        tool = "book"

        @property
        def side_effect(self):
            raise RuntimeError("derivada do próprio item")

    class _CarrierComItemPartido:
        metadata: dict = {}
        turn_executions = [_ExecQueExplode()]
        ego_result = NS(tools_executed=[escrita])

    assert committed_this_turn(_CarrierComItemPartido()) is True, (
        "o item partido matou a varredura — a proteção cobre a leitura da fonte e não a "
        "iteração sobre ela, e a segunda fonte tinha a resposta"
    )

    # E, sem NENHUMA fonte a falar, a resposta é False por ausência de prova — não por exceção.
    assert committed_this_turn(NS(metadata={}, turn_executions=[],
                                  ego_result=_EgoPartido())) is False


# ── a prosa sobre `committed_this_turn` tem de concordar consigo mesma ────────────────────
#
# O fato ("quantos lugares chamam o predicado?") está escrito em TRÊS sítios deste repo, em
# dois idiomas, e envelheceu nos três ao mesmo tempo: diziam "seis" enquanto a árvore tinha
# sete. Este repo não consegue CONTAR os chamadores — eles moram no cogno-host e no cogno-soma,
# que dependem daqui e não o contrário. Então ele fecha a metade que lhe cabe: os três sítios
# dizem o MESMO número. A outra metade ("esse número é o real") é pinada no host, onde os três
# pacotes são importáveis, por `tests/unit/test_committed_prose_matches_code.py`.
#
# Sozinho, isto não pega os três derivando JUNTOS — é o modo de falha gêmeo, e é exatamente por
# isso que a corrente precisa dos dois elos. Nenhum dos dois basta.

# Estendido para além de DEZ na terceira vez que a contagem passou do teto do mapa: um mapa
# curto transforma "a prosa está certa" em KeyError/falha opaca, que é a forma de falhar que
# esta suíte inteira existe para evitar.
# The count used to live in three files, and this test used to check that the three agreed.
# That check is gone because its SUBJECT is gone: `metakeys.py` and `CLAUDE.md` no longer state
# a count at all — they point at the docstring. Four consecutive PRs (#115-#118) each fixed the
# number in every copy and each left a stale sub-count or a missing name behind, because a test
# can pin a token and cannot read two hundred words of prose. Agreement between copies was
# always the weaker goal; having one copy is the stronger one.
#
# What replaces it is the inverse assertion: the other two sites must NOT restate the count.
# A future edit that "helpfully" re-inlines the list here is the exact regression, and it would
# have looked like documentation improving.
_CANONICO = "the count and the caller list live in `committed_this_turn`'s docstring only"


def test_the_other_sites_do_NOT_restate_the_caller_count():
    """`metakeys.py` and `CLAUDE.md` must POINT at the canonical enumeration, not copy it."""
    import pathlib
    import re

    from cogno_anima import metakeys as mk

    raiz = pathlib.Path(mk.__file__).resolve().parent.parent
    for nome, texto in (("metakeys.py", pathlib.Path(mk.__file__).read_text(encoding="utf-8")),
                        ("CLAUDE.md", (raiz / "CLAUDE.md").read_text(encoding="utf-8"))):
        achados = re.findall(r"(\w+) places CALL (?:this|that predicate)", texto)
        achados += re.findall(r"\*\*(\w+)\*\* CHAMAM este predicado", texto)
        achados += re.findall(r"uma definição, (\w+) consumidores", texto)
        assert not achados, (
            f"{nome} STATES the count again ({achados}) instead of pointing at the "
            f"docstring. {_CANONICO} — a second copy cost four consecutive PRs of drift, and "
            f"the regression would look like documentation improving."
        )
        assert "committed_this_turn" in texto, (
            f"{nome} stopped pointing at the canonical source — with neither the count NOR "
            f"the pointer, the reader is left with nothing."
        )


def test_the_canonical_docstring_still_states_a_count():
    """The other jaw of the pincer: pointing only works while the target still states it.

    The count itself is compared against the TREE by cogno-host
    (`test_committed_prose_matches_code`), the only repo that sees all three packages. This one
    guarantees only that the sentence exists — losing it would lose the fact, and two files
    would then point at nothing."""
    import re

    from cogno_anima.types import committed_this_turn

    assert re.search(r"(\w+) places CALL this", committed_this_turn.__doc__ or ""), (
        "the canonical docstring stopped declaring the count, and TWO files now point at it"
    )
