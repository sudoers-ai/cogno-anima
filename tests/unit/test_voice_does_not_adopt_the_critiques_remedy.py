"""A crítica é uma nota para o EXECUTOR. A voz estava a lê-la como o guião da resposta.

## O turno real

Um professor perguntou, quatro dias seguidos, quanto ia receber pelas suas aulas. Na véspera
a persona que fala com ele tinha chamado `transfer_persona` para o contabilista — a trava
abriu. Nos dois turnos seguintes foi o CONTABILISTA que executou: leu o livro-caixa, e
escreveu o rascunho verdadeiro («não há entradas financeiras registradas»).

O juiz rejeitou-o com esta crítica, verbatim:

    «A pergunta é financeira e está fora do escopo da persona Sofia; deveria ter sido
     encaminhada ao Senhor Barriga, em vez de respondida diretamente com um resumo financeiro.»

`judge_rejected_all` → `last_draft_voiced`, e o que chegou ao professor foi «essa questão
financeira deve ser direcionada ao Senhor Barriga». **O sistema já o tinha encaminhado ao
Barriga; foi o Barriga que executou o turno.**

## O que a secção dizia, e o que não dizia

A crítica aterra no prompt VERBATIM, debaixo de um cabeçalho que diz HARD RULE, e **nada à
volta dela diz para quem foi escrita**. As duas frases que lá estavam dizem o que se pode
afirmar e o que não se pode afirmar; nenhuma proíbe **adoptar o remédio que a crítica
propõe** — e foi esse o movimento medido.

Contado sobre 1514 traços de uma era comprovável (filtro `xmin`; uma linha reescrita depois de
criada é invisível a uma leitura ingénua e ~4,5% do corpus tem essa forma): dos turnos
carimbados `last_draft_voiced`, **11** entregaram uma resposta que manda o contacto para outro
lado — pessoa, departamento, canal — sobre um rascunho que não mandava. **5 desses 11 são o
mesmo professor.**

## Porque é INCONDICIONAL, ao contrário do `nothing_tried`

Aquela cláusula proíbe uma frase que às vezes é VERDADEIRA (uma escrita pode mesmo ter sido
tentada e ter falhado), logo tem de ser condicionada ao facto que a decide. Esta proíbe um
movimento que a voz nunca pode fazer **por autoridade da crítica**: a crítica não é, e não pode
ser, prova sobre para onde um contacto deve ser mandado. Um destino que as instruções da
PERSONA ou os DADOS DO EXECUTOR nomeiem fica intacto.

## As duas regras que foram MEDIDAS e RECUSADAS

Este ficheiro também existe para as pinar como recusadas, porque ambas são a coisa óbvia:

* **«vozear o último rascunho na exaustão»** — publicaria fabricações: 4 daqueles 11 turnos
  foram rejeitados precisamente PORQUE o rascunho inventou as figuras;
* **«disparar quando a tentativa sobrevivente tem leitura bem-sucedida»** — não discrimina:
  203 de 239 turnos exaustos têm uma, as 4 fabricações incluídas.

Por isso o rascunho **continua descartado** e o grounding continua a ser os dados do executor.
`test_o_rascunho_rejeitado_CONTINUA_a_nao_ser_re_oferecido` é o que impede a primeira de
entrar por descuido mais tarde.

Determinístico: as asserções são sobre o prompt RENDERIZADO, não sobre a resposta de um modelo.
"""

from __future__ import annotations

from cogno_anima import metakeys as mk
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.types import EgoResult, EgoStep, ToolExecution

from tests.unit.test_superego import _ctx, _m

# A crítica real do turno medido.
CRITICA = ("A pergunta é financeira e está fora do escopo da persona Sofia; deveria ter sido "
           "encaminhada ao Senhor Barriga, em vez de respondida diretamente com um resumo "
           "financeiro.")

# O que a cláusula acrescenta, asserido sobre o PROMPT.
NOTA_PARA_O_EXECUTOR = "a note about the EXECUTION, written for the executor"
PROIBE_MANDAR_EMBORA = "MUST NOT turn that into the answer by sending the contact away"
PERMITE_O_DESTINO_LEGITIMO = ("Name a destination only when the persona's own instructions or "
                              "the executor data give you one")
NUNCA_AFIRMAR_FEITO = "MUST NOT claim, imply or narrate that any action was performed"

# O rascunho verdadeiro que o contacto nunca viu.
RASCUNHO = ("Não há entradas financeiras registradas para as aulas entre setembro e dezembro "
            "de 2026.")
DADOS = "get_summary: Income:  R$ 0,00 (0 entries)\nExpense: R$ 0,00 (0 entries)"


def _turno_do_professor():
    """O turno medido: uma LEITURA correu, nada foi escrito, e o rascunho era verdadeiro."""
    ctx = _ctx(with_ego=False)
    ctx.user_input = ("quero saber, financeiramente, quanto eu vou ganhar em reais, pelas "
                      "aulas até o final do ano, agrupados por mês.")
    ctx.ego_result = EgoResult(
        steps=[EgoStep(index=0, path="native", assistant_text=RASCUNHO,
                       tool_calls=[ToolExecution(tool="get_summary", arguments={},
                                                 result="Income: R$ 0,00 (0 entries)",
                                                 ok=True, side_effect=False,
                                                 tool_mutating=False)])],
        metrics=_m("ego"))
    return ctx


def _rendered(ctx, *, kind=None, reason=CRITICA):
    carrier = {"reason": reason}
    if kind is not None:
        carrier["kind"] = kind
    ctx.metadata[mk.VOICE_CORRECTION] = carrier
    return SuperegoStage()._build_voice_prompt(ctx, DADOS, ["general:review"])


# ── o gémeo positivo: o turno real ────────────────────────────────────────────────────────

def test_a_critica_e_para_o_EXECUTOR_e_a_voz_e_avisada_disso():
    """O defeito, no prompt que a voz realmente recebeu.

    Mutação: tirar as quatro frases acrescentadas — é o estado de antes, e este teste cai.
    """
    prompt = _rendered(_turno_do_professor())
    assert CRITICA in prompt, "o andaime não pôs sequer a crítica no prompt"
    assert NOTA_PARA_O_EXECUTOR in prompt, (
        "a crítica aterra verbatim debaixo de um HARD RULE e nada diz para quem foi escrita")
    assert PROIBE_MANDAR_EMBORA in prompt, (
        "nada proíbe transformar «devia ter sido encaminhada a X» em «vá procurar o X»")
    # A regra que já lá estava não se perde: afirmar que se fez continua proibido.
    assert NUNCA_AFIRMAR_FEITO in prompt


# ── o gémeo que impede que isto vire mordaça ──────────────────────────────────────────────

def test_um_destino_LEGITIMO_continua_permitido():
    """A vítima fácil: um inquilino que lista o seu próprio financeiro na configuração.

    Proibir «mandar o contacto para outro lado» em bloco apagaria uma resposta correcta. A
    cláusula proíbe a crítica como FONTE, e nomeia as duas fontes que continuam a valer.
    """
    prompt = _rendered(_turno_do_professor())
    assert PERMITE_O_DESTINO_LEGITIMO in prompt, (
        "sem esta frase a cláusula é uma mordaça: um destino que a persona nomeia é uma "
        "resposta certa")


def test_uma_critica_que_nao_propoe_destino_nenhum_e_tratada_igual():
    """A cláusula é INCONDICIONAL dentro desta variante e por isso não precisa de ler a
    crítica — classificar aquela prosa está declarado como NÃO MEDIDO no `cogno_soma`, e um
    classificador que ninguém mediu é a fonte que esta cláusula existe para recusar."""
    prompt = _rendered(_turno_do_professor(),
                       reason="only asked for confirmation without recording")
    assert NOTA_PARA_O_EXECUTOR in prompt
    assert PROIBE_MANDAR_EMBORA in prompt


# ── as duas regras recusadas, pinadas como recusadas ──────────────────────────────────────

def test_o_rascunho_rejeitado_CONTINUA_a_nao_ser_re_oferecido():
    """A regra óbvia — «vozeia o último rascunho» — publicaria as fabricações que a rejeição
    apanhou. Esta cláusula não a implementa, e este teste é o que a mantém fora.

    Mutação que isto apanha: fazer `_draft_section` deixar de ver a rejeição.

    **Com o seu próprio CONTROLO**, e ele não é decoração: a primeira versão deste teste
    passava sobre a mutação, porque `_draft_section` tem outros portões (`JUDGE_CONVERSATIONAL`,
    `_judge_approved`) e neste andaime o rascunho não entrava de qualquer maneira. Um teste que
    afirma uma ausência tem de provar primeiro que sabe produzir a presença.
    """
    ctx = _turno_do_professor()
    ctx.metadata[mk.JUDGE_CONVERSATIONAL] = True
    controlo = SuperegoStage()._build_voice_prompt(ctx, DADOS, ["general:review"])
    assert RASCUNHO in controlo, (
        "o andaime não consegue mostrar o rascunho nem quando nada o retém — a asserção "
        "seguinte não discriminaria nada")

    prompt = _rendered(ctx)
    assert RASCUNHO not in prompt, (
        "o rascunho rejeitado voltou ao prompt — 4 dos 11 turnos medidos foram rejeitados "
        "por fabricarem as figuras, e re-oferecê-los publica-as")
    # E o grounding continua a ser os dados do executor, não o rascunho.
    assert DADOS in prompt


# ── os gémeos inversos: o que NÃO pode mudar ──────────────────────────────────────────────

def test_a_variante_da_afirmacao_NAO_VERIFICADA_fica_intacta():
    """Só uma variante mudou. A `unverified_claim` tem a sua própria regra («diz só o que o
    Contexto suporta») e os turnos medidos não estão nela — todos correram ferramentas."""
    prompt = _rendered(_turno_do_professor(), kind="unverified_claim")
    assert "# Review verdict (HARD RULE)" in prompt
    assert NOTA_PARA_O_EXECUTOR not in prompt
    assert PROIBE_MANDAR_EMBORA not in prompt


def test_um_turno_SEM_rejeicao_renderiza_como_antes():
    """Sem `voice_correction` não há secção nenhuma — a cláusula viaja com a rejeição."""
    ctx = _turno_do_professor()
    prompt = SuperegoStage()._build_voice_prompt(ctx, DADOS, ["general:review"])
    assert "# Execution verdict (HARD RULE)" not in prompt
    assert NOTA_PARA_O_EXECUTOR not in prompt


def test_nao_ha_cabecalho_novo_logo_o_inventario_persistido_nao_mexe():
    """Mesma forma que o `nothing_tried` tomou: a cláusula entra DENTRO da secção que já
    existe, portanto `_VOICE_BLOCKS` e o inventário que o host persiste ficam iguais."""
    prompt = _rendered(_turno_do_professor())
    slugs = [b["block"] for b in SuperegoStage.voice_prompt_inventory(prompt)]
    assert "execution_verdict" in slugs
    alfabeto = {slug for _, slug in SuperegoStage._VOICE_BLOCKS}
    assert all(s in alfabeto for s in slugs), slugs
