"""A company's registration number is not a person's data — and the cost of pretending it was
had nothing to do with privacy.

`pii.PII_RISK_MAP` fed `TAX_ID` (and therefore a Brazilian **CNPJ**) into `HIGH`. The ID stage
routes `pii_risk == "HIGH"` straight to the SUPEREGO, so on such a turn the EGO never runs and
**no tool is offered at all**. Put those two together on a contact who sends their company's
registration data and the system cannot write it down — and the model, holding no tools, says
what it would have done: *"Vou cadastrar a empresa …"*.

Measured on the owner's own scenario (2026-09-03, session `1a6cb213`, host `8042892`):

    t1  SECRETARY  pii=[]                        risco=NONE    rota=EGO   ferramentas: 6, com company_registration
    t2  SECRETARY  pii=[EMAIL, PHONE, TAX_ID]    risco=HIGH    rota=SUPEREGO   ferramentas: 0   <-- o cadastro
    t8  SECRETARY  pii=[]                        risco=NONE    rota=EGO   ferramentas: 6, com company_registration

The tool was on the table one turn before and one turn after. What removed it was the routing,
on the exact turn the data arrived — and `EMAIL`+`PHONE` alone score MEDIUM, so the CNPJ was the
whole of the escalation. The fabricated confirmation there is a CONSEQUENCE, not an invention.

So the split is by whose number it is: `COMPANY_ID` for an organisation's public registration
number, `TAX_ID` kept HIGH for the aliases that can be a natural person's.
"""
import pytest

from cogno_anima.security.detector import default_detector
from cogno_anima.security.pii import (PII_RISK_MAP, VALID_PII_TYPES, compute_pii_risk,
                                      normalize_pii_types)
from cogno_anima.stages.id import IDStage
from tests.unit.test_id import PlainEmbedder, _intent, make_ctx


# ── de quem é o número ───────────────────────────────────────────────────────────────────

def test_o_CNPJ_e_da_empresa():
    achado = default_detector().find("Nosso CNPJ é 11.222.333/0001-81")
    assert [m.pii_type for m in achado] == ["COMPANY_ID"]


def test_o_CPF_continua_a_ser_da_PESSOA_e_continua_ALTO():
    """O controlo que impede este PR de ser um afrouxamento: o que se moveu foi o número da
    empresa, e o da pessoa fica exactamente onde estava."""
    achado = default_detector().find("meu CPF é 529.982.247-25")
    assert [m.pii_type for m in achado] == ["NATIONAL_ID"]
    assert compute_pii_risk(["NATIONAL_ID"]) == "HIGH"


@pytest.mark.parametrize("alias, tipo", [
    ("CNPJ", "COMPANY_ID"), ("EIN", "COMPANY_ID"),
    ("NIF", "TAX_ID"), ("TAX_NUMBER", "TAX_ID"), ("TAXPAYER_ID", "TAX_ID"),
])
def test_o_ambiguo_fica_do_lado_seguro(alias, tipo):
    """`NIF` É o número fiscal de uma pessoa singular em Portugal, e `TAX_NUMBER`/`TAXPAYER_ID`
    não nomeiam jurisdição nenhuma. Só se separa o que é inequivocamente de uma organização."""
    assert normalize_pii_types([alias]) == [tipo]


def test_os_dois_riscos():
    assert PII_RISK_MAP["COMPANY_ID"] == "MEDIUM"
    assert PII_RISK_MAP["TAX_ID"] == "HIGH"
    assert "COMPANY_ID" in VALID_PII_TYPES


# ── e a consequência, que é o que se estava a pagar ──────────────────────────────────────

def test_o_turno_do_cadastro_deixa_de_escalar():
    """Exactamente os tipos do t2 medido."""
    assert compute_pii_risk(["EMAIL", "PHONE", "COMPANY_ID"]) == "MEDIUM"


def test_CONTROLO_o_mesmo_turno_com_o_tipo_ANTIGO_ainda_escala():
    """O gémeo: prova que era o tipo, e não outra coisa do turno, que produzia o HIGH."""
    assert compute_pii_risk(["EMAIL", "PHONE", "TAX_ID"]) == "HIGH"


@pytest.mark.asyncio
async def test_o_cadastro_com_CNPJ_chega_ao_EXECUTOR():
    """A ponta que importa: com MEDIUM o pedido de acção vai ao EGO — que é onde as ferramentas
    são oferecidas. Sem isto o resto do PR é uma reclassificação sem efeito."""
    ctx = make_ctx(_intent(intent_class="ACTION_REQUEST", pii_risk="MEDIUM"))
    out = await IDStage().process(ctx, PlainEmbedder())
    assert out.id_result.triad_route == "EGO"
    assert out.id_result.blocked is False


@pytest.mark.asyncio
async def test_CONTROLO_um_CPF_no_mesmo_pedido_continua_a_desviar():
    """O gémeo da protecção: a regra `HIGH -> SUPEREGO` não foi tocada, e não devia ser. O que
    mudou foi QUEM é classificado HIGH."""
    ctx = make_ctx(_intent(intent_class="ACTION_REQUEST", pii_risk="HIGH"))
    out = await IDStage().process(ctx, PlainEmbedder())
    assert out.id_result.triad_route == "SUPEREGO"


# ── o que o NER aprende ──────────────────────────────────────────────────────────────────

def test_o_prompt_ENSINA_o_tipo_novo_e_o_escalao():
    """Um tipo no código que o prompt não ensina é um tipo que o modelo nunca emite — e o
    `test_all_vocab_values_are_taught_by_the_prompt` guarda a mesma regra para os outros
    vocabulários. Aqui prende-se também o ESCALÃO, porque é dele que sai a rota."""
    from pathlib import Path
    texto = (Path(__file__).resolve().parents[2] / "cogno_anima" / "prompt_templates"
             / "ner" / "system.txt").read_text(encoding="utf-8")
    assert "COMPANY_ID" in texto
    linha = next(ln for ln in texto.splitlines() if "MEDIUM →" in ln or "MEDIUM  →" in ln)
    assert "COMPANY_ID" in linha, "o tipo está na lista mas o modelo não sabe que escalão dar"
