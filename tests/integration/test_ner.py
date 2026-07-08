"""
NER integration suite — real LLM (Ollama), the critical-layer quality bar.

Baseline model: env COGNO_NER_MODEL (default mistral:latest; qwen3:8b is the
recommended alternative). Auto-skips if Ollama is unavailable. Language is forced
to pt-BR (host/tenant-provided).

Two tiers, by design:
  • STRICT tests assert what the baseline reliably produces (intent_class,
    language propagation, entities, parole/verbs on clear cases) plus the
    DETERMINISTIC PII detector (model-independent) and structural validity.
    These MUST pass on the baseline.
  • NUANCED tests (xfail, strict=False) exercise the model-dependent fields
    (sentiment subtleties, temporal MIXED, modality, speech_act, is_composite,
    comparatives). Small/weak models miss these; they are reported, not fatal.

Run: pytest tests/integration/test_ner.py
     COGNO_NER_MODEL=qwen2.5:7b pytest tests/integration/test_ner.py -k pii
"""

from __future__ import annotations

import os
import asyncio
import pytest

from cognobench.harness import CognitivePipeline, build_ollama, ollama_available
from cogno_anima.stages.ner import (
    VALID_INTENTS, VALID_SENTIMENTS, VALID_TEMPORAL, VALID_TRIAD,
    VALID_MODALITY, VALID_SPEECH_ACTS, VALID_PAROLE,
    NER_KNOWLEDGE_DOMAINS, VALID_MANDATORY,
)
from cogno_anima.security.pii import PII_RISK_LEVELS

MODEL = os.environ.get("COGNO_NER_MODEL", "mistral:latest")
LANGUAGE = "pt-BR"


# ── Shared, cached pipeline (one LLM run per distinct input) ───────────────

@pytest.fixture(scope="module")
def ner():
    if not asyncio.run(ollama_available()):
        pytest.skip("Ollama not available at localhost:11434")
    backend, embedder = build_ollama(MODEL)
    pipe = CognitivePipeline(backend, embedder)
    cache: dict[str, object] = {}

    def run(text: str):
        if text not in cache:
            ctx = asyncio.run(pipe.run(text, force_language=LANGUAGE, stop_after="ner"))
            cache[text] = ctx.intent
        return cache[text]

    return run


# ════════════════════════════════════════════════════════════════════════
#  STRICT — must pass on the baseline
# ════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("text, expected", [
    ("o que é machine learning?", "INFORMATION_REQUEST"),
    ("me explica como funciona o TCP", "INFORMATION_REQUEST"),
    ("qual a diferença entre TCP e UDP?", "INFORMATION_REQUEST"),
    ("calcula 50 vezes 12 para mim", "ACTION_REQUEST"),
    ("instala o Docker no meu servidor", "ACTION_REQUEST"),
    ("crie um evento no calendário e mande um email pro cliente", "ACTION_REQUEST"),
    ("escreva uma história sobre um robô que aprende a sonhar", "CREATIVE_TASK"),
])
def test_intent_classification(ner, text, expected):
    assert ner(text).intent_class == expected


@pytest.mark.parametrize("text", [
    "o que é machine learning?",
    "explain how neural networks work",
    "explica qué es la inteligencia artificial",
])
def test_language_propagation(ner, text):
    """Forced tenant language (pt-BR) must propagate to langue regardless of input language."""
    assert (ner(text).langue or "").lower().startswith("pt")


@pytest.mark.parametrize("text, entity", [
    ("me fala sobre Albert Einstein e suas contribuições", "einstein"),
    ("como está o clima em São Paulo hoje?", "são paulo"),
    ("compare Python com Rust para programação de sistemas", "python"),
    ("quero configurar o Nginx no servidor", "nginx"),
])
def test_entities_present(ner, text, entity):
    intent = ner(text)
    pool = " ".join([
        *intent.entities_people, *intent.entities_concepts,
        *intent.entities_objects, intent.location or "",
    ]).lower()
    # Accent-fold both sides: the NOUMENO canonicalizes to English, so the model may return
    # "sao paulo" for "São Paulo" — the entity is present, just de-accented.
    import unicodedata

    def _fold(s: str) -> str:
        return "".join(c for c in unicodedata.normalize("NFKD", s)
                       if not unicodedata.combining(c))

    assert _fold(entity) in _fold(pool)


# ── Deterministic PII (model-independent: regex + check digits + Luhn) ─────

@pytest.mark.parametrize("text, must_contain, risk", [
    ("meu CPF é 111.444.777-35", {"NATIONAL_ID"}, "HIGH"),
    ("o CNPJ da empresa é 11.222.333/0001-81", {"TAX_ID"}, "HIGH"),
    ("manda um email pra joao.silva@example.com", {"EMAIL"}, "MEDIUM"),
    ("meu cartão é 4111 1111 1111 1111", {"CREDIT_CARD"}, "HIGH"),
    ("o servidor fica em 192.168.0.1", {"IP_ADDRESS"}, "MEDIUM"),
    ("meu telefone é (11) 98765-4321", {"PHONE"}, "MEDIUM"),
    ("meu CEP é 01310-100", {"ADDRESS"}, "MEDIUM"),
    ("sou o joão, cpf 111.444.777-35 e email a@b.com", {"NATIONAL_ID", "EMAIL"}, "HIGH"),
])
def test_pii_detected_deterministically(ner, text, must_contain, risk):
    intent = ner(text)
    assert must_contain <= set(intent.pii), f"{must_contain} not in {intent.pii}"
    assert intent.pii_risk == risk


def test_pii_none_when_absent(ner):
    intent = ner("qual a previsão do tempo para amanhã?")
    assert intent.pii == []
    assert intent.pii_risk == "NONE"


def test_invalid_cpf_not_flagged(ner):
    """An invalid CPF (bad check digits) must NOT raise PII risk — precision guard."""
    intent = ner("o número de protocolo é 123.456.789-00")
    assert "NATIONAL_ID" not in intent.pii


# ── Structural validity — sanitization must always yield in-vocab output ───

@pytest.mark.parametrize("text", [
    "o que é machine learning?",
    "URGENTE: o servidor de produção caiu, preciso de ajuda AGORA",
    "oi, tudo bem?",
    "asdf ghjk qwer zxcv",          # garbage
    "compare Python com Rust e me diga qual é mais rápido",
])
def test_structural_validity(ner, text):
    i = ner(text)
    assert i.intent_class in VALID_INTENTS
    assert i.sentiment in VALID_SENTIMENTS
    assert i.temporal_class in VALID_TEMPORAL
    assert i.triad_signal in VALID_TRIAD
    assert 0.0 <= i.confidence <= 1.0
    assert i.pii_risk in PII_RISK_LEVELS
    assert i.modality is None or i.modality in VALID_MODALITY
    assert i.speech_act is None or i.speech_act in VALID_SPEECH_ACTS
    assert i.parole is None or i.parole in VALID_PAROLE
    assert all(d in NER_KNOWLEDGE_DOMAINS for d in i.domains)
    assert all(t.split(".")[-1] in VALID_MANDATORY for t in i.mandatory_tags)
    assert (i.langue or "").lower().startswith("pt")     # forced language propagated
    # is_sequential implies is_composite would be the ideal contract (not yet enforced)


# ════════════════════════════════════════════════════════════════════════
#  NUANCED — model-dependent; exercised but not fatal on the baseline
# ════════════════════════════════════════════════════════════════════════

@pytest.mark.xfail(reason="model-dependent; smaller baselines are unreliable here "
                          "(strict=False → xpass tolerated on stronger models)", strict=False)
@pytest.mark.parametrize("text, field, expected", [
    # sentiment subtleties
    ("isso não funciona, já tentei 3 vezes e continua errado", "sentiment", "FRUSTRATED"),
    ("oi, tudo bem? como vai?", "sentiment", "POSITIVE"),
    # temporal MIXED
    ("compare a inflação atual com a de 2010", "temporal_class", "MIXED"),
    # modality
    ("acho que o deploy quebrou alguma coisa", "modality", "PROBABLE"),
    ("talvez o problema seja no cache do Redis", "modality", "POSSIBLE"),
    ("não sei se o erro é no front ou no backend", "modality", "UNCERTAIN"),
    # speech act
    ("vou implementar isso amanhã no servidor", "speech_act", "COMMISSIVE"),
    ("qual a diferença entre TCP e UDP?", "speech_act", "INTERROGATIVE"),
    # composite / comparatives
    ("busca o relatório, analisa os dados e gera um gráfico", "is_composite", True),
    ("crie um evento e mande um email pro cliente", "is_composite", True),
])
def test_nuanced_fields(ner, text, field, expected):
    actual = getattr(ner(text), field)
    assert actual == expected


# ── Basic end-to-end smoke (kept from the original suite) ──────────────────

def test_ner_e2e_smoke(ner):
    intent = ner("Quero agendar uma consulta com o veterinário amanhã, por favor.")
    assert intent is not None
    assert intent.intent_class in ("ACTION_REQUEST", "INFORMATION_REQUEST")
