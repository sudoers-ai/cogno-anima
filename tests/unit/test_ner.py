from cogno_anima.errors import StageParseError
import pytest
import json
from pathlib import Path
from pydantic import ValidationError

from cogno_anima.types import NoumenoResult, IntentResult
from cogno_anima.stages.ner import (
    IntentAnalyzer,
    NER_KNOWLEDGE_DOMAINS,
)
from tests.conftest import StubBackend

PROMPTS_DIR = Path(__file__).parent.parent.parent / "cogno_anima" / "prompt_templates"

def make_noumeno_result(
    original="Olá, quero lavar meu carro",
    rewritten="I want to wash my car",
    change_subject=False,
    context_turn="car washing",
) -> NoumenoResult:
    from cogno_anima.types import StageMetrics
    metrics = StageMetrics(stage="noumeno", elapsed_ms=10.0, tokens_in=5, tokens_out=5, model="test")
    return NoumenoResult(
        original=original,
        rewritten=rewritten,
        context_turn=context_turn,
        language="pt",
        canonical_language="en",
        drift_score=0.1,
        drift_tag="REWRITTEN",
        changed=True,
        confidence=0.9,
        change_subject=change_subject,
        subject_similarity=0.8,
        context_used=True,
        preserved_terms=[],
        rewrite_warnings=[],
        metrics=metrics,
    )

PERFECT_JSON = {
    "intent_class": "ACTION_REQUEST",
    "sentiment": "NEUTRAL",
    "confidence": 0.95,
    "temporal_class": "TIMELESS",
    "triad_signal": "EGO",
    "entities": {
        "people": ["Copernicus"],
        "pronouns": ["ele"],
        "possessives": ["meu"],
        "objects": ["carro"],
        "concepts": ["cleaning"]
    },
    "location": "São Paulo",
    "mandatory_tags": ["SYSTEM"],
    "abstract_tags": ["CAR_WASH", "CLEANING_SERVICE"],
    "aristotelian": {
        "SUBSTANCE": "USER_CAR | User's car needing cleaning",
        "ACTION": "WASH_CAR | Wash the car"
    },
    "goal": "wash the car",
    "causal_chain": ["user wants car washed"],
    "parole": "COLOQUIAL",
    "langue": "pt-BR",
    "negation": ["spending too much"],
    "constraints": ["cheap"],
    "modality": "CERTAIN",
    "speech_act": "DIRECTIVE",
    "verbs": ["wash"],
    "context_dependent": True,
    "is_composite": False,
    "is_sequential": False,
    "comparatives": ["walk vs drive"],
    "pii": ["EMAIL", "CPF"],
    "raw_intent_class": "ACTION_REQUEST",
    "raw_domains": ["LOGISTICS"],
    "raw_goal": "wash the car",
    "domains": ["LOGISTICS"]
}


@pytest.mark.asyncio
async def test_perfect_payload():
    """1. Extração estruturada com payload JSON perfeito."""
    backend = StubBackend(response=json.dumps(PERFECT_JSON))
    analyzer = IntentAnalyzer(backend=backend, prompts_dir=PROMPTS_DIR)
    # Original mentions São Paulo so the (grounded) location survives.
    noumeno = make_noumeno_result(original="Olá, quero lavar meu carro em São Paulo")
    result = await analyzer.analyze(noumeno)

    assert result.intent_class == "ACTION_REQUEST"
    assert result.entities_pronouns == ["he"]
    assert result.entities_possessives == ["my"]
    assert result.location == "São Paulo"
    assert result.mandatory_tags == ["NER.SYSTEM"]
    assert result.abstract_tags == ["NER.CAR_WASH", "NER.CLEANING_SERVICE"]
    assert "SUBSTANCE" in result.aristotelian
    assert result.pii == ["EMAIL", "NATIONAL_ID"]
    assert result.pii_risk == "HIGH"


@pytest.mark.asyncio
async def test_think_tag_stripping():
    """2. Recuperação de falha se o LLM injetar tags <think>...</think>."""
    think_response = f"<think>thinking logic here...</think>\n{json.dumps(PERFECT_JSON)}"
    backend = StubBackend(response=think_response)
    analyzer = IntentAnalyzer(backend=backend, prompts_dir=PROMPTS_DIR)
    result = await analyzer.analyze(make_noumeno_result())
    assert result.intent_class == "ACTION_REQUEST"


@pytest.mark.asyncio
async def test_pronoun_normalization():
    """3. Conversão e normalização de pronomes (ex: eu -> I, ele -> he)."""
    payload = PERFECT_JSON.copy()
    payload["entities"] = PERFECT_JSON["entities"].copy()
    payload["entities"]["pronouns"] = ["eu", "você", "ele", "ela", "nós", "eles"]
    backend = StubBackend(response=json.dumps(payload))
    analyzer = IntentAnalyzer(backend=backend, prompts_dir=PROMPTS_DIR)
    result = await analyzer.analyze(make_noumeno_result())
    assert "I" in result.entities_pronouns
    assert "you" in result.entities_pronouns
    assert "he" in result.entities_pronouns
    assert "she" in result.entities_pronouns
    assert "we" in result.entities_pronouns
    assert "they" in result.entities_pronouns


@pytest.mark.asyncio
async def test_possessive_normalization():
    """4. Conversão e normalização de possessivos (ex: meu -> my, nosso -> our)."""
    payload = PERFECT_JSON.copy()
    payload["entities"] = PERFECT_JSON["entities"].copy()
    payload["entities"]["possessives"] = ["meu", "minha", "seu", "sua", "nosso", "nossa"]
    backend = StubBackend(response=json.dumps(payload))
    analyzer = IntentAnalyzer(backend=backend, prompts_dir=PROMPTS_DIR)
    result = await analyzer.analyze(make_noumeno_result())
    assert "my" in result.entities_possessives
    assert "your" in result.entities_possessives
    assert "our" in result.entities_possessives


@pytest.mark.asyncio
async def test_domain_whitelist():
    """5. Filtragem de domains pela whitelist."""
    payload = PERFECT_JSON.copy()
    payload["domains"] = ["TECH", "INVALID_DOMAIN", "FINANCE"]
    backend = StubBackend(response=json.dumps(payload))
    analyzer = IntentAnalyzer(backend=backend, prompts_dir=PROMPTS_DIR)
    result = await analyzer.analyze(make_noumeno_result())
    assert "TECH" in result.domains
    assert "FINANCE" in result.domains
    assert "INVALID_DOMAIN" not in result.domains


@pytest.mark.asyncio
async def test_domain_aliases():
    """6. Mapeamento de aliases de domains (inclui colapsos CRYPTO→FINANCE, MATH→SCIENCE)."""
    payload = PERFECT_JSON.copy()
    payload["domains"] = ["ECONOMICS", "MEDICINE", "PROGRAMMING", "CRYPTO", "MATH", "GENERAL"]
    backend = StubBackend(response=json.dumps(payload))
    analyzer = IntentAnalyzer(backend=backend, prompts_dir=PROMPTS_DIR)
    result = await analyzer.analyze(make_noumeno_result())
    assert "FINANCE" in result.domains   # ECONOMICS and CRYPTO both collapse here
    assert "HEALTH" in result.domains
    assert "TECH" in result.domains
    assert "SCIENCE" in result.domains   # MATH collapses into SCIENCE
    assert "GENERAL" in result.domains   # passes through (regression: must not be dropped)
    # No alias target may fall outside the closed list.
    assert all(d in NER_KNOWLEDGE_DOMAINS for d in result.domains)
    assert "CRYPTO" not in result.domains
    assert "MATH" not in result.domains


@pytest.mark.asyncio
async def test_abstract_tags_sanitization():
    """7. Sanitização de abstract_tags para UPPER_SNAKE_CASE."""
    payload = PERFECT_JSON.copy()
    payload["abstract_tags"] = ["car wash!", "clean_service#1", "INVALID-TAG@"]
    backend = StubBackend(response=json.dumps(payload))
    analyzer = IntentAnalyzer(backend=backend, prompts_dir=PROMPTS_DIR)
    result = await analyzer.analyze(make_noumeno_result())
    assert "NER.CAR_WASH" in result.abstract_tags
    assert "NER.CLEAN_SERVICE1" in result.abstract_tags
    assert "NER.INVALIDTAG" in result.abstract_tags


@pytest.mark.asyncio
async def test_aristotelian_categories():
    """8. Filtragem de aristotelian para reter apenas as 10 categorias oficiais."""
    payload = PERFECT_JSON.copy()
    payload["aristotelian"] = {
        "SUBSTANCE": "USER_CAR | User's car",
        "ACTION": "WASH_CAR | Wash car",
        "INVALID_CAT": "some value",
    }
    backend = StubBackend(response=json.dumps(payload))
    analyzer = IntentAnalyzer(backend=backend, prompts_dir=PROMPTS_DIR)
    result = await analyzer.analyze(make_noumeno_result())
    assert "SUBSTANCE" in result.aristotelian
    assert "ACTION" in result.aristotelian
    assert "INVALID_CAT" not in result.aristotelian


@pytest.mark.asyncio
async def test_intent_safeguard_coercion():
    """9. Coerção de salvaguarda estrutural (UNKNOWN em intent_class remapeado)."""
    payload = PERFECT_JSON.copy()
    payload["intent_class"] = "UNKNOWN"
    payload["mandatory_tags"] = ["MATH"]
    backend = StubBackend(response=json.dumps(payload))
    analyzer = IntentAnalyzer(backend=backend, prompts_dir=PROMPTS_DIR)
    result = await analyzer.analyze(make_noumeno_result())
    assert result.intent_class == "ACTION_REQUEST"


@pytest.mark.asyncio
async def test_langue_inherited_from_noumeno():
    """10. langue é SEMPRE herdado de noumeno.language; o langue do LLM é ignorado."""
    # PERFECT_JSON carries langue="pt-BR" from the LLM — it must be ignored.
    backend = StubBackend(response=json.dumps(PERFECT_JSON))
    analyzer = IntentAnalyzer(backend=backend, prompts_dir=PROMPTS_DIR)

    noumeno = make_noumeno_result()  # language="pt"
    result = await analyzer.analyze(noumeno)
    assert result.langue == "pt"

    # A different noumeno language flows straight through.
    noumeno_es = make_noumeno_result()
    noumeno_es.language = "es"
    result_es = await analyzer.analyze(noumeno_es)
    assert result_es.langue == "es"


@pytest.mark.asyncio
async def test_json_decode_error():
    """11. Propagação estrita de json.JSONDecodeError."""
    backend = StubBackend(response="invalid json response")
    analyzer = IntentAnalyzer(backend=backend, prompts_dir=PROMPTS_DIR)
    with pytest.raises(StageParseError):
        await analyzer.analyze(make_noumeno_result())


@pytest.mark.asyncio
async def test_change_subject_logic_cleaning():
    """12. Limpeza lógica de contexto se change_subject=True no Noumeno."""
    class PromptCheckingBackend(StubBackend):
        async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
            self.generated_prompt = prompt
            return self.response, self.tokens_in, self.tokens_out
            
    check_backend = PromptCheckingBackend(response=json.dumps(PERFECT_JSON))
    analyzer = IntentAnalyzer(backend=check_backend, prompts_dir=PROMPTS_DIR)
    
    noumeno = make_noumeno_result(change_subject=True)
    await analyzer.analyze(noumeno, prior_goal="my-goal", active_domains=["FINANCE"], turn_number=2)

    assert "my-goal" not in check_backend.generated_prompt
    assert "FINANCE" not in check_backend.generated_prompt
    assert "TURN: 2" not in check_backend.generated_prompt


@pytest.mark.asyncio
async def test_legacy_tags_removal():
    """13. Não decodificação de tags com namespaces customizados (remover legado)."""
    payload = PERFECT_JSON.copy()
    payload["mandatory_tags"] = ["LEGACY.STUFF", "SYSTEM"]
    backend = StubBackend(response=json.dumps(payload))
    analyzer = IntentAnalyzer(backend=backend, prompts_dir=PROMPTS_DIR)
    result = await analyzer.analyze(make_noumeno_result())
    assert "NER.SYSTEM" in result.mandatory_tags
    assert len(result.mandatory_tags) == 1


@pytest.mark.asyncio
async def test_composite_sequential_comparatives():
    """14. Processamento de is_composite, is_sequential e comparativos."""
    payload = PERFECT_JSON.copy()
    payload["is_composite"] = True
    payload["is_sequential"] = True
    payload["comparatives"] = ["Python vs Rust"]
    backend = StubBackend(response=json.dumps(payload))
    analyzer = IntentAnalyzer(backend=backend, prompts_dir=PROMPTS_DIR)
    result = await analyzer.analyze(make_noumeno_result())
    assert result.is_composite is True
    assert result.is_sequential is True
    assert result.comparatives == ["Python vs Rust"]


@pytest.mark.asyncio
async def test_pii_types_normalization():
    """15. Conversão e normalização de PII types usando normalize_pii_types()."""
    payload = PERFECT_JSON.copy()
    payload["pii"] = ["CPF", "EMAIL_ADDRESS", "INVALID_PII"]
    backend = StubBackend(response=json.dumps(payload))
    analyzer = IntentAnalyzer(backend=backend, prompts_dir=PROMPTS_DIR)
    result = await analyzer.analyze(make_noumeno_result())
    assert "NATIONAL_ID" in result.pii
    assert "EMAIL" in result.pii
    assert "INVALID_PII" not in result.pii


@pytest.mark.asyncio
async def test_pii_risk_computation():
    """16. Cálculo de pii_risk no core."""
    payload = PERFECT_JSON.copy()
    payload["pii"] = ["CPF", "HEALTH_DATA"]
    backend = StubBackend(response=json.dumps(payload))
    analyzer = IntentAnalyzer(backend=backend, prompts_dir=PROMPTS_DIR)
    result = await analyzer.analyze(make_noumeno_result())
    assert result.pii_risk == "CRITICAL"


def test_validation_error():
    """17. Propagação de ValidationError se Pydantic falhar."""
    with pytest.raises(ValidationError):
        IntentResult(intent_class=123, sentiment=456)


@pytest.mark.asyncio
async def test_is_sequential_requires_composite():
    """18. is_sequential=True com is_composite=False é reconciliado para False."""
    payload = PERFECT_JSON.copy()
    payload["is_composite"] = False
    payload["is_sequential"] = True
    backend = StubBackend(response=json.dumps(payload))
    analyzer = IntentAnalyzer(backend=backend, prompts_dir=PROMPTS_DIR)
    result = await analyzer.analyze(make_noumeno_result())
    assert result.is_composite is False
    assert result.is_sequential is False   # reconciled

    # When composite, sequential is preserved.
    payload["is_composite"] = True
    payload["is_sequential"] = True
    backend2 = StubBackend(response=json.dumps(payload))
    analyzer2 = IntentAnalyzer(backend=backend2, prompts_dir=PROMPTS_DIR)
    result2 = await analyzer2.analyze(make_noumeno_result())
    assert result2.is_composite is True
    assert result2.is_sequential is True


@pytest.mark.asyncio
async def test_aristotelian_description_capped_preserving_tag():
    """19. A descrição aristotélica é limitada a 40 chars sem cortar a TAG."""
    payload = PERFECT_JSON.copy()
    payload["aristotelian"] = {"ACTION": "WASH_CAR | " + "d" * 100}
    backend = StubBackend(response=json.dumps(payload))
    analyzer = IntentAnalyzer(backend=backend, prompts_dir=PROMPTS_DIR)
    result = await analyzer.analyze(make_noumeno_result())
    assert result.aristotelian["ACTION"] == "WASH_CAR | " + "d" * 40
    assert result.aristo_tag("ACTION") == "WASH_CAR"   # tag intact


@pytest.mark.asyncio
async def test_entity_grounding_drops_hallucinated_people():
    """20. Nome de pessoa que não aparece no ORIGINAL é descartado (anti-alucinação)."""
    payload = PERFECT_JSON.copy()
    payload["entities"] = dict(PERFECT_JSON["entities"])
    payload["entities"]["people"] = ["Copernicus", "Napoleon"]
    backend = StubBackend(response=json.dumps(payload))
    analyzer = IntentAnalyzer(backend=backend, prompts_dir=PROMPTS_DIR)
    noumeno = make_noumeno_result(original="what did copernicus do in 1543")
    result = await analyzer.analyze(noumeno)
    assert "Copernicus" in result.entities_people
    assert "Napoleon" not in result.entities_people


@pytest.mark.asyncio
async def test_entity_grounding_drops_hallucinated_location():
    """21. Location que não aparece no ORIGINAL vira None."""
    payload = PERFECT_JSON.copy()
    payload["location"] = "Tokyo"
    backend = StubBackend(response=json.dumps(payload))
    analyzer = IntentAnalyzer(backend=backend, prompts_dir=PROMPTS_DIR)
    result = await analyzer.analyze(make_noumeno_result(original="qual a previsão do tempo amanhã?"))
    assert result.location is None


@pytest.mark.asyncio
async def test_entity_grounding_keeps_real_objects_and_concepts():
    """22. Grounding NÃO toca objects/concepts (podem ser traduzidos/derivados)."""
    payload = PERFECT_JSON.copy()
    payload["entities"] = dict(PERFECT_JSON["entities"])
    payload["entities"]["objects"] = ["car"]          # translated from 'carro'
    payload["entities"]["concepts"] = ["car washing"]  # derived concept
    backend = StubBackend(response=json.dumps(payload))
    analyzer = IntentAnalyzer(backend=backend, prompts_dir=PROMPTS_DIR)
    result = await analyzer.analyze(make_noumeno_result(original="quero lavar meu carro"))
    assert result.entities_objects == ["car"]
    assert result.entities_concepts == ["car washing"]


# ════════════════════════════════════════════════════════════════════════
#  P1: coercion / sanitization branches (the "never trust the LLM" paths)
# ════════════════════════════════════════════════════════════════════════

async def _analyze(payload: dict, **noumeno_kw):
    backend = StubBackend(response=json.dumps(payload))
    analyzer = IntentAnalyzer(backend=backend, prompts_dir=PROMPTS_DIR)
    return await analyzer.analyze(make_noumeno_result(**noumeno_kw))


@pytest.mark.asyncio
async def test_invalid_enums_fall_back_to_defaults():
    """Out-of-vocabulary enum values are coerced to their safe defaults."""
    payload = PERFECT_JSON.copy()
    payload.update(sentiment="ECSTATIC", temporal_class="YESTERDAY", triad_signal="HYPER")
    result = await _analyze(payload)
    assert result.sentiment == "NEUTRAL"
    assert result.temporal_class == "TIMELESS"
    assert result.triad_signal == "BALANCED"


@pytest.mark.asyncio
async def test_optional_enums_invalid_become_none():
    """modality/speech_act/parole outside the vocabulary become None (not coerced)."""
    payload = PERFECT_JSON.copy()
    payload.update(modality="VERYSURE", speech_act="SHOUTING", parole="ROBOTIC")
    result = await _analyze(payload)
    assert result.modality is None
    assert result.speech_act is None
    assert result.parole is None


@pytest.mark.asyncio
async def test_confidence_is_clamped_and_safe():
    payload = PERFECT_JSON.copy()
    payload["confidence"] = 5.0
    assert (await _analyze(payload)).confidence == 1.0
    payload["confidence"] = -2.0
    assert (await _analyze(payload)).confidence == 0.0
    payload["confidence"] = "not-a-number"
    assert (await _analyze(payload)).confidence == 0.5   # safe default


@pytest.mark.asyncio
async def test_list_fields_are_capped():
    payload = PERFECT_JSON.copy()
    payload["verbs"] = [f"verb{i}" for i in range(9)]                 # cap 5
    payload["mandatory_tags"] = ["SYSTEM", "ANALYSIS", "MATH", "CREATIVE"]  # cap 3
    payload["negation"] = [f"neg{i}" for i in range(8)]              # cap 4
    result = await _analyze(payload)
    assert len(result.verbs) == 5
    assert len(result.mandatory_tags) == 3
    assert len(result.negation) == 4


@pytest.mark.asyncio
async def test_goal_is_truncated():
    payload = PERFECT_JSON.copy()
    payload["goal"] = "g" * 200
    result = await _analyze(payload)
    assert len(result.goal) == 80


@pytest.mark.asyncio
async def test_context_dependent_accepts_string_boolean():
    payload = PERFECT_JSON.copy()
    payload["context_dependent"] = "true"
    assert (await _analyze(payload)).context_dependent is True
    payload["context_dependent"] = "nope"
    assert (await _analyze(payload)).context_dependent is False


@pytest.mark.asyncio
async def test_raw_intent_class_invalid_becomes_none():
    payload = PERFECT_JSON.copy()
    payload["raw_intent_class"] = "GIBBERISH"
    assert (await _analyze(payload)).raw_intent_class is None


@pytest.mark.asyncio
async def test_intent_class_falls_back_to_raw_when_unknown():
    """UNKNOWN intent_class is recovered from a valid raw_intent_class."""
    payload = PERFECT_JSON.copy()
    payload["intent_class"] = "UNKNOWN"
    payload["mandatory_tags"] = ["LINGUISTIC"]   # no MATH/SYSTEM/CREATIVE/ANALYSIS coercion
    payload["raw_intent_class"] = "SOCIAL"
    assert (await _analyze(payload)).intent_class == "SOCIAL"


@pytest.mark.asyncio
async def test_empty_mandatory_tags_defaults_to_unknown_tag():
    payload = PERFECT_JSON.copy()
    payload["mandatory_tags"] = ["NONSENSE_TAG"]   # filtered out → none valid
    result = await _analyze(payload)
    assert result.mandatory_tags == ["NER.UNKNOWN"]
