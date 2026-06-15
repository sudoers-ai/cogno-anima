"""
NER analysis quality cases — ported and adapted from the parent Cogno bench.

Tests the accuracy of the NER stage's semantic analysis against cogno-core's
`IntentResult` contract:
  - intent classification, sentiment, temporal class
  - entity extraction, language (inherited from NOUMENO), PII risk
  - speech act, modality, parole, verbs, composite/comparatives/negation
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NERCase:
    """A single NER quality benchmark case."""
    id: str
    input: str
    expect_intent: str = ""
    expect_sentiment: str = ""
    expect_temporal: str = ""
    expect_entities: list[str] = field(default_factory=list)
    expect_language: str = ""          # pt | en | es (matched against langue prefix)
    expect_pii_risk: str = ""          # NONE | LOW | MEDIUM | HIGH | CRITICAL
    expect_speech_act: str = ""
    expect_modality: str = ""
    expect_parole: str = ""
    expect_verbs: list[str] = field(default_factory=list)
    expect_is_composite: bool | None = None
    expect_comparatives: list[str] = field(default_factory=list)
    expect_negation: list[str] = field(default_factory=list)


NER_CASES: list[NERCase] = [
    # ── Intent classification ────────────────────────────────────────────
    NERCase(id="intent_info_request", input="o que é machine learning?",
            expect_intent="INFORMATION_REQUEST", expect_sentiment="NEUTRAL", expect_language="pt"),
    NERCase(id="intent_action_request", input="calcula 50 vezes 12 para mim",
            expect_intent="ACTION_REQUEST", expect_sentiment="NEUTRAL", expect_language="pt"),
    NERCase(id="intent_social", input="oi, tudo bem? como vai?",
            expect_intent="SOCIAL", expect_sentiment="POSITIVE", expect_language="pt"),
    NERCase(id="intent_creative", input="escreva uma história sobre um robô que aprende a sonhar",
            expect_intent="CREATIVE_TASK", expect_language="pt"),

    # ── Sentiment detection ──────────────────────────────────────────────
    NERCase(id="sentiment_positive", input="excelente! ficou perfeito, adorei o resultado!",
            expect_sentiment="POSITIVE"),
    NERCase(id="sentiment_frustrated", input="isso não funciona, já tentei 3 vezes e continua errado",
            expect_sentiment="FRUSTRATED"),
    NERCase(id="sentiment_urgent", input="URGENTE: preciso dos dados de vendas AGORA, o cliente está esperando",
            expect_sentiment="URGENT"),
    NERCase(id="sentiment_neutral", input="qual é a capital da Alemanha?",
            expect_sentiment="NEUTRAL"),

    # ── Temporal classification ──────────────────────────────────────────
    NERCase(id="temporal_recent", input="quais as notícias de hoje sobre tecnologia?",
            expect_temporal="RECENT"),
    NERCase(id="temporal_historical", input="quando foi a Revolução Francesa?",
            expect_temporal="HISTORICAL"),
    NERCase(id="temporal_timeless", input="quanto é a raiz quadrada de 144?",
            expect_temporal="TIMELESS"),
    NERCase(id="temporal_mixed", input="compare a inflação atual com a de 2010",
            expect_temporal="MIXED"),

    # ── Entity extraction ────────────────────────────────────────────────
    NERCase(id="entity_person", input="me fala sobre Albert Einstein e suas contribuições",
            expect_entities=["Albert Einstein"]),
    NERCase(id="entity_location", input="como está o clima em São Paulo hoje?",
            expect_entities=["São Paulo"]),
    NERCase(id="entity_concept_multi", input="compare Python com JavaScript para desenvolvimento web",
            expect_entities=["Python", "JavaScript"]),

    # ── Language detection (inherited from NOUMENO) ──────────────────────
    NERCase(id="lang_portuguese", input="me explica como funciona a fotossíntese",
            expect_language="pt"),
    NERCase(id="lang_english", input="explain how neural networks work",
            expect_language="en"),
    NERCase(id="lang_spanish", input="explica qué es la inteligencia artificial",
            expect_language="es"),

    # ── PII detection quality ────────────────────────────────────────────
    NERCase(id="ner_pii_none", input="qual a previsão do tempo para amanhã?",
            expect_pii_risk="NONE"),
    NERCase(id="ner_pii_credential", input="a senha do servidor é Admin@2024",
            expect_pii_risk="CRITICAL"),
    NERCase(id="ner_pii_email", input="meu email é joao.silva@example.com",
            expect_pii_risk="MEDIUM"),
    NERCase(id="ner_pii_national_id", input="meu CPF é 111.444.777-35",
            expect_pii_risk="HIGH"),   # valid CPF (check digits) → caught deterministically

    # ── Speech Act classification (Austin/Searle) ────────────────────────
    NERCase(id="speech_act_directive", input="me explica como funciona o TCP",
            expect_speech_act="DIRECTIVE", expect_intent="INFORMATION_REQUEST"),
    NERCase(id="speech_act_expressive", input="adorei a resposta, ficou perfeita! Obrigado!",
            expect_speech_act="EXPRESSIVE", expect_sentiment="POSITIVE"),
    NERCase(id="speech_act_commissive", input="vou implementar isso amanhã no servidor de produção",
            expect_speech_act="COMMISSIVE"),
    NERCase(id="speech_act_interrogative", input="qual a diferença entre TCP e UDP?",
            expect_speech_act="INTERROGATIVE", expect_intent="INFORMATION_REQUEST"),

    # ── Epistemic Modality ───────────────────────────────────────────────
    NERCase(id="modality_certain", input="o servidor caiu às 3 da manhã",
            expect_modality="CERTAIN"),
    NERCase(id="modality_probable", input="acho que o deploy quebrou alguma coisa no banco",
            expect_modality="PROBABLE"),
    NERCase(id="modality_possible", input="talvez o problema seja no cache do Redis",
            expect_modality="POSSIBLE"),
    NERCase(id="modality_uncertain", input="não sei se o erro é no front ou no backend",
            expect_modality="UNCERTAIN"),

    # ── Parole (register) ────────────────────────────────────────────────
    NERCase(id="parole_coloquial", input="e aí, esse negócio funciona ou tá bugado?",
            expect_parole="COLOQUIAL", expect_language="pt"),
    NERCase(id="parole_tecnico", input="configure o ingress controller com TLS termination no load balancer",
            expect_parole="TECNICO"),
    NERCase(id="parole_formal", input="solicito gentilmente a revisão do relatório financeiro trimestral",
            expect_parole="FORMAL"),

    # ── Verb extraction ──────────────────────────────────────────────────
    NERCase(id="verbs_single_action", input="calcula o preço total com desconto de 15%",
            expect_verbs=["calculate"]),
    NERCase(id="verbs_multi_action", input="busca o relatório, analisa os dados e gera um gráfico",
            expect_verbs=["search", "analyze", "generate"], expect_is_composite=True),

    # ── Composite & Comparatives ─────────────────────────────────────────
    NERCase(id="composite_multi_request", input="crie um evento no calendário e mande um email pro cliente",
            expect_is_composite=True, expect_intent="ACTION_REQUEST"),
    NERCase(id="comparative_tech", input="compare Python com Rust para programação de sistemas",
            expect_comparatives=["Python", "Rust"], expect_entities=["Python", "Rust"]),

    # ── Negation ─────────────────────────────────────────────────────────
    NERCase(id="negation_explicit", input="não quero usar JavaScript, prefiro TypeScript",
            expect_negation=["JavaScript"]),
]
