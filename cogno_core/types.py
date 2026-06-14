from __future__ import annotations

from typing import Optional, Any
from pydantic import BaseModel, Field


class StageMetrics(BaseModel):
    """Telemetry captured during execution of one LLM call or stage."""
    stage: str
    elapsed_ms: float
    tokens_in: int
    tokens_out: int
    tokens_total: int = 0
    model: str

    def model_post_init(self, __context: Any) -> None:
        self.tokens_total = self.tokens_in + self.tokens_out


class NoumenoResult(BaseModel):
    """Resultado da camada NOUMENO — percepção e normalização do input."""

    # ── Textos ────────────────────────────────────────────
    original: str               # Texto bruto do usuário, inalterado
    rewritten: str              # Texto reescrito em inglês
    context_turn: str           # Resumo curto do contexto ("" se 1º turno ou mudou assunto)

    # ── Idioma ────────────────────────────────────────────
    language: str               # Idioma detectado no original (BCP-47: "pt", "en")
    canonical_language: str = "en" # Idioma interno padrão (sempre "en")

    # ── Drift (distorção da reescrita) ────────────────────
    drift_score: float          # 1.0 - cosine(embed(original), embed(rewritten)) → [0.0, 1.0]
    drift_tag: str              # PASS_THROUGH | REWRITTEN | COMPRESSED | EXPANDED | DRIFT
    changed: bool               # True se o LLM realizou alterações estruturais/semânticas ativas
    confidence: float           # Confiança do LLM na preservação da intenção [0.0, 1.0]

    # ── Continuidade de assunto ───────────────────────────
    change_subject: bool        # True se houve mudança de assunto vs. histórico
    subject_similarity: float   # cosine(embed(input), embed(last_rewritten)) → [0.0, 1.0]
    context_used: bool          # True se o histórico foi usado (= bool(context_turn))

    # ── Preservação ──────────────────────────────────────
    preserved_terms: list[str]  # Termos preservados intactos (nomes, URLs, emails...)
    rewrite_warnings: list[str] # Alertas da reescrita (ambiguidade, perda potencial...)

    # ── Telemetria ───────────────────────────────────────
    metrics: StageMetrics


class IntentResult(BaseModel):
    """Resultado estruturado da análise semântica e intenções (NER Stage)."""

    # ── Classificação Semântica ──────────────────────────
    intent_class: str           # INFORMATION_REQUEST | ACTION_REQUEST | CLARIFICATION | CREATIVE_TASK | SOCIAL | UNKNOWN
    sentiment: str              # POSITIVE | NEGATIVE | NEUTRAL | CURIOUS | FRUSTRATED | URGENT | PLAYFUL
    confidence: float           # Confiança no mapeamento [0.0, 1.0]
    temporal_class: str         # RECENT | HISTORICAL | TIMELESS | MIXED
    triad_signal: str           # ID | EGO | SUPEREGO | BALANCED

    # ── Entidades Nomeadas ────────────────────────────────
    entities_people: list[str] = Field(default_factory=list)
    entities_pronouns: list[str] = Field(default_factory=list)
    entities_possessives: list[str] = Field(default_factory=list)
    entities_objects: list[str] = Field(default_factory=list)
    entities_concepts: list[str] = Field(default_factory=list)

    # ── Geolocalização ────────────────────────────────────
    location: Optional[str] = None

    # ── Tags Cognitivas ───────────────────────────────────
    mandatory_tags: list[str] = Field(default_factory=list) # NER.SYSTEM, NER.MATH...
    abstract_tags: list[str] = Field(default_factory=list)  # NER.XXX
    aristotelian: dict[str, str] = Field(default_factory=dict)
    domains: list[str] = Field(default_factory=list)

    # ── Cadeia Causal e Objetivos ────────────────────────
    goal: Optional[str] = None
    causal_chain: list[str] = Field(default_factory=list)

    # ── Linguagem e Fala ─────────────────────────────────
    parole: Optional[str] = None
    langue: Optional[str] = None

    # ── Restrições Pragmáticas ───────────────────────────
    negation: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    modality: Optional[str] = None
    speech_act: Optional[str] = None
    verbs: list[str] = Field(default_factory=list)

    # ── Complexidade Lexical ─────────────────────────────
    context_dependent: bool = False
    is_composite: bool = False
    is_sequential: bool = False
    comparatives: list[str] = Field(default_factory=list)

    # ── Informações Pessoais (PII) ────────────────────────
    pii: list[str] = Field(default_factory=list)
    pii_risk: str = "NONE"

    # ── Classificação Raw (Original em Isolamento) ───────
    raw_intent_class: Optional[str] = None
    raw_domains: list[str] = Field(default_factory=list)
    raw_goal: Optional[str] = None

    # ── Telemetria ───────────────────────────────────────
    metrics: StageMetrics
    raw_response: Optional[str] = None

    def aristo_tag(self, cat: str) -> str:
        """Retorna a tag Aristotélica."""
        val = self.aristotelian.get(cat, "")
        return val.split(" | ")[0].strip() if " | " in val else val.strip()

    def aristo_desc(self, cat: str) -> str:
        """Retorna a descrição Aristotélica."""
        val = self.aristotelian.get(cat, "")
        return val.split(" | ")[1].strip() if " | " in val else ""

    def aristo_parsed(self) -> dict[str, tuple[str, str]]:
        """Retorna o dicionário de categorias mapeadas em (tag, descrição)."""
        result: dict[str, tuple[str, str]] = {}
        for cat, val in self.aristotelian.items():
            if " | " in val:
                tag, desc = val.split(" | ", 1)
                result[cat] = (tag.strip(), desc.strip())
            else:
                result[cat] = (val.strip(), "")
        return result


class DriftMetrics(BaseModel):
    """Métricas de desvio semântico/epistemológico do pipeline."""
    # Stage 1: Epistemological drift (NOUMENO → NER)
    intent_changed: bool
    sentiment_changed: bool
    temporal_changed: bool
    word_count_original: int
    word_count_noumeno: int
    compression_ratio: float
    aristotelian_coverage: int
    drift_score: float

    # Stages 2–5 drift. None = "stage not computed yet" (distinct from 0.0 =
    # "computed, no drift"). compute_cumulative renormalizes over the stages
    # actually populated, so cumulative is on a full [0,1] scale at any point in
    # the pipeline build-out — not deflated by the stages that don't exist yet.
    ontological_drift: Optional[float] = None   # Stage 2 (NER)
    situational_drift: Optional[float] = None    # Stage 3 (ID)
    execution_drift: Optional[float] = None      # Stage 4 (EGO)
    synthesis_drift: Optional[float] = None       # Stage 5 (SUPEREGO)

    # Cumulative
    cumulative_drift: float = 0.0
    drift_action: str = "none"  # none | warn | ask_user | self_correct

    def to_tags(self) -> list[str]:
        """Gera tags de diagnóstico baseadas em drift."""
        tags: list[str] = []

        if self.drift_score >= 0.4:
            tags.append("NOUMENO.DRIFT")

        if self.compression_ratio == 1.0:
            tags.append("NOUMENO.PASS_THROUGH")
        elif self.compression_ratio < 0.8:
            tags.append("NOUMENO.COMPRESSED")
        elif self.compression_ratio > 1.3:
            tags.append("NOUMENO.EXPANDED")
        else:
            tags.append("NOUMENO.REWRITTEN")

        # Cumulative drift tags
        if self.drift_action == "ask_user":
            tags.append("DRIFT.ASK_USER")
        elif self.drift_action == "self_correct":
            tags.append("DRIFT.SELF_CORRECT")
        elif self.drift_action == "warn":
            tags.append("DRIFT.WARN")

        return tags


class PipelineContext(BaseModel):
    """Carrier object that flows through the entire pipeline carrying intermediate results."""
    user_input: str
    force_language: Optional[str] = None
    
    # Results populated by stages
    noumeno: Optional[NoumenoResult] = None
    intent: Optional[IntentResult] = None
    drift: Optional[DriftMetrics] = None
    
    # Custom metadata for the host to pass/read business or infra context
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    # Control info
    stop_reason: str = "completed"
    
    # Accumulators
    retry_metrics: list[StageMetrics] = Field(default_factory=list)

    @property
    def noumeno_metrics(self) -> Optional[StageMetrics]:
        return self.noumeno.metrics if self.noumeno else None

    @property
    def ner_metrics(self) -> Optional[StageMetrics]:
        return self.intent.metrics if self.intent else None

    @property
    def stage_metrics(self) -> list[StageMetrics]:
        base = [self.noumeno_metrics, self.ner_metrics]
        return [m for m in base if m is not None] + self.retry_metrics

    @property
    def total_tokens(self) -> int:
        return sum(m.tokens_total for m in self.stage_metrics)

    @property
    def total_elapsed_ms(self) -> float:
        return sum(m.elapsed_ms for m in self.stage_metrics)
