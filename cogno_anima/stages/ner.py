from __future__ import annotations

import re
import time
import json
import logging
from pathlib import Path
from typing import Optional

from cogno_anima.types import PipelineContext, NoumenoResult, IntentResult, StageMetrics
from cogno_synapse import LLMBackend
from cogno_anima.prompts import load_prompt
from cogno_anima.security.pii import compute_pii_risk, normalize_pii_types
from cogno_anima.security.detector import PiiDetector, default_detector
from cogno_anima.errors import StageParseError

# Closed vocabularies live in cogno_anima.vocab (single source of truth) and are
# re-exported here for backward compatibility. The prompt enumerates the same
# values; tests/unit/test_pipeline.py enforces prompt↔vocab alignment.
from cogno_anima.vocab import (  # noqa: F401
    VALID_INTENTS, VALID_SENTIMENTS, VALID_TEMPORAL, VALID_TRIAD,
    VALID_MODALITY, VALID_SPEECH_ACTS, VALID_PAROLE, VALID_MANDATORY,
    VALID_ARISTOTELIAN, NER_KNOWLEDGE_DOMAINS,
)

logger = logging.getLogger("cogno_anima.ner")

STAGE_NAME = "ner"

# Aliases mapping common LLM deviations onto the canonical closed list above.
# Every target value MUST be a member of NER_KNOWLEDGE_DOMAINS.
_DOMAIN_ALIASES: dict[str, str] = {
    # LAW
    "LEGAL": "LAW", "JURIDICAL": "LAW", "LEGISLATION": "LAW", "POLITICS": "LAW",
    # HEALTH
    "MEDICINE": "HEALTH", "MEDICAL": "HEALTH", "PHARMACY": "HEALTH",
    "PSYCHOLOGY": "HEALTH",
    # SCIENCE (math collapses into SCIENCE — there is no MATH domain)
    "ENGINEERING": "SCIENCE", "BIOLOGY": "SCIENCE", "PHYSICS": "SCIENCE",
    "CHEMISTRY": "SCIENCE", "ASTRONOMY": "SCIENCE", "ENVIRONMENT": "SCIENCE",
    "ECOLOGY": "SCIENCE", "CLIMATE": "SCIENCE",
    "MATH": "SCIENCE", "MATHEMATICS": "SCIENCE", "ARITHMETIC": "SCIENCE",
    "ALGEBRA": "SCIENCE", "GEOMETRY": "SCIENCE",
    # FINANCE (crypto collapses into FINANCE — there is no CRYPTO domain)
    "ECONOMICS": "FINANCE", "INVESTING": "FINANCE", "BANKING": "FINANCE",
    "CRYPTO": "FINANCE", "BLOCKCHAIN": "FINANCE", "DEFI": "FINANCE", "NFT": "FINANCE",
    # TECH
    "PROGRAMMING": "TECH", "SOFTWARE": "TECH", "HARDWARE": "TECH", "GAMING": "TECH",
    # TRAVEL
    "TOURISM": "TRAVEL",
    # CULTURE
    "SPORTS": "CULTURE", "MUSIC": "CULTURE", "ART": "CULTURE", "RELIGION": "CULTURE",
    "FOOD": "CULTURE", "COOKING": "CULTURE", "NUTRITION": "CULTURE", "RECIPES": "CULTURE",
    # EDUCATION
    "LANGUAGE": "EDUCATION",
    # GENERAL fallback
    "OTHER": "GENERAL",
}

# Fallback from a cognitive mandatory tag to a knowledge domain when the LLM
# omits `domains` entirely. Targets MUST be members of NER_KNOWLEDGE_DOMAINS.
_TAG_TO_DOMAIN: dict[str, str] = {
    "NER.MATH": "SCIENCE",
    "NER.SYSTEM": "TECH",
    "NER.ANALYSIS": "SCIENCE",
}


def _canonical_domains(raw: object) -> list[str]:
    """Normalize aliases, filter against the closed list, dedupe preserving order."""
    result: list[str] = []
    if isinstance(raw, list):
        for d in raw:
            if not isinstance(d, str):
                continue
            canonical = _DOMAIN_ALIASES.get(d.upper().strip(), d.upper().strip())
            if canonical in NER_KNOWLEDGE_DOMAINS and canonical not in result:
                result.append(canonical)
    return result


def make_tag(domain: str, name: str) -> str:
    """Gera uma tag namespaced."""
    return f"{domain}.{name}"


# Strong anaphoric back-references — forms that almost always continue a prior
# topic by pointing at entities mentioned earlier ("deles, qual o mais usado?").
# Used as a deterministic fallback for `context_dependent`: small LLMs frequently
# leave it False on these, which starves the ID stage's anaphoric continuity
# fast-path. Conservative on purpose (plural "of them" forms + "the same"), so it
# only ever flips False→True and won't mask a genuine topic change.
_ANAPHORA_RE = re.compile(
    r"\b(deles|delas|desses|dessas|destes|destas|daqueles|daquelas|"
    r"disso|nisso|neles|nelas|os mesmos|as mesmas|o mesmo|a mesma|"
    r"of them|of those|of these|the same|that one)\b",
    re.IGNORECASE,
)


def _has_anaphora(*texts: str) -> bool:
    """True if any text carries a strong anaphoric back-reference marker."""
    return any(bool(t) and _ANAPHORA_RE.search(t) is not None for t in texts)


class IntentAnalyzer:
    """
    IntentAnalyzer — NER Stage (Semantic Analysis).
    """
    name = STAGE_NAME

    def __init__(
        self,
        backend: Optional[LLMBackend] = None,
        prompts_dir: Optional[Path] = None,
        system_prompt_name: str = "system.txt",
        pii_detector: Optional[PiiDetector] = None,
    ):
        self._backend = backend
        # Deterministic PII backstop, unioned with the LLM's pii list. Defaults to
        # Brazil + international; pass a custom PiiDetector to add country packs.
        self._pii_detector = pii_detector or default_detector()

        # Load prompts
        self._system = load_prompt("ner", system_prompt_name, prompts_dir=prompts_dir)
        self._user_tpl = load_prompt("ner", "user.txt", prompts_dir=prompts_dir)

    async def process(self, ctx: PipelineContext, llm: LLMBackend) -> PipelineContext:
        """
        Runs the NER stage on the PipelineContext.
        """
        if not ctx.noumeno:
            raise ValueError("NoumenoResult must be populated before running IntentAnalyzer")

        prior_goal = ctx.metadata.get("last_goal")
        active_domains = ctx.metadata.get("active_domains")
        turn_number = ctx.metadata.get("turn_number")

        intent = await self.analyze(
            noumeno=ctx.noumeno,
            prior_goal=prior_goal,
            active_domains=active_domains,
            turn_number=turn_number,
            llm=llm,
        )

        ctx.intent = intent
        logger.info(
            "NER intent=%s sentiment=%s pii_risk=%s domains=%s composite=%s",
            intent.intent_class, intent.sentiment, intent.pii_risk,
            intent.domains, intent.is_composite,
        )
        return ctx

    async def analyze(
        self,
        noumeno: NoumenoResult,
        prior_goal: Optional[str] = None,
        active_domains: Optional[list[str]] = None,
        turn_number: Optional[int] = None,
        llm: Optional[LLMBackend] = None,
    ) -> IntentResult:
        """
        Analisa o NoumenoResult e extrai o IntentResult estruturado.

        O idioma (`langue`) do IntentResult é SEMPRE herdado de
        `noumeno.language` — o NER nunca redetecta idioma nem usa o `langue`
        devolvido pelo LLM.
        """
        backend = llm or self._backend
        if not backend:
            raise ValueError("LLMBackend must be provided either at init or analyze call")

        # 1. Regra de Mudança de Assunto (Subject Continuity / Shift Context Rules)
        #    Se houve mudança de assunto, todo o contexto anterior é limpo antes
        #    de montar o prompt.
        if noumeno.change_subject:
            prior_goal_line = ""
            domain_context_line = ""
            turn_context_line = ""
            context_turn = ""
        else:
            context_turn = noumeno.context_turn
            prior_goal_line = f"PRIOR GOAL: {prior_goal}" if prior_goal else ""
            domain_context_line = f"ACTIVE DOMAINS: {', '.join(active_domains)}" if active_domains else ""
            turn_context_line = f"TURN: {turn_number}" if turn_number is not None else ""

        # 2. Formatar Prompt do Usuário
        prompt = self._user_tpl.format(
            original_input=noumeno.original,
            noumeno_output=noumeno.rewritten,
            context_turn=context_turn,
            prior_goal_line=prior_goal_line,
            domain_context_line=domain_context_line,
            turn_context_line=turn_context_line,
        )

        # 3. Executar Chamada ao LLM
        t0 = time.perf_counter()
        raw_response, tokens_in, tokens_out = await backend.generate(self._system, prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        metrics = StageMetrics(
            stage=STAGE_NAME,
            elapsed_ms=round(elapsed_ms, 2),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=backend.model,
        )

        # 4. Parse da resposta. O idioma é herdado do NOUMENO, nunca do LLM.
        #    PII é detectado no texto ORIGINAL (não no rewrite, que pode mascarar).
        return self._parse(raw_response, metrics, language=noumeno.language,
                           original=noumeno.original, rewritten=noumeno.rewritten)

    def _parse(self, raw: str, metrics: StageMetrics, language: Optional[str] = None,
               original: str = "", rewritten: str = "") -> IntentResult:
        """
        Decodifica e sanitiza os campos do JSON gerado pelo LLM.
        """
        cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise StageParseError(STAGE_NAME, raw, exc) from exc
        # Valid JSON that is not an object (e.g. "5", "[]") would crash the field
        # coercion below with a raw AttributeError — treat it as a parse failure
        # so the contract stays "valid IntentResult OR StageParseError".
        if not isinstance(data, dict):
            raise StageParseError(STAGE_NAME, raw, TypeError("JSON is not an object"))

        # intent_class
        intent_class = str(data.get("intent_class", "UNKNOWN")).upper()
        if intent_class not in VALID_INTENTS or intent_class == "UNKNOWN":
            raw_ic = str(data.get("raw_intent_class", "UNKNOWN")).upper()
            if raw_ic in VALID_INTENTS:
                intent_class = raw_ic
            else:
                intent_class = "UNKNOWN"

        # Coerção de salvaguarda estrutural
        if intent_class == "UNKNOWN":
            raw_tags = [str(t).upper().split(".")[-1] for t in data.get("mandatory_tags", []) if isinstance(t, str)]
            if {"MATH", "SYSTEM"} & set(raw_tags):
                intent_class = "ACTION_REQUEST"
            elif {"CREATIVE"} & set(raw_tags):
                intent_class = "CREATIVE_TASK"
            elif {"ANALYSIS"} & set(raw_tags):
                intent_class = "INFORMATION_REQUEST"

        # sentiment
        sentiment = str(data.get("sentiment", "NEUTRAL")).upper()
        if sentiment not in VALID_SENTIMENTS:
            sentiment = "NEUTRAL"

        # confidence
        try:
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5

        # temporal_class
        temporal_class = str(data.get("temporal_class", "TIMELESS")).upper()
        if temporal_class not in VALID_TEMPORAL:
            temporal_class = "TIMELESS"

        # triad_signal
        triad_signal = str(data.get("triad_signal", "BALANCED")).upper()
        if triad_signal not in VALID_TRIAD:
            triad_signal = "BALANCED"

        # entities
        ent = data.get("entities")
        if not isinstance(ent, dict):
            ent = {}

        def _clean_list(v: object) -> list[str]:
            if not isinstance(v, list):
                return []
            return [str(x).strip() for x in v if x is not None and str(x).strip()]

        _PRONOUN_NORM: dict[str, str] = {
            "eu": "I", "tu": "you", "você": "you", "voce": "you",
            "ele": "he", "ela": "she", "nós": "we", "nos": "we",
            "vocês": "you", "voces": "you",
            "eles": "they", "elas": "they",
            "me": "me", "te": "you", "se": "him",
            "yo": "I", "tú": "you", "él": "he",
            "ella": "she", "nosotros": "we", "ellos": "they",
        }
        _POSSESSIVE_NORM: dict[str, str] = {
            "meu": "my", "minha": "my", "meus": "my", "minhas": "my",
            "seu": "your", "sua": "your", "seus": "your", "suas": "your",
            "nosso": "our", "nossa": "our", "nossos": "our", "nossas": "our",
            "dele": "his", "dela": "her", "deles": "their", "delas": "their",
            "mi": "my", "tu": "your", "su": "his",
        }
        _POSSESSIVE_WORDS = set(_POSSESSIVE_NORM.keys()) | {
            "my", "your", "his", "her", "its", "our", "their",
        }

        def _normalize_pronouns(items: list[str]) -> list[str]:
            result = []
            for w in items:
                lower = w.lower()
                if lower in _POSSESSIVE_WORDS:
                    continue
                normalized = _PRONOUN_NORM.get(lower, w)
                if normalized not in result:
                    result.append(normalized)
            return result

        def _normalize_possessives(items: list[str]) -> list[str]:
            result = []
            for w in items:
                normalized = _POSSESSIVE_NORM.get(w.lower(), w.lower())
                if normalized not in result:
                    result.append(normalized)
            return result

        entities_people      = _clean_list(ent.get("people",      []))
        entities_pronouns    = _normalize_pronouns(_clean_list(ent.get("pronouns",    [])))
        entities_possessives = _normalize_possessives(_clean_list(ent.get("possessives", [])))
        entities_objects     = _clean_list(ent.get("objects",     []))
        entities_concepts    = _clean_list(ent.get("concepts",    []))

        # Entity grounding (conservative): drop a proper-name entity only when
        # NONE of its words appear in the ORIGINAL text — this kills hallucinated
        # people/places without false drops from reordering or expansion. Applied
        # ONLY to people/location (proper nouns are language-invariant); objects
        # and concepts are left intact because they may be translated or derived.
        def _grounded(items: list[str], source: str) -> list[str]:
            low = source.lower()
            kept = []
            for item in items:
                tokens = [t for t in re.split(r"\W+", item.lower()) if len(t) >= 3]
                if not tokens or any(t in low for t in tokens):
                    kept.append(item)
            return kept

        if original:
            entities_people = _grounded(entities_people, original)

        # location
        loc_raw = data.get("location")
        location = str(loc_raw).strip() if loc_raw else None
        if original and location and not _grounded([location], original):
            location = None

        # mandatory_tags
        mandatory = []
        for t in data.get("mandatory_tags", []):
            if not isinstance(t, str):
                continue
            short = t.upper().split(".")[-1]
            if short in VALID_MANDATORY:
                mandatory.append(make_tag("NER", short))
        if not mandatory:
            mandatory = [make_tag("NER", "UNKNOWN")]
        mandatory = mandatory[:3]

        # abstract_tags
        abstract = []
        for t in data.get("abstract_tags", []):
            if not isinstance(t, str):
                continue
            short_name = t.upper().split(".")[-1].replace(' ', '_')
            clean = re.sub(r'[^A-Z0-9_]', '', short_name)[:30]
            if clean:
                abstract.append(make_tag("NER", clean))
        abstract = abstract[:5]

        # aristotelian
        aristo_raw = data.get("aristotelian") or {}
        aristotelian = {}
        if isinstance(aristo_raw, dict):
            for k, v in aristo_raw.items():
                k_upper = str(k).upper()
                if k_upper in VALID_ARISTOTELIAN and isinstance(v, str) and v.strip():
                    # Format is "TAG | description". The prompt caps the
                    # description at 40 chars — cap it without truncating the tag.
                    val = v.strip()
                    if " | " in val:
                        tag, desc = val.split(" | ", 1)
                        val = f"{tag.strip()} | {desc.strip()[:40]}"
                    else:
                        val = val[:40]
                    aristotelian[k_upper] = val

        # domains (aliased + filtered against the closed list; see module top)
        domains = _canonical_domains(data.get("domains", []))

        # fallback para domains se estiver vazio
        if not domains and mandatory:
            for tag in mandatory:
                d = _TAG_TO_DOMAIN.get(tag)
                if d and d not in domains:
                    domains.append(d)

        # goal
        goal_raw = data.get("goal")
        goal = (str(goal_raw).strip()[:80]
                if goal_raw and isinstance(goal_raw, str) and goal_raw.strip()
                else None)

        # causal_chain
        causal_raw = data.get("causal_chain", [])
        causal_chain: list[str] = []
        if isinstance(causal_raw, list):
            causal_chain = [str(x).strip()[:60] for x in causal_raw
                            if isinstance(x, str) and str(x).strip()][:4]

        # parole
        parole_raw = data.get("parole")
        parole: Optional[str] = None
        if parole_raw and isinstance(parole_raw, str):
            p = parole_raw.upper().strip()
            parole = p if p in VALID_PAROLE else None

        # langue — ALWAYS inherited from noumeno.language. The LLM's own `langue`
        # field (if any) is deliberately ignored: the NER must not redetect idioma.
        langue = language

        # negation
        negation_raw = data.get("negation", [])
        negation = []
        if isinstance(negation_raw, list):
            negation = [str(x).strip()[:40] for x in negation_raw
                        if isinstance(x, str) and str(x).strip()][:4]

        # constraints
        constraints_raw = data.get("constraints", [])
        constraints = []
        if isinstance(constraints_raw, list):
            constraints = [str(x).strip()[:40] for x in constraints_raw
                           if isinstance(x, str) and str(x).strip()][:4]

        # modality
        modality_raw = data.get("modality")
        modality: Optional[str] = None
        if modality_raw and isinstance(modality_raw, str):
            m = modality_raw.upper().strip()
            modality = m if m in VALID_MODALITY else None

        # speech_act
        speech_act_raw = data.get("speech_act")
        speech_act: Optional[str] = None
        if speech_act_raw and isinstance(speech_act_raw, str):
            s = speech_act_raw.upper().strip()
            speech_act = s if s in VALID_SPEECH_ACTS else None

        # verbs
        verbs_raw = data.get("verbs", [])
        verbs: list[str] = []
        if isinstance(verbs_raw, list):
            verbs = [str(x).strip()[:40] for x in verbs_raw
                     if isinstance(x, str) and str(x).strip()][:5]

        # context_dependent
        ctx_raw = data.get("context_dependent", False)
        if isinstance(ctx_raw, bool):
            context_dependent = ctx_raw
        elif isinstance(ctx_raw, str):
            context_dependent = ctx_raw.strip().lower() in ("true", "1", "yes")
        else:
            context_dependent = False
        # Deterministic fallback: strong anaphoric back-references ("deles, qual o
        # mais usado?") are context-dependent even when the LLM misses it. Only
        # flips False→True (never overrides a positive), so it can't mask a topic
        # change; feeds the ID stage's anaphoric continuity fast-path.
        if not context_dependent and _has_anaphora(original, rewritten):
            context_dependent = True

        # is_composite
        comp_flag_raw = data.get("is_composite", False)
        if isinstance(comp_flag_raw, bool):
            is_composite = comp_flag_raw
        elif isinstance(comp_flag_raw, str):
            is_composite = comp_flag_raw.strip().lower() in ("true", "1", "yes")
        else:
            is_composite = False

        # is_sequential
        seq_flag_raw = data.get("is_sequential", False)
        if isinstance(seq_flag_raw, bool):
            is_sequential = seq_flag_raw
        elif isinstance(seq_flag_raw, str):
            is_sequential = seq_flag_raw.strip().lower() in ("true", "1", "yes")
        else:
            is_sequential = False

        # Reconciliation: sequencing only makes sense for composite (multi-action)
        # requests. A single action cannot be "sequential". Enforce the contract
        # the prompt states but the LLM may violate.
        if not is_composite:
            is_sequential = False

        # comparatives
        comp_raw = data.get("comparatives", [])
        comparatives: list[str] = []
        if isinstance(comp_raw, list):
            comparatives = [str(x).strip()[:60] for x in comp_raw
                            if isinstance(x, str) and str(x).strip()][:4]

        # pii & pii_risk — union of the LLM's list with the deterministic detector
        # (run on the ORIGINAL text). pii_risk is always recomputed in-core.
        pii_llm = normalize_pii_types(data.get("pii", []))
        pii_regex = self._pii_detector.detect(original) if original else []
        pii = list(dict.fromkeys([*pii_llm, *pii_regex]))[:10]
        pii_risk = compute_pii_risk(pii)

        # raw fields
        raw_intent_raw = data.get("raw_intent_class")
        raw_intent_class: Optional[str] = None
        if raw_intent_raw and isinstance(raw_intent_raw, str):
            ric = raw_intent_raw.upper().strip()
            raw_intent_class = ric if ric in VALID_INTENTS else None

        raw_domains = _canonical_domains(data.get("raw_domains", []))

        raw_goal_raw = data.get("raw_goal")
        raw_goal: Optional[str] = (
            str(raw_goal_raw).strip()[:80]
            if raw_goal_raw and isinstance(raw_goal_raw, str) and raw_goal_raw.strip()
            else None
        )

        return IntentResult(
            intent_class=intent_class,
            sentiment=sentiment,
            confidence=confidence,
            temporal_class=temporal_class,
            triad_signal=triad_signal,
            entities_people=entities_people,
            entities_pronouns=entities_pronouns,
            entities_possessives=entities_possessives,
            entities_objects=entities_objects,
            entities_concepts=entities_concepts,
            location=location,
            mandatory_tags=mandatory,
            abstract_tags=abstract,
            aristotelian=aristotelian,
            domains=domains,
            goal=goal,
            causal_chain=causal_chain,
            parole=parole,
            langue=langue,
            negation=negation,
            constraints=constraints,
            modality=modality,
            speech_act=speech_act,
            verbs=verbs,
            context_dependent=context_dependent,
            is_composite=is_composite,
            is_sequential=is_sequential,
            comparatives=comparatives,
            pii=pii,
            pii_risk=pii_risk,
            raw_intent_class=raw_intent_class,
            raw_domains=raw_domains,
            raw_goal=raw_goal,
            metrics=metrics,
            raw_response=raw,
        )
