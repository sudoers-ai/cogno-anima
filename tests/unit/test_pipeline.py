"""
Cross-stage and contract tests for the NOUMENO → NER → Drift flow.

These tests use the zero-network stubs from conftest and assert the
architectural contracts the pipeline must respect:

  - stage responsibilities (NOUMENO prepares, NER consumes, Drift is pure);
  - langue inheritance from NOUMENO;
  - subject-change clears prior context, subject-continuity keeps it;
  - PII risk is recomputed deterministically in the core (LLM value ignored);
  - the `domains` closed list in code matches the NER prompt exactly;
  - errors propagate to the caller with no local fallback / LLM swap;
  - the NER carries no tool/skill routing responsibility.
"""

import re
import json
import pytest
from pathlib import Path

from cogno_anima.types import PipelineContext, IntentResult
from cogno_anima.errors import StageParseError
from cogno_anima.stages.base import BaseStage
from cogno_anima.stages.noumeno import Noumeno
from cogno_anima.stages.ner import IntentAnalyzer, NER_KNOWLEDGE_DOMAINS
from cogno_anima.stages.drift import DriftCalculator
from tests.conftest import StubBackend, StubEmbedder

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

NOUMENO_JSON = json.dumps({
    "rewritten": "I want to wash my car",
    "context_turn": "car washing",
    "confidence": 0.95,
    "changed": True,
    "preserved_terms": [],
    "rewrite_warnings": [],
})

NER_JSON = json.dumps({
    "intent_class": "ACTION_REQUEST",
    "sentiment": "NEUTRAL",
    "confidence": 0.95,
    "temporal_class": "TIMELESS",
    "triad_signal": "EGO",
    "entities": {"people": [], "pronouns": [], "possessives": ["my"],
                 "objects": ["car"], "concepts": ["car washing"]},
    "location": None,
    "mandatory_tags": ["SYSTEM"],
    "abstract_tags": ["CAR_WASH"],
    "aristotelian": {"ACTION": "WASH_CAR | wash the car"},
    "goal": "wash the car",
    "causal_chain": ["user wants car washed"],
    "parole": "COLOQUIAL",
    "langue": "pt-BR",          # must be ignored — langue comes from NOUMENO
    "negation": [],
    "constraints": [],
    "domains": ["LOGISTICS"],
    "modality": "CERTAIN",
    "speech_act": "DIRECTIVE",
    "is_composite": False,
    "is_sequential": False,
    "verbs": ["wash"],
    "context_dependent": False,
    "comparatives": [],
    "pii": [],
    "pii_risk": "NONE",
    "raw_intent_class": "ACTION_REQUEST",
    "raw_domains": ["LOGISTICS"],
    "raw_goal": "wash the car",
})


def _noumeno_stage(embedder=None) -> Noumeno:
    return Noumeno(embedder=embedder or StubEmbedder(), prompts_dir=PROMPTS_DIR)


def _ner_stage() -> IntentAnalyzer:
    return IntentAnalyzer(prompts_dir=PROMPTS_DIR)


# ────────────────────────────────────────────────────────────────────
#  BaseStage contract
# ────────────────────────────────────────────────────────────────────

def test_stages_satisfy_base_stage_protocol():
    """Noumeno and IntentAnalyzer must structurally satisfy BaseStage."""
    assert isinstance(_noumeno_stage(), BaseStage)
    assert isinstance(_ner_stage(), BaseStage)
    assert _noumeno_stage().name == "noumeno"
    assert _ner_stage().name == "ner"


# ────────────────────────────────────────────────────────────────────
#  NOUMENO → NER chaining
# ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_noumeno_then_ner_chains_through_context():
    """Running NOUMENO then NER populates both results on the same context."""
    # force_language keeps the NOUMENO-detected language deterministic for the test.
    ctx = PipelineContext(user_input="quero lavar meu carro", force_language="pt")

    ctx = await _noumeno_stage().process(ctx, StubBackend(response=NOUMENO_JSON))
    assert ctx.noumeno is not None
    assert ctx.noumeno.rewritten == "I want to wash my car"
    assert ctx.noumeno.language == "pt"

    ctx = await _ner_stage().process(ctx, StubBackend(response=NER_JSON))
    assert ctx.intent is not None
    assert ctx.intent.intent_class == "ACTION_REQUEST"
    # langue is inherited from NOUMENO (pt), NOT the LLM's "pt-BR".
    assert ctx.intent.langue == ctx.noumeno.language == "pt"


@pytest.mark.asyncio
async def test_ner_requires_noumeno_first():
    """NER must refuse to run before NOUMENO populated the context."""
    ctx = PipelineContext(user_input="hello")
    with pytest.raises(ValueError, match="NoumenoResult must be populated"):
        await _ner_stage().process(ctx, StubBackend(response=NER_JSON))


# ────────────────────────────────────────────────────────────────────
#  NOUMENO → NER → Drift chaining
# ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_flow_noumeno_ner_drift():
    """End-to-end (stubbed) NOUMENO → NER → Drift produces coherent metrics."""
    ctx = PipelineContext(user_input="quero lavar meu carro")
    ctx = await _noumeno_stage().process(ctx, StubBackend(response=NOUMENO_JSON))
    ctx = await _ner_stage().process(ctx, StubBackend(response=NER_JSON))

    calc = DriftCalculator()
    drift = calc.compute(ctx.noumeno, ctx.intent)
    calc.compute_ontological(drift, ctx.noumeno, ctx.intent)
    calc.compute_cumulative(drift)
    ctx.drift = drift

    # Epistemological drift is taken verbatim from NOUMENO, not recomputed.
    assert ctx.drift.drift_score == round(ctx.noumeno.drift_score, 3)
    assert 0.0 <= ctx.drift.ontological_drift <= 1.0
    assert 0.0 <= ctx.drift.cumulative_drift <= 1.0
    assert ctx.drift.drift_action in {"none", "warn", "ask_user", "self_correct"}


def test_drift_epistemological_comes_from_noumeno_unchanged():
    """DriftCalculator.compute must consume noumeno.drift_score, never recompute it."""
    from tests.unit.test_drift import make_noumeno_result, make_intent_result
    calc = DriftCalculator()
    noumeno = make_noumeno_result("a b c", "completely different longer rewrite text")
    noumeno.drift_score = 0.777
    drift = calc.compute(noumeno, make_intent_result())
    assert drift.drift_score == 0.777


# ────────────────────────────────────────────────────────────────────
#  Subject continuity: context cleared vs kept
# ────────────────────────────────────────────────────────────────────

class _PromptCapture(StubBackend):
    async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
        self.generated_prompt = prompt
        return self.response, self.tokens_in, self.tokens_out


@pytest.mark.asyncio
async def test_change_subject_false_uses_context():
    """change_subject=False → prior context (context_turn, goal, domains) is injected."""
    from tests.unit.test_ner import make_noumeno_result
    backend = _PromptCapture(response=NER_JSON)
    analyzer = _ner_stage()

    noumeno = make_noumeno_result(change_subject=False, context_turn="car washing")
    await analyzer.analyze(
        noumeno, prior_goal="keep-the-car-clean",
        active_domains=["LOGISTICS"], turn_number=7, llm=backend,
    )
    assert "car washing" in backend.generated_prompt
    assert "keep-the-car-clean" in backend.generated_prompt
    assert "LOGISTICS" in backend.generated_prompt
    assert "TURN: 7" in backend.generated_prompt


@pytest.mark.asyncio
async def test_change_subject_true_clears_context():
    """change_subject=True → all prior context is stripped before prompting."""
    from tests.unit.test_ner import make_noumeno_result
    backend = _PromptCapture(response=NER_JSON)
    analyzer = _ner_stage()

    noumeno = make_noumeno_result(change_subject=True, context_turn="car washing")
    await analyzer.analyze(
        noumeno, prior_goal="keep-the-car-clean",
        active_domains=["LOGISTICS"], turn_number=7, llm=backend,
    )
    assert "car washing" not in backend.generated_prompt
    assert "keep-the-car-clean" not in backend.generated_prompt
    assert "LOGISTICS" not in backend.generated_prompt
    assert "TURN: 7" not in backend.generated_prompt


# ────────────────────────────────────────────────────────────────────
#  PII risk recomputed deterministically (LLM value ignored)
# ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pii_risk_ignores_llm_value_and_recomputes():
    """The LLM's pii_risk is discarded; the core recomputes it from the pii list."""
    from tests.unit.test_ner import make_noumeno_result
    payload = json.loads(NER_JSON)
    payload["pii"] = ["EMAIL", "HEALTH_DATA"]   # HEALTH_DATA → CRITICAL
    payload["pii_risk"] = "LOW"                  # deliberately wrong; must be ignored
    backend = StubBackend(response=json.dumps(payload))
    result = await _ner_stage().analyze(make_noumeno_result(), llm=backend)
    assert result.pii == ["EMAIL", "HEALTH_DATA"]
    assert result.pii_risk == "CRITICAL"


# ────────────────────────────────────────────────────────────────────
#  domains: prompt ↔ code alignment, GENERAL regression
# ────────────────────────────────────────────────────────────────────

def _parse_prompt_domains() -> set[str]:
    """Extract the closed `domains` list declared in prompts/ner/system.txt."""
    text = (PROMPTS_DIR / "ner" / "system.txt").read_text(encoding="utf-8")
    anchor = "EXACT closed list:"
    start = text.index(anchor) + len(anchor)
    region = text[start:text.index("Do NOT invent", start)]
    return {tok.strip() for tok in re.split(r"[|\n]", region) if tok.strip().isupper()}


def test_code_domains_match_prompt_domains_exactly():
    """NER_KNOWLEDGE_DOMAINS must equal the prompt's closed domain list byte-for-byte."""
    assert NER_KNOWLEDGE_DOMAINS == _parse_prompt_domains()


def test_general_domain_is_accepted():
    """Regression: GENERAL (the prompt's fallback domain) must not be dropped."""
    assert "GENERAL" in NER_KNOWLEDGE_DOMAINS


def _parse_prompt_mandatory_tags() -> set[str]:
    """Extract the mandatory_tags vocabulary declared in prompts/ner/system.txt."""
    text = (PROMPTS_DIR / "ner" / "system.txt").read_text(encoding="utf-8")
    anchor = "mandatory_tags — 1 to 3 of:"
    start = text.index(anchor) + len(anchor)
    line = text[start:text.index("\n", start)]
    return {tok.strip() for tok in line.split("|") if tok.strip().isupper()}


def test_code_mandatory_tags_match_prompt_exactly():
    """VALID_MANDATORY must equal the prompt's mandatory_tags vocabulary (no stray LOGIC)."""
    from cogno_anima.stages.ner import VALID_MANDATORY
    assert VALID_MANDATORY == _parse_prompt_mandatory_tags()
    assert "LOGIC" not in VALID_MANDATORY


def test_all_vocab_values_are_taught_by_the_prompt():
    """Single-source guard: every value in cogno_anima.vocab must appear in the NER
    prompt. Adding a value to vocab without teaching the LLM (or vice-versa) fails here."""
    from cogno_anima import vocab
    text = (PROMPTS_DIR / "ner" / "system.txt").read_text(encoding="utf-8")
    sets = {
        "VALID_INTENTS": vocab.VALID_INTENTS,
        "VALID_SENTIMENTS": vocab.VALID_SENTIMENTS,
        "VALID_TEMPORAL": vocab.VALID_TEMPORAL,
        "VALID_TRIAD": vocab.VALID_TRIAD,
        "VALID_MODALITY": vocab.VALID_MODALITY,
        "VALID_SPEECH_ACTS": vocab.VALID_SPEECH_ACTS,
        "VALID_PAROLE": vocab.VALID_PAROLE,
        "VALID_MANDATORY": vocab.VALID_MANDATORY,
        "VALID_ARISTOTELIAN": vocab.VALID_ARISTOTELIAN,
        "NER_KNOWLEDGE_DOMAINS": vocab.NER_KNOWLEDGE_DOMAINS,
    }
    missing = {name: sorted(v for v in values if v not in text)
               for name, values in sets.items()}
    missing = {k: v for k, v in missing.items() if v}
    assert not missing, f"vocab values absent from the NER prompt: {missing}"


def test_ner_vocab_is_sourced_from_vocab_module():
    """The NER stage must re-export the SAME objects as cogno_anima.vocab (single source)."""
    from cogno_anima import vocab
    from cogno_anima.stages import ner
    assert ner.NER_KNOWLEDGE_DOMAINS is vocab.NER_KNOWLEDGE_DOMAINS
    assert ner.VALID_INTENTS is vocab.VALID_INTENTS
    assert ner.VALID_MANDATORY is vocab.VALID_MANDATORY


# ────────────────────────────────────────────────────────────────────
#  No tool / skill routing in NER
# ────────────────────────────────────────────────────────────────────

def test_intent_result_has_no_tool_fields():
    """IntentResult must not carry any tool/skill routing field."""
    forbidden = {"suggested_tools", "tools_section", "skill_names",
                 "skills", "tools", "tool_routing"}
    assert forbidden.isdisjoint(set(IntentResult.model_fields))


def test_ner_module_and_prompt_have_no_tool_routing():
    """The NER source and prompt must contain no tool/skill routing symbols."""
    import cogno_anima.stages.ner as ner_mod
    src = Path(ner_mod.__file__).read_text(encoding="utf-8").lower()
    prompt = (PROMPTS_DIR / "ner" / "system.txt").read_text(encoding="utf-8").lower()
    for needle in ("suggested_tools", "tools_section", "skill", "skillregistry",
                   "skillselector", "tool_routing"):
        assert needle not in src, f"{needle!r} leaked into ner.py"
        assert needle not in prompt, f"{needle!r} leaked into ner prompt"


# ────────────────────────────────────────────────────────────────────
#  Error propagation / no local fallback
# ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ner_llm_error_propagates_no_fallback():
    """An LLM failure in NER propagates; ctx.intent stays None (no silent fallback)."""
    from tests.unit.test_ner import make_noumeno_result

    class FailingLLM(StubBackend):
        async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
            raise RuntimeError("LLM exploded")

    ctx = PipelineContext(user_input="x")
    ctx.noumeno = make_noumeno_result()
    with pytest.raises(RuntimeError, match="LLM exploded"):
        await _ner_stage().process(ctx, FailingLLM())
    assert ctx.intent is None


@pytest.mark.asyncio
async def test_ner_invalid_json_propagates_no_fallback():
    """Invalid JSON from the LLM raises; the stage does not substitute a default."""
    from tests.unit.test_ner import make_noumeno_result
    ctx = PipelineContext(user_input="x")
    ctx.noumeno = make_noumeno_result()
    with pytest.raises(StageParseError):
        await _ner_stage().process(ctx, StubBackend(response="not json"))
    assert ctx.intent is None


@pytest.mark.asyncio
async def test_no_backend_raises_instead_of_swapping():
    """With no backend supplied at all, NER raises — it never picks an LLM itself."""
    from tests.unit.test_ner import make_noumeno_result
    analyzer = IntentAnalyzer(prompts_dir=PROMPTS_DIR)  # no backend at init
    with pytest.raises(ValueError, match="LLMBackend must be provided"):
        await analyzer.analyze(make_noumeno_result())  # and none at call
