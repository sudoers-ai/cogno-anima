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

from cogno_anima.types import StageMetrics

from cogno_anima.types import PipelineContext, IntentResult
from cogno_anima.errors import StageParseError
from cogno_anima.stages.base import BaseStage
from cogno_anima.stages.noumeno import Noumeno
from cogno_anima.stages.ner import IntentAnalyzer, NER_KNOWLEDGE_DOMAINS
from cogno_anima.stages.drift import DriftCalculator
from tests.conftest import StubBackend, StubEmbedder

PROMPTS_DIR = Path(__file__).parent.parent.parent / "cogno_anima" / "prompt_templates"

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
    "entities": {"people": [],
                 "objects": ["car"], "concepts": ["car washing"]},
    "location": None,
    "mandatory_tags": ["SYSTEM"],
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
    "pii": [],
    "pii_risk": "NONE",
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
async def test_change_subject_true_clears_the_THREADs_continuity():
    """change_subject=True → what the conversation WAS about is stripped before prompting.

    Prior goal, active domains and turn number describe the thread. On a genuine topic change
    they are noise at best and misdirection at worst, so they go. `context_turn` does not —
    see the test below; that split is the point.
    """
    from tests.unit.test_ner import make_noumeno_result
    backend = _PromptCapture(response=NER_JSON)
    analyzer = _ner_stage()

    noumeno = make_noumeno_result(change_subject=True, context_turn="car washing")
    await analyzer.analyze(
        noumeno, prior_goal="keep-the-car-clean",
        active_domains=["LOGISTICS"], turn_number=7, llm=backend,
    )
    assert "keep-the-car-clean" not in backend.generated_prompt
    assert "LOGISTICS" not in backend.generated_prompt
    assert "TURN: 7" not in backend.generated_prompt


@pytest.mark.asyncio
async def test_change_subject_true_KEEPS_the_frame_for_this_utterance():
    """`context_turn` survives, and this half is a deliberate change of contract.

    It used to be stripped with everything else. `change_subject` is a cosine against
    LAST_REWRITTEN, so after a short reply ("Claro" -> "Sure.") the anchor carries no content
    and everything said next reads as a topic change — measured firing on 8 of 10 real CLOSER
    turns, and 15 of 15 on the sequence that broke live (cosine 0.522). The transcript is
    already injected unconditionally for exactly that reason (anima #52); this field was left
    gated on the same boolean that fix declared unreliable.

    What it costs to drop it is concrete. On the live turn the model wrote "The user is
    PROVIDING INFORMATION about the average volume of daily customer service calls received"
    and the code discarded it one line later; the NER then read a lead ANSWERING a discovery
    question as one ASKING for an arithmetic mean, and the agent replied with 8,5.

    The distinction that makes keeping it safe: `context_turn` describes THIS utterance in
    light of what the assistant just said. Even a topic-changing message is interpreted
    against that. The thread's continuity is a different thing and still drops (test above).

    Measured, gpt-4o-mini + real embedder: goal naming its referent 2/15 -> 8/15 (p=0.050);
    SECRETARY suite 42/45 -> 41/45, one check on a holiday scenario, inside the run-to-run
    spread, with reschedule/cancel/long_meander/troll identical in both arms.
    """
    from tests.unit.test_ner import make_noumeno_result
    backend = _PromptCapture(response=NER_JSON)
    analyzer = _ner_stage()

    noumeno = make_noumeno_result(change_subject=True, context_turn="car washing")
    await analyzer.analyze(noumeno, prior_goal="keep-the-car-clean", llm=backend)
    assert "car washing" in backend.generated_prompt


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
    """Extract the closed `domains` list declared in cogno_anima/prompt_templates/ner/system.txt."""
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
    """Extract the mandatory_tags vocabulary declared in cogno_anima/prompt_templates/ner/system.txt."""
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


def test_stage_metrics_per_call_identity_defaults_to_inert():
    """`seq`/`attempt`/`prompt_sha` are stamped by the ORCHESTRATOR, which is the only layer
    that sequences the stages. Every stage in this package builds its own metrics and stamps
    nothing, so the defaults must be inert — an unstamped metric has to behave exactly as it
    did before the fields existed, or adding them changes every caller that never opted in.

    Mutation: give any of them a truthy default and this dies.
    """
    m = StageMetrics(stage="ego", elapsed_ms=1.0, tokens_in=10, tokens_out=2, model="fake")
    assert (m.seq, m.attempt, m.prompt_sha) == (0, 0, "")
    assert m.tokens_total == 12, "as novas chaves não entram na conta de tokens"

    stamped = StageMetrics(stage="ego", elapsed_ms=1.0, tokens_in=10, tokens_out=2,
                           model="fake", seq=4, attempt=2, prompt_sha="9f3c1a")
    assert (stamped.seq, stamped.attempt, stamped.prompt_sha) == (4, 2, "9f3c1a")
    assert stamped.tokens_total == 12


# ── "partially stamped" is a state, and it has to have a name ────────────────────────────
#
# Every canonical StageMetrics is built INSIDE its stage, which knows nothing of the turn; the
# orchestrator stamps them afterwards. So a mixed-version deployment, or a host driving the
# stages itself, produces a list where some entries carry `seq` and some do not — and sorting
# that mix is WORSE than not sorting: unstamped entries have seq 0 and land in FRONT, giving an
# order true for neither half while looking authoritative.

def _sm(stage, seq=0):
    return StageMetrics(stage=stage, elapsed_ms=1.0, tokens_in=1, tokens_out=1, model="fake",
                        seq=seq)


def test_ordered_stage_metrics_returns_the_TRUE_order_when_all_are_stamped():
    from cogno_anima.types import PipelineContext, ordered_stage_metrics
    ctx = PipelineContext(user_input="x")
    ctx.retry_metrics = [_sm("superego_judge", 3), _sm("ego", 2), _sm("superego_judge", 5)]
    assert [m.stage for m in ordered_stage_metrics(ctx)] == [
        "ego", "superego_judge", "superego_judge"]
    assert [m.seq for m in ordered_stage_metrics(ctx)] == [2, 3, 5]


def test_a_PARTIALLY_stamped_turn_is_left_ALONE_not_sorted():
    """The fixture is built so a naive sort WOULD move things: the unstamped entry sits LAST
    in the raw list, so sorting by `seq` (0 for it) drags it to the front.

    Mutation: sort unconditionally, or key on `m.seq` without the all-or-nothing guard, and
    this dies. A fixture where the unstamped entry already sits first proves nothing — the
    sorted and unsorted answers coincide, and the test passes with the bug in place.
    """
    from cogno_anima.types import PipelineContext, is_fully_sequenced, ordered_stage_metrics
    ctx = PipelineContext(user_input="x")
    ctx.retry_metrics = [_sm("ego", 2), _sm("superego_judge", 3), _sm("voice_unstamped", 0)]

    assert is_fully_sequenced(list(ctx.stage_metrics)) is False
    assert [m.stage for m in ordered_stage_metrics(ctx)] == [
        "ego", "superego_judge", "voice_unstamped"], "a ordem crua é preservada"


def test_an_unsequenced_turn_is_not_claimed_to_have_an_order():
    from cogno_anima.types import PipelineContext, is_fully_sequenced, ordered_stage_metrics
    ctx = PipelineContext(user_input="x")
    ctx.retry_metrics = [_sm("ner"), _sm("ego")]
    assert is_fully_sequenced(list(ctx.stage_metrics)) is False
    assert [m.stage for m in ordered_stage_metrics(ctx)] == ["ner", "ego"]
    assert is_fully_sequenced([]) is False, "vazio não tem ordem para afirmar"


def test_ordered_stage_metrics_never_mutates_the_context():
    from cogno_anima.types import PipelineContext, ordered_stage_metrics
    ctx = PipelineContext(user_input="x")
    ctx.retry_metrics = [_sm("b", 2), _sm("a", 1)]
    before = [m.stage for m in ctx.retry_metrics]
    ordered_stage_metrics(ctx)
    assert [m.stage for m in ctx.retry_metrics] == before
