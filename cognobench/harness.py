"""
Benchmark harness — a minimal reference pipeline that drives the cogno-anima
stages directly through dependency injection.

This deliberately does NOT live in the `cogno_anima` library: orchestration is
the host's responsibility. The harness exists only so the benchmark can run
NOUMENO → NER → Drift against a real (or stubbed) backend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cogno_synapse import (
    CachingEmbedder,
    Embedder,
    LLMBackend,
    OllamaBackend,
    create_embedder,
)
from cogno_anima.stages.noumeno import Noumeno
from cogno_anima.stages.ner import IntentAnalyzer
from cogno_anima.stages.id import IDStage
from cogno_anima.stages.drift import DriftCalculator
from cogno_anima.types import PipelineContext

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "cogno_anima" / "prompt_templates"

SLANGS = {"vc": "você", "pq": "porque", "blz": "beleza", "pfv": "por favor"}


@dataclass
class PipelineOutput:
    """Carrier for one benchmark run's stage outputs."""
    ctx: PipelineContext


class CognitivePipeline:
    """Reference NOUMENO → NER → Drift pipeline for benchmarking.

    ``ner_backend`` (plan 0.10) lets a slot sweep vary ONE stage's model while
    the other is pinned — with a single backend, "the NER score of model X" is
    really "NER given X's own NOUMENO rewrite", a contaminated measurement.
    Default None = same backend for both (the classic single-model run)."""

    def __init__(self, backend: LLMBackend, embedder: Embedder,
                 ner_backend: LLMBackend | None = None) -> None:
        self._backend = backend
        self._ner_backend = ner_backend or backend
        self._embedder = embedder
        self._noumeno = Noumeno(embedder=embedder, prompts_dir=PROMPTS_DIR, slangs=SLANGS)
        self._ner = IntentAnalyzer(prompts_dir=PROMPTS_DIR)
        self._id = IDStage()
        self._drift = DriftCalculator()

    async def run(
        self,
        user_input: str,
        history: Optional[list[str]] = None,
        force_language: Optional[str] = None,
        stop_after: str = "drift",
        metadata: Optional[dict] = None,
    ) -> PipelineContext:
        """Run the reference pipeline up to `stop_after` ('noumeno'|'ner'|'id'|'drift').

        `metadata` seeds `ctx.metadata` before the run — used by the multi-turn ID
        dimension to carry `id_state` and NER carry-over (`last_goal`,
        `active_domains`, `turn_number`) across turns. History seeding (the
        subject-continuity anchor) is applied on top.
        """
        ctx = PipelineContext(user_input=user_input, force_language=force_language)

        if metadata:
            ctx.metadata.update(metadata)

        # Seed multi-turn memory from history (cheap: use raw last turn as the
        # subject-continuity anchor; embeddings work on raw text).
        if history:
            ctx.metadata["last_rewritten"] = history[-1]
            ctx.metadata["last_context_turn"] = history[-1]

        ctx = await self._noumeno.process(ctx, self._backend)
        if stop_after == "noumeno":
            return ctx

        ctx = await self._ner.process(ctx, self._ner_backend)
        if stop_after == "ner":
            return ctx

        if stop_after == "id":
            # The ID stage seeds drift (epistemological + ontological) if absent,
            # then adds situational → cumulative → goal-aware downgrade.
            ctx = await self._id.process(ctx, self._embedder)
            return ctx

        drift = self._drift.compute(ctx.noumeno, ctx.intent)
        self._drift.compute_ontological(drift, ctx.noumeno, ctx.intent)
        self._drift.compute_cumulative(drift)
        ctx.drift = drift
        return ctx


# ──────────────────────────────────────────────────────────────────────────
#  Backend builders
# ──────────────────────────────────────────────────────────────────────────

def build_ollama(
    model: str,
    embed_model: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
    think: bool = False,
) -> tuple[LLMBackend, Embedder]:
    """Real Ollama backend + embedder (temperature 0, JSON-constrained output).

    ``think`` toggles the reasoning channel for reasoning models (qwen3, …). Under
    ``format="json"`` Ollama suppresses the thinking channel, so it is a safe no-op
    for these JSON-constrained ops — the visible effect is on the text path below.
    """
    backend = OllamaBackend(model=model, base_url=base_url, temperature=0.0,
                            format="json", think=think)
    # Through `build_embedder`, not a second construction: the `--embed-model` grammar has to
    # hold on BOTH branches. Hardcoding it here meant `--model qwen3:8b --embed-model
    # openai:text-embedding-3-small` silently became `OllamaEmbedder(model="openai:text-...")`
    # and 404'd on the first embed — while the flag's own help promised the grammar
    # unconditionally.
    return backend, build_embedder(embed_model, base_url)


def build_ollama_text(
    model: str, base_url: str = "http://localhost:11434", think: bool = False,
) -> LLMBackend:
    """Real Ollama backend for the EGO — NO JSON format, so the fallback path can
    emit free text with ``<TOOL_CALL>`` tags (JSON-constrained output would forbid them).

    ``think`` enables the reasoning channel; with no ``format`` constraint the model
    reasons internally and returns the answer in ``response`` (CognoBench's
    think-on/off comparison lives here and in the EGO loop)."""
    return OllamaBackend(model=model, base_url=base_url, temperature=0.0, think=think)


def build_cloud(spec: str) -> LLMBackend:
    """Cloud backend via the cogno-synapse factory (``"openai:gpt-4o-mini"``,
    ``"anthropic:…"``, any OpenAI-compatible prefix). ``temperature=0`` like the
    local runs. One instance serves BOTH the JSON stages (they regex-extract the
    object from plain text) and the text paths (EGO/voice) — cloud backends are
    not response-format-locked the way ``format="json"`` Ollama is. Note the EGO
    runs the NATIVE function-calling path on these (they satisfy
    ``ToolCallingBackend``) — the production-realistic cloud measurement, vs the
    text-fallback path the local column exercises. Raises ``MissingAPIKeyError``
    when the provider's key env is unset."""
    from cogno_synapse import create_backend
    return create_backend(spec, temperature=0.0)


def embedder_is_local(embed_model: str) -> bool:
    """Does this spec resolve to Ollama? ASKS the factory instead of re-deciding.

    It used to re-implement the parse, and got it wrong on the form this project actually
    documents: `partition(":")` reads `nomic-embed-text:latest` as provider `nomic-embed-text`
    and answers False, while the factory maps an unknown prefix straight back to Ollama. So a
    cloud-LLM run with the tagged local embedder — the exact spec named in CLAUDE.md — skipped
    the friendly preflight, built an OllamaEmbedder anyway, and died on the first embed with a
    raw transport error, after the cloud model had already been billed.
    Mirroring a rule is how it drifts; asking is how it cannot."""
    from cogno_synapse.factory import parse_model_string

    provider, _model = parse_model_string(embed_model)
    return provider == "ollama"


def build_embedder(
    embed_model: str = "nomic-embed-text", base_url: str = "http://localhost:11434",
) -> Embedder:
    """The bench's embedder, from a ``provider:model`` spec — same grammar as ``--model``.

    Local Ollama stays the DEFAULT and is still the better control: it is free and
    deterministic, and embedding similarity is not the model under test, so holding it
    fixed keeps an A/B measuring the one thing that changed.

    It is no longer the only option, because "free" stopped being the only cost. The
    machine that runs this bench also runs other GPU work, and a cloud LLM run was still
    refusing to start without Ollama up — the embedder alone pinned the whole suite to a
    busy GPU. A run can now be cloud end to end.

    Caching wraps whatever comes back: the LRU is what keeps a repeated goal comparison
    from being billed twice on a cloud embedder."""
    return CachingEmbedder(create_embedder(embed_model, base_url=base_url))


class TokenTally:
    """Wrap any ``LLMBackend`` and count tokens/calls across the whole run — the
    bench's cost meter (multiply by the provider's price table to get $ per
    sweep). Transparent: forwards ``generate``/``chat_with_tools`` untouched, so
    it satisfies ``ToolCallingBackend`` structurally; ``supports_native_tools``
    defers to the inner backend, keeping the EGO's native-vs-fallback gate
    exactly as if unwrapped."""

    def __init__(self, inner: LLMBackend) -> None:
        self._inner = inner
        self.tokens_in = 0
        self.tokens_out = 0
        self.calls = 0

    @property
    def model(self) -> str:
        return getattr(self._inner, "model", "unknown")

    async def generate(self, system: str, prompt: str):
        text, tin, tout = await self._inner.generate(system, prompt)
        self.tokens_in += tin
        self.tokens_out += tout
        self.calls += 1
        return text, tin, tout

    async def chat_with_tools(self, messages, tools, tool_choice=None):
        msg, tin, tout = await self._inner.chat_with_tools(messages, tools, tool_choice)
        self.tokens_in += tin
        self.tokens_out += tout
        self.calls += 1
        return msg, tin, tout

    def supports_native_tools(self) -> bool:
        inner = getattr(self._inner, "supports_native_tools", None)
        return bool(inner()) if callable(inner) else False


async def ollama_available(base_url: str = "http://localhost:11434") -> bool:
    import httpx
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{base_url}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────
#  Stub backend — plumbing smoke test only (fixed output, not for scoring)
# ──────────────────────────────────────────────────────────────────────────

_STUB_NOUMENO = json.dumps({
    "rewritten": "Explain how something works.",
    "context_turn": "",
    "confidence": 0.9,
    "changed": True,
    "preserved_terms": [],
    "rewrite_warnings": [],
})

_STUB_NER = json.dumps({
    "intent_class": "INFORMATION_REQUEST", "sentiment": "NEUTRAL", "confidence": 0.9,
    "temporal_class": "TIMELESS", "triad_signal": "EGO",
    "entities": {"people": [], "objects": [], "concepts": []},
    "location": None, "mandatory_tags": ["ANALYSIS"], "aristotelian": {}, "goal": None, "causal_chain": [], "parole": "COLOQUIAL",
    "negation": [], "constraints": [], "domains": ["GENERAL"], "modality": "CERTAIN",
    "speech_act": "INTERROGATIVE", "is_composite": False, "is_sequential": False,
    "verbs": [], "context_dependent": False, "pii": [],
    "pii_risk": "NONE", })


_STUB_EGO_CALL = '<TOOL_CALL>{"tool": "get_balance", "args": {}}</TOOL_CALL>'


class _StubBackend:
    model = "stub"

    async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
        # Route by which stage prompt this is: the NER user prompt contains "NOUMENO:".
        # An EGO fallback prompt renders the tool list + <TOOL_CALL> mechanics —
        # emit ONE read-only call so the ego_none sabotage has a measurable drop
        # (a stub that never calls tools is behaviorally identical to the
        # sabotage, which made that mutation dead machinery — review finding).
        # The duplicate-call guard terminates the loop on the repeat.
        blob = system + "\n" + prompt      # the EGO renders tools into SYSTEM
        if "<TOOL_CALL>" in blob and "get_balance" in blob:
            return _STUB_EGO_CALL, 10, 10
        if "NOUMENO:" in prompt or "ORIGINAL:" in prompt:
            return _STUB_NER, 10, 10
        return _STUB_NOUMENO, 10, 10


class _StubScopeBlock:
    """Stub for the SCOPE slot only: blocks everything. The plain stub's scope
    verdict is fail-open blocked=False — behaviorally identical to the
    scope_allow sabotage, which made that mutation unfalsifiable (review
    finding). A blocking baseline gives scope_allow a real directional flip,
    while conversations keep the plain stub (their plumbing needs the full
    EGO⇄judge loop, which an always-block scope would skip)."""

    model = "stub-scope-block"

    async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
        return json.dumps({"blocked": True, "refusal_message": "stub-blocked"}), 5, 5


class _StubEmbedder:
    async def embed(self, text: str) -> list[float]:
        n = float(len(text))
        return [n, n * 2, 1.0]

    async def similarity(self, a: str, b: str) -> float:
        return 1.0 if a == b else 0.8


def build_stub() -> tuple[LLMBackend, Embedder]:
    """Deterministic stub — proves the harness/report plumbing without a model."""
    return _StubBackend(), _StubEmbedder()


# ──────────────────────────────────────────────────────────────────────────
#  Sabotage stubs (plan 0.8, mutations M1/M3) — each breaks ONE stage on
#  purpose. A case whose checks stay green while its stage is sabotaged does
#  not observe that stage: it is dead weight, not a guard. Slot routing places
#  the sabotage exactly (no prompt sniffing): the sabotaged backend IS the slot.
# ──────────────────────────────────────────────────────────────────────────

_SABOTAGE_PAYLOADS = {
    # NOUMENO echoes nothing useful: constant rewrite, never marks a change —
    # every expect_in_rewrite / expect_changed=True check must flip red.
    "noumeno_echo": json.dumps({
        "rewritten": "unchanged input", "context_turn": "", "confidence": 0.1,
        "changed": False, "preserved_terms": [], "rewrite_warnings": [],
    }),
    # NER classifies nothing: UNKNOWN class, empty extractions. The DETERMINISTIC
    # safety checks must SURVIVE this (regex on the original text) — that
    # asymmetry is exactly what the mutation report shows.
    "ner_unknown": json.dumps({
        "intent_class": "UNKNOWN", "sentiment": "NEUTRAL", "confidence": 0.0,
        "temporal_class": "TIMELESS", "triad_signal": "BALANCED",
        "entities": {"people": [], "objects": [], "concepts": []}, "location": None,
        "mandatory_tags": [], "aristotelian": {}, "goal": None, "causal_chain": [],
        "parole": "COLOQUIAL", "negation": [], "constraints": [], "domains": [],
        "modality": "CERTAIN", "speech_act": "DECLARATIVE", "is_composite": False,
        "is_sequential": False, "verbs": [], "context_dependent": False,
        "pii": [], "pii_risk": "NONE",
    }),
    # Scope guard waves everything through — every must-block case flips.
    "scope_allow": json.dumps({"blocked": False, "refusal_message": ""}),
    # Judge approves everything with an empty critique — the exact failure mode
    # measured on mistral:latest (3/3 false approvals); must-reject cases flip.
    "judge_approve": json.dumps({"approved": True, "critique": ""}),
    # EGO emits no tool call ever — tool_selected/order checks flip.
    "ego_none": "I will not use any tool.",
    # Voice says nothing — response_nonempty flips.
    "voice_empty": "",
    # M3: garbage that is not JSON — must surface as scored model-fault failures
    # (StageParseError → 0.3 classification), never a silent pass.
    "garbage": "!!! not json at all {{{",
}

SABOTAGE_TARGET_SLOT = {
    "noumeno_echo": "noumeno", "ner_unknown": "ner", "scope_allow": "scope",
    "judge_approve": "judge", "ego_none": "ego", "voice_empty": "voice",
    "garbage": "ner",
}


class _SabotageBackend:
    """Returns one fixed payload for every call — placed into a single slot."""

    def __init__(self, mode: str) -> None:
        self.model = f"sabotage:{mode}"
        self._payload = _SABOTAGE_PAYLOADS[mode]

    async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
        return self._payload, 1, 1


def build_sabotage_stub(mode: str) -> LLMBackend:
    return _SabotageBackend(mode)


# ──────────────────────────────────────────────────────────────────────────
#  Preflight (plan 0.11) — a disarmed instrument scores a lying green.
# ──────────────────────────────────────────────────────────────────────────

async def preflight_embedder(embedder: Embedder) -> tuple[bool, float]:
    """The embedder must DISCRIMINATE. A constant-similarity embedder silently
    disabled change_subject for an entire bench's history (the closer stub
    incident) — assert two unrelated sentences do not read as near-identical."""
    sim = await embedder.similarity(
        "quero agendar uma consulta para quinta-feira",
        "the quarterly financial report shows increased revenue",
    )
    return sim < 0.95, sim


async def preflight_local_toks(model: str, base_url: str) -> float:
    """Measured GENERATION tok/s of a short real run. Ollama loses CUDA
    per-process and silently degrades to CPU (~10-20× slower) — a run started in
    that state produces timeouts and truncations that masquerade as model
    failures. The caller compares against a threshold and refuses to score below.

    Uses Ollama's ``eval_count/eval_duration`` (generation only), NOT wall-clock:
    wall-clock includes model LOAD, so a cold healthy model measured 0.4 tok/s
    and false-failed the probe — and the error's advice (restart) made it
    colder. Any transport failure returns 0.0 (the threshold then refuses)."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(f"{base_url}/api/generate", json={
                "model": model, "prompt": "Count: 1 2 3 4 5 6 7 8 9 10",
                "stream": False, "think": False, "options": {"num_predict": 40},
            })
            resp.raise_for_status()
            data = resp.json()
    except Exception:  # noqa: BLE001 — unreachable/timeout = cannot certify
        return 0.0
    tokens = data.get("eval_count") or 0
    dur_ns = data.get("eval_duration") or 0
    return (tokens / (dur_ns / 1e9)) if dur_ns > 0 else 0.0
