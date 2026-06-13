"""
Benchmark harness — a minimal reference pipeline that drives the cogno-core
stages directly through dependency injection.

This deliberately does NOT live in the `cogno_core` library: orchestration is
the host's responsibility. The harness exists only so the benchmark can run
NOUMENO → NER → Drift against a real (or stubbed) backend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cogno_core.llm import LLMBackend, Embedder, OllamaBackend, OllamaEmbedder
from cogno_core.stages.noumeno import Noumeno
from cogno_core.stages.ner import IntentAnalyzer
from cogno_core.stages.drift import DriftCalculator
from cogno_core.types import PipelineContext

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

SLANGS = {"vc": "você", "pq": "porque", "blz": "beleza", "pfv": "por favor"}


@dataclass
class PipelineOutput:
    """Carrier for one benchmark run's stage outputs."""
    ctx: PipelineContext


class CognitivePipeline:
    """Reference NOUMENO → NER → Drift pipeline for benchmarking."""

    def __init__(self, backend: LLMBackend, embedder: Embedder) -> None:
        self._backend = backend
        self._embedder = embedder
        self._noumeno = Noumeno(embedder=embedder, prompts_dir=PROMPTS_DIR, slangs=SLANGS)
        self._ner = IntentAnalyzer(prompts_dir=PROMPTS_DIR)
        self._drift = DriftCalculator()

    async def run(
        self,
        user_input: str,
        history: Optional[list[str]] = None,
        force_language: Optional[str] = None,
        stop_after: str = "drift",
    ) -> PipelineContext:
        """Run the reference pipeline up to `stop_after` ('noumeno'|'ner'|'drift')."""
        ctx = PipelineContext(user_input=user_input, force_language=force_language)

        # Seed multi-turn memory from history (cheap: use raw last turn as the
        # subject-continuity anchor; embeddings work on raw text).
        if history:
            ctx.metadata["last_rewritten"] = history[-1]
            ctx.metadata["last_context_turn"] = history[-1]

        ctx = await self._noumeno.process(ctx, self._backend)
        if stop_after == "noumeno":
            return ctx

        ctx = await self._ner.process(ctx, self._backend)
        if stop_after == "ner":
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
) -> tuple[LLMBackend, Embedder]:
    """Real Ollama backend + embedder (temperature 0 for determinism)."""
    backend = OllamaBackend(model=model, base_url=base_url, temperature=0.0)
    embedder = OllamaEmbedder(model=embed_model, base_url=base_url)
    return backend, embedder


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
    "entities": {"people": [], "pronouns": [], "possessives": [], "objects": [], "concepts": []},
    "location": None, "mandatory_tags": ["ANALYSIS"], "abstract_tags": ["SMOKE"],
    "aristotelian": {}, "goal": None, "causal_chain": [], "parole": "COLOQUIAL",
    "negation": [], "constraints": [], "domains": ["GENERAL"], "modality": "CERTAIN",
    "speech_act": "INTERROGATIVE", "is_composite": False, "is_sequential": False,
    "verbs": [], "context_dependent": False, "comparatives": [], "pii": [],
    "pii_risk": "NONE", "raw_intent_class": "INFORMATION_REQUEST",
    "raw_domains": ["GENERAL"], "raw_goal": None,
})


class _StubBackend:
    model = "stub"

    async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
        # Route by which stage prompt this is: the NER user prompt contains "NOUMENO:".
        if "NOUMENO:" in prompt or "ORIGINAL:" in prompt:
            return _STUB_NER, 10, 10
        return _STUB_NOUMENO, 10, 10


class _StubEmbedder:
    async def embed(self, text: str) -> list[float]:
        n = float(len(text))
        return [n, n * 2, 1.0]

    async def similarity(self, a: str, b: str) -> float:
        return 1.0 if a == b else 0.8


def build_stub() -> tuple[LLMBackend, Embedder]:
    """Deterministic stub — proves the harness/report plumbing without a model."""
    return _StubBackend(), _StubEmbedder()
