"""60-second quickstart: perception + routing, no tools, no DB.

The first three stages (NOUMENO → NER → ID) need only an LLMBackend + an Embedder.
Feed any language in; see it normalized to canonical English, read for intent and
PII, and routed — all with deterministic guardrails the core never trusts the LLM
to compute.

    # local + free (default):
    ollama pull mistral && ollama pull nomic-embed-text
    pip install cogno-anima
    python examples/quickstart_60s.py

    # or cloud, zero local install — set a key and pick a model:
    COGNO_BACKEND="groq:llama-3.3-70b-versatile" GROQ_API_KEY=... python examples/quickstart_60s.py
"""

import asyncio
import os

from cogno_anima.types import PipelineContext
from cogno_synapse import OllamaEmbedder, CachingEmbedder, create_backend
from cogno_anima.stages.noumeno import Noumeno
from cogno_anima.stages.ner import IntentAnalyzer
from cogno_anima.stages.id import IDStage

BACKEND = os.environ.get("COGNO_BACKEND", "mistral:latest")   # any create_backend() spec


async def perceive(text: str) -> None:
    llm = create_backend(BACKEND)
    embedder = CachingEmbedder(OllamaEmbedder(model="nomic-embed-text"))

    ctx = PipelineContext(user_input=text)
    ctx = await Noumeno(embedder).process(ctx, llm)      # rewrite → canonical English + drift
    ctx = await IntentAnalyzer().process(ctx, llm)       # intent, sentiment, PII (deterministic)
    ctx = await IDStage().process(ctx, embedder)         # strategic route (heuristic, no LLM)

    print(f"\ninput:   {ctx.user_input!r}  (lang={ctx.noumeno.language})")
    print(f"rewrite: {ctx.noumeno.rewritten!r}")
    print(f"intent:  {ctx.intent.intent_class} · sentiment={ctx.intent.sentiment}")
    print(f"pii:     {ctx.intent.pii or '—'}  (risk={ctx.intent.pii_risk})")
    print(f"route:   {ctx.id_result.triad_route}  (goal={ctx.id_result.goal_status})")
    print(f"tokens:  {ctx.total_tokens}  (LLM + embeddings)")


async def main() -> None:
    for text in [
        "quanto tá o meu saldo?",              # PT info request → EGO (tool gateway)
        "cancela meu cartão AGORA!!!",         # urgent action → routing shifts
        "oi, tudo bem?",                        # social → SUPEREGO
    ]:
        await perceive(text)


if __name__ == "__main__":
    asyncio.run(main())
