# 🧠 cogno-core

**The infrastructure-agnostic cognitive pipeline at the heart of [Cogno](https://github.com/sudoers-ai).**

`cogno-core` is a modular, dependency-light library that turns raw user input
into a routed, grounded, persona-voiced response — decoupled from any database,
message bus, or proprietary infra. It is the brain; your application is the body.

```
NOUMENO  →  NER  →  ID  →  EGO  →  SUPEREGO        (+ Drift, woven through)
   │         │       │      │          │
 rewrite   intent  route  execute    voice
 to EN     + PII   + goal  (tools)   + judge
```

- **NOUMENO** — perception: slang/normalization, language resolution, rewrite to canonical English, epistemological drift.
- **NER** — semantic analysis: intent, sentiment, entities, PII (detected deterministically, never trusted from the LLM), domains — all coerced to a closed vocabulary.
- **ID** — strategic router & continuity (heuristic, no LLM): goal lifecycle, BDI intentions, attention, safety gate, drift.
- **EGO** — executor: runs an agent loop and dispatches tools (native function calling *or* a `<TOOL_CALL>` text fallback). It gathers data; it does **not** write the reply.
- **SUPEREGO** — locutor: scope guard + judge (goal↔execution) + **writes** the final response in the persona's voice, grounded in the EGO's data.
- **Drift** — pure, no I/O: epistemological → ontological → situational → execution → synthesis → cumulative, emitting a `drift_action` signal.

## Philosophy: the core signals, the host decides

`cogno-core` deliberately ships **no business rules and no infrastructure**. It is
built on a few hard contracts:

- **Infra-agnostic** — no DB, no MCP, no queue. Tool execution is delegated to a host-injected `ToolDispatcher` ("EGO = brain, dispatcher = hands"). Persistence, transactions, rollback/outbox and atomicity are **host** concerns.
- **Never trust the LLM** — PII risk, routing, vocabularies and drift are computed deterministically; the LLM's self-assessment is ignored.
- **Signals, not exceptions** — the core emits `drift_action`, `blocked`, `interrupted`, `needs_handoff`, `stop_reason`; the host turns those into actions (retry, escalate, hand off to a human).
- **Stateless across turns** — all cross-turn state rides in `ctx.metadata` (e.g. `id_state`), a serializable dict the host persists — safe for multi-worker setups.
- **Errors propagate** — backends raise on transport/auth failure instead of degrading silently; a `FallbackBackend` is how you opt into failover.

What stays in the **host**: persona/MCP binding, RBAC, model ladders, the
retry/correction *loop*, session splitting, the real human handoff, billing,
semantic cache, and persistence. See `CLAUDE.md` for the full boundary map and
`cognobench/pipeline.py` (`ReferencePipeline`) for a reference orchestrator.

## Install

```bash
pip install -e .                      # core (pydantic, langdetect, httpx)
pip install -e ".[dev]"               # + test/lint/type-check tooling

# Optional cloud backends (SDKs are lazy-imported — install only what you use):
pip install "cogno-core[openai]"      # also: anthropic | groq | gemini | bedrock | llm (all)
```

Local inference uses [Ollama](https://ollama.com) (`mistral:latest` for the
cognitive stages, `nomic-embed-text:latest` for embeddings) by default.

## Quickstart (perception + routing)

The first three stages need only an `LLMBackend` + an `Embedder` — no dispatcher,
no DB:

```python
import asyncio
from cogno_core.types import PipelineContext
from cogno_core.llm import OllamaBackend, OllamaEmbedder, CachingEmbedder
from cogno_core.stages.noumeno import Noumeno
from cogno_core.stages.ner import IntentAnalyzer
from cogno_core.stages.id import IDStage

async def main():
    llm = OllamaBackend(model="mistral:latest", temperature=0.0, format="json")
    embedder = CachingEmbedder(OllamaEmbedder(model="nomic-embed-text"))

    ctx = PipelineContext(user_input="quanto tá o meu saldo?")
    ctx = await Noumeno(embedder).process(ctx, llm)
    ctx = await IntentAnalyzer().process(ctx, llm)
    ctx = await IDStage().process(ctx, embedder)

    print(ctx.noumeno.rewritten)        # "what is my balance?"
    print(ctx.intent.intent_class)      # INFORMATION_REQUEST
    print(ctx.id_result.triad_route)    # EGO  (tool gateway)
    print(ctx.total_tokens)             # LLM + embedding cost

asyncio.run(main())
```

To run the **full** pipeline (EGO tool dispatch + SUPEREGO voicing + the
correction loop), you provide a `ToolDispatcher`, persona prompts, and the
orchestration glue. A complete, runnable reference lives in
`cognobench/pipeline.py: ReferencePipeline`.

## LLM & embedder backends

Stages depend only on three runtime-checkable `Protocol`s in
`cogno_core/llm/base.py`: `LLMBackend`, the optional `ToolCallingBackend` (native
function calling), and `Embedder`. Anything matching the shape works.

- **Local**: `OllamaBackend`, `OllamaEmbedder` (+ `CachingEmbedder` adds a bounded LRU + token accounting to any embedder).
- **Cloud**: OpenAI, Anthropic, Groq, Gemini, Bedrock — each implements native function calling.
- **OpenAI-compatible** (DeepSeek, Moonshot/Kimi, xAI/Grok, OpenRouter, Together, Fireworks): reuse `OpenAIBackend` via its `base_url`, selected by `create_backend("deepseek:deepseek-chat")`.
- **Failover**: `FallbackBackend([...])` tries each in order; first success wins.

```python
from cogno_core.llm import create_backend
backend = create_backend("openai:gpt-4o-mini")   # or "deepseek:deepseek-chat", "qwen3:8b", …
```

## CognoBench

A self-contained, dependency-light cognitive benchmark (`cognobench/`), kept
decoupled from the library, scoring each stage end-to-end against a real model:

```bash
python3 cognobench.py                          # all dimensions vs local Ollama
python3 cognobench.py --only ner --limit 3     # one dimension, few cases
python3 cognobench.py --only conversations     # broad multi-turn e2e simulation
python3 cognobench.py --stub --limit 3         # fast plumbing smoke (no model)
```

Dimensions: `noumeno · ner · id · ego · superego · drift · conversations`. Hard
invariants are enforced; model-dependent "soft" checks are recalibratable with
`--calibrate`. The benchmark is **not** shipped in the wheel.

## Testing

```bash
python3 -m pytest                  # everything
python3 -m pytest tests/unit       # fast, no network (stubs)
python3 -m pytest tests/integration  # real Ollama; auto-skips if unavailable
```

Unit tests run on a coverage gate (`--cov-fail-under=85`) and use the
`StubBackend`/`StubEmbedder` doubles in `tests/conftest.py`. Integration tests
use real models at `temperature=0.0` for determinism.

## License

Licensed under the [Apache License 2.0](LICENSE).
