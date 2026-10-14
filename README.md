# 🧠 cogno-anima

**The infrastructure-agnostic cognitive pipeline at the heart of [Cogno](https://github.com/sudoers-ai).**

`cogno-anima` is a modular, dependency-light library that turns raw user input
into a routed, grounded, persona-voiced response — decoupled from any database,
message bus, or proprietary infra. It is the brain; the host — your application —
is the body that carries it.

![SECRETARY demo — the full pipeline booking an appointment on a free local model](docs/assets/secretary-demo.gif)

*The whole stack, live on a free 8B local model: NOUMENO rewrites Portuguese to
canonical English, the ID routes, the EGO calls real scheduling tools (and
self-corrects around a weekend), the judge approves, the voice replies in the
persona. Run it yourself in ~15 minutes — see
[Try it](#try-it--a-complete-agent-in-15-minutes) · quality numbers in
[BENCHMARKS.md](BENCHMARKS.md).*

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
  Its prompt leaves a record on the result — which parts it carried and how long each was (`EgoResult.prompt_blocks`, the closed `EGO_PROMPT_BLOCKS`), never the text (0.1.5).
- **SUPEREGO** — locutor: scope guard + judge (three branches — goal↔execution, truth, grounding — chosen per turn) + **writes** the final response in the persona's voice, grounded in the EGO's data.
  On a turn the scope guard refused, an optional strict selector over a host-built closed list (`cogno_anima.stages.scope_options`) either proves the refusal false or offers a closed "did you mean…?" (#200). A message the executor HOLDS for the contact's yes is judged before it is sent: the text the recipient will read (#183), the ask it records for them (#198), with the confirmation question left to the host (#199) and, when the host declares `ctx.force_language`, the language it is written in (#204).
  Only a JSON boolean is a verdict (`cogno_anima.verdict`, 0.1.3): a judge answering `"approved": "false"` in a string does not approve (fail-closed), a guard answering `"blocked": "false"` does not block (fail-open), and each result says how its verdict was read (`verdict_read`).
  What each JSON stage had to repair in the model's reply — a key left out, a value outside the vocabulary, a fence stripped — is counted on its result (`repairs`, a closed alphabet per stage in `cogno_anima.repairs`, 0.1.9); the count decides nothing.
  The JSON Schema of each of those replies is derived from the same vocabularies (`cogno_anima.schemas`, 0.1.10). Nothing reads it yet: no stage sends a schema.
- **Untrusted text** — a tool result, a held message, the persona's rules, the contact's note and the contact's own message are made inert before they reach the executor, the judge or the voice: no tool-call trigger survives, no fence can be opened or closed from inside, no line opens with a header the core renders (`cogno_anima.security.prompt_guard`: `sanitize_untrusted`, `defang_structure`). What other people wrote in the turn's context travels on a carrier of its own (`mk.EGO_CONTEXT_UNTRUSTED`) and is rendered inside a fence. This is structural: it does not change what a model does with an instruction it reads.
- **Tool-dispatch adapters** (`cogno_anima.tools`) — the wrappers a host stacks around its dispatcher: `CompositeDispatcher` (many sources, one table), the commit sink, the confirm-argument recorder, id provenance, idempotency, and `PreJudgeDispatcher`, which has the judge read a WRITE before it goes out — in shadow by default, enforced per tool only where the host asks (`cogno_anima.stages.ProposalJudge` is the callback this package ships).
- **Drift** — pure, no I/O: epistemological → ontological → situational → execution → synthesis → cumulative, emitting a `drift_action` signal.

## Philosophy: the core signals, the host decides

`cogno-anima` deliberately ships **no business rules and no infrastructure**. It is
built on a few hard contracts:

- **Infra-agnostic** — no DB, no MCP, no queue. Tool execution is delegated to a host-injected `ToolDispatcher` ("EGO = brain, dispatcher = hands"). Persistence, transactions, rollback/outbox and atomicity are **host** concerns.
- **Never trust the LLM** — PII risk, routing, vocabularies and drift are computed deterministically; the LLM's self-assessment is ignored.
- **Signals, not exceptions** — the core emits `drift_action`, `blocked`, `interrupted`, `needs_handoff`, `stop_reason`; the host turns those into actions (retry, escalate, hand off to a human).
- **Stateless across turns** — all cross-turn state rides in `ctx.metadata` (e.g. `id_state`), a serializable dict the host persists — safe for multi-worker setups.
- **Errors propagate** — backends raise on transport/auth failure instead of degrading silently; a `FallbackBackend` is how you opt into failover.

What stays in the **host**: persona/MCP binding, RBAC, model ladders, the
retry/correction *loop*, session splitting, the real human handoff, billing,
semantic cache, and persistence. See **[docs/HOST_INTEGRATION.md](docs/HOST_INTEGRATION.md)**
for how to wire the core into a host, `CLAUDE.md` for the full boundary map, and
`cognobench/pipeline.py` (`ReferencePipeline`) for a reference orchestrator.

## Try it — a complete agent in 15 minutes

The fastest way to *feel* the pipeline is the SECRETARY demo in
[`cogno-praxis`](https://github.com/sudoers-ai/cogno-praxis): the full
cognitive loop over local Ollama, with a real scheduling vertical served as an
MCP subprocess — free, offline, any language in:

```bash
ollama pull qwen3:8b && ollama pull nomic-embed-text     # the download IS most of the 15 min

pip install "git+https://github.com/sudoers-ai/cogno-homeo" \
            "git+https://github.com/sudoers-ai/cogno-synapse" \
            "git+https://github.com/sudoers-ai/cogno-anima" \
            "git+https://github.com/sudoers-ai/cogno-soma" \
            "cogno-mcp[mcp] @ git+https://github.com/sudoers-ai/cogno-mcp"

git clone https://github.com/sudoers-ai/cogno-praxis && cd cogno-praxis
pip install -e .
python examples/secretary_demo.py --trace     # --trace shows each cognitive stage live
```

(PyPI packages are coming; the git chain above is the interim install.)

## Install

```bash
pip install -e .                      # core (pydantic, langdetect, cogno-synapse — which brings httpx)
pip install -e ".[dev]"               # + test/lint/type-check tooling

# Optional cloud backends. The SDKs belong to cogno-synapse and are lazy-imported — install only
# what you use. This package has ONE extra for them, `llm`, which pulls every provider:
pip install "cogno-anima[llm]"         # all of: openai | anthropic | groq | gemini | bedrock
pip install "cogno-synapse[openai]"    # or a single provider, from the lib that owns it
```

Local inference uses [Ollama](https://ollama.com). The library itself has no default model: a
backend is built with the model you name. What the rest of the repository names is `qwen3:8b`
for CognoBench and the integration suites, `mistral:latest` in the two scripts under `examples/`,
and `nomic-embed-text` for embeddings.

## Quickstart (perception + routing)

The first three stages need only an `LLMBackend` + an `Embedder` — no dispatcher,
no DB:

```python
import asyncio
from cogno_anima.types import PipelineContext
from cogno_synapse import OllamaBackend, OllamaEmbedder, CachingEmbedder
from cogno_anima.stages.noumeno import Noumeno
from cogno_anima.stages.ner import IntentAnalyzer
from cogno_anima.stages.id import IDStage

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

Stages depend only on three runtime-checkable `Protocol`s defined in the sibling
[`cogno-synapse`](https://github.com/sudoers-ai/cogno-synapse) lib
(`cogno_synapse.base`): `LLMBackend`, the optional `ToolCallingBackend` (native
function calling), and `Embedder`. Anything matching the shape works. (`cogno-anima`
re-exports them, so `from cogno_anima import LLMBackend` keeps working.)

- **Local**: `OllamaBackend`, `OllamaEmbedder` (+ `CachingEmbedder` adds a bounded LRU + token accounting to any embedder).
- **Cloud**: OpenAI, Anthropic, Groq, Gemini, Bedrock — each implements native function calling.
- **OpenAI-compatible** (DeepSeek, Moonshot/Kimi, xAI/Grok, OpenRouter, Together, Fireworks): reuse `OpenAIBackend` via its `base_url`, selected by `create_backend("deepseek:deepseek-chat")`.
- **Failover**: `FallbackBackend([...])` tries each in order; first success wins.

```python
from cogno_synapse import create_backend
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

Dimensions: `noumeno · ner · id · safety · ego · superego · drift · conversations`
(`superego` is a selector — it SCORES as three per-op dimensions,
`superego_scope · superego_judge · superego_voice`). Hard
invariants are enforced; model-dependent "soft" checks are recalibratable with
`--calibrate`. The benchmark is **not** shipped in the wheel.

**Consolidated scoreboard — quality per cognitive function × model, and where
small models break: [BENCHMARKS.md](BENCHMARKS.md).**

## Documentation

Every `.md` of this repository, by what it is for.

- [`docs/HOST_INTEGRATION.md`](docs/HOST_INTEGRATION.md) — how to wire the core into a host: the
  stages in order, what the host injects, the metakeys, the correction loop, the tool adapters.
- [`docs/ACT_CONFIRM_READONLY.md`](docs/ACT_CONFIRM_READONLY.md) — the three confirmation gates
  of the EGO (the read-only mask, the confirmation gate, the skill that asks), what a hold is and
  how long it lives, and the pre-judge's enforcement.
- [`docs/NETWORK_PERSONA_CHANNEL.md`](docs/NETWORK_PERSONA_CHANNEL.md) — the contract for the
  three things this library does NOT own: the persona, the channel and the network between
  personas. Each section ends with who decides.
- [`CLAUDE.md`](CLAUDE.md) — the full map of the pipeline, stage by stage, with the measurement
  behind each rule. Long, and the place where a boundary is argued.
- [`LOGGING.md`](LOGGING.md) — what the library logs and what it never logs.
- [`BENCHMARKS.md`](BENCHMARKS.md) — the consolidated scoreboard, and how to read a number from
  this bench. Per-dimension result notes, each one run and dated, live beside the bench:
  [`NER`](cognobench/NER_BENCH_RESULTS.md), [`ID`](cognobench/ID_BENCH_RESULTS.md),
  [`EGO`](cognobench/EGO_BENCH_RESULTS.md), [`SUPEREGO`](cognobench/SUPEREGO_BENCH_RESULTS.md),
  [`conversations`](cognobench/CONVERSATION_BENCH_RESULTS.md) and
  [`bookkeeper`](cognobench/BOOKKEEPER_BENCH_RESULTS.md). [`cognobench/CLAUDE.md`](cognobench/CLAUDE.md)
  is the bench's own design note, and
  [`cognobench/results/quarantine/README.md`](cognobench/results/quarantine/README.md) says what
  the quarantined runs are.
- [`examples/README.md`](examples/README.md) — the two runnable examples: a 60-second quickstart
  and a minimal host.
- [`PYPI_SETUP.md`](PYPI_SETUP.md) — publishing the ecosystem to PyPI.
- [`CHANGELOG.md`](CHANGELOG.md) — every change, newest first.

## Environment

The library reads **no environment variable**: every setting is an argument, and provider keys
are read by `cogno-synapse` when it builds a backend. What the rest of the repository reads:

| name | read by | what |
| --- | --- | --- |
| `COGNO_TEST_MODEL` | `tests/integration/backends.py` | the `provider:model` the integration suites run on (default: Ollama `qwen3:8b`) |
| `COGNO_NER_MODEL` | `tests/integration/backends.py` | the same for the NER suite, read first |
| `COGNO_TEST_EMBED_MODEL` | `tests/integration/backends.py` | the embedder of the integration suites (default: Ollama `nomic-embed-text:latest`) |
| `COGNO_OLLAMA_TIMEOUT` | `cogno-synapse` | the per-call timeout of the Ollama client; the nightly CI job sets it, because its runner has no GPU |
| `COGNO_BACKEND` | `examples/quickstart_60s.py` | the backend spec of the quickstart (default `mistral:latest`) |
| `OLLAMA_BASE_URL` / `COGNO_OLLAMA_URL` / `OLLAMA_HOST` | `tests/unit/_ollama_gate.py` | read only to know which address the unit suite must REFUSE to connect to |

## The Cogno ecosystem

`cogno-anima` is one organ of **[Cogno](https://github.com/sudoers-ai)** — a family of
small, composable, Apache-2.0 libraries that together form a complete
conversational-agent platform. Each library owns a single concern and stays
infra-agnostic; a **host** assembles them into a running agent:

![The Cogno ecosystem](docs/assets/cogno-ecosystem.svg)

The open-source libraries are the organs; the **host is the body** that joins
them. Our reference host — `cogno-host`, with its `cogno-ui` dashboard — is the
private product layer, but it holds no special powers: everything it does rides
on the public seams documented in each library's `docs/HOST_INTEGRATION.md`, so
you can assemble a body of your own.

## Testing

```bash
python3 -m pytest                  # the unit suite ONLY (testpaths = tests/unit)
python3 -m pytest tests/unit       # the same, named: fast, no network (stubs)
python3 -m pytest tests/integration  # real Ollama; auto-skips if unavailable
```

A bare `pytest` never collects `tests/integration`: integration is opt-in by path because
it uses the local GPU, which may be serving real traffic.

The unit suite cannot reach a local Ollama: `tests/unit/conftest.py` installs `tests/unit/_ollama_gate.py`, which refuses any connection to Ollama's port or to the address in `OLLAMA_BASE_URL`/`COGNO_OLLAMA_URL`/`OLLAMA_HOST`, and fails the test even when the code under test swallowed the refusal (#202).

Unit tests run on a coverage gate (`--cov-fail-under=85`) and use the
`StubBackend`/`StubEmbedder` doubles in `tests/conftest.py`. Integration tests
use real models at `temperature=0.0` — reproducible against the default local
Ollama (measured), but `temperature=0` only *requests* greedy decoding, so a
suite pointed at a hosted provider via `COGNO_TEST_MODEL` is sampling, not
replaying.

## Inspiration

Cogno models a conversational agent on a coarse map of **human perception and the
psyche** rather than on a flat prompt→response call. The stage names are the map:

- **NOUMENO** — after Kant's *noumenon*: the raw thing-in-itself is normalized into
  a *perceived* phenomenon (canonical English) before the mind reasons over it.
- **ID · EGO · SUPEREGO** — Freud's structural model of the psyche: the **ID** as the
  strategic drive/router, the **EGO** as the executor mediating with the outside world
  (tools), the **SUPEREGO** as the internalized judge and voice of limits.
- **Aristotelian** intent tags — the categories/causal analysis borrowed from Aristotle.
- **Drift** — an epistemological check on how far perception has strayed from the input.

It is an engineering metaphor, not a claim of psychological fidelity — but the analogy
is load-bearing: it is *why* perception, drive, execution and conscience are separate,
deterministically-gated stages instead of one prompt.

## Credits

Created by **Vinicius Vale** — [Sudoers](https://github.com/sudoers-ai).
Contributions welcome under the Apache-2.0 license below.

## License

Licensed under the [Apache License 2.0](LICENSE).
