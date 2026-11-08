# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`cogno-anima` is a modular, infrastructure-agnostic library implementing **Stage 1 (NOUMENO / Perception Layer)** of the Cogno cognitive intelligence pipeline. It normalizes raw user input, detects language, rewrites it into canonical English, measures semantic drift, and extracts structured intent/NER metadata — all decoupled from any proprietary infra (DB, message bus, etc.).

## Common commands

```bash
# Install with dev dependencies (editable)
pip install -e ".[dev]"

# A bare pytest runs ONLY the unit suite (`testpaths = ["tests/unit"]`); integration is
# asked for by path, below. Why: it uses the local GPU, which serves real traffic.
python3 -m pytest

# The same unit suite, named explicitly (fast, no network/LLM required)
python3 -m pytest tests/unit

# Run a single test file / test
python3 -m pytest tests/unit/test_noumeno.py
python3 -m pytest tests/unit/test_noumeno.py::TestNoumenoStage::test_metrics_populated

# Run integration tests (default: a local Ollama at localhost:11434; auto-skip if
# it is unavailable)
python3 -m pytest tests/integration

# ...or point them somewhere else entirely — `provider:model`, two axes, and the
# skip then asks about THAT provider's key instead of the local server. Use this
# when the box's GPU is busy serving real traffic: the suite asks for the same
# qwen3:8b the deployment does, so a local run queues live turns behind it.
COGNO_TEST_MODEL=openai:gpt-4o-mini \
COGNO_TEST_EMBED_MODEL=openai:text-embedding-3-small \
        python3 -m pytest tests/integration

# Cognitive benchmark (CognoBench) — scores NOUMENO/NER/ID/Drift quality
python3 cognobench.py                    # full run vs local Ollama
python3 cognobench.py --only ner --limit 3   # one dimension, few cases
python3 cognobench.py --only id              # multi-turn goal continuity + routing
python3 cognobench.py --stub --limit 3       # fast plumbing smoke (no model)
python3 cognobench.py --calibrate --only drift id   # record drift/goal_status actuals
python3 cognobench.py --only ego superego --model qwen3:8b --think  # reasoning on/off compare
# 100% NUVEM — nada local, nem o embedder (a caixa pode estar com a GPU ocupada):
python3 cognobench.py --only id --model openai:gpt-4o-mini \
        --embed-model openai:text-embedding-3-small
```

Notes on these commands (their full story is in `docs/DEVELOPMENT.md`):

- `--embed-model` takes `provider:model`, the grammar of `--model`. The default stays the local
  Ollama embedder: it is the control variable when an A/B measures something else.
- **CognoBench's default `--model` is `qwen3:8b`.** Integration tests ask for `temperature=0.0`;
  a hosted provider is not obliged to honour it, so a cloud run is sampling.
- **The integration suites take a `provider:model` spec on two axes**, `COGNO_TEST_MODEL` (NER
  reads `COGNO_NER_MODEL` first) and `COGNO_TEST_EMBED_MODEL`, resolved in ONE place,
  `tests/integration/backends.py`. The skip travels with the spec: the local server for an
  Ollama spec, the provider's key for a cloud one.
- **The version is written once**, in `cogno_anima/_version.py`; `pyproject.toml` reads it.
  `tests/unit/test_version_has_one_source.py` compares the package, the build configuration and
  the changelog's first heading.
- CognoBench's design is `cognobench/CLAUDE.md` (loaded when working there).
- What this library does NOT own — the persona, the channel, the network between personas — is
  `docs/NETWORK_PERSONA_CHANNEL.md`. The ecosystem map is `docs/ARCHITECTURE.md`.

## Architecture

### Pipeline flow

The pipeline operates on a single mutable carrier object, `PipelineContext` (`cogno_anima/types.py`), which flows through stages sequentially:

1. **NOUMENO** (`cogno_anima/stages/noumeno.py`, `Noumeno`) — slang expansion, the language
   (`ctx.force_language` → the stage's `default_language` → `langdetect`), subject continuity,
   the LLM rewrite into canonical English (JSON), and the epistemological drift
   (`classify_drift`). An embedder failure never kills the turn: the drift it could not take is
   `UNKNOWN` (`None`), never 0.0 or 1.0. Populates `ctx.noumeno`. → `docs/stages/NOUMENO.md`
2. **NER** (`cogno_anima/stages/ner.py`, `IntentAnalyzer`) — the `IntentResult`, every field
   coerced against a closed vocabulary, with fallbacks; the PII risk is computed
   deterministically, never taken from the model. Populates `ctx.intent`. → `docs/stages/NER.md`
3. **ID** (`cogno_anima/stages/id.py`, `IDStage`) — heuristic, no LLM; stateless across turns
   (everything rides in `ctx.metadata["id_state"]`); goal continuity, the route, and signals the
   host decides on. Populates `ctx.id_result`. → `docs/stages/ID.md`
4. **EGO** (`cogno_anima/stages/ego.py`, `EgoStage`) — the executor: an agent loop over the
   host's `ToolDispatcher`, native function calling or the text fallback, the three confirmation
   gates (`docs/ACT_CONFIRM_READONLY.md`), and a record of its prompt that never holds its text.
   It does not write the reply. → `docs/stages/EGO.md`
5. **SUPEREGO** (`cogno_anima/stages/superego.py`, `SuperegoStage`) — the scope guard (fails
   OPEN, and says so), the judge (fails CLOSED) and the voice, which WRITES the reply from the
   executor's data and runs the deterministic backstops on it. → `docs/stages/SUPEREGO.md` (the
   stage, the verdict reader, the stop reasons), `docs/stages/SUPEREGO_SCOPE.md`,
   `docs/stages/SUPEREGO_JUDGE.md`, `docs/stages/SUPEREGO_VOICE.md`
6. **Drift** (`cogno_anima/stages/drift.py`, `DriftCalculator`) — pure; `None` means "not
   measured", never "no drift". → `docs/stages/DRIFT.md`

### Where each contract lives

**Before changing a stage or one of these areas, read its file, and update it in the same pull
request.** Each file holds what this one used to: the contract, and the measurement behind each
rule.

| Area | File |
| --- | --- |
| The six stages | `docs/stages/` (one file per stage; the SUPEREGO has four) |
| Tool dispatch adapters (`cogno_anima/tools/`): the router, the recorders, idempotency, id provenance, the pre-judge | `docs/TOOLS.md` |
| The confirmation gates and how long a hold lives | `docs/ACT_CONFIRM_READONLY.md` |
| The model transport (`cogno-synapse`) | `docs/TRANSPORT.md` |
| Untrusted text in prompts (`security/prompt_guard.py`) | `docs/PROMPT_GUARD.md` |
| Repairs of a model's JSON reply / the reply schemas | `docs/REPAIRS.md` / `docs/SCHEMAS.md` |
| What the library logs: the policy / the event list | `docs/LOGGING_POLICY.md` / `LOGGING.md` |
| Prompt order and the provider's prompt cache | `docs/PROMPTS.md` |
| The turn predicates (`committed_this_turn` and its family) and the models | `docs/TYPES.md` |
| Integration suites, CognoBench defaults, the version | `docs/DEVELOPMENT.md` |
| Wiring the core into a host; the whole ecosystem | `docs/HOST_INTEGRATION.md`; `docs/ARCHITECTURE.md` |

### Rules that hold for every change

- **A prompt does not move by a byte unless the change is deliberate.** The proof is digests
  taken on the base tree, with a control that renders the old bytes again; regenerate them only
  when the prompt changes on purpose. A change to what a model reads brings the model half: in
  the cloud, n=3, with the cost written before the run.
- **The judge fails CLOSED, the scope guard fails OPEN and is marked**, and only a JSON boolean is
  a verdict (`cogno_anima/verdict.py`).
- **A missing measurement is `None`**, never 0.0. **An absent host carrier makes a rule
  stricter**, never switches it off.
- **No log line carries text somebody else wrote** (`docs/LOGGING_POLICY.md`, `LOGGING.md` and
  its three tests).
- **Third-party text reaches a prompt only through the prompt guard**; the reserved headers are
  derived from the block tables, never copied (`docs/PROMPT_GUARD.md`).
- **One definition per rule.** A fact the stages and the host share lives in one function — for
  example `committed_this_turn` in `cogno_anima/types.py`, whose docstring alone enumerates its
  callers. Documents point at it; they never restate it.

### Prompts

Prompt templates live under `cogno_anima/prompt_templates/<stage>/` (`noumeno/`, `ner/`) as plain text files, loaded via `cogno_anima.prompts.load_prompt(stage, prompt_name, prompts_dir=...)`. The loader strips YAML frontmatter (`---\n...\n---\n`) and any `TODO(docs)` lines. `IntentAnalyzer` can load an alternate system prompt via `system_prompt_name` (default `system.txt`); the default NER prompt is concise (~3.4k tokens) so it fits Ollama's default `num_ctx=8192` with room for input/output.

The `domains` closed list inside `cogno_anima/prompt_templates/ner/system.txt` is the source of truth and **must stay byte-for-byte aligned with `NER_KNOWLEDGE_DOMAINS` in `cogno_anima/stages/ner.py`** — `tests/unit/test_pipeline.py::test_code_domains_match_prompt_domains_exactly` enforces this. `langue` is no longer detected by the LLM; the NER inherits it from `noumeno.language`.

### Testing conventions

- `tests/conftest.py` provides `StubBackend` and `StubEmbedder` fixtures (`stub_backend`, `stub_embedder`) — zero-network test doubles for unit tests.
- `tests/unit/` — pure unit tests using stubs, no network. **Enforced for Ollama, not promised** (2026-09-30): `tests/unit/conftest.py` installs `tests/unit/_ollama_gate.py` for the whole unit session, so any attempt to open a connection to Ollama (port 11434 on any host, or the host:port of `OLLAMA_BASE_URL`/`COGNO_OLLAMA_URL`/`OLLAMA_HOST`, read at connect time) raises `OllamaGateError`, and the test FAILS at teardown even when the code under test swallowed the refusal. The local Ollama is the GPU serving live traffic; the incident was a host unit test that left a backend `None`, got the real `OllamaBackend` default and called `localhost:11434`. It cuts at `socket.socket.connect` (every client ends there) plus httpcore's `connect_tcp` (so the httpx path raises the gate's error unwrapped rather than inside anyio's `ExceptionGroup`); `tests/unit/test_ollama_gate.py` pins the twins, the controls and the swallowed case.
- `tests/integration/` — real Ollama-backed tests; check `is_ollama_available()` and skip if unreachable. Always use `temperature=0.0` — it buys determinism against the local default (measured: zero unstable checks over three cognobench sweeps on qwen3:8b) but only REQUESTS greedy decoding from a hosted provider, which nothing obliges to deliver it; a suite pointed at one via `COGNO_TEST_MODEL` is sampling. See `BENCHMARKS.md` § *Reading a number from this bench*.
- Async tests use `@pytest.mark.asyncio`.

### Language conventions

All docstrings, comments, identifiers, prompts, and documentation are in **English** — the project's canonical language for its open-source audience. The one deliberate exception is **Portuguese domain data** that must stay Portuguese to function: e.g. the PT/ES pronoun map and slang dictionary, the birth-context regex in `security/pii.py`, and Portuguese user-input fixtures in `tests/`/`cognobench/` (they are the input under test for the PT→English rewrite — translating them would defeat the test). English comments may quote a Portuguese example string to illustrate such input. (Historically stage-logic comments were written in Portuguese; that convention was retired — do not add new Portuguese prose.)
