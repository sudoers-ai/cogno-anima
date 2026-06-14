# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`cogno-core` is a modular, infrastructure-agnostic library implementing **Stage 1 (NOUMENO / Perception Layer)** of the Cogno cognitive intelligence pipeline. It normalizes raw user input, detects language, rewrites it into canonical English, measures semantic drift, and extracts structured intent/NER metadata — all decoupled from any proprietary infra (DB, message bus, etc.).

## Common commands

```bash
# Install with dev dependencies (editable)
pip install -e ".[dev]"

# Run the full test suite
python3 -m pytest

# Run only unit tests (fast, no network/LLM required)
python3 -m pytest tests/unit

# Run a single test file / test
python3 -m pytest tests/unit/test_noumeno.py
python3 -m pytest tests/unit/test_noumeno.py::TestNoumenoStage::test_metrics_populated

# Run integration tests (require a local Ollama instance at localhost:11434;
# auto-skip if Ollama is unavailable)
python3 -m pytest tests/integration

# Cognitive benchmark (CognoBench) — scores NOUMENO/NER/Drift quality
python3 cognobench.py                    # full run vs local Ollama
python3 cognobench.py --only ner --limit 3   # one dimension, few cases
python3 cognobench.py --stub --limit 3       # fast plumbing smoke (no model)
python3 cognobench.py --calibrate --only drift   # record drift actuals
```

Integration tests use real models (`llama3.1:8b`, `nomic-embed-text:latest`) via Ollama and are written to be deterministic (`temperature=0.0`).

### CognoBench (`cognobench/`)

A self-contained, dependency-light cognitive benchmark for the implemented stages (NOUMENO → NER → Drift), kept **decoupled** from the library: the harness (`cognobench/harness.py`) drives the stages directly via dependency injection (any `LLMBackend` + `Embedder`), with no `PipelineRunner`/infra. Curated cases live in `cognobench/{ner,drift,noumeno}_cases.py` (ported from the parent Cogno SaaS bench). It is **not** shipped in the library wheel (`packages.find` includes only `cogno_core*`). A stub-mode smoke test (`tests/unit/test_cognobench_smoke.py`) guards the plumbing in CI without needing Ollama. Drift numeric bands are **soft/recalibratable** — the parent's bands were calibrated against a heuristic drift model, but cogno-core's drift is embedding-based and pure, so only the hard invariants (valid action, cumulative ∈ [0,1]) are enforced; use `--calibrate` to record actuals. ID/EGO/SUPEREGO dimensions will be added as those stages land.

## Architecture

### Pipeline flow

The pipeline operates on a single mutable carrier object, `PipelineContext` (`cogno_core/types.py`), which flows through stages sequentially:

1. **NOUMENO** (`cogno_core/stages/noumeno.py`, class `Noumeno`) — perception/normalization:
   - Expands slang via `expand_slangs` (utils)
   - Resolves language by precedence: per-request `ctx.force_language` (the tenant/session language) → stage `default_language` (host/global config, e.g. the SaaS sets `Noumeno(default_language="pt-BR")`) → `langdetect` fallback. The library ships no business default (`default_language=None`); `langue` in NER is inherited from this resolved `noumeno.language`
   - Checks subject continuity against `ctx.metadata["last_rewritten"]` using an `Embedder` (cosine similarity vs `subject_threshold`)
   - Calls the LLM to rewrite input into canonical English, returning JSON (`rewritten`, `context_turn`, `confidence`, `changed`, `preserved_terms`, `rewrite_warnings`)
   - Computes `drift_score` = `1 - cosine(embed(original), embed(rewritten))` and classifies it via `classify_drift()` into `PASS_THROUGH | REWRITTEN | COMPRESSED | EXPANDED | DRIFT` (reconciliation rule: drift > 0.50 forces `changed=True` and tag `DRIFT`)
   - Populates `ctx.noumeno: NoumenoResult`

2. **NER** (`cogno_core/stages/ner.py`, class `IntentAnalyzer`) — semantic analysis:
   - Requires `ctx.noumeno` to be populated first
   - Builds prompt context from `ctx.noumeno` plus carry-over metadata (`last_goal`, `last_pii_risk`, `active_domains`, `turn_number`); if `noumeno.change_subject` is true, prior context is dropped
   - Calls the LLM and parses/sanitizes the JSON response into an `IntentResult` — every field is validated/coerced against a fixed vocabulary (`VALID_INTENTS`, `VALID_SENTIMENTS`, `VALID_TEMPORAL`, `VALID_TRIAD`, `VALID_MODALITY`, `VALID_SPEECH_ACTS`, `VALID_PAROLE`, `NER_KNOWLEDGE_DOMAINS`, etc.), with fallback heuristics when the LLM omits/garbles fields (e.g., `intent_class=UNKNOWN` falls back based on `mandatory_tags`; empty `domains` falls back from `mandatory_tags` via `_TAG_TO_DOMAIN`)
   - PII detection: raw `pii` strings are normalized/aliased (`security/pii.py: normalize_pii_types`) and risk is computed deterministically (`compute_pii_risk`) — never trust the LLM's own risk judgment
   - Populates `ctx.intent: IntentResult`

3. **Drift** (`cogno_core/stages/drift.py`, class `DriftCalculator`) — pure, no I/O:
   - `compute()` seeds `DriftMetrics` from `noumeno.drift_score` (epistemological drift) plus word-count/compression stats
   - `compute_ontological()`, `compute_situational()`, `compute_execution()`, `compute_synthesis()` fill in drift for later pipeline stages (NER/ID/EGO/SUPEREGO) that live outside this library — callers invoke these incrementally as those stages complete
   - `compute_cumulative()` applies fixed weights (`_CUMULATIVE_WEIGHTS`, must sum to 1.0) across all 5 stages and sets `drift_action` (`none|warn|ask_user|self_correct`) based on thresholds (`0.50/0.70/0.85`)
   - `DriftMetrics.to_tags()` turns scores into diagnostic tags (`NOUMENO.DRIFT`, `NOUMENO.PASS_THROUGH`, `DRIFT.ASK_USER`, etc.)

### LLM/Embedder abstraction

`cogno_core/llm/base.py` defines two `Protocol`s (runtime-checkable, structurally typed — no inheritance required):
- `LLMBackend`: `async generate(system, prompt) -> (text, tokens_in, tokens_out)`, plus a `model` attribute
- `Embedder`: `async embed(text) -> list[float]`, `async similarity(a, b) -> float`

`cogno_core/llm/ollama.py` provides concrete implementations (`OllamaBackend`, `OllamaEmbedder`), talking to a local Ollama server over `httpx`. `OllamaEmbedder` is a thin, stateless client; it also exposes `embed_with_usage`/`similarity_with_usage` returning `(vector, tokens)` (from Ollama's `prompt_eval_count`). Caching is **backend-agnostic**: `cogno_core/llm/cache.py: CachingEmbedder` wraps *any* `Embedder` to add a bounded LRU cache (by lowercased text) plus token/call accounting (`EmbeddingUsage`) — e.g. `CachingEmbedder(OllamaEmbedder(...))`. New backends (OpenAI, Bedrock, etc.) implement the same protocol shape and get caching for free by composition — stages depend only on the protocol, not on Ollama.

### Prompts

Prompt templates live under `prompts/<stage>/` (`noumeno/`, `ner/`) as plain text files, loaded via `cogno_core.prompts.load_prompt(stage, prompt_name, prompts_dir=...)`. The loader strips YAML frontmatter (`---\n...\n---\n`) and any `TODO(docs)` lines. `IntentAnalyzer` can load an alternate system prompt via `system_prompt_name` (default `system.txt`); the default NER prompt is concise (~3.4k tokens) so it fits Ollama's default `num_ctx=8192` with room for input/output.

The `domains` closed list inside `prompts/ner/system.txt` is the source of truth and **must stay byte-for-byte aligned with `NER_KNOWLEDGE_DOMAINS` in `cogno_core/stages/ner.py`** — `tests/unit/test_pipeline.py::test_code_domains_match_prompt_domains_exactly` enforces this. `langue` is no longer detected by the LLM; the NER inherits it from `noumeno.language`.

### Models (`cogno_core/types.py`)

All cross-stage data is `pydantic.BaseModel`. Key types: `StageMetrics` (per-call telemetry; carries LLM `tokens_in`/`tokens_out` **and** `embedding_tokens`/`embedding_calls`; `tokens_total` auto-computed in `model_post_init` as `tokens_in + tokens_out + embedding_tokens`), `NoumenoResult`, `IntentResult` (with helper methods `aristo_tag`/`aristo_desc`/`aristo_parsed` for parsing `"TAG | description"`-style aristotelian fields), `DriftMetrics`, and `PipelineContext` (the carrier, with derived properties `total_tokens` (incl. embeddings), `total_llm_tokens`, `total_embedding_tokens`, `total_elapsed_ms`, `stage_metrics`). NOUMENO records embedding cost from its similarity calls; NER records its LLM generate tokens.

### Testing conventions

- `tests/conftest.py` provides `StubBackend` and `StubEmbedder` fixtures (`stub_backend`, `stub_embedder`) — zero-network test doubles for unit tests.
- `tests/unit/` — pure unit tests using stubs, no network.
- `tests/integration/` — real Ollama-backed tests; check `is_ollama_available()` and skip if unreachable. Always use `temperature=0.0` for determinism.
- Async tests use `@pytest.mark.asyncio`.

### Language conventions

Docstrings/comments in stage logic (`noumeno.py`, type field comments in `types.py`) are frequently written in Portuguese (the original domain language); code identifiers, prompts, and LLM I/O are in English. Match the existing language of the file/section you're editing.
