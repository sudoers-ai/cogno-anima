# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`cogno-anima` is a modular, infrastructure-agnostic library implementing **Stage 1 (NOUMENO / Perception Layer)** of the Cogno cognitive intelligence pipeline. It normalizes raw user input, detects language, rewrites it into canonical English, measures semantic drift, and extracts structured intent/NER metadata — all decoupled from any proprietary infra (DB, message bus, etc.).

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

# Cognitive benchmark (CognoBench) — scores NOUMENO/NER/ID/Drift quality
python3 cognobench.py                    # full run vs local Ollama
python3 cognobench.py --only ner --limit 3   # one dimension, few cases
python3 cognobench.py --only id              # multi-turn goal continuity + routing
python3 cognobench.py --stub --limit 3       # fast plumbing smoke (no model)
python3 cognobench.py --calibrate --only drift id   # record drift/goal_status actuals
python3 cognobench.py --only ego superego --model qwen3:8b --think  # reasoning on/off compare
```

Integration tests use real models via Ollama and are written to be deterministic (`temperature=0.0`). Default NOUMENO/NER model is `mistral:latest` (top scorer on the ID bench; `qwen3:8b` is the recommended alternative — override NER via `COGNO_NER_MODEL`); embeddings use `nomic-embed-text:latest`.

The **CognoBench** cognitive benchmark lives in `cognobench/` — see `cognobench/CLAUDE.md` for its design (loaded when working there).

## Architecture

### Pipeline flow

The pipeline operates on a single mutable carrier object, `PipelineContext` (`cogno_anima/types.py`), which flows through stages sequentially:

1. **NOUMENO** (`cogno_anima/stages/noumeno.py`, class `Noumeno`) — perception/normalization:
   - Expands slang via `expand_slangs` (utils)
   - Resolves language by precedence: per-request `ctx.force_language` (the tenant/session language) → stage `default_language` (host/global config, e.g. the SaaS sets `Noumeno(default_language="pt-BR")`) → `langdetect` fallback. The library ships no business default (`default_language=None`); `langue` in NER is inherited from this resolved `noumeno.language`
   - Checks subject continuity against `ctx.metadata["last_rewritten"]` using an `Embedder` (cosine similarity vs `subject_threshold`)
   - Calls the LLM to rewrite input into canonical English, returning JSON (`rewritten`, `context_turn`, `confidence`, `changed`, `preserved_terms`, `rewrite_warnings`)
   - Computes `drift_score` = `1 - cosine(embed(original), embed(rewritten))` and classifies it via `classify_drift()` into `PASS_THROUGH | REWRITTEN | COMPRESSED | EXPANDED | DRIFT` (reconciliation rule: drift > 0.50 forces `changed=True` and tag `DRIFT`)
   - Populates `ctx.noumeno: NoumenoResult`

2. **NER** (`cogno_anima/stages/ner.py`, class `IntentAnalyzer`) — semantic analysis:
   - Requires `ctx.noumeno` to be populated first
   - Builds prompt context from `ctx.noumeno` plus carry-over metadata (`last_goal`, `last_pii_risk`, `active_domains`, `turn_number`); if `noumeno.change_subject` is true, prior context is dropped
   - Calls the LLM and parses/sanitizes the JSON response into an `IntentResult` — every field is validated/coerced against a fixed vocabulary (`VALID_INTENTS`, `VALID_SENTIMENTS`, `VALID_TEMPORAL`, `VALID_TRIAD`, `VALID_MODALITY`, `VALID_SPEECH_ACTS`, `VALID_PAROLE`, `NER_KNOWLEDGE_DOMAINS`, etc.), with fallback heuristics when the LLM omits/garbles fields (e.g., `intent_class=UNKNOWN` falls back based on `mandatory_tags`; empty `domains` falls back from `mandatory_tags` via `_TAG_TO_DOMAIN`)
   - PII detection: raw `pii` strings are normalized/aliased (`security/pii.py: normalize_pii_types`) and risk is computed deterministically (`compute_pii_risk`) — never trust the LLM's own risk judgment
   - Populates `ctx.intent: IntentResult`

3. **ID** (`cogno_anima/stages/id.py`, class `IDStage`) — strategic router & continuity:
   - Requires `ctx.intent` (and `ctx.noumeno`) to be populated first. **Heuristic — no LLM**; signature is `process(ctx, embedder)` (the embedder is used only for goal similarity).
   - **Stateless across turns:** all cross-turn state rides in `ctx.metadata["id_state"]` (a serializable dict the *host* persists to its DB/Redis) — there is no live per-session instance, so it survives a multi-worker HTTP setup. `turn_number` comes from `ctx.metadata["turn_number"]` (host authoritative) or auto-increments from `id_state` when absent.
   - Composes three pure helpers in `cogno_anima/routing/`: `GoalManager` (goal continuity NEW→ONGOING→COMPLETED→ABANDONED via staged checks: CLARIFICATION → domain match → anaphoric-PII → `context_dependent` → semantic cosine with **one-sided enrichment**, Jaccard fallback; `update()` returns `(status, goal, similarity)`), `IntentionTracker` (BDI intentions, max 5, FIFO), `AttentionFilter` (additive scoring over host-injected `ctx.metadata["attention_candidates"]`).
   - Routing priority (`_resolve_route`): `pii_risk=CRITICAL`→SUPEREGO (`blocked=True`) → `HIGH`→SUPEREGO → `emotional_override`→SUPEREGO → `CREATIVE_TASK`→SUPEREGO → `ACTION_REQUEST`/`INFORMATION_REQUEST`→EGO → `SOCIAL`→SUPEREGO → trust `triad_signal` → `BALANCED`. The EGO is the **tool gateway**: both action *and* information requests route to it (it no-ops to a draft when no tool is needed) — widened from the old `ACTION+SYSTEM` rule, which starved tool-requiring info queries ("what's my balance?"); surfaced by the e2e conversation simulation.
   - Cross-turn signals: temporal stickiness (a follow-up under an ONGOING goal keeps the prior turn's higher temporal — recorded on `IdResult.temporal_class`, **does not mutate the `IntentResult`**); `emotional_override` from a `frustration_streak` counter (≥ threshold of consecutive `FRUSTRATED` turns; host may inject its own); `complexity` (advisory `LOW|MEDIUM|HIGH|EXPERT`; `complex_domains` configurable, core default none — the core never escalates models); `needs_confirmation` (`_needs_confirmation`) — flags an ACTION the *user* framed tentatively (`speech_act=INTERROGATIVE` or `modality∈{POSSIBLE,UNCERTAIN}`; `INFORMATION_REQUEST` never flags). **SIGNAL ONLY** — the ID never touches `ego_readonly`; the host decides (ask directly, or route to the EGO read-only — see EGO gate A). Two more **signal-only** doubt flags (host decides): `confidence_divergence` (`|noumeno.confidence − intent.confidence| ≥ threshold`, default 0.4 — the *disagreement* between stages is the robust signal, not the absolute LLM confidence the core otherwise distrusts) and `clarification_suggested` (the NOUMENO rewriter raised `rewrite_warnings`).
   - Feeds drift: seeds `ctx.drift` if absent, then `compute_situational(goal_similarity)` → `compute_cumulative()` → `downgrade_for_intentional_shift(goal_status)`. Records embedding cost (from the goal-similarity calls, via a usage-aware closure) in `StageMetrics(model="heuristic")`; tokens are 0 on fast-paths/first turn.
   - Output (`triad_route`/`goal_status`/`complexity`) is sanitized against the closed vocab (`VALID_TRIAD`, `VALID_GOAL_STATUS`, `VALID_COMPLEXITY`). Populates `ctx.id_result: IdResult`.
   - **Out of scope (host/EGO):** persona↔MCP binding, skill selection/execution, model-ladder escalation, session splitting, and the dynamic `ask_user` text — the ID emits signals (`drift_action`, `blocked`, `complexity`), the host decides.

4. **EGO** (`cogno_anima/stages/ego.py`, class `EgoStage`) — executor & tool dispatch:
   - Requires `ctx.noumeno` + `ctx.intent`. Signature is `process(ctx, backend, dispatcher, *, system_prompt)` — `backend: LLMBackend`, `dispatcher: ToolDispatcher` (host-injected), `system_prompt` is the persona's **execution** prompt (host).
   - **EGO = executor, SUPEREGO = locutor.** The EGO runs an agent loop (decide tool → `dispatcher.execute` → feed result back → repeat) and gathers data; it does **NOT** write the user reply. `EgoResult` is a trace (`steps` = source of truth; `tools_executed`/`draft`/`has_side_effects` derived) + a `draft` (the model's last text) for the SUPEREGO to voice. There is no `response`/`response_source` and no `returns_raw_json` (presentation left the EGO).
   - **NER signal enrichment in `_task_context`** (prompt hints): user `intent.constraints` / `intent.negation` are surfaced as `Constraints (must respect)` / `Must NOT` so the loop honors explicit restrictions.
   - **Two confirmation gates (capability, not prompt-hints — `docs/ACT_CONFIRM_READONLY.md`).** Tool classification is host-declared via the optional `ToolPolicyDispatcher` Protocol (`is_mutating`/`requires_confirmation`, probed with `isinstance` like `ToolCallingBackend`); the core never hardcodes which tools mutate. **(A) Read-only mask** — when the host sets `ctx.metadata["ego_readonly"]` (acting on the ID's `needs_confirmation` signal because the *user* was tentative), the EGO offers ONLY non-mutating tools and `force_first` is off → it consults + proposes, never commits. **Fail-safe:** no policy → mask ALL tools (propose via draft). **(B) Confirmation gate** — a `requires_confirmation` (destructive) tool the model picks is **never executed** without `ctx.metadata["ego_confirmed"]`; the core holds it, records `EgoResult.pending_confirmation`, and stops to propose (the host runs its confirm UX → next turn with `ego_confirmed` → executes). **Opt-in** (no policy → no gate). Symmetric: A is broad (user unsure → mask all writes), B is surgical (dangerous tool → hold that call); both are propose/commit, both keep "core signals, host decides".
   - **Dual-path FC:** a backend satisfying `ToolCallingBackend` (separate optional `Protocol` in `llm/base.py`, NOT folded into `LLMBackend`) uses native function calling; a plain `LLMBackend` (a stub, the distilled student, the default `OllamaBackend`) uses the **text-fallback path** — the model emits `<TOOL_CALL>` tags read by `parse_tool_calls_from_text` (`llm/tool_parsing.py`, ported from the parent; also rescues FC leaks). On fallback the tool list + `<TOOL_CALL>` mechanics are rendered into the prompt; on native they travel via the API.
   - **Execution is delegated** ("EGO = brain, dispatcher = hands"): the core never touches the DB/MCP/API, so atomicity/rollback/outbox are **host** concerns. `ToolResult` (`output, ok, error, side_effect`) carries no `compensating_tool` (compensation is host-internal). Error contract: recoverable failure (`ToolResult(ok=False)`) is fed back so the model self-corrects; fatal (`MCPDispatchError` raised by the dispatcher) propagates; a stray exception is wrapped in `ToolExecutionError` and propagated (the EGO never guesses recoverability).
   - **Loop policy:** `max_steps = ctx.metadata.get("ego_max_steps", default)` where `default` is 5, or 8 when `intent.is_composite` (a multi-task request gets more budget; the host's explicit `ego_max_steps` always wins); `intent.is_sequential` adds an **ordering instruction** + the `causal_chain` as a numbered supporting plan in the task context (the loop still decides real tool order). Stop when the model emits no tool_calls (→ `draft`); `tool_choice="required"` on the 1st iteration for `ACTION_REQUEST`; hallucinated tool name → recoverable feed-back; **duplicate-call detection** (ported from parent: same `(tool,args)` blocked after 2 repeats, abort after 2 all-blocked steps). Budget/convergence bounds are **signals** (`interrupted=True` + partial result), not exceptions. Correction loop (host-orchestrated EGO↔SUPEREGO): `ctx.metadata["ego_correction"]={reason,attempt}` drives an `[ACTIONS ALREADY EXECUTED]` block built from the prior `ctx.ego_result`; the core renders whatever trace the host hands back (host rollback → empty trace → fresh retry).
   - Tokens summed across the loop into one `StageMetrics(model=backend.model)` (embeddings 0); folded into `PipelineContext` totals via `ctx.ego_result` + `ego_metrics` (like `id_metrics`). **Out of scope (host):** persona/MCP/skill-selection/RBAC, KG/history retrieval, mobile brevity + voicing (SUPEREGO), model-ladder, atomicity. A standalone embedding `ToolRetriever` for large catalogs is left to the host (the EGO trusts `dispatcher.tools_schema()` as final). When a persona draws tools from several sources at once (in-process skills via `cogno-cortex`, an MCP client, host-native functions), the host merges them with `cogno_anima.tools.CompositeDispatcher` — a `ToolDispatcher` (+`ToolPolicyDispatcher`) whose `tools_schema()` is the union (first source wins on a name collision) and whose `execute()`/policy methods route to the owning source (a non-policy source is treated conservatively: assumed mutating, no confirmation gate). The EGO still sees one flat tool set; the *contract* is the unifier — skill/MCP/native are just sources behind it.

5. **SUPEREGO** (`cogno_anima/stages/superego.py`, class `SuperegoStage`) — guardrails, judge & voicer:
   - **EGO=executor, SUPEREGO=locutor:** the SUPEREGO **writes** the final user response from the EGO's gathered data (it does NOT review a pre-written one — that is the key divergence from the parent). Three LLM ops, **A2** (host injects a backend per op, may differ — e.g. big judge + small voicer, or the same model):
     - `check_input_scope(ctx, backend, *, scope_prompt) -> ScopeCheckResult` — pre-EGO cheap ALLOW/BLOCK relevance guard; NER-assisted bypass for SOCIAL/CLARIFICATION (no LLM call); **fail-OPEN** (a cost guard must never refuse a legit user on error); the `refusal_message` is generated in the same call.
     - `evaluate(ctx, backend, *, limits_prompt) -> SuperegoResult` — the **JUDGE**: approve the EGO execution or return a `critique`. **Criterion #1 is goal↔execution** ("asked X, did X not Y"), then **#2 user constraints** (honored every `intent.constraints`, did NOT do what `intent.negation` forbade — rendered into the prompt via `_format_restrictions`), then completeness, grounding (incl. NOUMENO `preserved_terms` reproduced exactly — `_format_preserved`), safety/limits. **Fail-CLOSED** (never approve unverified — a false-pass is worse than a retry). The `critique` feeds the EGO's `ctx.metadata["ego_correction"].reason` (the correction loop closes).
     - `voice(ctx, backend, *, voice_prompt) -> SuperegoResult` — **writes** `response` in the persona's voice **and limits** (limits go to the voice too, not only the judge), grounded in the tool data (exact figures verbatim); `strip_cot`; runs a **deterministic PII backstop on the OUTPUT** (flags `pii:flagged_in_output`, does NOT auto-redact — avoids over-redaction) **and a preserved-term backstop** (flags `preserved:mutated_in_output` when a critical NOUMENO-preserved term — figure/email/URL — that the executor grounded shows up *altered* in the reply; flag-only, mutation-of-present not mere absence, so it never forces a nonsensical echo); feeds **lexical** `compute_synthesis` drift + recomputes cumulative. Also surfaces a **register-accommodation** signal (`# Register`, derived from NER `parole`): match the user's formality *where it does not conflict with the persona* — **the persona voice/limits always win**. Raises on backend failure (errors propagate).
   - Deterministic utils (pure): `strip_cot` (`<think>`/`<thinking>`), `detect_adjustments` (tone hints from sentiment/intent/emotional_override/pii_risk **+ a `register:*` hint from `parole` via `_parole_to_register`** — a formality axis distinct from sentiment's *emotional* tone; `MIXED`/`None` → no hint). `_blocked_response` (PII-CRITICAL → host `block_message` + core fallback). **No Embedder** (synthesis drift is lexical word-overlap, not embedding).
   - **Tokens** like the other stages, per-call with distinct labels (`superego_scope`/`superego_judge`/`superego_voice`): the final `voice` → `ctx.superego_result` → `superego_metrics` (folded into totals); the host accumulates `scope` + each `judge` attempt + EGO retries into `ctx.retry_metrics`.
   - **Handoff/early-exit:** closed `vocab.VALID_STOP_REASONS` (`completed|human_handoff|semantic_cache|scope_blocked|pii_blocked`) + `ctx.needs_handoff` signal — the core **signals**, the host **escalates** (retry exhaustion / confidence floor) and does the actual handoff. **Out of scope (host):** persona scope/limits/voice prompt text, the retry LOOP + `max_corrections`, billing, the real human handoff, semantic cache, session split.

6. **Drift** (`cogno_anima/stages/drift.py`, class `DriftCalculator`) — pure, no I/O:
   - `compute()` seeds `DriftMetrics` from `noumeno.drift_score` (epistemological drift) plus word-count/compression stats
   - `compute_ontological()`, `compute_situational()`, `compute_execution()`, `compute_synthesis()` fill in drift for the other pipeline stages (NER/ID/EGO/SUPEREGO) — callers invoke these incrementally as those stages complete (the ID stage calls `compute_situational` itself)
   - `compute_cumulative()` applies weights (`DEFAULT_CUMULATIVE_WEIGHTS`, injectable via `DriftCalculator(weights=..., thresholds=...)`) across the 5 stages and sets `drift_action` (`none|warn|ask_user|self_correct`) based on `DriftThresholds` (default `0.50/0.70/0.85`). Weights are **relative** — cumulative is renormalized over the stages actually computed, so the "sum to 1.0" invariant is relaxed (5 keys, non-negative, positive sum); a host can plug a risk profile per-instance (multi-tenant safe).
   - `downgrade_for_intentional_shift(drift, goal_status)` softens `ask_user`→`warn` on a deliberate topic change (NEW/ABANDONED); `compute_cumulative` stays goal-agnostic and the ID invokes this explicitly.
   - `DriftMetrics.to_tags()` turns scores into diagnostic tags (`NOUMENO.DRIFT`, `NOUMENO.PASS_THROUGH`, `DRIFT.ASK_USER`, etc.)

### LLM/Embedder abstraction — extracted to **`cogno-synapse`**

The model-transport layer (formerly `cogno_anima/llm/`) was **extracted into the sibling `cogno-synapse` lib** (`cogno_synapse` package; repo `/home/vinicius/git/cogno-synapse`). `cogno-anima` now declares `cogno-synapse` as a runtime dependency and imports the protocols/backends from it (`from cogno_synapse import LLMBackend, OllamaBackend, …`). For backward compatibility the top-level `cogno_anima` package **re-exports** the same names (`from cogno_anima import OllamaBackend` still works), and `cogno_anima.errors` re-exports `InvalidAPIKeyError`/`MissingAPIKeyError` (now `cogno_synapse.SynapseError` subclasses, no longer `CognoError`). Resilience (`FallbackBackend`) runs over the `cogno-homeo` kernel — see below.

`cogno_synapse/base.py` defines three `Protocol`s (runtime-checkable, structurally typed — no inheritance required):
- `LLMBackend`: `async generate(system, prompt) -> (text, tokens_in, tokens_out)`, plus a `model` attribute
- `ToolCallingBackend` (extends `LLMBackend`): adds `async chat_with_tools(messages, tools, tool_choice) -> (message_dict, tin, tout)` + `supports_native_tools()` for native function calling. **Optional and separate** so a text-only backend (a stub, the distilled student) satisfies only `LLMBackend` and the EGO auto-uses the text-fallback path (`isinstance(backend, ToolCallingBackend)`); NOUMENO/NER/ID never touch it.
- `Embedder`: `async embed(text) -> list[float]`, `async similarity(a, b) -> float`

`cogno_synapse/ollama.py` provides concrete implementations (`OllamaBackend`, `OllamaEmbedder`), talking to a local Ollama server over `httpx`. `OllamaBackend` sends `think=false` by default: reasoning models (qwen3, deepseek, …) otherwise route their output to a separate `thinking` field and leave `response` empty, which would make the JSON stages raise `StageParseError`; the cognitive stages want direct JSON, not chain-of-thought (`generate()` also falls back to `thinking` if `response` is empty). Set `OllamaBackend(..., think=True)` to opt back in. `OllamaEmbedder` is a thin, stateless client; it also exposes `embed_with_usage`/`similarity_with_usage` returning `(vector, tokens)` (from Ollama's `prompt_eval_count`). Caching is **backend-agnostic**: `cogno_synapse/cache.py: CachingEmbedder` wraps *any* `Embedder` to add a bounded LRU cache (by lowercased text) plus token/call accounting (`EmbeddingUsage`) — e.g. `CachingEmbedder(OllamaEmbedder(...))`. New backends implement the same protocol shape and get caching for free by composition — stages depend only on the protocol, not on Ollama.

**Cloud backends** (`cogno_synapse/{openai,anthropic,groq,gemini,bedrock}_backend.py`): adapted from the parent — each implements `LLMBackend` + `ToolCallingBackend` (`generate` + `chat_with_tools` converting the unified OpenAI-format messages/tools to the provider's shape + `supports_native_tools`). **SDKs are lazy-imported** (optional extras: `pip install "cogno-synapse[openai|anthropic|groq|gemini|bedrock|llm]"`; the convenience extra `cogno-anima[llm]` pulls `cogno-synapse[llm]`). Two deliberate divergences from the parent: (1) **errors propagate** — they **raise** on transport/auth failure (`InvalidAPIKeyError` for 401/403) instead of returning `("",0,0)`, matching the core contract; (2) **no tenant-key contextvar** (host owns key rotation). `cogno_synapse/fallback.py: FallbackBackend` is a slim, infra-agnostic failover chain (try each, first success wins, last error propagates; skips non-FC backends for `chat_with_tools`) whose loop now runs over `cogno_homeo.resilient_call`, so a host can opt into a circuit breaker / retry / metrics by passing them to the constructor — with none supplied it behaves exactly as before (the parent's Redis circuit-breaker + probe threads remain host concerns and were NOT ported). `cogno_synapse/factory.py: create_backend("provider:model")` instantiates a single backend (raises `MissingAPIKeyError` for a cloud provider without a key); the business `_FALLBACK_MATRIX` (model ladders) is host, not core. **OpenAI-compatible providers** (DeepSeek, Moonshot/Kimi, xAI/Grok, OpenRouter, Together, Fireworks) reuse `OpenAIBackend` via its `base_url` param instead of a class each — the factory's `_OPENAI_COMPATIBLE` registry maps each prefix to `(base_url, key_env)`, e.g. `create_backend("deepseek:deepseek-chat")`. There is deliberately **no `mistral:` prefix** (it would clobber Ollama's `mistral:latest`, the default local model). Unit tests mock the SDK clients (no network); integration tests are gated on real keys (auto-skip). All of these tests live in the `cogno-synapse` repo now.

### Prompts

Prompt templates live under `cogno_anima/prompt_templates/<stage>/` (`noumeno/`, `ner/`) as plain text files, loaded via `cogno_anima.prompts.load_prompt(stage, prompt_name, prompts_dir=...)`. The loader strips YAML frontmatter (`---\n...\n---\n`) and any `TODO(docs)` lines. `IntentAnalyzer` can load an alternate system prompt via `system_prompt_name` (default `system.txt`); the default NER prompt is concise (~3.4k tokens) so it fits Ollama's default `num_ctx=8192` with room for input/output.

The `domains` closed list inside `cogno_anima/prompt_templates/ner/system.txt` is the source of truth and **must stay byte-for-byte aligned with `NER_KNOWLEDGE_DOMAINS` in `cogno_anima/stages/ner.py`** — `tests/unit/test_pipeline.py::test_code_domains_match_prompt_domains_exactly` enforces this. `langue` is no longer detected by the LLM; the NER inherits it from `noumeno.language`.

### Models (`cogno_anima/types.py`)

All cross-stage data is `pydantic.BaseModel`. Key types: `StageMetrics` (per-call telemetry; carries LLM `tokens_in`/`tokens_out` **and** `embedding_tokens`/`embedding_calls`; `tokens_total` auto-computed in `model_post_init` as `tokens_in + tokens_out + embedding_tokens`), `NoumenoResult`, `IntentResult` (with helper methods `aristo_tag`/`aristo_desc`/`aristo_parsed` for parsing `"TAG | description"`-style aristotelian fields), `IdResult` (ID routing/continuity output: `triad_route`, `active_goal`, `goal_status`, `goal_similarity`, `active_intentions`, `attention_focus`, `blocked`/`block_reason`, `turn_number`, `temporal_class`, `emotional_override`, `complexity`, `metrics`), `DriftMetrics`, and `PipelineContext` (the carrier — `noumeno`/`intent`/`id_result`/`drift` — with derived properties `total_tokens` (incl. embeddings), `total_llm_tokens`, `total_embedding_tokens`, `total_elapsed_ms`, `stage_metrics`, and per-stage `noumeno_metrics`/`ner_metrics`/`id_metrics`). NOUMENO records embedding cost from its similarity calls; NER records its LLM generate tokens; ID records embedding cost from goal similarity (no LLM tokens).

### Testing conventions

- `tests/conftest.py` provides `StubBackend` and `StubEmbedder` fixtures (`stub_backend`, `stub_embedder`) — zero-network test doubles for unit tests.
- `tests/unit/` — pure unit tests using stubs, no network.
- `tests/integration/` — real Ollama-backed tests; check `is_ollama_available()` and skip if unreachable. Always use `temperature=0.0` for determinism.
- Async tests use `@pytest.mark.asyncio`.

### Language conventions

All docstrings, comments, identifiers, prompts, and documentation are in **English** — the project's canonical language for its open-source audience. The one deliberate exception is **Portuguese domain data** that must stay Portuguese to function: e.g. the PT/ES pronoun map and slang dictionary, the birth-context regex in `security/pii.py`, and Portuguese user-input fixtures in `tests/`/`cognobench/` (they are the input under test for the PT→English rewrite — translating them would defeat the test). English comments may quote a Portuguese example string to illustrate such input. (Historically stage-logic comments were written in Portuguese; that convention was retired — do not add new Portuguese prose.)
