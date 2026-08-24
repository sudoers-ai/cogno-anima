# Host Integration Guide

How to wire `cogno-anima` into a real application (the "host" — e.g. the Cogno
SaaS runtime). The library ships the **cognition**; the host owns
**orchestration and infrastructure**. This guide is the human-facing companion to
`cognobench/pipeline.py: ReferencePipeline`, which is the *executable* reference
for everything below.

> TL;DR — the core is a set of stateless stages over a `PipelineContext`. The
> host decides the control flow, persists cross-turn state, provides backends +
> tools + prompts, and turns the core's **signals** into actions.

---

## 1. The boundary

| Concern | Owner |
| --- | --- |
| Stage logic (NOUMENO/NER/ID/EGO/SUPEREGO/Drift) | **core** |
| Vocabularies, PII detection, drift math, routing heuristics | **core** |
| Orchestration order, the EGO⇄SUPEREGO correction *loop* | **host** |
| Tool execution (DB/MCP/API) via `ToolDispatcher` | **host** |
| Persistence of cross-turn state, sessions/turns/memories | **host** |
| Transactions, rollback, outbox, atomicity | **host** |
| Persona prompts (scope/ego/limits/voice), persona↔MCP binding | **host** |
| Model selection / ladders / key rotation | **host** |
| Retry budget, human handoff, billing, semantic cache | **host** |

The core never imports a DB, a queue, or an SDK eagerly. It emits **signals**;
the host acts on them.

---

## 2. What the host must provide

1. **LLM backend(s)** implementing `LLMBackend` (and optionally
   `ToolCallingBackend` for native function calling). Use `OllamaBackend`, a
   cloud backend, or `create_backend("provider:model")`. You may use different
   backends per role (a JSON-capable model for NOUMENO/NER/scope/judge, a
   text/FC model for EGO, a small model for voice).
2. **An `Embedder`** (`OllamaEmbedder`, wrapped in `CachingEmbedder` for an LRU +
   token accounting). Used by NOUMENO (subject continuity) and ID (goal
   similarity).
3. **A `ToolDispatcher`** — your "hands". The EGO calls
   `dispatcher.tools_schema()` and `await dispatcher.execute(name, args)`; you do
   the actual DB/MCP/API work and return a `ToolResult`.
4. **Persona prompts** (plain strings you store/version): a scope prompt, an EGO
   execution prompt, a limits prompt, and a voice prompt.
5. **Persistence of `ctx.metadata`** between turns (see §5). This is a plain
   serializable dict — put it in Postgres/Redis next to your `sessions`/`turns`
   rows.

```python
from cogno_anima.tools import ToolDispatcher
from cogno_anima.types import ToolResult

class MyDispatcher:                       # satisfies ToolDispatcher (structural)
    def tools_schema(self) -> list[dict]:
        return [...]                      # OpenAI-format function schemas
    async def execute(self, name: str, arguments: dict) -> ToolResult:
        # do the real work inside YOUR transaction / MCP call
        return ToolResult(output="...", ok=True, side_effect=True)
```

---

## 3. The orchestration (mirrors `ReferencePipeline.run_turn`)

```
NOUMENO → NER → ID → [PII gate] → [scope guard] → EGO ⇄ SUPEREGO(judge) → SUPEREGO(voice)
```

```python
# perception + routing (no tools, no DB)
ctx = await noumeno.process(ctx, gen_backend)
ctx = await ner.process(ctx, gen_backend)
ctx = await id_stage.process(ctx, embedder)

# 1) safety gate — ID flags PII-CRITICAL
if ctx.id_result.blocked:
    ctx.superego_result = superego._blocked_response(ctx)
    ctx.stop_reason = "pii_blocked"
    return ctx

# 2) cheap scope guard (fail-OPEN: never refuse a legit user on error)
scope = await superego.check_input_scope(ctx, gen_backend, scope_prompt=scope_prompt)
if scope.blocked:
    ctx.stop_reason = "scope_blocked"
    return ...(scope.refusal_message)

# 3) EGO route (tool gateway): execute, then the correction loop
if ctx.id_result.triad_route == "EGO":
    attempt = 1
    while True:
        ctx = await ego.process(ctx, ego_backend, dispatcher, system_prompt=ego_prompt)
        judge = await superego.evaluate(ctx, gen_backend, limits_prompt=limits_prompt)  # fail-CLOSED
        if judge.approved or attempt >= max_corrections:
            break
        on_rollback(ctx)                                   # undo side effects (host)
        ctx.metadata["ego_correction"] = {"reason": judge.critique, "attempt": attempt + 1}
        attempt += 1
    if not judge.approved:                                 # exhausted → escalate
        ctx.needs_handoff = True
        ctx.stop_reason = "human_handoff"
        return ctx
    on_commit(ctx)                                          # persist side effects (host)

# 4) SUPEREGO writes the final reply (EGO data → persona voice), for EGO and chat paths
ctx.superego_result = await superego.voice(ctx, voice_backend, voice_prompt=voice_prompt)
return ctx
```

Key points:

- **EGO = executor, SUPEREGO = locutor.** The EGO gathers data and never writes
  the user reply; `superego.voice(...)` writes `ctx.superego_result.response`.
- **The correction loop is the host's.** The core gives you the judge's
  `critique`; you decide how many retries (`max_corrections`) and feed the
  critique back via `ctx.metadata["ego_correction"]`.
- Both `ACTION_REQUEST` and `INFORMATION_REQUEST` route to EGO (the tool
  gateway). Pure social/creative turns skip EGO and go straight to voice.

---

## 4. Atomicity (the part the core can't do)

Never hold a DB transaction open across an LLM call. The recommended shape:

1. The EGO dispatches tools through your `ToolDispatcher`. Buffer side effects
   (write-behind / outbox) rather than committing immediately.
2. The **judge runs before commit**. If it rejects and you retry, call your
   `on_rollback` to discard the buffered/uncommitted effects.
3. On approval, **commit** (`on_commit`), then voice the reply.

`ReferencePipeline` represents these as `on_rollback`/`on_commit` callbacks
(no-ops in the bench). A real host opens its tx / flushes its outbox there. If a
tool raises `MCPDispatchError`, it propagates (fatal); a recoverable
`ToolResult(ok=False)` is fed back so the model self-corrects.

---

## 5. Cross-turn state (`ctx.metadata`)

The core is **stateless across turns** — everything that must survive lives in
`ctx.metadata`, a serializable dict you persist and re-inject next turn. The keys
the core reads/writes:

| Key | Direction | Purpose |
| --- | --- | --- |
| `id_state` | core writes / host re-injects | goal lifecycle, intentions, frustration streak, turn counter |
| `turn_number` | host sets (authoritative) | the turn index (`turns.turn_n`); else ID auto-increments |
| `last_rewritten`, `last_context_turn` | host sets | previous turn's canonical text → NOUMENO subject continuity |
| `attention_candidates` | host injects | items the `AttentionFilter` scores |
| `ego_context` | host injects | retrieved memories/KG facts the voice should ground in |
| `emotional_override` | host may inject | force de-escalation (else ID derives it from a frustration streak) |
| `pii_session_hint` | host may inject | a known-PII session, tightens goal continuity |
| `ego_correction` | host sets in the loop | `{reason, attempt}` fed back to the EGO on retry |
| `ego_max_steps` | host may set | EGO agent-loop bound (default 5) |
| `circling_streak` | host may inject | how many CONSECUTIVE turns the host's anti-repeat guard fired (an int, counted by the host, **reset to 0 on a clean turn**) → from 2, the EGO tells the executor it keeps arriving at the same answer. A stale value keeps warning forever, so it must be recomputed every turn, like the core does for `id_state`'s frustration streak |
| `voice_traits` | host may inject | the persona's DECLARED personality traits (list, or comma string) from the closed `vocab.VALID_VOICE_TRAITS` (`warm`, `reserved`, `direct`, `formal`, `casual`, `humorous`, `concise`, `detailed`, `empathetic`; max 4; both sides of a contradicting pair are dropped — the rule is the pure `cogno_anima.sanitize_voice_traits(raw) → (kept, dropped)`, which a host should call at save time so the dashboard refuses what the voice would drop; a list, a comma string, a JSON array or a Postgres `text[]` literal all parse) → the SUPEREGO voice MODULATES them for the turn (`_modulate_traits`: the safety floor first, then this contact's own baseline) and renders them as a `# Persona traits` section and emits `trait:*` adjustments. Delivery only: they never touch figures, limits or refusals, and they outrank the contact's register hint. Read by the voice only — the EGO's tools stay neutral |
| `contact_state` | host may inject | the contact's emotional NEUTRAL — `{valence_ema: -1..1, arousal_ema: 0..1, n}`, an exponential moving average the host keeps per identity from the NER sentiment (`vocab.SENTIMENT_VALENCE`), LLM-free. The voice reads the turn as a DELTA against it: a negative turn far below the contact's own normal is an escalation (empathy, no detail); the same sentiment inside their normal is not. Ignored below `n`=5 (cold start = the persona as declared). **Stamp the value as it stood BEFORE this turn** — take the EMA step after it, or the message's own sentiment lands in the baseline it is compared against and every delta shrinks toward zero. Delivery only — never routing |

Minimal threading between turns:

```python
saved = ctx.metadata.get("id_state")        # persist this with your turn row
# next turn:
ctx = PipelineContext(user_input=text, force_language=tenant_lang)
ctx.metadata["id_state"] = saved
ctx.metadata["turn_number"] = turn_n
ctx.metadata["last_rewritten"] = prev_rewritten
if memories: ctx.metadata["ego_context"] = "[MEMORIES]\n" + "\n".join(memories)
```

Because all state is in `ctx.metadata` (not a live object), this is safe across
multiple stateless HTTP workers.

---

## 6. Signals the host must handle

The core never decides policy — it raises flags you act on:

| Signal | Where | Host action |
| --- | --- | --- |
| `ctx.id_result.blocked` + `block_reason` | ID (PII-CRITICAL) | short-circuit, block message |
| `ctx.stop_reason` (`vocab.VALID_STOP_REASONS`) | terminal | route/log the terminal (`completed`/`pii_blocked`/`scope_blocked`/`human_handoff`/`semantic_cache`) |
| `ctx.needs_handoff` | host policy (e.g. judge exhaustion) | escalate to a human agent |
| `ctx.drift.drift_action` (`none/warn/ask_user/self_correct`) | Drift | warn / ask clarification / trigger a correction |
| `ctx.ego_result.interrupted` + `interrupt_reason` | EGO | budget/convergence hit → surface partial result |

The core *sets the vocabulary*; the host *implements the consequence* (the real
handoff, the retry budget, the clarification text).

---

## 7. Token accounting

Every stage records a `StageMetrics` (LLM `tokens_in/out` + `embedding_tokens`).
`PipelineContext` aggregates:

- `ctx.stage_metrics` — the per-stage list for the final turn.
- `ctx.retry_metrics` — host-accumulated extras (scope check, each judge attempt,
  failed EGO retries) so retries are billed without double-counting the main path.
- `ctx.total_tokens`, `ctx.total_llm_tokens`, `ctx.total_embedding_tokens`,
  `ctx.total_elapsed_ms`.

Bill `total_tokens + sum(retry_metrics)` per turn.

---

## 8. Backends & model ladders

Selecting the model per stage/tenant and building failover ladders is **host
policy**. The core gives you the parts:

```python
from cogno_synapse import create_backend, FallbackBackend

gen   = create_backend("openai:gpt-4o-mini")     # NOUMENO/NER/scope/judge (JSON)
ego   = create_backend("deepseek:deepseek-chat") # EGO (native FC or text fallback)
voice = create_backend("mistral:latest")         # small/cheap voicer (Ollama)

# optional failover:
gen = FallbackBackend([create_backend("openai:gpt-4o-mini"),
                       create_backend("groq:llama-3.1-8b-instant")])
```

OpenAI-compatible providers (DeepSeek, Moonshot/Kimi, xAI/Grok, OpenRouter,
Together, Fireworks) work through `OpenAIBackend`'s `base_url` automatically.

---

## 9. Canonical reference

- `cognobench/pipeline.py: ReferencePipeline` — the executable version of this
  guide (the host glue, not shipped in the wheel).
- `tests/unit/test_e2e_pipeline.py` — deterministic seam tests (routing, the
  correction loop, exhaustion→handoff, multi-turn state).
- `tests/integration/test_e2e_pipeline.py` — the full pipeline against a real
  model.
- `CLAUDE.md` — the exhaustive per-stage contract and core↔host boundary map.
