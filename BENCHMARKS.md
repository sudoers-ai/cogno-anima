# Cogno — measured cognition

Every cognitive function in Cogno is benchmarked against real models with
curated, reproducible suites (`cognobench/` here, plus sibling benches in
[`cogno-persona`](https://github.com/sudoers-ai/cogno-persona) and
[`cogno-engram`](https://github.com/sudoers-ai/cogno-engram)). All runs are
local Ollama at `temperature=0.0`; hard invariants (valid vocab, no
hallucinated tool dispatch, safety gates) are enforced on every model —
the numbers below are the model-dependent quality on top of those guarantees.

> Reproduce any cell: `python3 cognobench.py --only <dimension> --model <model>`

## Scoreboard — quality by cognitive function × local model

| Model (local, free) | EGO — tool selection | SUPEREGO — scope·judge·voice | ID — routing & goal continuity | Conversations — multi-turn e2e | BOOKKEEPER — financial tools |
|---|---|---|---|---|---|
| **mistral:latest** (7B, default) | **100%** (64/64) | 93.1% (54/58)³ | **99.0%** (103/104) | 96.0% (144/150)¹ | — |
| **qwen3:8b** (recommended) | **100%** (64/64) | **100%** (58/58)³ | 98.1% (102/104) | 95.4% (144/151) | **100%** (16/16) |
| qwen3.5:8b | **100%** (64/64) | 97.6% (41/42)⁴ | **99.0%** (103/104) | — | 94% (15/16) |
| qwen3.5:4b | **100%** (64/64) | 97.6% (41/42)⁴ | 98.1% (102/104) | 94.2% (129/137)² | — |
| llama3.1:8b | **100%** (64/64) | 90.5% (38/42)⁴ | 94.2% (98/104) | 93.4% (141/151) | — |
| qwen2.5:7b-instruct | — | — | 93.3% (97/104) | — | — |
| phi3:mini (3.8B) | 98.4% (63/64) | 90.5% (38/42)⁴ | 77.9% (81/104) | — | — |

¹ predates one added soft check (150- vs 151-check suite).
² two heaviest composite sessions hit client `ReadTimeout` on the 4B model
(latency artifact, not logic) and were excluded.
³ 2026-07-12 **judge clause-pair suite** (21→29 cases, 58 checks): every judge
prompt exception clause (trust-the-tools, honest-failure, no-fabrication,
mid-flow) is fenced by a SAVE+GUARD pair. mistral keeps 100% on the old 42
checks but **falsely APPROVES 3** of the new fabrication cases — the dangerous
direction, and exactly the surface the host's deterministic grounding backstops
cover. Route the JUDGE slot to `qwen3:8b` (per-stage routing) even when mistral
drives NOUMENO/NER.
⁴ 42-check suite (predates the clause pairs) — re-run pending.

## Cloud column (same suites, `--model provider:model`)

Same harness, cloud backends via `cogno-synapse` (`python3 cognobench.py --model
openai:gpt-4o-mini`): the EGO runs **native function calling** here (the
production-realistic path) vs the `<TOOL_CALL>` text fallback the local column
exercises; embeddings stay local. The run's token footer is the cost meter.

| Model (cloud) | EGO | SUPEREGO | ID | Conversations | Overall | Tokens (in/out) | ~$/sweep |
|---|---|---|---|---|---|---|---|
| openai:gpt-4o-mini | **100%** (64/64) | **100%** (42/42)⁵ | 97.0% (98/101) | 98.7% (154/156) | **96.8%** (491/507) | 581k / 46.5k | $0.12 |

⁵ full sweep predates the clause pairs; on the 58-check suite gpt-4o-mini scores
**98.3% (57/58)** — one *safe* false-reject (the honest-failure relay).
5 case errors in this sweep were a strict NOUMENO/NER JSON parser tripping on
cloud "extra data" output — fixed (raw_decode fallback), so the scores above are
conservative. gpt-5-nano / gpt-5-mini sweeps: in progress (reasoning-family
latency), rows land as they complete.

Full per-case analyses: [`cognobench/EGO_BENCH_RESULTS.md`](cognobench/EGO_BENCH_RESULTS.md) ·
[`SUPEREGO_BENCH_RESULTS.md`](cognobench/SUPEREGO_BENCH_RESULTS.md) ·
[`ID_BENCH_RESULTS.md`](cognobench/ID_BENCH_RESULTS.md) ·
[`CONVERSATION_BENCH_RESULTS.md`](cognobench/CONVERSATION_BENCH_RESULTS.md) ·
[`BOOKKEEPER_BENCH_RESULTS.md`](cognobench/BOOKKEEPER_BENCH_RESULTS.md) ·
persona routing: [`ROUTING_BENCH_RESULTS.md`](https://github.com/sudoers-ai/cogno-persona/blob/main/cognobench/ROUTING_BENCH_RESULTS.md).

## Where small models break (and where they don't)

The point of sweeping small local models is mapping their **failure shapes**,
not crowning a winner:

- **Tool selection is NOT the bottleneck.** Every model from 4B up scores 100%
  on EGO tool selection — even `phi3:mini` (3.8B) hits 98.4% — because the
  deterministic runtime (closed vocabularies, duplicate-call detection,
  capability gates) does the heavy lifting; the model only has to pick the
  right tool. When `phi3:mini` *does* degrade, the shape is instructive: it
  loops re-calling the same tool, and the **duplicate-call guard + step budget**
  contain it (`interrupted=True`, a signal — never a crash or a runaway bill).
- **The judge is where quality shows.** `llama3.1:8b` and `phi3:mini` each lose
  4 points on SUPEREGO — all **false rejections** of correct executions
  (`qwen3.5` models lose 1 the same way). That is the *safe* failure direction
  for a fail-closed judge (a needless retry, never a false approve), but it
  costs latency/tokens.
- **Goal continuity degrades gracefully with size.** ID stays ≥93% down to 7B —
  and `qwen3.5:8b` ties the best score (99.0%, its one miss a soft goal-status
  call on the longest chain). `phi3:mini` (3.8B) drops to 77.9% — weak goal
  extraction, and in one case a missed CRITICAL credential PII (`llama3.1:8b`
  also missed it once). The deterministic PII risk floor catches the common
  cases regardless of model.
- **Safety gates never depend on the model.** Read-only masks, confirmation
  holds and valid-dispatch invariants held on *every* model in *every* sweep —
  they are code, not model goodwill.

## Embedding-side benches (deterministic / embedder-only)

- **Persona routing** (`cogno-persona`): `nomic-embed-text`, threshold 0.25 —
  **21/21** on the single-tenant + ported parent suites (English post-NOUMENO
  input; SOCIAL messages skip embedding entirely).
- **Memory substrate** (`cogno-engram`): retrieval / buffer / consolidation /
  graph / lifecycle are deterministic dimensions (no model) gated at 100% in CI;
  LLM-dependent Tier-2 extraction is an opt-in dimension.

## Cost

Local models above cost **R$ 0 upstream** — tokens are still counted per stage
(`StageMetrics`) and folded into `PipelineContext.total_tokens`, so a host can
meter them against a plan allowance with
[`cogno-meter`](https://github.com/sudoers-ai/cogno-meter). Cloud-model sweeps
(same suites via `cogno-synapse` cloud backends) are the natural next column —
they need API keys and are run per-provider.
