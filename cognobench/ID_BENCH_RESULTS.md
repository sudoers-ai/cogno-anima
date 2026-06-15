# CognoBench — ID dimension results (multi-model sweep)

Run: 2026-06-15, local Ollama, `nomic-embed-text:latest` for goal similarity,
language forced `pt-BR`. Each run is the **full 14-case** ID dimension
(`python3 cognobench.py --only id --model <M>`), where `<M>` drives **both**
NOUMENO and NER. 104 checks/run = per-turn hard invariants (valid goal_status +
route) + soft `goal_status` lifecycle + deterministic `expect_route`/`expect_blocked`.

Scoring is **direct off `IdResult.goal_status`** — the parent inferred goal
continuity indirectly from the EGO skill (`current_skill == prev_skill`). So this
is a stricter, more direct metric over more checks (104 vs the parent's 32).

## Results

| Model | Score | Correct/Total | ~Wall time | Notes |
|-------|-------|---------------|-----------|-------|
| `mistral:latest` | **99.0%** | 103/104 | ~17 min | Best. Only the universal `anaphoric_deep` miss. |
| `qwen3:8b` | **98.1%** | 102/104 | ~12 min | `anaphoric_deep` + `math_sequence` continuation. |
| `llama3.1:8b` (default) | 94.2% | 98/104 | ~21 min | 4 soft continuation/farewell + **NER missed CRITICAL credential PII** (safety gate not triggered). |
| `qwen2.5:7b-instruct` | 93.3% | 97/104 | ~14 min | Soft farewell→COMPLETED and continuation misses. |
| `phi3:mini` | 77.9% | 81/104 | ~15 min | Small model; weakest NER goal extraction. |
| `qwen3.5:4b` | ⚠ **ERROR** | 0/0 (14 errors) | ~1 min | NOUMENO returns empty output → `StageParseError`. Model incompatible with the harness `format="json"` (likely thinking-only output stripped to empty). Not an ID bug. |

## Comparison to the parent (`cogno/bench_results/`)

The parent's **`goals`** dimension (32 checks, skill-inference proxy) over time:

| Parent config | goals score |
|---|---|
| `NER=qwen2.5:7b-instruct` + `NOUMENO=phi3:mini` | 96.9–100% (most stable) |
| `ALL=gpt-4o-mini` | 93.8–100% (after the MCP-collision bug fix; dipped to 15.6% while buggy) |
| `NER=gpt-4.1-nano` / `gpt-4.1-mini` | 96.9% |

**Validation:** cogno-core's ID reaches the parent's top band (`mistral` 99%,
`qwen3:8b` 98%) while using a **single** model for both NOUMENO+NER (the parent
split them) and a **stricter, direct** `goal_status` metric over 3× the checks.
The port + decoupling holds, and the direct metric is at least as discriminating
as the parent's skill-inference proxy.

Note the parent's best goal config leaned on `qwen2.5:7b-instruct` *as NER only*
(with the cheap `phi3:mini` doing NOUMENO). Here a single model does both, so
`qwen2.5`'s 93.3% is not contradictory — splitting roles (cheap NOUMENO + strong
NER) would likely lift it, matching the parent.

## Recurring findings (model-independent)

- **`anaphoric_deep` ("deles, qual o mais usado?") fails on every working model.**
  The Stage 1.6 fast-path needs NER `context_dependent=true`, which no model
  reliably sets here. This is the strongest case for the optional follow-up:
  use NOUMENO's `change_subject`/subject similarity as a cheap continuity prior
  in the ID, independent of NER flags.
- **Soft farewell ("perfeito, era isso", "thanks, that's all I needed") → COMPLETED**
  is missed by weaker NERs (llama3.1, qwen2.5) that don't classify it `SOCIAL`;
  caught by `mistral`/`qwen3:8b`.
- **CRITICAL credential PII** ("minha senha é …") is only as good as the NER's
  PII extraction: `llama3.1:8b` missed it (gate not fired); `mistral`, `qwen2.5`,
  `qwen3:8b` caught it. The ID gate itself is deterministic given the NER PII set.

## How to reproduce

```bash
for m in mistral:latest qwen3:8b llama3.1:8b qwen2.5:7b-instruct phi3:mini; do
  python3 cognobench.py --only id --model "$m"
done
# record actuals without failing soft checks:
python3 cognobench.py --only id --model mistral:latest --calibrate
```
