# EGO dimension — CognoBench results

The EGO is an **executor**: it selects and runs tools (via a host dispatcher) and
gathers data; it does **not** write the user reply (the SUPEREGO voices it). This
dimension therefore scores **tool selection** + **agent-loop hygiene**, not prose.

It is **decoupled from NER quality**: each case hand-builds the NOUMENO+NER
context (the NER dimension already covers extraction), so a miss reflects the
model's tool-use ability, not upstream noise. The default `OllamaBackend` has no
native FC, so the EGO runs the **text-fallback path** (`<TOOL_CALL>` tags) — the
same path the distilled student will use.

Toolset: `record_expense`, `record_income`, `get_balance`, `get_summary`,
`convert_currency`, `delete_all_records` (destructive). Checks:
- **hard** (always enforced): `steps_present` (≥1 loop step), `dispatched_tools_valid`
  (the loop never dispatches a hallucinated tool name);
- **hard capability gates** (deterministic — see `docs/ACT_CONFIRM_READONLY.md`):
  `no_mutation` (a read-only turn dispatches no mutating tool — they are masked),
  `held_for_confirmation` (a destructive tool is held, never executed, without
  `ego_confirmed`);
- **soft** (model-dependent, `--calibrate`able): `tool_selected` (right tool for an
  action), `no_tool` (a greeting/thank-you calls nothing), or `order` (an
  order-dependent multi-task request dispatches its tools in the right sequence —
  2R-B: `is_composite` raises the loop budget, `is_sequential` adds the order hint).

The two capability-gate cases (`readonly_propose`, `destructive_needs_confirmation`)
replace the earlier *soft* "act-confirm" cases, which a strong host execution
prompt overrode on the fallback path: a hopeful "the model will restrain itself"
check became an **enforceable** "the mutating/destructive tool is not available /
not executed" invariant. The classification (mutating / destructive) is
host-declared via `ToolPolicyDispatcher`.

## Results (2026-06-22, 21 cases / 64 checks, text-fallback path, temperature 0.0)

> **2026-06-22 expansion.** The case set grew from 12→21 (+9) to widen selection
> breadth and gate coverage — ported *in spirit* (not 1:1) from the parent's
> `execution_cases.py` (income/expense disambiguation, balance-vs-summary, reversed
> currency, a second read-only mask over a plain mutating tool, a read-only-over-
> destructive defence-in-depth case, and a non-sequential composite). Adapted to
> anima's contract (EGO scores tool *selection*, not the parent's response prose).
> Full **5-model re-sweep on the 21-case set (2026-06-22)** below.

| Model               | EGO accuracy          |
| ------------------- | --------------------- |
| mistral:latest      | 100.0% (64/64)        |
| qwen3:8b            | 100.0% (64/64)        |
| qwen3:8b (`--think`) | 100.0% (64/64)        |
| qwen3.5:4b          | 100.0% (64/64)        |
| llama3.1:8b         | 100.0% (64/64)        |
| qwen3.5:8b          | 100.0% (64/64)        |
| phi3:mini (3.8B)    | 98.4% (63/64)         |

> **2026-07-10 sweep**: `qwen3.5:8b` joins the 100% club. `phi3:mini` (3.8B) misses one
> soft `tool_selected` (`summary_period` — answers without dispatching) and, on the
> composite currency cases, loops re-calling `convert_currency` until the
> **duplicate-call guard + `max_steps` budget** interrupt it (`interrupted=True`) —
> the deterministic containment doing exactly its job on a too-small model.

> **Every model selects the right tool on all 21 cases and passes all gates.** The
> single qwen3:8b balance miss seen on the 12-case run did **not** recur on the
> re-sweep (Ollama at temp 0 is not bit-deterministic across runs/context); EGO tool
> selection is robust across the whole model range. The capability gates
> (read-only mask, confirmation hold, read-only-over-destructive) hold by
> construction for every model — they are deterministic, not model goodwill.

> `qwen3.5:4b` and `llama3.1:8b` added in the 2026-06-18 full-suite sweep (post
> synapse/homeo extraction). Both pick the correct tool on every action case —
> the EGO path (tool selection + the `parse_tool_calls_from_text` text-fallback,
> now sourced from `cogno-synapse`) is unchanged by the extraction.

mistral picks the correct tool for every action case, stays conversational on
chat cases, and the capability gates hold by construction (read-only masks the
mutating tools; the confirmation gate holds the destructive call as
`pending_confirmation`). On the sequential case it dispatches `convert_currency`
before `record_income` (order hint honored).

`qwen3:8b` (text-fallback, reasoning off) misses **one** soft check: on the
`balance` case it answers conversationally instead of calling `get_balance`
(`tool_selected` want `get_balance`, got `[]`). With `--think` (reasoning channel
on) it recovers the tool call → 37/37 — see *Thinking on/off* below.

### Thinking on/off (qwen3:8b, wall-clock over the 12 cases)

| Mode             | Accuracy       | Wall-clock |
| ---------------- | -------------- | ---------- |
| reasoning off    | 97.3% (36/37)  | 34.4s      |
| reasoning on     | 100.0% (37/37) | 260.6s     |

Reasoning buys the single tool-selection check (+2.7pp) at **~7.6× latency** on
the multi-step EGO loop (each loop iteration pays the think tax). On the EGO the
trade is poor: the default `mistral:latest` already scores 100% with reasoning
off and far lower latency. Reserve `think=True` for a host that has *only* a
reasoning model and sees tool-selection misses. (Under JSON ops thinking is a
no-op — Ollama suppresses the channel; the cost is entirely on this text path.)

Re-run: `python3 cognobench.py --only ego --model qwen3:8b [--think]`.

Re-run for another model:

```bash
python3 cognobench.py --only ego --model <M>
python3 cognobench.py --only ego --model <M> --calibrate   # record soft actuals
```
