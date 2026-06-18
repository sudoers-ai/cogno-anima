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

## Results (2026-06, 12 cases / 37 checks, text-fallback path, temperature 0.0)

| Model               | EGO accuracy   |
| ------------------- | -------------- |
| mistral:latest      | 100.0% (37/37) |
| qwen3:8b            | 97.3% (36/37)  |
| qwen3:8b (`--think`) | 100.0% (37/37) |

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
