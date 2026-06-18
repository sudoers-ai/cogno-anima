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

| Model            | EGO accuracy |
| ---------------- | ------------ |
| mistral:latest   | 100.0% (37/37) |

mistral picks the correct tool for every action case, stays conversational on
chat cases, and the capability gates hold by construction (read-only masks the
mutating tools; the confirmation gate holds the destructive call as
`pending_confirmation`). On the sequential case it dispatches `convert_currency`
before `record_income` (order hint honored). Re-record `qwen3:8b` when convenient.

Re-run for another model:

```bash
python3 cognobench.py --only ego --model <M>
python3 cognobench.py --only ego --model <M> --calibrate   # record soft actuals
```
