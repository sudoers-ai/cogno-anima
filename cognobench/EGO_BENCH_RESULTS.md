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
`convert_currency`. 9 cases × 3 checks = **27 checks**:
- **hard** (always enforced): `steps_present` (≥1 loop step), `dispatched_tools_valid`
  (the loop never dispatches a hallucinated tool name);
- **soft** (model-dependent, `--calibrate`able): `tool_selected` (right tool for an
  action) or `no_tool` (a greeting/thank-you calls nothing).

## Results (2026-06, 9 cases / 27 checks, text-fallback path, temperature 0.0)

| Model            | EGO accuracy |
| ---------------- | ------------ |
| mistral:latest   | 100.0% (27/27) |
| qwen3:8b         | 100.0% (27/27) |

Both defaults pick the correct tool for every action case and correctly stay
conversational (no tool) on the chat cases, through the text-fallback path. The
hard invariants (valid steps, no hallucinated dispatch) hold by construction —
the loop blocks unknown tool names and feeds the block back.

Re-run for another model:

```bash
python3 cognobench.py --only ego --model <M>
python3 cognobench.py --only ego --model <M> --calibrate   # record soft actuals
```
