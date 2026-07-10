# BOOKKEEPER bench — results

Tool-selection bench for the **BOOKKEEPER** persona (ported from the parent SaaS ANALYST
`persona_cases`), run through CognoBench's EGO harness (text-fallback `<TOOL_CALL>` path) against
the **real 8 bookkeeper tools** (`cogno_praxis/bookkeeper/server.py`): `add_income`, `add_outcome`,
`get_summary`, `list_clients`, `search`, `remove_by_search`, `get_usage`, `help`.

16 cases: income (5), expense (3), summary (3), list_clients, search, `remove_by_search` (destructive
→ must be **held** for confirmation, not executed), `get_usage` (tokens ≠ expenses), scope `help`.

Run: `python -m cognobench.bookkeeper_cases [model]` (needs Ollama).

## Results (2026-07-10, local Ollama, temperature=0.0)

| Model | Tool-selection | Valid-dispatch invariant | Notes |
|---|---|---|---|
| **qwen3:8b** | **16/16 (100%)** | 16/16 | clean — the recommended local default |
| qwen3.5:8b | 15/16 (94%) | 16/16 | only miss: `help` ("what can you do?") answered conversationally, no tool — a defensible soft miss; all 15 financial ops correct |

The **destructive-hold** invariant (`remove_by_search` held for confirmation, never executed) and the
**valid-dispatch** invariant (only real tools dispatched) held on every model — these are hard,
model-independent guarantees from the vertical's MCP annotations + the EGO's confirmation gate.

## vs the parent

The parent's ANALYST `persona_cases` measured the same thing (utterance → correct financial tool).
The port lands in the same band as the parent's persona benches (~90–100% on a capable local model);
qwen3:8b reproduces a **clean 100%**. The only cross-model soft miss is the meta `help` case, not a
financial-operation error — the money-moving tools (`add_income`/`add_outcome`/`get_summary`) select
correctly across models.

## Scope

This is the **cognobench (EGO tool-selection)** half of PR-3 — decoupled, self-contained (hand-built
tool schemas, no host/vertical import needed). The **hostbench `bookkeeper_bench`** (full host + real
`BookkeeperService`, cost-per-model meter, mirroring `secretary_bench`) is the complementary half and
lands with PR-2 (the host wiring), since it drives the assembled host end-to-end.
