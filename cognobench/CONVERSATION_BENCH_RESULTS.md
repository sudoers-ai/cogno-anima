# conversations dimension — CognoBench results

The broad **end-to-end** simulation: whole multi-turn **sessions** modelling the
host's `sessions` / `turns` / `memories` tables (persona + `mcp_module`
metadata, conversation history, and injected memories) driven through the FULL
pipeline (NOUMENO → NER → ID → scope guard → EGO ⇄ judge correction loop →
voice) via `ReferencePipeline.run_turn`, with `id_state` + NER carry-over
threaded across turns.

Scored on hard invariants (valid route per turn, terminal reached, no
hallucinated dispatch) + soft (route / blocked / tool / goal_status /
grounding), all `--calibrate`-able.

## Results (2026-06, 5 sessions / 58 checks, temperature 0.0)

| Model            | conversations accuracy |
| ---------------- | ---------------------- |
| mistral:latest   | 96.6% (56/58)          |
| qwen3:8b         | 93.1% (54/58)          |

This is up from **84.1%** before the routing widening
(`INFORMATION_REQUEST → EGO`, so tool-requiring info queries like "what's my
balance?" reach the tool gateway instead of going straight to SUPEREGO). After
the fix, **every remaining miss is a soft check** — no hard-invariant failure on
either model.

### Remaining soft misses

mistral (2):

- `finance_full_session.t2` — `goal_status` want `NEW` got `ABANDONED` (NER +
  embedding dependent; soft / calibratable)
- `finance_full_session.t3` — `tool` want `record_income` got `[]` (model chose
  not to dispatch)

qwen3:8b (4): the same two, plus

- `memory_grounded_reply.t1` — `tool` want `get_balance` got `[]`
- `memory_grounded_reply.t1` — `grounded` want `1000` got `''` (no figure to
  ground because no tool ran)

These are model-behaviour soft signals (tool selection + lifecycle labels), not
contract violations — record actuals with `--calibrate` if the bands drift.

## Re-run

```bash
python3 cognobench.py --only conversations --model mistral:latest
python3 cognobench.py --only conversations --model qwen3:8b
python3 cognobench.py --only conversations --model <M> --calibrate   # record soft actuals
```
