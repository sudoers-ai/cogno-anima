# SUPEREGO dimension — CognoBench results

EGO=executor, SUPEREGO=locutor: the SUPEREGO **writes** the final response from
the EGO's data (it does not review a pre-written one). This dimension scores the
three SUPEREGO operations, decoupled from NER (contexts hand-built):

- **scope** — Early Input Scope Guard: in/out-of-scope → ALLOW/BLOCK (JSON backend).
- **judge** (`evaluate`) — quality gate; **criterion #1 is goal↔execution** ("asked
  X, did X not Y") → approve/reject (JSON backend).
- **voice** — writes the final response **grounded** in the tool data (text backend).

9 cases → 18 checks: hard invariants (`blocked`/`approved` are bool, `response`
non-empty) + soft (`scope` = expected ALLOW/BLOCK, `judge` = expected approve/
reject, `grounded` = a required substring appears).

## Results (2026-06, 9 cases / 18 checks, temperature 0.0)

| Model            | SUPEREGO accuracy |
| ---------------- | ----------------- |
| mistral:latest   | 100.0% (18/18)    |

mistral blocks off-topic (recipe/trivia) and allows finance/greetings; the judge
**catches income-recorded-instead-of-expense** (the goal↔execution miss) and
approves correct executions; voice grounds the figure (e.g. "1000") in the reply.

Re-run for another model:

```bash
python3 cognobench.py --only superego --model <M>
python3 cognobench.py --only superego --model <M> --calibrate   # record soft actuals
```
