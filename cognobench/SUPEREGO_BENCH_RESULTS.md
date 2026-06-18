# SUPEREGO dimension — CognoBench results

EGO=executor, SUPEREGO=locutor: the SUPEREGO **writes** the final response from
the EGO's data (it does not review a pre-written one). This dimension scores the
three SUPEREGO operations, decoupled from NER (contexts hand-built):

- **scope** — Early Input Scope Guard: in/out-of-scope → ALLOW/BLOCK (JSON backend).
- **judge** (`evaluate`) — quality gate; **criterion #1 is goal↔execution** ("asked
  X, did X not Y"), **#2 user constraints** (honored `constraints`, did NOT do what
  `negation` forbade) → approve/reject (JSON backend).
- **voice** — writes the final response **grounded** in the tool data, with a
  **register-accommodation** signal from NER `parole` (text backend) and a
  deterministic **preserved-term backstop** (2R-A: a critical figure/email/URL the
  NOUMENO preserved must survive verbatim — the judge also sees them as grounding
  evidence).

13 cases → 26 checks: hard invariants (`blocked`/`approved` are bool, `response`
non-empty) + soft (`scope` = expected ALLOW/BLOCK, `judge` = expected approve/
reject, `grounded` = a required substring appears).

## Results (2026-06, 13 cases / 26 checks, temperature 0.0)

| Model               | SUPEREGO accuracy |
| ------------------- | ----------------- |
| mistral:latest      | 100.0% (26/26)    |
| qwen3:8b            | 92.3% (24/26)     |
| qwen3:8b (`--think`) | 96.2% (25/26)     |

mistral blocks off-topic (recipe/trivia) and allows finance/greetings; the judge
**catches income-recorded-instead-of-expense** (the goal↔execution miss) **and
rejects a `negation`-violating execution** (categorized when told not to) while
approving a constraint-honored one; voice grounds the figure (e.g. "1000") in the
reply, keeps a preserved figure ("1234.56") verbatim, and carries the user's
register (`parole`) into the voice prompt.

### qwen3:8b — the 2 misses are locale formatting, not grounding failures

`qwen3:8b` "misses" `voice_academic_register` (want `1000`) and
`voice_preserved_figure` (want `1234.56`) — but it **did** ground both figures;
it just renders them in the forced `pt-BR` locale (`1.000,00`, `1.234,56`). The
`grounded` soft check is a **literal substring match**, so the thousands/decimal
separators break it. mistral happens to emit the bare `1000`/`1234.56` and passes.
In substance qwen3:8b is 26/26 on grounding. **Known bench gap:** the `grounded`
check is locale-naive — normalizing digit runs (strip `.`/`,` grouping) before the
substring match would score this fairly across locales. Flagged, not yet fixed.

### Thinking on/off (qwen3:8b, wall-clock over the 13 cases)

| Mode             | Accuracy      | Wall-clock |
| ---------------- | ------------- | ---------- |
| reasoning off    | 92.3% (24/26) | 17.9s      |
| reasoning on     | 96.2% (25/26) | 45.6s      |

Reasoning recovers one of the two locale "misses" (it emitted a canonical figure
that turn) at **~2.5× latency** — a single LLM call per op, so the think tax is
milder than the EGO's loop. The remaining miss is the locale-formatting artifact
above, which reasoning does not change. Since both apparent failures are the
check's locale-naivety rather than real quality gaps, **thinking is not worth the
latency for the SUPEREGO** at this fidelity; `mistral:latest` (reasoning off,
fastest) remains the default. (JSON ops — scope/judge — see no thinking effect;
Ollama suppresses the channel under `format=json`.)

Re-run for another model:

```bash
python3 cognobench.py --only superego --model <M> [--think]
python3 cognobench.py --only superego --model <M> --calibrate   # record soft actuals
```
