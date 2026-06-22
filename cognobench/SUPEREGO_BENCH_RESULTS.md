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

21 cases → 42 checks: hard invariants (`blocked`/`approved` are bool, `response`
non-empty) + soft (`scope` = expected ALLOW/BLOCK, `judge` = expected approve/
reject, `grounded` = a required substring appears).

## Results (2026-06-22, 21 cases / 42 checks, temperature 0.0)

> **2026-06-22 expansion.** The case set grew from 13→21 (+8) — ported *in spirit*
> from the parent's `safety_cases.py` (adversarial/out-of-scope → the scope-guard
> BLOCK seam: prompt injection, off-topic health, coding) plus more judge
> goal↔execution failures (wrong amount, wrong entity) and a multi-figure grounding
> case. Full **5-model re-sweep on the 21-case set (2026-06-22)** below.

| Model               | SUPEREGO accuracy     |
| ------------------- | --------------------- |
| mistral:latest      | 100.0% (42/42)        |
| qwen3:8b            | 100.0% (42/42)        |
| qwen3:8b (`--think`) | 100.0% (42/42)        |
| qwen3.5:4b          | 97.6% (41/42)         |
| llama3.1:8b         | 90.5% (38/42)         |

> The two larger reasoners and mistral are perfect. **`qwen3.5:4b`** misses **one**
> `judge(soft)`: it rejects `judge_correct_summary` (a correct multi-figure summary)
> — over-strict on the new multi-figure judge case. **`llama3.1:8b`** misses **four**,
> *all the same shape*: `judge(soft)` **false rejections** of correct executions
> (`judge_correct_expense/balance/honors_constraint/summary` — it returns reject where
> the bigger models approve). That is the **safe** failure direction for a fail-closed
> judge (a false reject triggers a needless correction loop; a false *approve* would
> ship a wrong answer), and it is consistent with llama3.1:8b's prior 88.5% — a
> model-quality gap in the judge, not a wiring regression. The scope guard and voice
> grounding pass on every model. **Takeaway: a weak judge model over-rejects; size the
> judge backend up (mistral / qwen3:8b) for the goal↔execution gate.**

> qwen3:8b rows reflect the locale-tolerant `grounded` check (below). Pre-fix it
> scored 92.3% / 96.2% — both "misses" were the literal substring match tripping
> on pt-BR number formatting, never a quality gap.

mistral blocks off-topic (recipe/trivia) and allows finance/greetings; the judge
**catches income-recorded-instead-of-expense** (the goal↔execution miss) **and
rejects a `negation`-violating execution** (categorized when told not to) while
approving a constraint-honored one; voice grounds the figure (e.g. "1000") in the
reply, keeps a preserved figure ("1234.56") verbatim, and carries the user's
register (`parole`) into the voice prompt.

### Locale-tolerant `grounded` check (fixed 2026-06-18)

`qwen3:8b` originally "missed" `voice_academic_register` (want `1000`) and
`voice_preserved_figure` (want `1234.56`) — but it **did** ground both figures; it
renders them in the forced `pt-BR` locale (`1.000,00`, `1.234,56`), and the
`grounded` soft check was a **literal substring match** that tripped on the
thousands/decimal separators (mistral happened to emit the bare digits and passed).
`dimensions._grounded_match` now falls back, for a purely-numeric needle, to comparing
**digit runs** with grouping/decimal separators stripped (per-run, so unrelated
numbers don't fuse) — so `1234.56` matches `1.234,56` but not `1.234,57`. qwen3:8b is
now 26/26 (confirmed against Ollama), and the check scores grounding fairly across
locales for any model.

### Thinking on/off (qwen3:8b, wall-clock over the 13 cases)

Measured **before** the locale fix (raw accuracy is now 26/26 in both modes —
reasoning changes only latency here, not quality):

| Mode             | Accuracy (pre-fix) | Wall-clock |
| ---------------- | ------------------ | ---------- |
| reasoning off    | 92.3% (24/26)      | 17.9s      |
| reasoning on     | 96.2% (25/26)      | 45.6s      |

The think tax is **~2.5×** — milder than the EGO's loop because the SUPEREGO is a
single LLM call per op. With the grounded check fixed, both modes are 26/26, so
reasoning buys nothing on the SUPEREGO: **thinking is not worth the latency**;
`mistral:latest` (reasoning off, fastest) remains the default. (JSON ops —
scope/judge — see no thinking effect; Ollama suppresses the channel under
`format=json`.)

Re-run for another model:

```bash
python3 cognobench.py --only superego --model <M> [--think]
python3 cognobench.py --only superego --model <M> --calibrate   # record soft actuals
```
