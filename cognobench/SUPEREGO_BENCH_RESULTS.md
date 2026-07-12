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

29 cases → 58 checks: hard invariants (`blocked`/`approved` are bool, `response`
non-empty) + soft (`scope` = expected ALLOW/BLOCK, `judge` = expected approve/
reject, `grounded` = a required substring appears).

## Judge clause pairs (added 2026-07-12, 21→29 cases)

The judge prompt accretes **exception clauses** bug-by-bug (TRUST THE TOOLS, an
honestly-relayed failure is valid, NO FABRICATION after a failure, MID-FLOW is
valid) — and each new clause risks regressing a neighbour. Every clause is now
fenced by its own **pair**: one *SAVE* case the clause must rescue, one *GUARD*
case the clause must not weaken. The pairs needed the bench case to carry a
failing tool (`tool_ok`/`error`/`side_effect`), the EGO `draft` the judge reads,
and host-injected `context` (the clock anchor).

| Clause | SAVE (must pass because of it) | GUARD (must still fail despite it) |
| --- | --- | --- |
| TRUST THE TOOLS | approve a tool-resolved relative date vs the context clock | reject a draft contradicting the tool's own figure |
| honest failure = valid | approve a truthfully-relayed business refusal | reject a draft claiming success over an ERROR |
| no fabrication post-failure | approve relaying the tool's OWN alternative | reject invented substitute options |
| MID-FLOW = valid | approve a gathering step + question | reject a step whose data mismatches the request |

**Results on the 29-case suite (2026-07-12):** `qwen3:8b` **100% (58/58)** — its
critiques name the exact violation (the contradicted figure, the fabricated
slots, the wrong date). `mistral:latest` **93.1% (54/58)**: it keeps 100% on the
pre-existing 42 checks but misses 4 of the new clause checks — **3 false
APPROVES** (draft contradicting the tool, false success after an ERROR,
fabricated alternatives) and 1 safe false-reject (mid-flow). That is the
*dangerous* failure direction, and it is precisely the fabrication surface the
host's deterministic grounding backstops cover in production — the bench now
measures what those nets are for. **Takeaway: mistral remains the NOUMENO/NER
default, but the JUDGE slot deserves qwen3:8b (per-stage routing exists for
exactly this).**

## Results (2026-06-22, 21 cases / 42 checks, temperature 0.0)

> Rows below predate the 2026-07-12 clause pairs (42-check suite); re-run with
> `--only superego` for a 58-check row.

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
| qwen3.5:8b          | 97.6% (41/42)         |
| phi3:mini (3.8B)    | 90.5% (38/42)         |

> **2026-07-10 sweep**: `qwen3.5:8b` matches its 4b sibling (one over-strict judge
> rejection). `phi3:mini` mirrors llama3.1's shape — 3 judge false-rejections + 1
> scope miss — again the *safe* fail-closed direction (a needless retry, never a
> false approve).

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
