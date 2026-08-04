# NER prompt slimming — results (2026-08-04)

Phase 1 of `docs/PROMPT_SLIMMING_PLAN.md`, branch `exp/prompt-slimming-spike`.
Scaffolding + how to reproduce: `cognobench/prompt_variants/README.md`.
Raw data: `cognobench/prompt_variants/prompt_sweep.jsonl` — 41 runs, $1.36 total.

**The question.** The NER system prompt is 3,149 tokens. Can it be smaller without
losing accuracy, and is the answer the same for every model?

**The answer.** Yes it can, and no it is not. Two *different kinds* of cut behave
differently, and the split is the whole result:

- **Cutting rules prose is free on both models** (−14% tokens, no measurable loss).
- **Cutting worked examples splits them**: the cheap model gets *better*, the
  stronger model gets steadily worse.

## Setup

- Dimension `ner`, 55 cases / **127 checks** per run, full runs (no `--limit`).
- 6 prompt variants × 2 models, **2–5 repetitions per cell**, `temperature=0.0`.
- Embedder pinned to local `nomic-embed-text` (control). Variants differ **only**
  in `ner/system.txt` (verified by `diff -rq`), so the NOUMENO stage is constant.

The variants sit on two axes — *examples* (3 → 2 → 1 → 0) and *rules* (full →
compressed):

| Variant | Tokens | Δ | Examples | Field rules |
| --- | ---: | ---: | :-: | :-: |
| `default` | 3,149 | — | 3 | full |
| `v4-lean-rules` | 2,734 | −14% | 3 | **compressed** |
| `v1-no-ex3` | 2,717 | −14% | 2 | full |
| `v2-one-example` | 2,316 | −27% | 1 | full |
| `v3-schema-only` | 2,100 | −34% | 0 + skeleton | full |
| `v5-min` | 1,901 | **−40%** | 1 | **compressed** |

## Results

Raw per-run check counts (out of 127), sorted; the **range matters more than the
mean** at these sample sizes.

`gpt-4.1-nano`:

| Variant | Tokens | n | Runs | Mean | Δ | $/run |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `default` | 3,149 | 5 | 109, 109, 111, 112, 113 | 110.8 | — | 0.0326 |
| `v4-lean-rules` | 2,734 | 3 | 110, 110, 111 | 110.3 | −0.5 | 0.0304 |
| `v1-no-ex3` | 2,717 | 2 | 112, 112 | 112.0 | +1.2 | 0.0301 |
| `v2-one-example` | 2,316 | 5 | 111, 113, 113, 114, 116 | 113.4 | +2.6 | 0.0282 |
| `v3-schema-only` | 2,100 | 2 | 109, 114 | 111.5 | +0.7 | 0.0266 |
| **`v5-min`** | **1,901** | 3 | **112, 115, 115** | **114.0** | **+3.2** | **0.0259** |

`gpt-4o-mini`:

| Variant | Tokens | n | Runs | Mean | Δ | $/run |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `default` | 3,149 | 5 | 114, 116, 117, 117, 117 | 116.2 | — | 0.0491 |
| **`v4-lean-rules`** | **2,734** | 3 | **116, 117, 118** | **117.0** | **+0.8** | **0.0458** |
| `v1-no-ex3` | 2,717 | 2 | 113, 114 | 113.5 | −2.7 | 0.0452 |
| `v2-one-example` | 2,316 | 2 | 111, 111 | 111.0 | −5.2 | 0.0420 |
| `v3-schema-only` | 2,100 | 2 | 108, 110 | 109.0 | −7.2 | 0.0394 |
| `v5-min` | 1,901 | 3 | 113, 115, 116 | 114.7 | −1.5 | 0.0389 |

## Recommendation

| Model | Prompt | Tokens | Accuracy vs default | Cost |
| --- | --- | ---: | --- | --- |
| `gpt-4.1-nano` | **`v5-min`** | 1,901 (−40%) | **better** (114.0 vs 110.8) | −21% |
| `gpt-4o-mini` | **`v4-lean-rules`** | 2,734 (−14%) | same (117.0 vs 116.2) | −7% |

Both models get a smaller prompt. Neither gets a worse one. But they get *different*
smaller prompts — which is the finding that justifies binding a prompt to a model
rather than shipping one for everyone.

## What the two axes mean

**1. Rules prose is dead weight on both models.** Compressing `=== FIELD RULES ===`
costs nothing anywhere: nano −0.5 (inside its own 109–113 baseline spread), mini
+0.8 (its worst `v4` run, 116, beats its worst `default` run, 114). −14% of the
prompt for no measurable accuracy change, on either model. **This is the safest cut
in the experiment and it is model-independent.**

Part of it is provably free: the baseline spends five lines teaching the `pii_risk`
severity mapping, but `ner.py:555` does `pii_risk = compute_pii_risk(pii)` — the
model's own `pii_risk` is **discarded in-core**. The prompt was paying to teach an
output nobody reads. The `langue` block was likewise stated twice.

**2. Worked examples split the models, in opposite directions.** On `gpt-4o-mini`
every example removed costs accuracy, monotonically (−2.7 → −5.2 → −7.2). On
`gpt-4.1-nano` removing them *helps* (+2.6 at one example). Same cut, opposite sign.

**3. The direction contradicts the prior — for a reason the plan predicted.** Earlier
work found slimming hurt *weak* models because few-shot examples act as convergence
steering. That was measured on the **EGO agent loop**, where an example that saves an
iteration saves 3–4k tokens. The NER is a **single-shot JSON call**: no iterations to
save, so that mechanism is absent, and what remains is the opposite effect — the weak
model over-anchors on the examples' *content* instead of analysing the input.
**Do not generalise a slimming result across the loop/single-shot boundary.**

**4. One worked example is load-bearing; zero is not "fewer".** On nano,
`v3-schema-only` (111.5) sits *below* `v2-one-example` (113.4) despite being smaller,
and its two runs were 109 and 114 — a 5-check spread. Dropping to zero worked
examples makes the output **unstable**, which a mean hides.

## Method notes worth keeping

**Two repetitions underestimate the noise, and it misled this very analysis.** At
n=2 the nano baseline read 109/111 and the noise floor looked like 2 checks; at n=5
it reads 109–113, a spread of **4**. An interim conclusion here ("v2 clears the noise
floor") was stated with more confidence than n=2 could support. Two reps detect that
a difference exists; they do not size it. **Report ranges, not just means.**

**The 8-case screening pass gave the wrong answer — the sign flipped.** It ranked
`v3-schema-only` *best* on `gpt-4o-mini` (+5.3pp); the full 127-check run ranked it
**last** (−7.2 checks). Screening may shake out plumbing; it must never eliminate a
variant.

**`v3` is a schema skeleton, not "no examples", by design.** `ner/user.txt` never
shows the JSON envelope and `=== FIELD RULES ===` documents vocabulary but not
nesting, so the output contract lives *only* inside the worked examples. Deleting
them outright would have measured whether the model can guess an undocumented schema
— a different question whose near-certain failure would have looked like "examples
are essential" and closed the investigation on a confound.

## Before this ships

**Confidence is uneven across the three claims.** Highest on "examples hurt mini"
(monotone, −7.2 at the extreme, far outside overlap). Good on "rules prose is free"
(both models, and the direction of the small deltas differs by model, which is what
noise looks like). Weakest on "fewer examples help nano": `v5`'s runs (112, 115, 115)
mostly clear the baseline's max (113), but n=3 against a 4-check spread is thin.
**Run 5 reps of `nano`/`v5-min` and `nano`/`default` before trusting the size of that
gain** (~$0.20).

**Scope.** One dimension (NER), one stage, two models, one provider, curated bench
fixtures rather than production traffic. The bench drives the *real* `Noumeno` and
`IntentAnalyzer` classes, but not the production host orchestration — a prompt tuned
on these 55 cases could be fitting their mix. The test that would close that gap is
running the two candidates through `cogno-host`'s `hostbench`, which drives the full
pipeline with a real persona and tools.

**The shipped `system.txt` stays as-is regardless.** It also serves self-hosted
deployments on local models, where slimming has regressed twice before, and nothing
here was measured on a local model.
