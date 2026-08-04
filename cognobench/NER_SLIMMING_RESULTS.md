# NER prompt slimming — results (2026-08-04)

Phase 1 of `docs/PROMPT_SLIMMING_PLAN.md`, run on branch `exp/prompt-slimming-spike`.
Scaffolding + how to reproduce: `cognobench/prompt_variants/README.md`.
Raw data: `cognobench/prompt_variants/prompt_sweep.jsonl` (every run, append-only).

**The question.** The NER system prompt is 3,149 tokens and 40% of it is three
worked examples. Do they earn their tokens?

**The answer: it depends on the model, and the two tested models point in opposite
directions.** That is not a wash — it is the result that justifies binding a prompt
to a model instead of shipping one prompt for everyone.

## Setup

- Dimension `ner`, 55 cases / **127 checks**, full runs (no `--limit`).
- 4 prompt variants × 2 models × **2 repetitions** = 16 runs, `temperature=0.0`.
- Embedder pinned to local `nomic-embed-text` throughout (control variable).
- Variants differ **only** in `ner/system.txt` — verified by `diff -rq`, so the
  NOUMENO stage is constant and cannot confound the comparison.
- Total cost: **$0.61** for the whole experiment, ~4 min/run.

**Noise floor.** Cloud runs are nondeterministic, so the baseline's own two
repetitions define the floor: **2 checks on nano, 3 on mini**. A delta inside that
band is not a result. Every verdict below is stated against that floor, not against
zero.

## Results

`gpt-4.1-nano` — noise floor 2 checks:

| Variant | Checks /127 | Δ | Prompt tokens | $/run | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `default` | 110.0 | — | 3,149 | 0.0326 | — |
| `v1-no-ex3` | 112.0 | +2.0 | −14% | 0.0301 (−8%) | noise |
| **`v2-one-example`** | **113.5** | **+3.5** | **−27%** | **0.0282 (−13%)** | **BETTER** |
| `v3-schema-only` | 111.5 | +1.5 | −34% | 0.0266 (−19%) | noise, **unstable** |

`gpt-4o-mini` — noise floor 3 checks:

| Variant | Checks /127 | Δ | Prompt tokens | $/run | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| **`default`** | **115.5** | — | 3,149 | 0.0491 | **best** |
| `v1-no-ex3` | 113.5 | −2.0 | −14% | 0.0452 (−8%) | noise |
| `v2-one-example` | 111.0 | −4.5 | −27% | 0.0420 (−14%) | worse |
| `v3-schema-only` | 109.0 | −6.5 | −34% | 0.0394 (−20%) | worse |

## What it means

**1. The optimum diverges by model, and both directions clear the noise floor.**
`gpt-4o-mini` degrades monotonically as examples are removed (−6.5 checks at the
deepest cut, more than twice its noise floor). `gpt-4.1-nano` *improves* — its best
prompt is 27% smaller than the shipped one. There is no single winning prompt, which
is precisely the condition that makes a per-model binding worth building.

**2. The direction contradicts the prior — for a reason the plan predicted.** Earlier
work found that slimming hurt *weak* models, because few-shot examples act as
convergence steering. That was measured on the **EGO agent loop**, where an example
that saves an iteration saves 3–4k tokens. The NER is a **single-shot JSON call**:
there are no iterations to save, so that mechanism is simply absent, and what remains
is the opposite effect — the weaker model over-anchors on the examples' *content*
instead of analysing the input. Two different mechanisms, and the measurement
separated them. **Do not generalise a slimming result across the loop/single-shot
boundary.**

**3. One worked example is load-bearing; zero is not the same as "fewer".** On nano,
`v3-schema-only` scores *below* `v2-one-example` despite being smaller, and its two
repetitions differ by **5 checks — more than double its own noise floor**. Mean
accuracy hides that: dropping to zero worked examples makes the output *unstable*,
not merely slightly worse. The schema skeleton preserves the format but loses
something the single example still supplies.

**4. Slimming does not close the model gap.** nano at its best (113.5) still trails
mini at its best (115.5). What it buys is cost: **$0.0282 vs $0.0491, 43% cheaper for
2 checks less**. That is a real point on the cost/quality frontier, not a free win.

## Recommendation

| Slot | Model | Prompt |
| --- | --- | --- |
| NER, cost-sensitive | `gpt-4.1-nano` | `v2-one-example` (−27% tokens) |
| NER, quality-first | `gpt-4o-mini` | `default` (unchanged) |

**Do not ship this yet.** Two repetitions is the floor for detecting a difference,
not for sizing one, and nano's decisive margin (+3.5) clears its noise floor (2) by
1.5 checks. Before this table goes into `DEFAULT_MODEL_CATALOG`, run **3–4 more
repetitions on the two decisive cells** (`nano`/`v2-one-example` and
`mini`/`default`). That costs about $0.25 and half an hour.

**Scope limits.** One dimension (NER), one layer, two models, one provider. Nothing
here says anything about NOUMENO, the EGO, the SUPEREGO, or any local model — and
the shipped `system.txt` must stay as-is regardless, because it also serves
self-hosted deployments where slimming has regressed twice before.

## Method notes worth keeping

**The 8-case screening pass gave the wrong answer.** It ranked `v3-schema-only` as
the *best* variant on `gpt-4o-mini` (+5.3pp); the full 127-check run ranked it
**last** (−5.1pp). A cheap screening pass is fine for shaking out plumbing, but it
must not be allowed to eliminate a variant — the sign itself flipped.

**`v3` is a schema skeleton, not "no examples", by design.** `ner/user.txt` never
shows the JSON envelope and `=== FIELD RULES ===` documents vocabulary but not
nesting, so the output contract lives *only* inside the worked examples. Deleting
them outright would have measured whether the model can guess an undocumented
schema — a different question whose near-certain failure would have looked like
"examples are essential" and closed the investigation on a confound.
