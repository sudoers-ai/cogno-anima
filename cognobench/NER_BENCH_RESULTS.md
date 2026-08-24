# NER dimension — CognoBench results

The NER (Stage 2) turns the NOUMENO's canonical English into structured semantics:
intent class, sentiment, temporal framing, speech act, modality, `parole` register,
entities/verbs, PII, and the composite/sequential/negation/constraint signals the EGO
and SUPEREGO consume downstream.

55 cases, every check a **field-level equality** against a closed vocabulary
(`VALID_INTENTS`, `VALID_SENTIMENTS`, `VALID_MODALITY`, `VALID_SPEECH_ACTS`,
`VALID_PAROLE`, …). Language is forced to pt-BR (host/tenant-provided), so `langue`
verifies **propagation** from the NOUMENO, not detection.

Reproduce any cell (the runs below predate the per-slot flags; re-runs should pin
the other slots — see BENCHMARKS.md "Choosing a model PER SLOT"):

    python3 cognobench.py --only ner --model <model>

> **Suite provenance.** These runs predate the suite-versioning mechanism; the
> case universe is the one now pinned as **ner-v1** (125 checks). The
> pre-field-diet rows ran the 127-check predecessor (two `comparatives` checks
> removed by anima #65). That is exactly why the table compares **absolute
> correct counts, never percentages**. The raw sweep outputs were lost to a
> /tmp cleanup; every figure here was re-verified against the structured
> extraction made from those files while they existed (session records,
> 2026-08-06). Future runs persist via `--out cognobench/results/` — committed,
> never /tmp.

## Results (2026-08-05, 125 checks, temperature 0.0)

| Model | NER | correct | fallback¹ | tokens in | cost/sweep² |
|---|---|---|---|---|---|
| **openai:gpt-4.1-mini** | **97.6%** | 122/125 | 9 | 232,578 | $0.1256 |
| openai:gpt-4o-mini | 93.6% | 117/125 | 8 | 232,352 | $0.0465 |
| **qwen3:8b** (local) | *pending* | — | — | — | **free** |
| openai:gpt-4.1-nano | 88.8% | 111/125 | 3 | 232,328 | $0.0308 |

¹ `intent_class_fallback` events — see [Fallback rate](#fallback-rate-the-blind-spot).
² 110 LLM calls per sweep, priced off `cogno_meter/pricing.py`. Embeddings are local
(free). The qwen3:8b post-diet row needs a GPU-healthy Ollama (its CPU run was
cancelled); its pre-diet score was 115/127 with 0 fallbacks.

**`gpt-4.1-mini` leads by 5 checks over `gpt-4o-mini`** and is what production routes
to the NER slot today. It costs 2.7× more per sweep — a decision driven by per-turn
volume, not by this table, since the NER runs once per turn (the cheapest slot in the
pipeline to upgrade).

**`gpt-4.1-nano` is the cheapest and the weakest.** Saving there costs perception
quality, which contaminates every stage downstream — and in the earlier full-suite
measurement it was the only model to **miss PII** (`ner_pii_address`: risk `MEDIUM`
read as `NONE`). PII feeds safety routing; under-reading it is the dangerous
direction. Do not route the NER slot to nano.

### Where each model loses

| Model | failing fields |
|---|---|
| gpt-4.1-mini | `speech_act` ×2, `modality` ×1 |
| gpt-4o-mini | `sentiment` ×3, `speech_act` ×2, `modality` ×2, `is_sequential` ×1 |
| gpt-4.1-nano | `negation` ×3, `speech_act` ×2, `sentiment` ×2, `modality` ×2, `verb` ×1, `temporal` ×1, `pii_risk` ×1, `parole` ×1, `constraint` ×1 |

The losses concentrate in the **pragmatic** fields — `speech_act`, `modality`,
`sentiment` — not the structural ones. Every model scores `intent_class`, `langue`,
`entity` and `pii_risk` cleanly (nano excepted). `pii_risk` is never trusted from the
LLM anyway: it is recomputed deterministically in `security/pii.py`.

> **Instrument note:** `speech_act_directive` and `speech_act_commissive`
> failed on **every model in every run** (8/8); `modality_probable` in 7 of 8
> (the post-diet nano run passed it). A check that everyone stable-fails is an
> instrument defect until proven otherwise (wrong expected value or
> prompt-vocabulary mismatch) — quarantine candidates under the mutation-gate
> rules, not evidence about models.

## Fallback rate — the blind spot

`fallback` counts `intent_class_fallback` events: the model returned
`intent_class=UNKNOWN` and the deterministic heuristic coerced it from
`mandatory_tags`.

**This is not a bench check.** It is raw-output quality reported next to the score,
because the score cannot see it — the fallback is *supposed* to rescue the turn, so a
model that degrades gets caught by the safety net and still marks the same number.
Measured on the 2026-08-05 full sweep, `gpt-4o-mini` emitted 18 fallbacks against
qwen3:8b's 3 while scoring within one check of it. (The harness now surfaces this
per run — the `fallbacks=` column in the report and `meta.ner_fallbacks` in
persisted artifacts.)

## 2026-08-24 — the few-shot diet, measured and REFUSED

The field diet (#65) removed seven fields no code reads from the parser and from the prompt's
field SPEC — but left all seven in the three few-shot examples, which is the stronger
instruction. So the model kept emitting `entities.pronouns`, `entities.possessives`,
`abstract_tags`, `comparatives`, `raw_intent_class`, `raw_domains` and `raw_goal` on every turn,
and the parser kept dropping them: **75 of 365 tokens of Example 1 — 21% of the OUTPUT**
(tiktoken `cl100k_base`). #65's measured gain was **input-only** (−6.3%); output costs 4× and is
the latency bottleneck.

Cutting them from the examples was measured, `--repeat 3` per cell, and **refused**.

| | input | **output** | stable score (majority) | unstable checks |
|---|---|---|---|---|
| **gpt-4o-mini** | −4.8% | **−19.4%** | 115 → **111** | 1 → **4** |
| **gpt-4.1-nano** | −4.8% | **−16.4%** | 112 → **113** | 6 → **3** |

**Read the stable score, not the median.** By the median of three runs the cut cost gpt-4o-mini
2 checks; by this suite's own majority instrument (strict, ties fail) it cost **4**, and the
instability tripled. The median is optimistic by construction — the aggregate table is why
`--repeat` exists.

**The two models move in opposite directions on BOTH metrics**, which is the prior in the
project's `prompt-slimming-loop-vs-singleshot` note, measured again with a different cut: the
examples act as scaffolding for gpt-4o-mini and as distraction for the nano, whose instability
halves without them.

**What refuses it is the deployed configuration, not the average.** The live tenant routes
`ner_model = openai:gpt-4o-mini` (`noumeno_model` is the nano). The one model that actually runs
this stage in production is the one that loses four stable checks — and one of the checks the
cut destabilizes is `parole`, the single pragmatic field this suite scores clean before it, and
the one a per-contact register profile would have to rest on.

**What reopens it**, with these numbers in hand: (a) a narrower cut measured per group — the
`raw_*` trio is most of the text and is a different KIND of removal from
`pronouns`/`possessives`/`comparatives`/`abstract_tags`, and nothing here separates them; (b)
the NER slot moving to a nano-class model, where the cut is a gain on both axes. Until then the
21% is a known, priced waste rather than an unknown one.

**Instrument notes, paid for here:** never pipe a bench run through `head` — it cut the
AGGREGATE table off the baseline, and the `--out` artifacts persist per-run totals but not
per-run checks, so the "before" aggregate had to be re-run against a prompt that had already
moved. And "clean field" from n=1 is not a statement about stability: `parole` showed 0 failures
in a single run and is stable across three — true, but only the repetition could say so.

## 2026-08-24 re-run — the numbers above were stale relative to the prompt

`gpt-4o-mini`, current prompt, `--embed-model openai:text-embedding-3-small`, n=1:
**116/125, fallbacks=9**, in=248,567 out=19,474 over 110 calls.

Run because the failure profile was about to be used as EVIDENCE — the question was whether
`speech_act`/`modality`/`parole` are solid enough to carry a per-contact moving average
(cogno-anima #98's emotional baseline, and the communication profile that would follow). A
document is not a measurement once the prompt underneath it has moved.

| field | then (post-diet) | now | reading |
|---|---|---|---|
| `speech_act` | ×2 (`directive`, `commissive`) | **×2, the same two** | the prompt now quotes the bench inputs **verbatim** — `DIRECTIVE (… "explain X", "me explica")`, `COMMISSIVE (… "vou implementar" — the speaker acts, not the assistant)` — and the model still answers INTERROGATIVE and DIRECTIVE. Prompt-resistant. The instrument note below called these quarantine candidates; they are not — the expected values are coherent and the prompt teaches them. It is the model. |
| `modality` | ×2 | ×1, and it **moved** (`probable` now passes, `possible` now fails) | the error sits on the PROBABLE↔POSSIBLE boundary and drifts between runs — both values mean "hedged". A profile that collapses the axis to *asserts* vs *hedges* would read this field cleanly; one that distinguishes the two would be reading noise. |
| `sentiment` | ×3 | ×3, in **both** directions | not one bias but two, and the bigger one is the dangerous one. Two are `NEUTRAL → CURIOUS`, worth **+0.2** on `vocab.SENTIMENT_VALENCE`. The third is `POSITIVE → NEUTRAL`, worth **−1.0** — a *missed* positive. A contact whose warm turns read as neutral accumulates a baseline that is too LOW, and then their genuinely upset turn produces a SMALLER delta against it, so #98's escalation branch **under**-fires precisely for the person it was built for: they are told "this matches how you usually write" when it does not. Read a baseline as an estimate with a −1.0 tail, never as ground truth. |
| `parole` | ×1 (nano only) | **0** | the only clean field of the four — the one register axis a profile can rest on today. |
| new | — | `verbs` ×1, `constraint` ×1 | inside the ±2 noise floor; the `constraint` miss (`"3 linhas"` → `[]`) is the same coverage gap the SUPEREGO judge's `_format_restrictions` depends on. |

**Do not read 117 → 116 as a regression.** It is one check inside the ±2 floor measured below,
across a prompt change and a different embedder. What is outside the floor is the *profile*: two
speech-act failures that survive being told the answer, and a modality error that changes which
case it lands on.

## Effect of the field diet (#65)

anima `19af7da` cut seven fields no code read (`entities_pronouns`,
`entities_possessives`, `abstract_tags`, `comparatives`, `raw_intent_class`,
`raw_domains`, `raw_goal`). `comparatives` carried two bench checks, so the suite
went **127 → 125 checks**. Same 55 cases, same code otherwise.

| Model | correct before (of 127) | correct after (of 125) | Δ |
|---|---|---|---|
| gpt-4.1-mini | 122 | 122 | 0 |
| gpt-4o-mini | 117 | 117 | 0 |
| gpt-4.1-nano | 112 | 111 | −1 |
| qwen3:8b | 115 | *pending* | — |

**Compare absolute correct counts, not percentages.** Dropping two checks raises
every percentage on its own: `gpt-4o-mini` reads 92.1% → 93.6% while answering
exactly as many questions right.

Every Δ above sits **inside the ±2 noise floor measured below**, so the honest
reading is not "the diet cost exactly zero" but "any effect on correctness is
smaller than this suite can resolve at n=1". Separating a real 1–2 check effect from
drift needs n≥3 per cell (`--repeat 3` exists now).

The diet's benefit, by contrast, is outside the noise and on the input side:
**248,238 → 232,578 input tokens per sweep (−6.3%)**, roughly 143 tokens per call.

### Run-to-run variance is ±2 checks — measured, not assumed

Two runs of `gpt-4o-mini` against identical code, temperature 0.0:

| Run | correct | fallback | failing fields |
|---|---|---|---|
| first | 117/125 | 8 | `sentiment` ×3, `speech_act` ×2, `modality` ×2, `is_sequential` ×1 |
| repeat | 115/125 | 8 | the same 8, plus one more `is_sequential` and one `constraint` |

Same fallback count, same failure *profile*, two checks of drift. The pragmatic
fields fail consistently; the drift lands on the boundary cases (`is_sequential`,
`constraint`).

The `gpt-4.1-mini` pre/post pair shows the same magnitude independently: it passed
both `comparatives` checks before the diet, so losing them should have taken it
122 → 120; it stayed at 122, having recovered `sentiment` and `parole`.

**Do not read a difference of one or two checks as a model difference.** The
paired-comparison rule in `compare.py` is the instrument for that question.

## Notes

- **Wall-clock is not comparable across concurrent runs.** The embedder is always
  local Ollama, even for a cloud model, so a local sweep running at the same time
  queues the cloud run's embedding calls behind its own inference (a 369s cloud
  sweep took ~4× that while sharing the GPU).
- Local runs use `qwen3:8b`, never `mistral:latest` — the latter is not
  judge-capable (see `SUPEREGO_BENCH_RESULTS.md`).
