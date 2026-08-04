# Provider profiles + prompt slimming — experiment plan (cloud-only, OpenAI)

**Status: proposal, not started.** Experimental work, to run on its own branch. It
ships only if the numbers justify it; a negative result is a valid outcome and
should be recorded, not discarded.

**Product goal:** a tenant picks a **provider profile** ("GPT") and every pipeline
layer is filled with the model measured best *for that layer*, plus the prompt
variant tuned for that model. One pick, six slots, six prompts — see §5.

**Measurement goal:** produce the table that profile reads. Two axes from one grid:
*which model per layer*, and *what is the smallest prompt that holds its score on
that model*.

**Scope decision (2026-08-04):** this round tests **cloud models only, OpenAI only**.
No local models. That is also the only feasible matrix today — `OPENAI_API_KEY` is
the sole provider key present in `cogno-host/.env`. The profile mechanism is
provider-generic; other providers get their own profile when a key exists and the
same sweep has been run for them. **An unmeasured provider profile must not ship** —
guessing the cells is how you get a bot that refuses its users (§5.0).

---

## 1. The targets — only the large prompts

Measured with `cl100k_base`. Two repos hold them: the perception prompts are core
(`cogno-anima`), the persona prompts are host-side (`cogno-praxis`).

| Layer | File | Tokens | Calls/turn | Prod share¹ |
| --- | --- | ---: | --- | ---: |
| **EGO** (executor) | `cogno-praxis/.../scheduler/prompts/system.txt` | 2,738 | **× loop steps (5–8)** | ~47% |
| **NER** (analysis) | `cogno-anima/prompt_templates/ner/system.txt` | 3,149 | 1 (single-shot) | ~20% |
| **SUPEREGO** (voicer) | `cogno-praxis/.../scheduler/prompts/voice.txt` | 1,816 | 1 | ~10% |
| **NOUMENO** (rewrite) | `cogno-anima/prompt_templates/noumeno/system.txt` | 1,296 | 1 (single-shot) | ~8% |
| **SUPEREGO** (judge) | `cogno-praxis/.../scheduler/prompts/limits.txt` | 667 | 1 + per correction | ~10% |

¹ Share of turn input tokens, from the 2026-07-11 prod ledger measurement.

Secondary targets, same method if the primaries pay off: `closer/system.txt`
(1,925), `host/prompts/landing_sdr.txt` (2,557), `closer/voice.txt` (858).
Everything else is under ~500 tokens and not worth the experiment cost.

**Not a target:** `scheduler/scope.txt` (490) — already small, and it guards cost
rather than spending it.

### Why the two families behave differently

This is the central hypothesis of the experiment.

- **NER and NOUMENO are single-shot JSON calls.** No agent loop. Prompt tokens map
  **linearly** to cost. Cutting 1,000 tokens saves 1,000 tokens per turn, full stop.
- **The EGO prompt is re-sent on every loop iteration.** Its cost is
  `size × steps`. The 2026-07-11 and 2026-07-13 measurements showed that cutting
  the EGO prompt **backfired on weak local models**: tool calls doubled (34→71)
  because the few-shot examples act as *convergence steering*, and each extra
  iteration costs ~3–4k tokens — far more than the prompt bytes saved. On cloud
  with native function calling, the surgical `slim2` cut was worth ~12%.

Two consequences of going cloud-only:

1. **The EGO's main downside risk is out of scope.** The convergence-steering
   failure was a small-model phenomenon (`gpt-4o` scored 19/19 even on the
   aggressive slim). Cloud-only makes phase 3 *more* attractive, not less — but
   `tool_calls` must still be measured, because it is the real cost driver.
2. **Nothing found here may be promoted into the shipped default prompt.** The
   defaults in `cogno_anima/prompt_templates/` and the praxis personas also serve
   self-hosted/local deployments, where slimming has already regressed twice. A
   cloud win ships as a **per-slot variant**, or it goes through a local
   re-validation gate first. This is exactly the `slim2` lesson.

---

## 2. Method

### 2.1 Versioning — variants as directory trees

Both stages already accept `prompts_dir=` (`Noumeno.__init__`, `IntentAnalyzer.__init__`),
and the NER also accepts `system_prompt_name=`. **No core change is needed to swap
a prompt.** Variants live as complete mirrors of the template tree:

```
cognobench/prompt_variants/
  v0-baseline/          # copy of prompt_templates/ at the experiment SHA
  v1-no-ex3/ner/system.txt
  v2-one-example/ner/system.txt
  v3-lean-fieldrules/ner/system.txt
  ...
```

Each variant carries a `MANIFEST.md`: hypothesis, parent variant, token count,
what was removed. Git gives history; the manifest gives intent. A variant without
a stated hypothesis is not an experiment, it is a guess.

> This full-tree layout is for the **experiment only** — it keeps each run's inputs
> trivially reproducible from one path. What eventually *ships* is the narrower
> sibling-file layout in §5.4; duplicated trees would let the unchanged files drift
> apart over time. The winning variant is transcribed into that layout, not copied
> across as a tree.

### 2.2 Ablation by section, not rewrite

Remove **one block at a time** and measure. A full rewrite that scores worse tells
you nothing about *which* part carried the score. The NER prompt decomposes cleanly:

| Block | Lines | Tokens | Share |
| --- | --- | ---: | ---: |
| header + CRITICAL SENTIMENT / LANGUAGE rules | 1–21 | 201 | 6% |
| `=== EXAMPLES ===` — 3 full JSON examples | 22–157 | **1,260** | **40%** |
| — ex1 composite / ex2 back-ref / ex3 contamination | | 422 / 401 / 432 | |
| `=== FIELD RULES ===` — vocabularies + guidance | 158–285 | 1,687 | 53% |

Planned NER variants:

| Variant | Change | Est. tokens | Hypothesis |
| --- | --- | ---: | --- |
| `v0` | baseline | 3,149 | reference |
| `v1` | drop ex3 (contamination) | ~2,717 | least-used example |
| `v2` | keep ex1 only | ~2,326 | one example anchors the JSON shape |
| `v3` | 3 examples compressed to field-diffs | ~2,400 | examples exist for *format*, not content |
| `v4` | FIELD RULES prose trimmed, vocab lists intact | ~2,600 | vocab is load-bearing, prose is not |
| `v5` | v2 + v4 combined | ~1,900 | best of both, if both hold |

**Hard constraint:** the `domains` closed list inside FIELD RULES must stay
byte-for-byte identical to `NER_KNOWLEDGE_DOMAINS`
(`tests/unit/test_pipeline.py::test_code_domains_match_prompt_domains_exactly`).
The sweep runner must run that test against each variant **before** spending
API tokens on it.

### 2.3 The model matrix — OpenAI capability ladder

Three tiers, chosen so the result answers *"does the tolerable cut scale with model
strength?"* — which is the per-slot question:

| Slot | Model | $/1M in | $/1M out | Role in the matrix |
| --- | --- | ---: | ---: | --- |
| floor | `gpt-4.1-nano` | 0.10 | 0.40 | cheapest viable; scored 96.9% on the ID bench |
| **reference** | `gpt-4o-mini` | 0.15 | 0.60 | the recommended cloud backbone (best EGO cost/benefit) |
| upper-mid | `gpt-4.1-mini` | 0.40 | 1.60 | scored 96.9% NER but **worse + pricier on the EGO** |
| ceiling (finalists only) | `gpt-4o` | 2.50 | 10.00 | upper bound on how much can be cut |

**Excluded: the whole `gpt-5-*` family.** Measured reasoning tax — huge output
token counts and ~3× wall-clock. It would confound the measurement (you would be
measuring reasoning, not the prompt) and it is already ruled out of the hot path.

**Control variable — the embedder stays local `nomic-embed-text`.** Embeddings feed
drift and subject continuity, not prompt size. Swapping it would move the drift
numbers and break comparability with every recorded baseline. Keep it fixed; it
also means Ollama must still be running, though the load is negligible.

### 2.4 Harness changes needed

> **Corrected 2026-08-04 (measured, not assumed).** An earlier draft of this plan
> claimed cognobench had no cloud path. **That was wrong.** `runner.py` already
> parses the `provider:` prefix and routes to `build_cloud`, so
> `--model openai:gpt-4o-mini` worked before this experiment started — and the
> embedder was already pinned to local Ollama on cloud runs, exactly the control
> the plan asks for. Only `--prompts-dir` was actually missing. The spike shrank
> from ~5 h to ~2 h because of it. Verify the seam before pricing the work.

Historically `cognobench/harness.py` hardcoded `PROMPTS_DIR`. Remaining additions:

1. `--prompts-dir PATH` — thread through `CognitivePipeline` and `ReferencePipeline`.
2. `--backend provider:model` — use `cogno_synapse.create_backend`; the stages
   already accept any `LLMBackend`, so cloud comes free through the protocol.
3. **Per-stage backend override** (`--backend-ner`, `--backend-noumeno`, … or one
   `stage=spec` map). Required by the profile: a single `--backend` drives NOUMENO
   and NER with the same model and cannot separate their columns. See §5.0.1.
4. **Ledger**: append each run's `--json` output plus `(variant, per-stage backends,
   git SHA, prompt tokens, cost)` to `cognobench/prompt_sweep.jsonl`. The report
   reads the ledger and plots score × prompt-tokens per model per layer.

For the praxis/persona prompts the harness already exists:
`cogno-host/hostbench/secretary_bench.py` has a committed `--persona {full,slim,slim2}`
flag that monkeypatches the persona loader to strip a named block. Extend it to take
an arbitrary variant directory rather than three hardcoded modes.

### 2.5 Statistical discipline — stricter now

Going cloud-only removes the deterministic leg of the experiment. Every cell is now
nondeterministic, so the noise discipline is mandatory everywhere:

- **Minimum 2 runs per cell, report the mean.** The 2026-07-12 sweep drifted ±1–2
  checks and ±$0.01 between identical runs; a single-run comparison misled us once
  already.
- **Treat a ±1-check gap as noise.** Promote a variant only on a gap that survives
  a third run.
- Set `temperature=0.0` anyway — it narrows but does not eliminate the variance.
- Screen all variants with `--limit 20` first, then run the full case set only on
  finalists. Cloud runs are cheap enough that this is a time optimisation, not a
  cost one.

---

## 3. Phases and gates

Each phase ends in a go/no-go. A phase that fails its gate stops there; the
recorded negative result is the deliverable for that layer.

> **Phase 0 and the NER spike are BUILT** (branch `exp/prompt-slimming-spike`,
> 2026-08-04) — `--prompts-dir` + variant generator + contract guard + sweep ledger.
> See `cognobench/prompt_variants/README.md` for how to run it and
> `NER_SLIMMING_RESULTS.md` for what it measured.

| # | Phase | Work | Gate to continue |
| --- | --- | --- | --- |
| **0** | Instrumentation ✅ | `--prompts-dir`, variant generator, contract guard, sweep ledger | baseline reproduces the known cloud scores |
| **1** | **NER** (3,149 tok) | 6 variants × 3 tiers × 2 reps | a variant holds within −1 check at ≥20% fewer tokens |
| **2** | **NOUMENO** (1,296 tok) | 4 variants, same matrix | same |
| **3** | **EGO persona** (2,738 tok × steps) | 3 variants, `secretary_bench` | **must measure `tool_calls`, not just prompt size** |
| **4** | **SUPEREGO** voice + judge | 4 variants; **voice is unmeasured on OpenAI — treat as first-class, not an afterthought** | no grounding / preserved-term regression |
| **5** | Consolidation | Pareto frontier per tier, recommendation | — |

**Phase 3 prerequisite — check prompt caching first.** OpenAI discounts a stable
prompt prefix ≥1024 tokens by 50% automatically, and the bench prices at list rate,
ignoring it. The real cloud cost may already be lower than the ledger says, which
would shrink the prize for the whole experiment. Verify prefix stability and
capture `cached_tokens` before optimizing further — a stable-prefix reordering may
beat any slimming, at zero quality risk. **This is the single highest-value check
in the plan and it is cheap; do it in phase 0 if possible.**

---

## 4. Cost

### 4.1 Per-run token cost

Measured: a 5-case NER run consumed 28,780 input / 2,327 output tokens across 10
LLM calls. Extrapolated to the full 55-case NER dimension: **~316k in / ~26k out
per run**.

| Model | Full NER run | Full NOUMENO run | `secretary_bench` run |
| --- | ---: | ---: | ---: |
| `gpt-4.1-nano` | $0.04 | ~$0.01 | ~$0.06 |
| `gpt-4o-mini` | $0.06 | ~$0.01 | **$0.128** (measured) |
| `gpt-4.1-mini` | $0.17 | ~$0.03 | **$0.212** (measured) |
| `gpt-4o` | $1.05 | ~$0.18 | **$1.34** (measured) |

### 4.2 Budget per phase

| Phase | Matrix | Runs | Cost |
| --- | --- | ---: | ---: |
| 1 — NER, 3 tiers | 6 variants × 3 × 2 reps | 36 | **~$3.30** |
| 1b — NER finalists on `gpt-4o` | 3 × 1 × 2 | 6 | **~$6.30** |
| 2 — NOUMENO | 4 × 3 × 2 | 24 | **~$0.30** |
| 3 — EGO persona | 3 × 3 × 2 | 18 | **~$2.40** |
| 4 — SUPEREGO | 4 × 3 × 2 | 24 | **~$1.20** |
| Screening passes (`--limit 20`) | all phases | ~30 | **~$1.00** |
| | | **~140 runs** | **~$15** |

**Budget $40 and it will not be reached.** The `gpt-4o` ceiling runs are the only
line worth watching — restrict them to finalists, never to the full variant sweep.

### 4.3 Wall-clock

Cloud-only removes the GPU bottleneck that dominated the earlier estimate. A full
NER run is ~110 sequential API calls; at ~2s each that is **~4–5 min per run**.

| Phase | Runs | Sequential | With 4-way parallelism |
| --- | ---: | --- | --- |
| 1 — NER (+ finalists) | 42 | ~3.5 h | ~1 h |
| 2 — NOUMENO | 24 | ~0.5 h | ~10 min |
| 3 — EGO persona | 18 | ~1.5 h | ~25 min |
| 4 — SUPEREGO | 24 | ~1 h | ~20 min |
| **Total** | **~140** | **~7 h** | **~2 h** |

Runs are independent processes, so parallelism is free — the limit is the **OpenAI
rate limit for the account tier**, which should be checked before launching a
4-way sweep. If throttling appears, fall back to 2-way; the whole sweep still fits
in an afternoon.

### 4.4 Calendar

| Phase | Engineering | Machine |
| --- | --- | --- |
| 0 — Instrumentation (`--backend` is new work) | ~4–5 h | — |
| 1 — NER | ~2 h analysis | ~1 h |
| 2 — NOUMENO | ~1 h | ~10 min |
| 3 — EGO persona | ~3 h | ~25 min |
| 4 — SUPEREGO | ~2 h | ~20 min |
| 5 — Consolidation (fill the profile table) | ~3 h | — |
| **Experiment total** | **~15 h hands-on** | **~2 h** |
| 6 — Profile + binding wiring (§5.6) | ~17 h | — |
| **Grand total** | **~32 h** | **~2 h** |

The experiment is **~2 working days** and the machine time is no longer the
constraint — engineering is. The wiring adds **~2 more days**, but it is gated:
build it only if phases 1–4 produce a table worth binding to. Dropping phases 3–4
after a negative phase 1 brings the experiment to ~1 day.

---

## 5. The product goal — a provider profile that fills everything

The experiment is not the deliverable. The deliverable is a **provider profile**:
the tenant picks *"GPT"* once, and every layer is already filled with the model
measured best for that layer **and** the prompt tuned for that model. One pick,
six slots, six prompts. No manual pairing, no per-layer knob-twiddling.

```
pick profile "GPT"
  ├─ noumeno → gpt-4.1-nano   + prompt variant "lean"
  ├─ ner     → gpt-4o-mini    + prompt variant "lean"
  ├─ ego     → gpt-4o-mini    + prompt variant "default"
  ├─ scope   → gpt-4o-mini    + prompt variant "default"
  ├─ judge   → gpt-4o-mini    + prompt variant "default"
  └─ voice   → gpt-4o-mini    + prompt variant "default"
```

(Illustrative — the actual picks are what the sweep must produce.)

### 5.0 The catalog already has the right home, on the wrong axis

`cogno-host/cogno_host/model_catalog.py` already ships `DEFAULT_MODEL_CATALOG` with
one entry per provider, each carrying `tiers` (a price ladder) and `routing`. But
`routing` is keyed by **complexity band** (`LOW`/`MEDIUM`/`HIGH`/`EXPERT`) — the
escalation axis. What the profile needs is the **pipeline-layer** axis, which does
not exist yet. It is an additive change to a structure that is already the accepted
home for model policy:

```python
"openai": {
    "tiers":   [...],                              # exists — price ladder
    "routing": {"LOW": ..., "EXPERT": ...},        # exists — complexity axis
    "layers":  {"noumeno": "gpt-4.1-nano",         # NEW — pipeline-layer axis
                "ner": "gpt-4o-mini", "ego": "gpt-4o-mini",
                "scope": "gpt-4o-mini", "judge": "gpt-4o-mini",
                "voice": "gpt-4o-mini"},
    "prompts": {"ner": "lean", "noumeno": "lean"}, # NEW — prompt variant per layer
}
```

Precedence, extending what `resolve_specs` already does — most specific wins:

```
<layer>_model explicit pin  >  profile.layers[layer]  >  cogno_model  >  host default
```

A tenant who picks a profile and then pins one layer keeps the pin; a tenant who
picks nothing behaves exactly as today. Nothing regresses by adding this.

**The catalog also already documents why a naive profile breaks.** Its `LOW` band
deliberately skips `gpt-5-nano` despite it being cheapest, because the measured
reasoning tax (571k out vs 515k in, ~3× wall-clock) makes cheapest-on-paper the
most expensive to answer with. The same trap applies per layer: `gpt-4.1-nano` is
known to **over-block on scope/judge**. A "cheapest everywhere" GPT profile would
ship a bot that refuses its users. Every cell in `layers` has to be measured, not
assumed.

### 5.0.1 This needs two measurement axes, and one grid gives both

The profile has two unknowns: *which model per layer*, and *which prompt per model*.
Those are different questions, and the plan so far only measures the second.

The good news is that **the sweep grid already answers both**. Running
`variants × models` per layer produces a score for every cell; reading it down a
column gives the best model for that layer, reading it across a row gives the best
prompt for that model. No second sweep is needed — but two things follow:

1. **Phase 0 must support per-layer model isolation.** A single `--backend` flag
   drives NOUMENO and NER with the same model, so it cannot separate their columns.
   The flag needs to accept a per-stage mapping (`--backend-ner`, `--backend-noumeno`,
   …, or one `stage=spec` map). This is a small extension, but it is load-bearing
   for the profile and must not be discovered in phase 2.
2. **Existing per-layer evidence is thin and scattered.** For OpenAI we have the EGO
   (`gpt-4o-mini` best, `gpt-4.1-mini` worse *and* pricier), the NER
   (`gpt-4.1-nano`/`4.1-mini` at 96.9%), and a negative for nano on scope/judge.
   **The voice layer has never been measured on any OpenAI model.** That gap is the
   profile's weakest cell and phase 4 should treat it as a first-class target, not
   an afterthought.

### 5.1 The prompt half — the seam already exists

`cogno-host/cogno_host/model_router.py` resolves the tenant's dashboard picks into
one spec per stage:

```python
resolve_specs(routing) -> {"ner": "openai:gpt-4o-mini", "ego": "openai:gpt-4o", ...}
```

The prompt binding is that function's twin, consuming its output:

```python
resolve_prompt_variants(specs) -> {"ner": "lean", "ego": "default", ...}
```

Same shape, same precedence, same fail-open behaviour. It belongs beside
`resolve_specs`, and it runs once per turn on data the router already computed.

### 5.2 Ownership split — pack in the lib, mapping in the host

This follows the doctrine the rest of the system already uses (*core ships
capability, host decides policy*), with one deliberate exception:

| Artefact | Owner | Why |
| --- | --- | --- |
| Prompt **variants** for NOUMENO/NER | **`cogno-anima`** | the prompt is coupled to the parsing contract (`NER_KNOWLEDGE_DOMAINS`, the JSON field set). A host-authored variant that drifts silently breaks the stage. |
| Prompt **variants** for persona slots (EGO system / voice / limits) | **`cogno-praxis`** | they are persona content, already per-persona. |
| The **model → profile mapping** | **`cogno-host`** | it is a business/cost policy, tenant-overridable, and changes as models are added. |

The exception matters: prompts are usually host content, but the NER prompt is
effectively part of the stage's wire format. Shipping its variants next to the code
that parses them is what keeps the contract test meaningful.

### 5.3 Bind to a prompt *variant class*, not to a model id

> **Terminology.** Two different things are in play and they must not share a word.
> A **provider profile** is the tenant-facing preset ("GPT") that fills all six
> layers. A **variant class** is the small internal label a prompt file carries
> (`default` / `lean` / `verbose`). A profile *selects* a variant class per layer.

The single most important design decision, because it decides the maintenance bill.

Binding one variant per exact model id makes N variants grow with the catalogue, and
every contract change (a new domain, a new field) must then be applied N times. Bind
instead to a small closed set of **variant classes** — expected to be about three:

| Variant class | Intent | Likely members |
| --- | --- | --- |
| `default` | the shipped prompt, full guidance | anything unknown, all local/self-hosted |
| `lean` | examples trimmed, vocab intact | strong models that do not need convergence steering |
| `verbose` | extra steering, if a cheap tier needs it | `gpt-4.1-nano`-class, only if the sweep shows it helps |

Model ids resolve to a variant class by **longest-prefix match**, so
`gpt-4o-mini-2024-07-18` lands on `gpt-4o-mini`'s class without a catalogue edit.
That matching logic is already solved and battle-tested in
`cogno-meter/cogno_meter/pricing.py` — reuse the approach rather than reinventing it.

### 5.4 Storage layout

A variant is a **sibling file**, not a duplicated tree — duplicating the tree would
also copy the unchanged `user.txt` and let the copies drift:

```
cogno_anima/prompt_templates/ner/
  system.txt            # default — always the fallback, never deleted
  system.lean.txt       # profile variant
  user.txt              # shared, not duplicated
```

`IntentAnalyzer` already accepts `system_prompt_name=`, so **the NER needs no core
change at all.** `Noumeno` hardcodes `"system.txt"` (`stages/noumeno.py:59`) and
needs the same two-line parameter for symmetry. That is the entire core change.

### 5.5 Two failure modes to design against

**A silent empty prompt.** `load_prompt` returns `""` for a missing file
(`prompts.py:33`). A typo in a profile name would therefore hand the stage an empty
system prompt — degrading it into garbage output rather than failing cleanly. The
resolver **must** check existence and fall back to `system.txt`, logging the miss.
Never let `""` reach a stage.

**Variant rot.** A new entry in `NER_KNOWLEDGE_DOMAINS` currently breaks one prompt
and one test. With a pack, it breaks N prompts and the test only checks the default.
`test_code_domains_match_prompt_domains_exactly` must be parametrised over **every**
variant in the pack before the first variant is merged. This is the price of the
feature and it should be paid up front, not after the first silent divergence.

### 5.6 Implementation phases

These follow the measurement phases — there is no point wiring a mapping before
knowing whether the variants earn their place.

| # | Step | Effort |
| --- | --- | --- |
| **6a** | Parametrise the domains contract test over all variants; add `system_prompt_name` to `Noumeno` | ~1 h |
| **6b** | `layers` + `prompts` keys in `DEFAULT_MODEL_CATALOG` per provider | ~1 h |
| **6c** | Profile precedence in `resolve_specs` (pin > profile > `cogno_model` > default) | ~2 h |
| **6d** | `resolve_prompt_variants()` in `model_router.py` + prefix matcher + fail-open fallback | ~3 h |
| **6e** | Thread the resolved prompt variant into the soma `TurnConfig` per stage | ~2 h |
| **6f** | Persona-side equivalent in `cogno_host/persona.py` (`load_persona` slot loading) | ~2 h |
| **6g** | Dashboard: one **profile picker** that fills the six per-layer knobs, still overridable | ~3 h |
| **6h** | Tests: unknown model → default, missing file → default, prefix match, pin-beats-profile | ~3 h |

**~17 h**, on top of the ~15 h of experiment work. Unlike the prompt binding (which
is invisible by design), the profile itself **is** a dashboard change — it is the
one thing the tenant actually clicks. The six existing per-layer knobs stay, as the
override path for anyone who wants to depart from the preset.

---

## 6. What "success" looks like

A variant enters the pack only if it is **smaller and not worse**. Concretely:

- ≥20% fewer prompt tokens, **and**
- score within −1 check of baseline on the tier it targets, sustained over ≥2 runs, **and**
- for the EGO: **no increase in `tool_calls`** (the real cost driver).

If the winner differs by tier — which the priors suggest it will, with stronger
models tolerating deeper cuts — that is not a disappointing result, it is **the
intended one**: it is what makes the per-profile binding worth building at all. A
single winning prompt across all tiers would mean the prompt half of the feature is
unnecessary and the right move is to ship that prompt as the new default — the
**model** half of the profile (one pick fills six layers) still stands on its own.

**The real deliverable of phases 1–5 is one filled-in table**, the `layers` +
`prompts` block for the `openai` provider in `DEFAULT_MODEL_CATALOG`. Everything
else in the experiment exists to justify each of its twelve cells. A cell with no
measurement behind it should carry the safe default, not a guess — the catalog's
own `gpt-5-nano` comment is the precedent for writing down *why* a cheap option was
rejected.

**Shipping constraint:** a cloud-only win does not go into `system.txt`, which also
serves self-hosted deployments where slimming has already regressed twice. It ships
as a **profile variant** whose members are cloud models only, leaving local
deployments on `default` untouched. That constraint is not a limitation of the
design — it is precisely what the profile mechanism buys.

## 7. Recorded priors (do not re-discover)

- Aggressive EGO slimming regresses weak models: examples are convergence steering,
  and token cost is `prompt × steps`. **Measure steps.** (`gpt-4o` is immune;
  `gpt-4o-mini` is not — it went 30→27.)
- `slim2` (surgical, cut only the redundant tool list) is cloud-only: −12% cloud,
  **+6% tokens locally**. Never promote it to a shared prompt.
- `gpt-4.1-mini` is **worse and 1.65× pricier than `gpt-4o-mini`** on the EGO
  (over-explores, stops short of booking). Keep it in the matrix as a data point,
  not as a candidate backbone.
- The bench is noisy on cloud; single-run comparisons have misled us. ≥2 runs.
- Empty/weak `tools_schema()` descriptions sabotage any prompt-pruning measurement —
  the tool descriptions are a prerequisite, not a variable.
- The `gpt-5-*` family carries a measured reasoning tax; excluded from this sweep.
