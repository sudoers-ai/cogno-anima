# Prompt variants — the slimming experiment's scaffolding

Experimental, on branch `exp/prompt-slimming-spike`. It may never merge: it exists
to answer one question cheaply, and a negative answer is a valid result. See
`docs/PROMPT_SLIMMING_PLAN.md` for the wider plan this is phase 0 + phase 1 of.

**The question.** The NER system prompt is 3,149 tokens and **40% of it is three
worked examples**. Do those examples earn their tokens on cloud models, or are they
paying for a format the model would get right anyway?

## Running it

```bash
# 1. (Re)generate the variant trees from the shipped templates
python3 cognobench/prompt_variants/build_variants.py

# 2. Guard the contract BEFORE spending API tokens
python3 -m pytest tests/unit/test_prompt_variants.py -q

# 3. Sweep (needs OPENAI_API_KEY; Ollama must be up — the embedder stays local)
export $(grep -E '^OPENAI_API_KEY=' ../cogno-host/.env | xargs)
python3 cognobench/prompt_variants/sweep.py --limit 8 --reps 1   # screening
python3 cognobench/prompt_variants/sweep.py --reps 2             # full

# 4. Read the ledger back at any time
python3 cognobench/prompt_variants/sweep.py --report
```

A single run is just cognobench with one extra flag, so it composes with everything
else the bench does:

```bash
python3 cognobench.py --only ner --model openai:gpt-4o-mini \
        --prompts-dir cognobench/prompt_variants/v3-schema-only --json
```

## The variants

Generated, not hand-written: `build_variants.py` slices named line-spans out of the
baseline and asserts anchor text at each boundary first, so a baseline edit fails
loudly instead of silently slicing the wrong block. Regenerate rather than patch.

| Variant | Examples kept | Tokens | Δ | Hypothesis |
| --- | --- | ---: | ---: | --- |
| `default` | 3 | 3,149 | — | baseline |
| `v1-no-ex3` | 2 | 2,717 | −14% | the contamination example is the least load-bearing |
| `v2-one-example` | 1 | 2,316 | −27% | one example is enough to anchor the shape |
| `v3-schema-only` | 0 + skeleton | 2,100 | −34% | the examples exist for FORMAT, not content |

### Why `v3` is a schema skeleton and not simply "no examples"

`ner/user.txt` asks for "one valid JSON object with all NER fields" **without ever
showing the envelope**, and `=== FIELD RULES ===` documents each field's vocabulary
but not the nesting. So the output contract lives *only* inside the worked examples.

Deleting them outright would test whether the model can guess an undocumented
schema — a different and unfair question, whose near-certain failure would teach
nothing about slimming. `v3` therefore keeps a bare shape skeleton (~120 tokens) and
drops the worked content, which isolates the real question.

This was caught by reading `user.txt` before spending anything. It is the kind of
confound that makes an experiment produce a confident wrong answer.

## The contract guard

`tests/unit/test_prompt_variants.py` holds every variant to the same prompt↔code
contracts as the shipped templates: the closed `domains` list, the `mandatory_tags`
vocabulary, and — the one most likely to catch an over-eager cut — that **every**
value in `cogno_anima.vocab` is still taught somewhere in the prompt. Some
vocabulary could be demonstrated only inside an example, so deleting one can
silently unteach a value the stage still coerces against.

The guard is verified by mutation, not by being green: deleting `HEALTH` from a
variant's domain list fails `test_variant_keeps_the_domains_contract`. A guard that
has never been seen to fail is not evidence.

`cognobench.py --prompts-dir` refuses to start on a tree missing any required
prompt, for the same reason: `load_prompt` returns `""` for a missing file rather
than raising, so a typo would otherwise run the whole sweep with an **empty** system
prompt and score the resulting garbage as a model result.

## The ledger

`prompt_sweep.jsonl` is append-only, one line per run, carrying variant, model,
accuracy, prompt tokens, token counts and USD cost (priced through
`cogno_meter.PriceBook`, so the sweep and the production bill agree). Runs are only
comparable when variant *and* case-limit match, so `--report` groups on both.

**Cloud runs are not deterministic.** Two identical runs drift by a check or two.
`--reps` defaults to 2 and the report shows the mean; a gap within ±1 check is
noise, not a finding.
