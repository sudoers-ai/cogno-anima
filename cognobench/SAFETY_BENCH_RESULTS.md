# CognoBench — `safety` dimension

The dimension has run for months and had **no results file**: nobody was tracking its score.
Found 2026-08-17 while filing a sweep — a dimension whose number nobody records is one whose
regression nobody notices, which is the same silent-instrument failure this bench exists to
avoid.

## Baseline

| model | score | cases | note |
|-------|-------|-------|------|
| `qwen3:8b` | **100.0%** | 44/44 | 2026-08-17, same sweep as `id` (overall 147/148; 222k in / 17.8k out over 100 calls) |

## How to read a 100%

A full score here means the deterministic gates held on the cases the suite has — **not** that
the pipeline is safe. The gates it exercises (PII risk classification, the CRITICAL block, the
scope guard) are code; what the suite varies is the model feeding them. A model that misses a
CRITICAL credential is what turns a green gate red, and `ID_BENCH_RESULTS.md` records exactly
that happening to `llama3.1:8b`.

So the useful signal from this dimension is **comparative**: it separates models that feed the
gates correctly from models that do not. Reading it as an absolute safety claim would be the
"score means safe" mistake the rest of this directory keeps warning about.

## Reproduce

    python3 cognobench.py --only safety            # default model: qwen3:8b
    python3 cognobench.py --only id safety         # the sweep this baseline came from
