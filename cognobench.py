#!/usr/bin/env python3
"""
cognobench.py — thin CLI shim for the cogno-core cognitive benchmark.

Usage:
    python3 cognobench.py                      # full run vs local Ollama
    python3 cognobench.py --only ner           # one dimension
    python3 cognobench.py --stub --limit 3     # fast plumbing smoke test
    python3 cognobench.py --model qwen2.5:7b   # different model
    python3 cognobench.py --calibrate --only drift   # record drift actuals

See `cognobench/` for the harness, cases, dimensions and report.
"""

from cognobench.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
