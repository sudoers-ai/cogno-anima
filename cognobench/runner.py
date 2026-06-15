"""
cognobench CLI runner.

Drives the cogno-core cognitive stages (NOUMENO → NER → ID → Drift) over curated
case sets and prints a scored report. Defaults to a local Ollama backend;
`--stub` runs a fast plumbing smoke test with no model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from cognobench.harness import (
    CognitivePipeline, build_ollama, build_stub, ollama_available,
)
from cognobench.dimensions import run_noumeno, run_ner, run_id, run_drift
from cognobench.types import BenchReport
from cognobench.report import render
from cognobench.ner_cases import NER_CASES
from cognobench.drift_cases import DRIFT_CASES
from cognobench.noumeno_cases import NOUMENO_CASES
from cognobench.id_cases import ID_CASES

# Pipeline order: NOUMENO → NER → ID → Drift.
ALL_DIMENSIONS = ("noumeno", "ner", "id", "drift")


async def run_bench(
    model: str,
    embed_model: str,
    base_url: str,
    only: list[str],
    stub: bool,
    limit: int | None,
    calibrate: bool,
    language: str | None = "pt-BR",
) -> BenchReport:
    if stub:
        backend, embedder = build_stub()
        model_label = "stub"
    else:
        if not await ollama_available(base_url):
            print(f"✗ Ollama not reachable at {base_url}. "
                  f"Start it, or run with --stub for a plumbing check.", file=sys.stderr)
            sys.exit(2)
        backend, embedder = build_ollama(model, embed_model, base_url)
        model_label = model

    pipe = CognitivePipeline(backend, embedder)
    report = BenchReport(model=model_label)

    dims = [d for d in ALL_DIMENSIONS if d in only] if only else list(ALL_DIMENSIONS)

    def cap(cases):
        return cases[:limit] if limit else cases

    if "noumeno" in dims:
        report.dimensions.append(await run_noumeno(pipe, cap(NOUMENO_CASES), language=language))
    if "ner" in dims:
        report.dimensions.append(await run_ner(pipe, cap(NER_CASES), language=language))
    if "id" in dims:
        report.dimensions.append(
            await run_id(pipe, cap(ID_CASES), calibrate=calibrate, language=language))
    if "drift" in dims:
        report.dimensions.append(
            await run_drift(pipe, cap(DRIFT_CASES), calibrate=calibrate, language=language))

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cognobench",
        description="Cognitive benchmark for cogno-core (NOUMENO → NER → ID → Drift)",
    )
    parser.add_argument("--model", "-m", default="llama3.1:8b",
                        help="Ollama model for NOUMENO/NER (default: llama3.1:8b)")
    parser.add_argument("--embed-model", default="nomic-embed-text",
                        help="Ollama embedding model (default: nomic-embed-text)")
    parser.add_argument("--base-url", default="http://localhost:11434",
                        help="Ollama base URL")
    parser.add_argument("--language", "-l", default="pt-BR",
                        help="Host/tenant language forced on every case (default: pt-BR). "
                             "Language checks then verify propagation, not detection.")
    parser.add_argument("--detect", action="store_true",
                        help="Disable forced language; fall back to langdetect "
                             "and score per-case expected language (flaky on short text)")
    parser.add_argument("--only", nargs="+", choices=ALL_DIMENSIONS, default=[],
                        help="Run only these dimensions")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap cases per dimension (smoke runs)")
    parser.add_argument("--stub", action="store_true",
                        help="Fast plumbing smoke test (no model, scores meaningless)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Drift/ID: record actuals (cumulative band, goal_status) "
                             "without failing the soft checks")
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON summary instead of the table")
    parser.add_argument("--no-failures", action="store_true",
                        help="Hide the per-failure breakdown")
    args = parser.parse_args(argv)

    report = asyncio.run(run_bench(
        model=args.model, embed_model=args.embed_model, base_url=args.base_url,
        only=args.only, stub=args.stub, limit=args.limit, calibrate=args.calibrate,
        language=None if args.detect else args.language,
    ))

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(render(report, show_failures=not args.no_failures))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
