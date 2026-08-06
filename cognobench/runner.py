"""
cognobench CLI runner.

Drives the cogno-anima cognitive stages (NOUMENO → NER → ID → Drift) over curated
case sets and prints a scored report. Defaults to a local Ollama backend;
`--stub` runs a fast plumbing smoke test with no model.

Fase-0 mechanics live here: per-run persistence (``--out``), the repeat mode with
its stable-score aggregate (``--repeat``), the NER-fallback side metric (a
logging handler — the coercion event is log-only in the stage), and non-zero
exit when any dimension came back INVALID (transport failure ≠ a smaller
denominator, and CI/sweep scripts must be able to see the difference).
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import logging
import re
import sys
from pathlib import Path

from cognobench.harness import (
    CognitivePipeline, TokenTally, build_cloud, build_local_embedder, build_ollama,
    build_ollama_text, build_stub, ollama_available,
)
from cognobench.dimensions import (
    run_noumeno, run_ner, run_id, run_ego, run_superego, run_drift, run_conversations,
    run_safety,
)
from cognobench.types import BenchReport, aggregate_runs
from cognobench.report import render, render_aggregate
from cognobench import suites as suite_registry
from cognobench.ner_cases import NER_CASES
from cognobench.drift_cases import DRIFT_CASES
from cognobench.noumeno_cases import NOUMENO_CASES
from cognobench.id_cases import ID_CASES
from cognobench.ego_cases import EGO_CASES
from cognobench.superego_cases import SUPEREGO_CASES
from cognobench.conversation_cases import CONVERSATION_CASES
from cognobench.safety_cases import SAFETY_CASES

# Pipeline order: NOUMENO → NER → ID → EGO → SUPEREGO → Drift, then the broad
# end-to-end conversation simulation (full pipeline, multi-turn). "superego"
# is the SELECTOR for the trio of scored sub-dimensions (scope/judge/voice).
ALL_DIMENSIONS = ("noumeno", "ner", "id", "safety", "ego", "superego", "drift",
                  "conversations")


class _FallbackCounter(logging.Handler):
    """Counts the NER's coercion events (plan 0.4a).

    ``intent_class_fallback`` is a log-only event in the stage (no field on
    ``IntentResult`` records it), and the score cannot see it — measured
    2026-08-05, a model emitted 6× more fallbacks than another while scoring
    within one check. The bench runs the stage in-process, so a handler on the
    stage's logger captures it without stdout parsing."""

    EVENTS = ("intent_class_fallback", "domains_fallback")

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.counts: dict[str, int] = {}

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        for ev in self.EVENTS:
            if ev in msg:
                self.counts[ev] = self.counts.get(ev, 0) + 1

    def snapshot(self) -> dict[str, int]:
        return dict(self.counts)


async def run_bench(
    model: str,
    embed_model: str,
    base_url: str,
    only: list[str],
    stub: bool,
    limit: int | None,
    calibrate: bool,
    language: str | None = "pt-BR",
    think: bool = False,
) -> BenchReport:
    # "openai:gpt-4o-mini" etc. → cloud column (the synapse factory owns the prefix
    # registry; a bare/unknown prefix — mistral:latest, qwen3:8b — stays Ollama).
    from cogno_synapse.factory import parse_model_string
    provider, _ = parse_model_string(model)
    cloud = not stub and provider != "ollama"

    text_backend: object  # the EGO/voice path (no JSON constraint)
    if stub:
        backend, embedder = build_stub()
        text_backend = backend
        model_label = "stub"
    elif cloud:
        # Embeddings stay LOCAL Ollama even on a cloud run (free; not the model
        # under test) — so Ollama must still be up for the embedder.
        if not await ollama_available(base_url):
            print(f"✗ Ollama not reachable at {base_url} (the embedder is local even "
                  f"on a cloud run). Start it, or run with --stub.", file=sys.stderr)
            sys.exit(2)
        backend = TokenTally(build_cloud(model))
        text_backend = backend           # cloud backends serve JSON and text alike
        embedder = build_local_embedder(embed_model, base_url)
        model_label = model
    else:
        if not await ollama_available(base_url):
            print(f"✗ Ollama not reachable at {base_url}. "
                  f"Start it, or run with --stub for a plumbing check.", file=sys.stderr)
            sys.exit(2)
        json_be, embedder = build_ollama(model, embed_model, base_url, think=think)
        backend = TokenTally(json_be)
        text_backend = TokenTally(build_ollama_text(model, base_url, think=think))
        model_label = f"{model} (think)" if think else model

    pipe = CognitivePipeline(backend, embedder)
    report = BenchReport(model=model_label)

    dims = [d for d in ALL_DIMENSIONS if d in only] if only else list(ALL_DIMENSIONS)

    def cap(cases):
        return cases[:limit] if limit else cases

    # NER fallback side metric: attach for the whole run; per-dimension counts by
    # snapshot delta. DEBUG level so domains_fallback (a debug record) is seen.
    ner_logger = logging.getLogger("cogno_anima.ner")
    counter = _FallbackCounter()
    prev_level = ner_logger.level
    ner_logger.addHandler(counter)
    ner_logger.setLevel(logging.DEBUG)

    def _delta(before: dict[str, int]) -> dict[str, int]:
        now = counter.snapshot()
        return {k: v - before.get(k, 0) for k, v in now.items()
                if v - before.get(k, 0) > 0}

    def _add(dim_result, before):
        d = _delta(before)
        if d:
            dim_result.meta["ner_fallbacks"] = d
        report.dimensions.append(dim_result)

    try:
        if "noumeno" in dims:
            before = counter.snapshot()
            _add(await run_noumeno(pipe, cap(NOUMENO_CASES), language=language), before)
        if "ner" in dims:
            before = counter.snapshot()
            _add(await run_ner(pipe, cap(NER_CASES), language=language), before)
        if "id" in dims:
            before = counter.snapshot()
            _add(await run_id(pipe, cap(ID_CASES), calibrate=calibrate, language=language),
                 before)
        if "safety" in dims:
            before = counter.snapshot()
            _add(await run_safety(pipe, cap(SAFETY_CASES), language=language), before)
        if "ego" in dims:
            # EGO needs a TEXT backend: <TOOL_CALL> fallback on Ollama, native FC on
            # cloud. In stub mode the JSON stub yields a no-tool result — enough for
            # plumbing.
            before = counter.snapshot()
            _add(await run_ego(text_backend, cap(EGO_CASES), calibrate=calibrate,
                               language=language), before)
        if "superego" in dims:
            # scope/judge consume JSON (json backend); voice needs free text.
            # Three scored sub-dimensions from one suite (plan 0.5).
            before = counter.snapshot()
            for sub in await run_superego(backend, text_backend, cap(SUPEREGO_CASES),
                                          calibrate=calibrate, language=language):
                report.dimensions.append(sub)
        if "drift" in dims:
            before = counter.snapshot()
            _add(await run_drift(pipe, cap(DRIFT_CASES), calibrate=calibrate,
                                 language=language), before)
        if "conversations" in dims:
            # Full-pipeline multi-turn simulation: gen=JSON, ego/voice=text backend.
            before = counter.snapshot()
            _add(await run_conversations(backend, text_backend, embedder,
                                         cap(CONVERSATION_CASES),
                                         calibrate=calibrate, language=language), before)
    finally:
        ner_logger.removeHandler(counter)
        ner_logger.setLevel(prev_level)

    report.meta["ner_fallbacks_total"] = counter.snapshot()

    # Cost meter: total LLM tokens across every backend call this run (the tallies
    # wrap both the JSON and the text paths; on cloud they are the same instance).
    tallies = {id(b): b for b in (backend, text_backend) if isinstance(b, TokenTally)}
    report.tokens_in = sum(t.tokens_in for t in tallies.values())
    report.tokens_out = sum(t.tokens_out for t in tallies.values())
    report.llm_calls = sum(t.calls for t in tallies.values())
    return report


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def _stamp_metadata(report: BenchReport, args: argparse.Namespace, repeat_index: int,
                    dims: list[str]) -> None:
    """Run metadata (plan 0.1): every persisted run is attributable to an exact
    config + case universe. ``--limit`` slices positionally, so the EFFECTIVE
    case ids go in, not just the suite hash."""
    now = _dt.datetime.now(_dt.timezone.utc)
    report.timestamp = now.isoformat(timespec="seconds")
    # Slug from the run's LABEL (model_label), not args.model — a --stub run must
    # say "stub", never wear the default model's name.
    report.run_id = f"{now.strftime('%Y%m%dT%H%M%S')}-{_slug(report.model)}-r{repeat_index}"
    report.config = {
        "model": args.model, "embed_model": args.embed_model,
        "language": None if args.detect else args.language,
        "only": args.only, "limit": args.limit, "stub": args.stub,
        "calibrate": args.calibrate, "think": args.think,
        "repeat_index": repeat_index, "repeat_total": args.repeat,
    }
    registry = suite_registry.registry()
    for dim, info in registry.items():
        if dim not in dims:
            continue
        ids = info["case_ids"][:args.limit] if args.limit else info["case_ids"]
        report.suites[dim] = {"suite_id": info["suite_id"],
                              "suite_hash": info["suite_hash"],
                              "effective_case_ids": ids}


def _write_out(report: BenchReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{report.run_id}.json"
    path.write_text(json.dumps(report.to_dict(include_checks=True), indent=2,
                               ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cognobench",
        description="Cognitive benchmark for cogno-anima (NOUMENO → NER → ID → Drift)",
    )
    parser.add_argument("--model", "-m", default="qwen3:8b",
                        help="Model for NOUMENO/NER (default: qwen3:8b — every local "
                             "run uses it; 'provider:model' prefixes go to the cloud)")
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
    parser.add_argument("--think", action="store_true",
                        help="Enable the reasoning channel (qwen3, deepseek, …). No-op "
                             "under JSON ops (scope/judge); visible on the text path "
                             "(EGO loop, SUPEREGO voice). Use to compare accuracy×latency.")
    parser.add_argument("--repeat", type=int, default=1, metavar="N",
                        help="Run the whole selection N times and report the stable "
                             "score + noise floor (plan 0.2; N=3 for model screening)")
    parser.add_argument("--out", type=Path, default=None, metavar="DIR",
                        help="Persist one JSON per run (full per-check results + run "
                             "metadata) into DIR — the input to compare.py and the "
                             "noise-floor/discrimination reports")
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON summary instead of the table")
    parser.add_argument("--no-failures", action="store_true",
                        help="Hide the per-failure breakdown")
    args = parser.parse_args(argv)

    dims = [d for d in ALL_DIMENSIONS if d in args.only] if args.only \
        else list(ALL_DIMENSIONS)

    reports: list[BenchReport] = []
    for i in range(1, max(1, args.repeat) + 1):
        report = asyncio.run(run_bench(
            model=args.model, embed_model=args.embed_model, base_url=args.base_url,
            only=args.only, stub=args.stub, limit=args.limit, calibrate=args.calibrate,
            language=None if args.detect else args.language, think=args.think,
        ))
        _stamp_metadata(report, args, i, dims)
        if args.out:
            path = _write_out(report, args.out)
            print(f"  → {path}", file=sys.stderr)
        reports.append(report)

    aggregate = aggregate_runs(reports) if len(reports) > 1 else None

    if args.json:
        if aggregate:
            print(json.dumps({"runs": [r.to_dict() for r in reports],
                              "aggregate": aggregate},
                             indent=2, ensure_ascii=False))
        else:
            print(json.dumps(reports[0].to_dict(), indent=2, ensure_ascii=False))
    else:
        for r in reports:
            print(render(r, show_failures=not args.no_failures))
        if aggregate:
            print(render_aggregate(aggregate))

    # Non-zero when any run had an invalid dimension: a transport-degraded run
    # must be distinguishable from a scored one by scripts and CI.
    if any(r.invalid_dimensions for r in reports):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
