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
    CognitivePipeline, TokenTally, build_cloud, build_embedder,
    build_ollama, embedder_is_local,
    build_ollama_text, build_sabotage_stub, build_stub, ollama_available,
    preflight_embedder, preflight_local_toks, SABOTAGE_TARGET_SLOT, _StubScopeBlock,
)
from cognobench.dimensions import (
    run_noumeno, run_ner, run_id, run_ego, run_superego, run_drift, run_conversations,
    run_safety, systemic_guard,
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


def _build_one(spec: str, base_url: str, think: bool, *, text: bool = False):
    """One backend for one slot spec ('qwen3:8b' → Ollama, 'openai:…' → cloud)."""
    from cogno_synapse.factory import parse_model_string
    provider, _ = parse_model_string(spec)
    if provider != "ollama":
        return TokenTally(build_cloud(spec))
    if text:
        return TokenTally(build_ollama_text(spec, base_url, think=think))
    json_be, _ = build_ollama(spec, "nomic-embed-text", base_url, think=think)
    return TokenTally(json_be)


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
    slot_models: dict[str, str] | None = None,
    mutate: str | None = None,
    preflight: bool = False,
    min_toks: float = 10.0,
) -> BenchReport:
    # "openai:gpt-4o-mini" etc. → cloud column (the synapse factory owns the prefix
    # registry; a bare/unknown prefix — mistral:latest, qwen3:8b — stays Ollama).
    from cogno_synapse.factory import parse_model_string
    provider, _ = parse_model_string(model)
    cloud = not stub and provider != "ollama"
    slot_models = slot_models or {}

    text_backend: object  # the EGO/voice path (no JSON constraint)
    if stub or mutate:
        # --mutate implies the stub base (M1/M3 are deterministic by design: the
        # sabotage must be the only variable, not model noise on top of it).
        backend, embedder = build_stub()
        text_backend = backend
        model_label = f"stub+mutate:{mutate}" if mutate else "stub"
    elif cloud:
        # Ollama is required only if the EMBEDDER is still Ollama — which it is by default,
        # since a local embedder is free and deterministic and is not the model under test.
        # It used to be required unconditionally, so a fully cloud run refused to start on a
        # machine whose GPU was busy with something else: the embedder alone pinned the suite
        # to local hardware. Pass `--embed-model openai:text-embedding-3-small` (any
        # `provider:model`, same grammar as `--model`) for a run with no local anything.
        if embedder_is_local(embed_model) and not await ollama_available(base_url):
            print(f"✗ Ollama not reachable at {base_url} — the LLM is cloud but the embedder "
                  f"({embed_model}) is local. Start it, pass --embed-model with a "
                  f"provider:model spec (e.g. openai:text-embedding-3-small), or --stub.",
                  file=sys.stderr)
            sys.exit(2)
        backend = TokenTally(build_cloud(model))
        text_backend = backend           # cloud backends serve JSON and text alike
        embedder = build_embedder(embed_model, base_url)
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

    report = BenchReport(model=model_label)

    # ── Preflight (plan 0.11): a disarmed instrument scores a lying green ──
    if not stub and not mutate:
        emb_ok, emb_sim = await preflight_embedder(embedder)
        report.meta["preflight_embedder_similarity"] = round(emb_sim, 4)
        if not emb_ok:
            print(f"✗ preflight: embedder does not discriminate (unrelated-sentence "
                  f"similarity {emb_sim:.3f} ≥ 0.95) — run INVALID.", file=sys.stderr)
            sys.exit(3)
        if preflight and not cloud:
            toks = await preflight_local_toks(model, base_url)
            report.meta["preflight_toks_per_s"] = round(toks, 1)
            if toks < min_toks:
                print(f"✗ preflight: {model} at {toks:.1f} tok/s < {min_toks} — Ollama "
                      f"has likely lost the GPU (restart the service). Run INVALID.",
                      file=sys.stderr)
                sys.exit(3)

    # ── Slot routing (plan 0.10) ──
    # A slot sweep varies ONE stage's model and pins the rest — with a single
    # --model, "the NER score of X" is really "NER given X's own NOUMENO".
    # --mutate places a sabotage backend into exactly one slot (plan 0.8).
    slots: dict[str, object] = {}
    for name, spec in slot_models.items():
        slots[name] = _build_one(spec, base_url, think,
                                 text=name in ("ego", "voice"))
    if mutate:
        slots[SABOTAGE_TARGET_SLOT[mutate]] = build_sabotage_stub(mutate)

    pipe = CognitivePipeline(slots.get("noumeno", backend), embedder,
                             ner_backend=slots.get("ner", backend))
    ego_be = slots.get("ego", text_backend)
    # Stub scope baseline blocks (harness._StubScopeBlock) so the scope_allow
    # sabotage is falsifiable; conversations keep the plain stub via `slots`.
    scope_default = _StubScopeBlock() if (stub or mutate) else backend
    scope_be = slots.get("scope", scope_default)
    judge_be = slots.get("judge", backend)
    voice_be = slots.get("voice", text_backend)
    if slots:
        report.config["slots"] = {k: getattr(v, "model", "?") for k, v in slots.items()}
    if slot_models:
        # The label is compare.py's join key: a slot-swept run pooled with a
        # pure run of the same base model would average two instruments.
        report.model = model_label + "[" + ",".join(
            f"{k}={v}" for k, v in sorted(slot_models.items())) + "]"
    else:
        report.model = model_label

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

    def _add(dim_result, before, planned: int = 0):
        systemic_guard(dim_result, planned)
        d = _delta(before)
        if d:
            dim_result.meta["ner_fallbacks"] = d
        report.dimensions.append(dim_result)

    try:
        if "noumeno" in dims:
            before = counter.snapshot()
            _add(await run_noumeno(pipe, cap(NOUMENO_CASES), language=language), before,
                 planned=len(cap(NOUMENO_CASES)))
        if "ner" in dims:
            before = counter.snapshot()
            _add(await run_ner(pipe, cap(NER_CASES), language=language), before,
                 planned=len(cap(NER_CASES)))
        if "id" in dims:
            before = counter.snapshot()
            _add(await run_id(pipe, cap(ID_CASES), calibrate=calibrate, language=language),
                 before, planned=len(cap(ID_CASES)))
        if "safety" in dims:
            before = counter.snapshot()
            saf = await run_safety(pipe, cap(SAFETY_CASES), language=language)
            if stub or mutate:
                # *_llm checks measure a real model; under the stub they are noise, not
                # signal — excluded from the score, and the excluded count is PRINTED
                # by the report (a case that does not count must say it does not count).
                llm_checks = [c for c in saf.checks if c.field.endswith("_llm")]
                saf.checks = [c for c in saf.checks if not c.field.endswith("_llm")]
                saf.meta["llm_checks_excluded"] = len(llm_checks)
            _add(saf, before, planned=len(cap(SAFETY_CASES)))
        if "ego" in dims:
            # EGO needs a TEXT backend: <TOOL_CALL> fallback on Ollama, native FC on
            # cloud. In stub mode the JSON stub yields a no-tool result — enough for
            # plumbing.
            before = counter.snapshot()
            _add(await run_ego(ego_be, cap(EGO_CASES), calibrate=calibrate,
                               language=language), before, planned=len(cap(EGO_CASES)))
        if "superego" in dims:
            # scope/judge consume JSON (json backend); voice needs free text.
            # Three scored sub-dimensions from one suite (plan 0.5).
            before = counter.snapshot()
            # Positional --limit sliced whole sub-dimensions away (the first
            # cases are all scope-kind) — cap per kind instead.
            se_cases = SUPEREGO_CASES if not limit else [
                c for kind in ("scope", "judge", "voice")
                for c in [x for x in SUPEREGO_CASES if x.kind == kind][:limit]]
            kind_counts = {k: sum(1 for c in se_cases if c.kind == k)
                           for k in ("scope", "judge", "voice")}
            for sub in await run_superego(judge_be, voice_be, se_cases,
                                          calibrate=calibrate, language=language,
                                          scope_backend=scope_be):
                systemic_guard(sub, kind_counts.get(sub.name.split("_")[1], 0))
                report.dimensions.append(sub)
        if "drift" in dims:
            before = counter.snapshot()
            _add(await run_drift(pipe, cap(DRIFT_CASES), calibrate=calibrate,
                                 language=language), before, planned=len(cap(DRIFT_CASES)))
        if "conversations" in dims:
            # Full-pipeline multi-turn simulation: gen=JSON, ego/voice=text backend.
            before = counter.snapshot()
            _add(await run_conversations(backend, ego_be, embedder,
                                         cap(CONVERSATION_CASES),
                                         calibrate=calibrate, language=language,
                                         slots={k: v for k, v in slots.items()
                                                if k != "ego"} or None), before,
                 planned=len(cap(CONVERSATION_CASES)))
    finally:
        ner_logger.removeHandler(counter)
        ner_logger.setLevel(prev_level)

    report.meta["ner_fallbacks_total"] = counter.snapshot()

    # Cost meter: total LLM tokens across every backend call this run (the tallies
    # wrap the JSON, text and slot paths; on cloud json/text are the same instance).
    tallies = {id(b): b for b in (backend, text_backend, *slots.values())
               if isinstance(b, TokenTally)}
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
    report.run_id = (f"{now.strftime('%Y%m%dT%H%M%S%f')}-{_slug(report.model)}"
                 f"-r{repeat_index}")
    # update(), not assignment — run_bench may have recorded config["slots"].
    report.config.update({
        "model": args.model, "embed_model": args.embed_model,
        "language": None if args.detect else args.language,
        "only": args.only, "limit": args.limit, "stub": args.stub,
        "calibrate": args.calibrate, "think": args.think, "mutate": args.mutate,
        "repeat_index": repeat_index, "repeat_total": args.repeat,
    })
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
        epilog="Exit codes: 0 = scored clean; 1 = scored but ≥1 dimension INVALID "
               "(transport failure — artifacts persisted, totals exclude it); "
               "2 = environment/config (Ollama unreachable, missing API key, usage); "
               "3 = preflight refused (embedder does not discriminate / tok/s below "
               "threshold).",
    )
    parser.add_argument("--model", "-m", default="qwen3:8b",
                        help="Model for NOUMENO/NER (default: qwen3:8b — every local "
                             "run uses it; 'provider:model' prefixes go to the cloud)")
    parser.add_argument("--embed-model", default="nomic-embed-text",
                        help="embedding model as 'provider:model' (default: nomic-embed-text "
                             "= local Ollama). Use e.g. openai:text-embedding-3-small for a "
                             "run that touches no local service")
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
    for slot in ("noumeno", "ner", "ego", "scope", "judge", "voice"):
        parser.add_argument(f"--{slot}-model", metavar="SPEC", default=None,
                            help=f"Model for the {slot} slot only (plan 0.10) — the "
                                 f"other slots stay on --model. A slot sweep varies "
                                 f"ONE stage and pins the rest.")
    parser.add_argument("--mutate", choices=sorted(SABOTAGE_TARGET_SLOT), default=None,
                        help="Sabotage ONE stage over the stub base (plan 0.8, "
                             "M1/M3): a case whose checks stay green under its "
                             "stage's sabotage does not observe that stage.")
    parser.add_argument("--preflight", action="store_true",
                        help="Before scoring a local model, probe tok/s and refuse "
                             "to run below --min-toks (Ollama silently loses the "
                             "GPU; a CPU run masquerades as model failures)")
    parser.add_argument("--min-toks", type=float, default=10.0,
                        help="Minimum tok/s for the local preflight (default 10)")
    parser.add_argument("--out", type=Path, default=None, metavar="DIR",
                        help="Persist one JSON per run (full per-check results + run "
                             "metadata) into DIR — the input to compare.py and the "
                             "noise-floor/discrimination reports")
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON summary instead of the table")
    parser.add_argument("--no-failures", action="store_true",
                        help="Hide the per-failure breakdown")
    args = parser.parse_args(argv)

    _user_slots = [s for s in ("noumeno", "ner", "ego", "scope", "judge", "voice")
                   if getattr(args, f"{s}_model") is not None]
    if _user_slots and (args.stub or args.mutate):
        parser.error("--<slot>-model needs a real run: the stub/mutate base "
                     "replaces every slot (a 'no model' smoke was silently "
                     "making live calls through slot flags)")

    dims = [d for d in ALL_DIMENSIONS if d in args.only] if args.only \
        else list(ALL_DIMENSIONS)

    slot_models = {s: v for s in ("noumeno", "ner", "ego", "scope", "judge", "voice")
                   if (v := getattr(args, f"{s}_model")) is not None}

    from cogno_synapse.errors import SynapseError
    reports: list[BenchReport] = []
    for i in range(1, max(1, args.repeat) + 1):
        try:
            report = asyncio.run(run_bench(
            model=args.model, embed_model=args.embed_model, base_url=args.base_url,
            only=args.only, stub=args.stub, limit=args.limit, calibrate=args.calibrate,
            language=None if args.detect else args.language, think=args.think,
                slot_models=slot_models, mutate=args.mutate,
                preflight=args.preflight, min_toks=args.min_toks,
            ))
        except SynapseError as exc:
            # Missing key / provider config = environment, not a scored run.
            print(f"✗ {exc}", file=sys.stderr)
            return 2
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
