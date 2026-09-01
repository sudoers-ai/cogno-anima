"""Console report rendering for cognobench."""

from __future__ import annotations

from cognobench.types import BenchReport


def _bar(pct: float, width: int = 24) -> str:
    filled = int(round(pct / 100.0 * width))
    return "█" * filled + "·" * (width - filled)


def render(report: BenchReport, show_failures: bool = True) -> str:
    lines: list[str] = []
    lines.append("")
    lines.append("═" * 60)
    header = f"  CognoBench (cogno-anima)   model={report.model or 'n/a'}"
    if report.run_id:
        header += f"   run={report.run_id}"
    lines.append(header)
    lines.append("═" * 60)

    for dim in report.dimensions:
        if not dim.valid:
            # An invalid dimension NEVER renders a score — a transport failure is
            # not a smaller denominator (plan 0.3).
            lines.append(f"  {dim.name:<15} {'▓' * 24} INVALID — {dim.invalid_reason}")
            continue
        if dim.total == 0 and not dim.errors:
            # "0.0% (0/0)" reads as a catastrophic score for a dimension that was
            # never exercised (a --limit slice) — say what it is instead.
            lines.append(f"  {dim.name:<15} {'·' * 24} (no cases)")
            continue
        fb = dim.meta.get("ner_fallbacks", {})
        fb_note = f"  fallbacks={sum(fb.values())}" if fb else ""
        # A number that mixes deterministic and LLM-assisted checks must say so: the
        # two halves have different meanings (regression vs live model measurement),
        # and a reader subtracting one from the other needs both denominators.
        det = [c.correct for c in dim.checks if c.field.endswith("_deterministic")]
        llm = [c.correct for c in dim.checks if c.field.endswith("_llm")]
        split_note = (f"  [det {sum(det)}/{len(det)} · llm {sum(llm)}/{len(llm)}]"
                      if det and llm else "")
        excluded = dim.meta.get("llm_checks_excluded")
        excl_note = (f"  [llm checks excluded from score: {excluded} — stub run]"
                     if excluded else "")
        lines.append(
            f"  {dim.name:<15} {_bar(dim.accuracy)} "
            f"{dim.accuracy:5.1f}%  ({dim.correct_count}/{dim.total})"
            + (f"  ⚠ {len(dim.errors)} errors" if dim.errors else "")
            + fb_note + split_note + excl_note
        )

    lines.append("-" * 60)
    lines.append(
        f"  {'OVERALL':<15} {_bar(report.accuracy)} "
        f"{report.accuracy:5.1f}%  ({report.correct_count}/{report.total})"
        + ("  [excludes INVALID dims]" if report.invalid_dimensions else "")
    )
    if report.llm_calls:
        lines.append(
            f"  {'tokens':<15} in={report.tokens_in:,}  out={report.tokens_out:,}  "
            f"calls={report.llm_calls}  (× provider price table = $ per sweep)"
        )
    fb_total = report.meta.get("ner_fallbacks_total") or {}
    if fb_total:
        pairs = "  ".join(f"{k}={v}" for k, v in sorted(fb_total.items()))
        lines.append(f"  {'degradation':<15} {pairs}  (raw-output quality — the "
                     f"score cannot see this)")
    for dim in report.dimensions:
        fails = dim.meta.get("tool_failures") or {}
        if fails:
            calls = dim.meta.get("tool_calls", {})
            pairs = "  ".join(f"{t}={n}/{calls.get(t, '?')}" for t, n in fails.items())
            lines.append(f"  {'tool ok=False':<15} [{dim.name}] {pairs}")
    lines.append("═" * 60)

    if show_failures:
        for dim in report.dimensions:
            fails = dim.failures
            if not fails and not dim.errors:
                continue
            lines.append(f"\n  ▼ {dim.name} — {len(fails)} failed checks")
            for c in fails:
                lines.append(
                    f"    ✗ {c.case_id:<28} {c.field:<20} "
                    f"want={c.expected!r} got={c.actual!r}"
                )
            for case_id, msg in dim.errors:
                lines.append(f"    ⚠ {case_id:<28} ERROR {msg}")

    # Calibrate visibility (plan 0.9): the recorded actuals ARE the product of a
    # calibrate run — before this, render showed only failures and calibrate
    # forced everything green, so the actuals were invisible.
    if report.config.get("calibrate"):
        lines.append("\n  ▼ calibrate — recorded actuals (soft checks)")
        for dim in report.dimensions:
            for c in dim.checks:
                if "(soft)" in c.field:
                    lines.append(f"    · {c.case_id:<28} {c.field:<22} "
                                 f"expected={c.expected!r} actual={c.actual!r}")

    lines.append("")
    return "\n".join(lines)


def render_aggregate(aggregate: dict) -> str:
    """Render the --repeat stable-score/noise-floor table (plan 0.2)."""
    lines = ["", "═" * 72,
             "  AGGREGATE over repeats — stable score (majority) + noise floor",
             "═" * 72,
             f"  {'dimension':<16} {'runs':>4} {'stable':>7} {'min':>5} {'max':>5} "
             f"{'unstable':>9} {'floor':>6}"]
    for name, s in aggregate.items():
        lines.append(
            f"  {name:<16} {s['runs']:>4} {s['stable_correct']:>4}/{s['checks']:<4}"
            f"{s['correct_min']:>4} {s['correct_max']:>5} {s['unstable_checks']:>9} "
            f"{s['noise_floor']:>6}"
            + (f"  ⚠ {s['invalid_runs']} invalid runs" if s["invalid_runs"] else "")
        )
    lines.append("-" * 72)
    lines.append("  A difference between two models is REAL only when the paired")
    lines.append("  comparison clears both floors (compare.py) — never read a gap")
    lines.append("  smaller than the floor as a model difference.")
    unstable = [f"{name}: {', '.join(s['unstable_detail'][:4])}"
                + (" …" if len(s["unstable_detail"]) > 4 else "")
                for name, s in aggregate.items() if s["unstable_detail"]]
    if unstable:
        lines.append("\n  ▼ unstable checks (diverged across repeats)")
        for u in unstable:
            lines.append(f"    ~ {u}")
    lines.append("═" * 72)
    lines.append("")
    return "\n".join(lines)
