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
    lines.append(f"  CognoBench (cogno-core)   model={report.model or 'n/a'}")
    lines.append("═" * 60)

    for dim in report.dimensions:
        lines.append(
            f"  {dim.name:<9} {_bar(dim.accuracy)} "
            f"{dim.accuracy:5.1f}%  ({dim.correct_count}/{dim.total})"
            + (f"  ⚠ {len(dim.errors)} errors" if dim.errors else "")
        )

    lines.append("-" * 60)
    lines.append(
        f"  {'OVERALL':<9} {_bar(report.accuracy)} "
        f"{report.accuracy:5.1f}%  ({report.correct_count}/{report.total})"
    )
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

    lines.append("")
    return "\n".join(lines)
