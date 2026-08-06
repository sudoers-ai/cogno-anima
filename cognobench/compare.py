"""compare.py — reads persisted run JSONs (--out) and answers the two questions
a single run cannot (plan 0.2 / 0.7):

* **Is model A really better than B?** Paired per-check comparison: with n=3
  and sd≈1.5 checks, aggregate gaps under ~4/125 are indistinguishable — but the
  noise concentrates in a small flaky fringe, so pairing per check recovers the
  power. A > B only when b−c > 2·√(b+c) (≈McNemar, p≈0.05) AND b−c clears both
  models' noise floors.
* **Which checks discriminate?** Per (dimension, case_id, field) across models:
  *discriminator* = ≥1 model stable-fails AND ≥1 stable-passes (what selection
  ranks on); *guard* = everyone stable-passes (regression net — an EGO-style
  100%-everywhere dimension is saturated and cannot pick a model); *suspect* =
  everyone stable-fails (instrument defect until proven otherwise → quarantine).

Only runs whose suite_id+suite_hash match are ever joined — different suite
versions never share a table (plan principle 4).

Usage:
    python -m cognobench.compare results/ --pair modelA modelB
    python -m cognobench.compare results/ --discriminate
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def load_runs(paths: list[Path]) -> list[dict]:
    runs = []
    for p in paths:
        files = sorted(p.glob("*.json")) if p.is_dir() else [p]
        for f in files:
            try:
                runs.append(json.loads(f.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  ! skipping {f}: {exc}", file=sys.stderr)
    return runs


def _stable(results: list[bool]) -> bool | None:
    """Majority verdict across repeats; None for an empty list."""
    if not results:
        return None
    return sum(results) * 2 >= len(results)


def _index(runs: list[dict]) -> dict[str, dict]:
    """model → {suites, dims: {dim → {(case_id, field) → [bool per run]}}}.

    Skips invalid dimensions (they never scored) and mutate runs (sabotage
    artifacts are not model measurements)."""
    by_model: dict[str, dict] = {}
    for run in runs:
        if run.get("config", {}).get("mutate"):
            continue
        model = run.get("model", "?")
        slot = by_model.setdefault(model, {"suites": {}, "dims": defaultdict(dict)})
        for dim in run.get("dimensions", []):
            if not dim.get("valid", True) or "checks" not in dim:
                continue
            name = dim["dimension"]
            suite = run.get("suites", {}).get(name.split("_")[0] if
                                              name.startswith("superego") else name)
            if suite:
                pin = (suite["suite_id"], suite["suite_hash"])
                prev = slot["suites"].setdefault(name, pin)
                if prev != pin:
                    print(f"  ! {model}/{name}: mixed suite versions {prev} vs {pin} "
                          f"— refusing to join (re-baseline instead)", file=sys.stderr)
                    continue
            for c in dim["checks"]:
                slot["dims"][name].setdefault((c["case_id"], c["field"]), []) \
                    .append(bool(c["correct"]))
    return by_model


def _floors(model_slot: dict) -> dict[str, int]:
    floors = {}
    for dim, checks in model_slot["dims"].items():
        unstable = sum(1 for v in checks.values() if len(set(v)) > 1)
        floors[dim] = max(unstable, 2)
    return floors


def pair(by_model: dict, a: str, b: str) -> int:
    """Paired comparison per dimension. Exit 0 always (informational)."""
    sa, sb = by_model.get(a), by_model.get(b)
    if not sa or not sb:
        print(f"models found: {sorted(by_model)}", file=sys.stderr)
        return 2
    fa, fb = _floors(sa), _floors(sb)
    print(f"\n  PAIRED  {a}  vs  {b}")
    print(f"  {'dimension':<16} {'A>B':>4} {'B>A':>4} {'floorA':>7} {'floorB':>7}  verdict")
    for dim in sorted(set(sa["dims"]) & set(sb["dims"])):
        if sa["suites"].get(dim) and sb["suites"].get(dim) \
                and sa["suites"][dim] != sb["suites"][dim]:
            print(f"  {dim:<16} suite versions differ — not comparable")
            continue
        ca, cb = sa["dims"][dim], sb["dims"][dim]
        common = set(ca) & set(cb)
        bwins = cwins = 0
        for key in common:
            va, vb = _stable(ca[key]), _stable(cb[key])
            if va is True and vb is False:
                bwins += 1          # b in McNemar's notation: A passes, B fails
            elif va is False and vb is True:
                cwins += 1
        diff = bwins - cwins
        floor = max(fa.get(dim, 2), fb.get(dim, 2))
        significant = abs(diff) > 2 * math.sqrt(bwins + cwins) if (bwins + cwins) else False
        clears_floor = abs(diff) > floor
        if significant and clears_floor:
            verdict = f"{a if diff > 0 else b} WINS"
        else:
            verdict = "TIE (decide by cost / error direction, or escalate BOTH to n=10)"
        print(f"  {dim:<16} {bwins:>4} {cwins:>4} {fa.get(dim, 2):>7} {fb.get(dim, 2):>7}  {verdict}")
    return 0


def discriminate(by_model: dict) -> int:
    """Tag every check across the model matrix; flag saturated dimensions."""
    checks: dict[tuple[str, str, str], dict[str, bool]] = defaultdict(dict)
    for model, slot in by_model.items():
        for dim, dchecks in slot["dims"].items():
            for key, results in dchecks.items():
                v = _stable(results)
                if v is not None:
                    checks[(dim, *key)][model] = v

    tags: dict[str, list] = {"discriminator": [], "guard": [], "suspect": []}
    for key, verdicts in checks.items():
        vals = set(verdicts.values())
        if vals == {True}:
            tags["guard"].append(key)
        elif vals == {False}:
            tags["suspect"].append(key)
        else:
            tags["discriminator"].append(key)

    n_models = len(by_model)
    print(f"\n  DISCRIMINATION over {n_models} models "
          f"({', '.join(sorted(by_model))})")
    for tag, items in tags.items():
        print(f"\n  ▼ {tag} ({len(items)})")
        for dim, case_id, fld in sorted(items)[:40]:
            print(f"    {dim:<16} {case_id:<30} {fld}")
        if len(items) > 40:
            print(f"    … +{len(items) - 40} more")

    # Saturation alarm (the EGO-100%-across-9-models case): a dimension with
    # zero discriminators cannot rank models — it is a qualification floor.
    by_dim: dict[str, dict[str, int]] = defaultdict(lambda: {"d": 0, "g": 0, "s": 0})
    for (dim, _, _f) in tags["discriminator"]:
        by_dim[dim]["d"] += 1
    for (dim, _, _f) in tags["guard"]:
        by_dim[dim]["g"] += 1
    for (dim, _, _f) in tags["suspect"]:
        by_dim[dim]["s"] += 1
    print("\n  ▼ per-dimension summary (saturated = 0 discriminators)")
    for dim, c in sorted(by_dim.items()):
        note = "  ⚠ SATURATED — cannot rank models" if c["d"] == 0 else ""
        note += "  ⚠ has suspects (quarantine: everyone stable-fails)" if c["s"] else ""
        print(f"    {dim:<16} disc={c['d']:<4} guard={c['g']:<4} suspect={c['s']}{note}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cognobench.compare")
    ap.add_argument("paths", nargs="+", type=Path,
                    help="results dirs or run JSONs (from --out)")
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"),
                    help="paired per-check comparison between two model labels")
    ap.add_argument("--discriminate", action="store_true",
                    help="tag guard/discriminator/suspect across all models found")
    args = ap.parse_args(argv)

    by_model = _index(load_runs(args.paths))
    if not by_model:
        print("no valid runs found", file=sys.stderr)
        return 2
    if args.pair:
        return pair(by_model, *args.pair)
    if args.discriminate:
        return discriminate(by_model)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
