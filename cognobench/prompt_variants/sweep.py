#!/usr/bin/env python3
"""Run the prompt-variant × model matrix and append every run to a ledger.

Each run is an independent ``cognobench.py`` process, so runs parallelise freely;
the ledger (``prompt_sweep.jsonl``) is append-only and every line is one run. The
ledger — not the terminal output — is the experiment's record: a score with no
recorded (variant, model, prompt tokens, cost) beside it cannot be compared later.

    export $(grep -E '^OPENAI_API_KEY=' ../cogno-host/.env | xargs)
    python3 cognobench/prompt_variants/sweep.py --models openai:gpt-4o-mini --reps 2
    python3 cognobench/prompt_variants/sweep.py --report        # summarise the ledger

Cloud runs are NOT deterministic: two identical runs drift by a check or two, so
the matrix takes ``--reps`` and the report shows the mean. A one-check gap between
variants is noise, not a result.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
LEDGER = ROOT / "prompt_sweep.jsonl"
BASELINE = REPO / "cogno_anima" / "prompt_templates"


def _prompt_tokens(tree: Path) -> int:
    """Tokens in the tree's NER system prompt — the quantity under test."""
    try:
        import tiktoken
    except ImportError:
        return -1
    text = (tree / "ner" / "system.txt").read_text(encoding="utf-8")
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def _cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    """Price via cogno-meter's PriceBook so the sweep and production bill agree."""
    try:
        from cogno_meter.pricing import PriceBook
    except ImportError:
        return -1.0
    return PriceBook.default().llm_cost_usd(model, tokens_in, tokens_out)


def _trees(names: list[str] | None) -> list[tuple[str, Path | None]]:
    """(label, prompts_dir) pairs; None means the shipped baseline templates."""
    out: list[tuple[str, Path | None]] = [("default", None)]
    found = sorted(p for p in ROOT.iterdir()
                   if p.is_dir() and (p / "ner" / "system.txt").exists())
    for tree in found:
        if names is None or tree.name in names:
            out.append((tree.name, tree))
    return out


def _run_one(dimension: str, model: str, label: str, tree: Path | None,
             rep: int, limit: int | None) -> dict | None:
    cmd = [sys.executable, str(REPO / "cognobench.py"),
           "--only", dimension, "--model", model, "--json", "--no-failures"]
    if tree is not None:
        cmd += ["--prompts-dir", str(tree)]
    if limit:
        cmd += ["--limit", str(limit)]

    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"✗ {label:16s} {model:22s} rep{rep} FAILED: "
              f"{proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else 'no stderr'}",
              file=sys.stderr)
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(f"✗ {label:16s} {model:22s} rep{rep}: unparseable output", file=sys.stderr)
        return None

    row = {
        "dimension": dimension, "variant": label, "model": model, "rep": rep,
        "limit": limit,
        "accuracy": data["overall_accuracy"], "correct": data["correct"],
        "total": data["total"],
        "tokens_in": data["tokens_in"], "tokens_out": data["tokens_out"],
        "llm_calls": data["llm_calls"],
        "prompt_tokens": _prompt_tokens(tree or BASELINE),
        "cost_usd": round(_cost_usd(model, data["tokens_in"], data["tokens_out"]), 5),
    }
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  {label:16s} {model:22s} rep{rep}  "
          f"{row['accuracy']:5.1f}%  {row['correct']:3d}/{row['total']:<3d} "
          f"{row['prompt_tokens']:5d} tok  ${row['cost_usd']:.4f}")
    return row


def report() -> int:
    if not LEDGER.exists():
        print("no ledger yet — run the sweep first", file=sys.stderr)
        return 1
    rows = [json.loads(l) for l in LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        print("ledger is empty", file=sys.stderr)
        return 1

    grouped: dict = defaultdict(list)
    for r in rows:
        grouped[(r["dimension"], r["model"], r["limit"], r["variant"])].append(r)

    baselines = {k[:3]: g for k, g in grouped.items() if k[3] == "default"}

    print(f"\n{'dimension':<12} {'model':<22} {'variant':<16} {'runs':>4} "
          f"{'acc%':>7} {'Δacc':>6} {'p.tok':>6} {'Δtok':>7} {'$/run':>8}")
    print("─" * 96)
    for key in sorted(grouped, key=lambda k: (k[0], k[1], str(k[2]), k[3] != "default", k[3])):
        dim, model, limit, variant = key
        g = grouped[key]
        acc = sum(r["accuracy"] for r in g) / len(g)
        ptok = g[0]["prompt_tokens"]
        cost = sum(r["cost_usd"] for r in g) / len(g)

        base = baselines.get((dim, model, limit))
        if base and variant != "default":
            b_acc = sum(r["accuracy"] for r in base) / len(base)
            d_acc, d_tok = f"{acc - b_acc:+.1f}", f"{100 * ptok // base[0]['prompt_tokens'] - 100:+d}%"
        else:
            d_acc, d_tok = "—", "—"
        suffix = f" (limit {limit})" if limit else ""
        print(f"{dim + suffix:<12} {model:<22} {variant:<16} {len(g):>4} "
              f"{acc:>7.1f} {d_acc:>6} {ptok:>6} {d_tok:>7} {cost:>8.4f}")

    print(f"\ntotal spend across the ledger: ${sum(r['cost_usd'] for r in rows):.4f} "
          f"over {len(rows)} runs")
    print("Cloud runs are noisy — treat a Δacc within ±1 check as no difference.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Prompt-variant sweep over the cognitive bench")
    p.add_argument("--models", nargs="+", default=["openai:gpt-4o-mini"])
    p.add_argument("--variants", nargs="+", default=None,
                   help="variant directory names; default = every generated tree + baseline")
    p.add_argument("--dimension", default="ner")
    p.add_argument("--reps", type=int, default=2,
                   help="repeats per cell — cloud is nondeterministic, 2 is the floor")
    p.add_argument("--limit", type=int, default=None, help="cap cases (screening pass)")
    p.add_argument("--jobs", "-j", type=int, default=4, help="parallel runs")
    p.add_argument("--report", action="store_true", help="summarise the ledger and exit")
    args = p.parse_args()

    if args.report:
        return report()

    trees = _trees(args.variants)
    cells = [(args.dimension, m, label, tree, rep, args.limit)
             for m in args.models for label, tree in trees
             for rep in range(1, args.reps + 1)]
    print(f"{len(cells)} runs: {len(trees)} variants × {len(args.models)} models "
          f"× {args.reps} reps  (jobs={args.jobs}, limit={args.limit or 'full'})")

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda c: _run_one(*c), cells))

    failed = sum(1 for r in results if r is None)
    if failed:
        print(f"\n⚠ {failed}/{len(cells)} runs failed — the ledger is incomplete.",
              file=sys.stderr)
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
