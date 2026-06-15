#!/usr/bin/env python3
"""Run the drift cases through the real pipeline and print per-case actuals.

Used to recalibrate the soft cumulative bands in cognobench/drift_cases.py
against cogno-core's embedding-based drift (the original bands came from the
parent's heuristic drift model and do not transfer).
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognobench.harness import CognitivePipeline, build_ollama, ollama_available
from cognobench.drift_cases import DRIFT_CASES

LANGUAGE = "pt-BR"


async def main() -> None:
    if not await ollama_available():
        raise SystemExit("Ollama not reachable at localhost:11434")
    backend, embedder = build_ollama("llama3.1:8b")
    pipe = CognitivePipeline(backend, embedder)

    print(f"{'case_id':<34} {'cumul':>6} {'action':>13} {'epist':>6} {'onto':>6}")
    print("-" * 74)
    rows = []
    for case in DRIFT_CASES:
        ctx = await pipe.run(case.input, history=case.history,
                             force_language=LANGUAGE, stop_after="drift")
        d = ctx.drift
        onto = d.ontological_drift
        rows.append((case.id, d.cumulative_drift, d.drift_action, d.drift_score, onto))
        onto_str = f"{onto:>6.3f}" if onto is not None else "  None"
        print(f"{case.id:<34} {d.cumulative_drift:>6.3f} {d.drift_action:>13} "
              f"{d.drift_score:>6.3f} {onto_str}")

    cums = [r[1] for r in rows]
    print("-" * 74)
    print(f"min={min(cums):.3f}  max={max(cums):.3f}  "
          f"avg={sum(cums)/len(cums):.3f}  n={len(cums)}")


if __name__ == "__main__":
    asyncio.run(main())
