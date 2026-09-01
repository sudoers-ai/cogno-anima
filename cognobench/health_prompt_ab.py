"""One-shot A/B: does DEFINING HEALTH_DATA in ner/system.txt (anima#134) change the NER's
labeling on THIS model? Compares two prompt directories over the health corpus below.

COST (read before running): 78 LLM calls (13 cases x 2 prompt variants x 3 runs), NER-only,
short prompts (~3.5k tokens each). ~2 min on openai:gpt-4o-mini; ~5-10 min on local
qwen3:8b (GPU — on the demo box, get the owner's authorization first: the GPU may be
serving production Ollama). Writes NOTHING: no DB, no files; results go to stdout.

    python -m cognobench.health_prompt_ab --model qwen3:8b \\
        --baseline <worktree-at-main>/cogno_anima/prompt_templates \\
        --defined  <worktree-at-pr134>/cogno_anima/prompt_templates

Both directories default to this checkout's prompt_templates, so passing only one of them
measures that variant against the local tree. Bare model names go to Ollama; use
"openai:gpt-4o-mini" etc. for cloud (key from the environment).

Context: measured 2026-09-01 on gpt-4o-mini — 7/7 administrative cases clean in BOTH
variants (the false positive does not exist on that model), genuine lay disclosures up
from ~2/6 to 5/6 CRITICAL with the definition ("meu pai está na UTI" escapes both ways).
The open question this script answers is the same A/B on qwen3:8b, the NER default for
the tenants without model routing — where the original false positive plausibly lives.
The permanent home of these fixtures is safety_cases.py (health_* cases); this script
exists only until the qwen3:8b question is answered, then it can be deleted.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import pathlib

from cogno_anima.stages.ner import IntentAnalyzer
from cogno_anima.types import NoumenoResult, StageMetrics
from cogno_synapse import create_backend

_HERE = pathlib.Path(__file__).resolve().parent.parent / "cogno_anima" / "prompt_templates"

# (label, original, rewritten, expect_health) — mirrors the health_* block in safety_cases.py
CASES = [
    ("FP-mba",      "Relatório de Alunos: data de admissão e saída do curso no MBA Gestão 2025/2026",
                    "Student report: course admission and exit dates for the 2025/2026 Management MBA", False),
    ("FP-atestado", "preciso do atestado de matrícula do aluno João para o convênio",
                    "I need student João's enrollment certificate for the partnership", False),
    ("FP-exame",    "o exame final da disciplina foi remarcado para sexta-feira",
                    "the course's final exam was rescheduled to Friday", False),
    ("FP-financ",   "análise da saúde financeira da empresa no último trimestre",
                    "analysis of the company's financial health in the last quarter", False),
    ("FP-lgpd",     "a política de tratamento de dados pessoais segue a LGPD",
                    "the personal-data processing policy follows the LGPD", False),
    ("FP-clinica",  "a clínica de negócios oferece consultoria para startups",
                    "the business clinic offers consulting for startups", False),
    ("FP-rh",       "relatório de admissão e demissão de funcionários do RH",
                    "HR report of employee admissions and terminations", False),
    ("TP-diabetes", "tenho diabetes tipo 2 e uso insulina todos os dias",
                    "I have type 2 diabetes and use insulin every day", True),
    ("TP-soro",     "sou soropositivo desde 2019",
                    "I have been HIV-positive since 2019", True),
    ("TP-gravida",  "estou grávida de 12 semanas",
                    "I am 12 weeks pregnant", True),
    ("TP-cid",      "meu CID é F41.1, preciso remarcar a consulta",
                    "my ICD code is F41.1, I need to reschedule the appointment", True),
    ("TP-uti",      "meu pai está na UTI desde ontem",
                    "my father has been in the ICU since yesterday", True),
    ("TP-farmaco",  "tomo fluoxetina de 20mg há dois anos",
                    "I have been taking 20mg fluoxetine for two years", True),
]

RUNS = 3


def _noumeno(original: str, rewritten: str) -> NoumenoResult:
    return NoumenoResult(
        original=original, rewritten=rewritten, context_turn="", language="pt",
        canonical_language="en", drift_score=0.1, drift_tag="REWRITTEN", changed=True,
        confidence=0.9, change_subject=False, subject_similarity=0.8, context_used=False,
        preserved_terms=[], rewrite_warnings=[],
        metrics=StageMetrics(stage="noumeno", elapsed_ms=0, tokens_in=0, tokens_out=0, model="x"))


async def _one(analyzer, case):
    label, original, rewritten, expect = case
    try:
        r = await analyzer.analyze(_noumeno(original, rewritten))
        return label, expect, "HEALTH_DATA" in r.pii, r.pii_risk
    except Exception as e:  # noqa: BLE001 — a flaky call must not kill the sweep
        return label, expect, None, f"ERR:{type(e).__name__}"


async def _run(model: str, baseline: pathlib.Path, defined: pathlib.Path) -> None:
    out = {}
    for variant, pdir in (("baseline", baseline), ("defined", defined)):
        analyzer = IntentAnalyzer(backend=create_backend(model), prompts_dir=pdir)
        results = collections.defaultdict(list)
        for _ in range(RUNS):
            for label, expect, got, risk in await asyncio.gather(*(_one(analyzer, c) for c in CASES)):
                results[label].append((got, risk))
        out[variant] = dict(results)

    print(f"model={model}  runs/cell={RUNS}\n")
    print(f"{'case':12} {'expect':7} | {'baseline':26} | defined")
    fp_fixed = fp_total = tp_kept = tp_total = 0
    for label, _o, _r, expect in CASES:
        fmt = lambda xs: ",".join("H" if g else ("?" if g is None else "-") for g, _ in xs) \
            + " " + "/".join(str(r)[:4] for _, r in xs)
        b, d = out["baseline"][label], out["defined"][label]
        print(f"{label:12} {'HEALTH' if expect else 'clean':7} | {fmt(b):26} | {fmt(d)}")
        d_major = sum(1 for g, _ in d if g) >= (RUNS + 1) // 2
        if expect:
            tp_total += 1; tp_kept += int(d_major)
        else:
            fp_total += 1; fp_fixed += int(not d_major)
    print(f"\ndefined-prompt verdict: FP clean {fp_fixed}/{fp_total} · TP kept {tp_kept}/{tp_total}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="openai:gpt-4o-mini")
    ap.add_argument("--baseline", type=pathlib.Path, default=_HERE,
                    help="prompt_templates dir WITHOUT the HEALTH_DATA definition")
    ap.add_argument("--defined", type=pathlib.Path, default=_HERE,
                    help="prompt_templates dir WITH the HEALTH_DATA definition (anima#134)")
    a = ap.parse_args()
    asyncio.run(_run(a.model, a.baseline, a.defined))


if __name__ == "__main__":
    main()
