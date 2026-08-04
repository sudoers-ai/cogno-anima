#!/usr/bin/env python3
"""Generate the NER prompt-variant trees for the slimming experiment.

Regenerating from the shipped templates (rather than hand-editing each tree) keeps
every variant a pure, reproducible function of the baseline: when the baseline
prompt changes, rerun this and the variants follow instead of silently rotting.

Each tree is a full mirror of ``cogno_anima/prompt_templates`` so a run is
reproducible from one ``--prompts-dir`` path; only ``ner/system.txt`` differs.

    python3 cognobench/prompt_variants/build_variants.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent.parent / "cogno_anima" / "prompt_templates"

# 1-indexed, inclusive line spans of the baseline ner/system.txt. Asserted below
# against anchor text so a baseline edit fails loudly here instead of silently
# slicing the wrong block into a variant.
EXAMPLES_HEADER = (22, 23)
EX1 = (24, 69)
EX2 = (70, 114)
EX3 = (115, 157)
FIELD_RULES_START = 158

_ANCHORS = {
    22: "=== EXAMPLES ===",
    24: "--- Example 1:",
    70: "--- Example 2:",
    115: "--- Example 3:",
    158: "=== FIELD RULES ===",
}

# The output contract lives ONLY inside the worked examples: ner/user.txt asks for
# "one valid JSON object with all NER fields" without ever showing the envelope,
# and FIELD RULES documents each field's vocabulary but not the nesting. So a
# no-examples variant would be testing whether the model can guess the schema, not
# whether the worked content earns its tokens. This skeleton isolates the real
# question: keep the shape, drop the content.
SCHEMA_SKELETON = """--- Output schema (shape only — fill every field) ---
{
  "intent_class": "", "sentiment": "", "confidence": 0.0,
  "temporal_class": "", "triad_signal": "",
  "entities": {"people": [], "pronouns": [], "possessives": [],
               "objects": [], "concepts": []},
  "location": null,
  "mandatory_tags": [], "abstract_tags": [],
  "aristotelian": {"CATEGORY": "TAG | short description"},
  "goal": "", "causal_chain": [], "parole": "",
  "negation": [], "constraints": [], "domains": [],
  "modality": "", "speech_act": "",
  "is_composite": false, "is_sequential": false,
  "verbs": [], "context_dependent": false, "comparatives": [],
  "pii": [], "pii_risk": "",
  "raw_intent_class": "", "raw_domains": [], "raw_goal": ""
}
"""


def _check_anchors(lines: list[str]) -> None:
    for lineno, expected in _ANCHORS.items():
        actual = lines[lineno - 1]
        if not actual.startswith(expected):
            sys.exit(f"✗ baseline ner/system.txt changed: line {lineno} is {actual!r}, "
                     f"expected it to start with {expected!r}. Update the spans in "
                     f"{Path(__file__).name} before regenerating.")


def _drop(lines: list[str], *spans: tuple[int, int]) -> str:
    """Return the text with the given 1-indexed inclusive line spans removed."""
    drop = {n for start, end in spans for n in range(start, end + 1)}
    return "\n".join(l for i, l in enumerate(lines, start=1) if i not in drop)


def _replace_examples_with_skeleton(lines: list[str]) -> str:
    head = lines[:EXAMPLES_HEADER[0] - 1]              # everything before === EXAMPLES ===
    tail = lines[FIELD_RULES_START - 1:]               # === FIELD RULES === onward
    return "\n".join(head + [SCHEMA_SKELETON, ""] + tail)


VARIANTS = {
    # 3 examples → 2: is the contamination example carrying its weight?
    "v1-no-ex3": lambda L: _drop(L, EX3),
    # → 1: does a single example anchor the shape well enough?
    "v2-one-example": lambda L: _drop(L, EX2, EX3),
    # → 0 worked examples, shape preserved: do the examples exist for FORMAT or CONTENT?
    "v3-schema-only": _replace_examples_with_skeleton,
}


def main() -> int:
    baseline_ner = (BASELINE / "ner" / "system.txt").read_text(encoding="utf-8")
    lines = baseline_ner.split("\n")
    _check_anchors(lines)

    for name, transform in VARIANTS.items():
        tree = ROOT / name
        if tree.exists():
            shutil.rmtree(tree)
        shutil.copytree(BASELINE, tree, ignore=shutil.ignore_patterns("__pycache__"))
        (tree / "ner" / "system.txt").write_text(transform(lines), encoding="utf-8")

        chars = len((tree / "ner" / "system.txt").read_text(encoding="utf-8"))
        base = len(baseline_ner)
        print(f"{name:20s} {chars:6d} B  ({chars - base:+6d}, {100 * chars // base:3d}% of baseline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
