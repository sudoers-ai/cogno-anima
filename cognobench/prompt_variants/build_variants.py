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


# Compressed FIELD RULES. Every vocabulary enumeration is preserved verbatim —
# tests/unit/test_prompt_variants.py fails the build if any value stops being taught.
# What is cut is prose, not contract:
#   * the pii_risk → severity mapping. `ner.py:555` does `pii_risk = compute_pii_risk(pii)`,
#     so the model's own pii_risk is DISCARDED in-core; teaching the mapping buys nothing.
#     The field stays (the parser reads a shape) but the 5-line table goes.
#   * the `langue` block, which the baseline states twice (header + field rule) to say the
#     same thing: the core assigns it. Once is enough.
#   * restatements of a rule the enumeration on the same line already carries.
# Kept deliberately: the ACTION vs INFORMATION distinction, the "math is NEVER RECENT"
# warning, the entities exclusions, and the raw_* independence rule — all of them encode
# behaviour the bench has caught models getting wrong.
LEAN_FIELD_RULES = """=== FIELD RULES ===

intent_class — EXACTLY one of:
  INFORMATION_REQUEST | ACTION_REQUEST | CLARIFICATION |
  CREATIVE_TASK | SOCIAL | UNKNOWN
  → ACTION_REQUEST: user wants something DONE (compute, execute, create, send,
    translate, schedule, search). Often imperative.
  → INFORMATION_REQUEST: user wants to LEARN or be EXPLAINED. Imperative phrasing
    does NOT make it an ACTION ("me explica o que é X" → INFORMATION_REQUEST).
  → SOCIAL: pure greeting/thanks/farewell, or a short confirmation with no request
    ("oi", "obrigado", "sim", "ok", "claro"). Any question or task → not SOCIAL.
  → UNKNOWN if none fits clearly.

sentiment — EXACTLY one of (judge the ORIGINAL, not the rewrite):
  POSITIVE | NEGATIVE | NEUTRAL | CURIOUS | FRUSTRATED | URGENT | PLAYFUL
  → NEUTRAL if undeterminable.

confidence — float [0.0, 1.0] in your intent_class.

temporal_class — EXACTLY one of: RECENT | HISTORICAL | TIMELESS | MIXED
  → RECENT = time-sensitive TARGET (current prices, weather, news, today/now).
    WARNING: math is NEVER RECENT.
  → HISTORICAL = a specific past event/date/era. TIMELESS = permanent concept,
    definition, theory, mathematical truth. MIXED = several frames in one input.

triad_signal — EXACTLY one of: ID | EGO | SUPEREGO | BALANCED
  → ID = needs planning/decomposition. EGO = concrete action or tool to run.
    SUPEREGO = ethical/tonal/emotional handling. BALANCED = full pipeline.

entities.people      — proper names of REAL individuals. [] if none.
entities.pronouns    — THIRD PERSON / referential pronouns ONLY, in English.
                       EXCLUDE generic I/me/you/we. [] if none.
entities.possessives — my, your, his, her, its, our, their (English). [] if none.
entities.objects     — objects, systems, products, technologies; named entities
                       ("Bitcoin", "Docker"). No generic discourse nouns. Max 5.
entities.concepts    — abstract ideas/theories CENTRAL to the input. No generic
                       discourse words. Max 5.

location — geographic reference, or null.

mandatory_tags — 1 to 3 of: SYSTEM | ANALYSIS | MATH | CREATIVE | LINGUISTIC | UNKNOWN
  → MOST SPECIFIC wins; ANALYSIS is the LAST RESORT. Return the short form.

abstract_tags — 2 to 5 topic tags in UPPERCASE_SNAKE_CASE.

aristotelian — ONLY the relevant categories (not all 10):
  SUBSTANCE | QUANTITY | QUALITY | RELATION | PLACE | TIME | POSITION | STATE | ACTION | PASSION
  → each value: "UPPERCASE_SNAKE_CASE_TAG | short description (max 40 chars)"

goal — the underlying objective (English, max 80 chars). null if none.
causal_chain — ordered cause → context → consequence. Max 4, each max 60 chars.
parole — EXACTLY one of: COLOQUIAL | TECNICO | ACADEMICO | FORMAL | GIRIA | POETICO | MIXED
  → null if truly ambiguous.
langue — ASSIGNED BY THE CORE. Do not detect it; any value you send is ignored.

negation — concepts explicitly negated. English, max 4.
constraints — explicit restrictions on the answer. Max 4.

domains — KNOWLEDGE DOMAINS from this EXACT closed list:
  TECH | SCIENCE | HEALTH | FINANCE | LOGISTICS | TRAVEL |
  HISTORY | LAW | PHILOSOPHY | EDUCATION | CULTURE | NEWS | GENERAL
  → Do NOT invent domains. Use GENERAL if none fits. [] if domain-agnostic.

modality — EXACTLY one of: CERTAIN | PROBABLE | POSSIBLE | UNCERTAIN | MIXED. null if undetectable.
speech_act — EXACTLY one of: DIRECTIVE | EXPRESSIVE | COMMISSIVE | CONSTATIVE | INTERROGATIVE | MIXED
  → null if undetectable.

verbs — main action verbs from ORIGINAL, English infinitive. Max 5.
is_composite — true if MULTIPLE distinct requests.
is_sequential — true if the actions DEPEND on each other chronologically.
context_dependent — true if the query back-references prior context.
comparatives — explicit comparative expressions. Max 4, each max 60 chars.

pii — PII categories present in the ORIGINAL. Use ONLY:
  NATIONAL_ID | TAX_ID | EMAIL | PHONE | CREDIT_CARD | BANK_ACCOUNT |
  DATE_OF_BIRTH | HEALTH_DATA | ADDRESS | IP_ADDRESS | PASSPORT |
  CREDENTIAL | BIOMETRIC | NAME
  → Max 10. [] if no PII.
pii_risk — EXACTLY one of: NONE | LOW | MEDIUM | HIGH | CRITICAL

=== RAW CLASSIFICATION (Session Splitting Support) ===
Also classify the ORIGINAL IN ISOLATION — as if no NOUMENO rewrite and no
conversational context existed. This detects history biasing the main answer.

raw_intent_class — same rules as intent_class, ORIGINAL text only.
  → EXACTLY one of: INFORMATION_REQUEST | ACTION_REQUEST | CLARIFICATION | CREATIVE_TASK | SOCIAL | UNKNOWN
  → UNKNOWN when the ORIGINAL is ambiguous without context.
raw_domains — same closed list above, ORIGINAL text only. [] if too vague.
raw_goal — same rules as goal, ORIGINAL text only. null if none detectable.

IMPORTANT: the raw_* fields MUST be judged independently. Do NOT copy the
contextual values — they may legitimately differ.

=== OUTPUT ===
Return ONLY the JSON object. No markdown, no explanation, no code blocks.
/no_think
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


def _lean_rules(lines: list[str]) -> str:
    """Swap FIELD RULES for the compressed edition, keeping the examples."""
    return "\n".join(lines[:FIELD_RULES_START - 1] + [LEAN_FIELD_RULES])


def _lean_rules_one_example(lines: list[str]) -> str:
    kept = _drop(lines, EX2, EX3).split("\n")
    # _drop shifts the tail up, so re-find the boundary by anchor instead of index.
    cut = next(i for i, l in enumerate(kept) if l.startswith("=== FIELD RULES ==="))
    return "\n".join(kept[:cut] + [LEAN_FIELD_RULES])


VARIANTS = {
    # ── examples axis (FIELD RULES untouched) ──
    # 3 examples → 2: is the contamination example carrying its weight?
    "v1-no-ex3": lambda L: _drop(L, EX3),
    # → 1: does a single example anchor the shape well enough?
    "v2-one-example": lambda L: _drop(L, EX2, EX3),
    # → 0 worked examples, shape preserved: do the examples exist for FORMAT or CONTENT?
    "v3-schema-only": _replace_examples_with_skeleton,
    # ── rules axis (examples untouched) — isolates the FIELD RULES cut ──
    "v4-lean-rules": _lean_rules,
    # ── both: the smallest prompt that still teaches the whole contract ──
    "v5-min": _lean_rules_one_example,
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
