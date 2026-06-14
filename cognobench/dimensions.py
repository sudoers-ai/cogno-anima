"""
Dimension runners — execute cases through the reference pipeline and score them.

Adapted from the parent Cogno `eval_ner` mixin, but decoupled: no PipelineRunner,
no SkillRegistry, no infra. Scoring targets cogno-core's `IntentResult`,
`NoumenoResult` and `DriftMetrics` contracts directly.
"""

from __future__ import annotations

from cognobench.harness import CognitivePipeline
from cognobench.types import CheckResult, DimensionResult
from cognobench.ner_cases import NERCase
from cognobench.drift_cases import DriftCase, VALID_ACTIONS
from cognobench.noumeno_cases import NoumenoCase, VALID_DRIFT_TAGS


def _lang_prefix(value: str) -> str:
    return (value or "").lower().split("-")[0]


def _language_check(field_name: str, actual: str, case_expect: str, forced: str | None):
    """Score language as PROPAGATION when host-provided, else as DETECTION.

    With a tenant/host language (the SaaS default — currently pt-BR for all),
    `force_language` is set, so we verify the language *propagates* unchanged
    through the stages rather than testing langdetect (flaky on short text).
    """
    if forced:
        ok = _lang_prefix(actual) == _lang_prefix(forced)
        return (f"{field_name}_propagated", forced, actual or "", ok)
    if case_expect:
        ok = _lang_prefix(actual) == case_expect.lower()
        return (field_name, case_expect, actual or "", ok)
    return None


# ──────────────────────────────────────────────────────────────────────────
#  NOUMENO
# ──────────────────────────────────────────────────────────────────────────

async def run_noumeno(
    pipe: CognitivePipeline, cases: list[NoumenoCase], language: str | None = None,
) -> DimensionResult:
    dim = DimensionResult(name="noumeno")
    for case in cases:
        try:
            ctx = await pipe.run(case.input, force_language=language, stop_after="noumeno")
            n = ctx.noumeno
            checks: list[tuple[str, str, str, bool]] = []

            checks.append(("rewrite_nonempty", "non-empty", n.rewritten[:30],
                           bool(n.rewritten.strip())))
            checks.append(("drift_tag_valid", "valid", n.drift_tag,
                           n.drift_tag in VALID_DRIFT_TAGS))

            lang_check = _language_check("language", n.language, case.expect_language, language)
            if lang_check:
                checks.append(lang_check)
            if case.expect_changed is not None:
                checks.append(("changed", str(case.expect_changed), str(n.changed),
                               n.changed == case.expect_changed))

            for field, expected, actual, correct in checks:
                dim.checks.append(CheckResult(case.id, field, expected, actual, correct))
        except Exception as exc:  # noqa: BLE001
            dim.errors.append((case.id, repr(exc)))
    return dim


# ──────────────────────────────────────────────────────────────────────────
#  NER
# ──────────────────────────────────────────────────────────────────────────

async def run_ner(
    pipe: CognitivePipeline, cases: list[NERCase], language: str | None = None,
) -> DimensionResult:
    dim = DimensionResult(name="ner")
    for case in cases:
        try:
            ctx = await pipe.run(case.input, force_language=language, stop_after="ner")
            intent = ctx.intent
            if intent is None:
                dim.errors.append((case.id, "intent is None"))
                continue

            checks: list[tuple[str, str, str, bool]] = []

            if case.expect_intent:
                a = (intent.intent_class or "").upper()
                checks.append(("intent_class", case.expect_intent, a, a == case.expect_intent.upper()))
            if case.expect_sentiment:
                a = (intent.sentiment or "").upper()
                checks.append(("sentiment", case.expect_sentiment, a, a == case.expect_sentiment.upper()))
            if case.expect_temporal:
                a = (intent.temporal_class or "").upper()
                checks.append(("temporal", case.expect_temporal, a, a == case.expect_temporal.upper()))
            lang_check = _language_check("langue", intent.langue or "", case.expect_language, language)
            if lang_check:
                checks.append(lang_check)
            if case.expect_pii_risk:
                a = (intent.pii_risk or "NONE").upper()
                checks.append(("pii_risk", case.expect_pii_risk, a, a == case.expect_pii_risk.upper()))
            if case.expect_speech_act:
                a = (intent.speech_act or "").upper()
                checks.append(("speech_act", case.expect_speech_act, a, a == case.expect_speech_act.upper()))
            if case.expect_modality:
                a = (intent.modality or "").upper()
                checks.append(("modality", case.expect_modality, a, a == case.expect_modality.upper()))
            if case.expect_parole:
                a = (intent.parole or "").upper()
                checks.append(("parole", case.expect_parole, a, a == case.expect_parole.upper()))
            if case.expect_is_composite is not None:
                checks.append(("is_composite", str(case.expect_is_composite),
                               str(intent.is_composite), intent.is_composite == case.expect_is_composite))

            # Entities (substring match against people/concepts/objects/location)
            if case.expect_entities:
                pool = [e.lower() for e in (
                    list(intent.entities_people or [])
                    + list(intent.entities_concepts or [])
                    + list(intent.entities_objects or [])
                    + ([intent.location] if intent.location else [])
                ) if e]
                for want in case.expect_entities:
                    found = any(want.lower() in e for e in pool)
                    checks.append(("entity", want, str(pool[:4]), found))

            if case.expect_verbs:
                verbs = [v.lower() for v in (intent.verbs or [])]
                for want in case.expect_verbs:
                    checks.append(("verb", want, str(verbs[:5]),
                                   any(want.lower() in v for v in verbs)))

            if case.expect_comparatives:
                comps = " ".join(intent.comparatives or []).lower()
                for want in case.expect_comparatives:
                    checks.append(("comparative", want, str(intent.comparatives or []),
                                   want.lower() in comps))

            if case.expect_negation:
                negs = " ".join(intent.negation or []).lower()
                for want in case.expect_negation:
                    checks.append(("negation", want, str(intent.negation or []),
                                   want.lower() in negs))

            for field, expected, actual, correct in checks:
                dim.checks.append(CheckResult(case.id, field, expected, actual, correct))
        except Exception as exc:  # noqa: BLE001
            dim.errors.append((case.id, repr(exc)))
    return dim


# ──────────────────────────────────────────────────────────────────────────
#  DRIFT
# ──────────────────────────────────────────────────────────────────────────

async def run_drift(
    pipe: CognitivePipeline, cases: list[DriftCase], calibrate: bool = False,
    language: str | None = None,
) -> DimensionResult:
    dim = DimensionResult(name="drift")
    for case in cases:
        try:
            ctx = await pipe.run(case.input, history=case.history,
                                 force_language=language, stop_after="drift")
            d = ctx.drift
            cum = d.cumulative_drift

            # Hard invariants (always checked)
            dim.checks.append(CheckResult(case.id, "action_valid", "in set",
                                          d.drift_action, d.drift_action in VALID_ACTIONS))
            dim.checks.append(CheckResult(case.id, "cumulative_range", "[0,1]",
                                          f"{cum:.3f}", 0.0 <= cum <= 1.0))

            # Soft band (skipped in calibrate mode — just records the actual)
            in_band = case.min_cumulative <= cum <= case.max_cumulative
            dim.checks.append(CheckResult(
                case.id, "cumulative_band(soft)",
                f"[{case.min_cumulative:.2f},{case.max_cumulative:.2f}]",
                f"{cum:.3f}", True if calibrate else in_band))
        except Exception as exc:  # noqa: BLE001
            dim.errors.append((case.id, repr(exc)))
    return dim
