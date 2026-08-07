"""
Dimension runners — execute cases through the reference pipeline and score them.

Adapted from the parent Cogno `eval_ner` mixin, but decoupled: no PipelineRunner,
no SkillRegistry, no infra. Scoring targets cogno-anima's `IntentResult`,
`NoumenoResult` and `DriftMetrics` contracts directly.
"""

from __future__ import annotations

import re

import httpx
from pydantic import ValidationError

from cognobench.harness import CognitivePipeline
from cognobench.types import CheckResult, DimensionResult
from cogno_anima.errors import StageParseError
from cogno_synapse.errors import SynapseError
from cognobench.ner_cases import NERCase
from cognobench.drift_cases import DriftCase, VALID_ACTIONS
from cognobench.noumeno_cases import NoumenoCase, VALID_DRIFT_TAGS
from cognobench.id_cases import IdCase, VALID_GOAL_STATUS, VALID_ROUTES
from cognobench.ego_cases import (
    EgoCase, BenchDispatcher, EGO_SYSTEM, VALID_TOOLS, SIDE_EFFECT_TOOLS,
)
from cognobench.safety_cases import SafetyCase
from cognobench.superego_cases import SuperegoCase
from cognobench.conversation_cases import (
    ConversationCase, BenchDispatcher as ConvDispatcher, INHERIT_LANGUAGE,
    EGO_PROMPT, LIMITS_PROMPT, VOICE_PROMPT, VALID_TOOLS as CONV_TOOLS,
)
from cognobench.harness import PROMPTS_DIR, SLANGS
from cognobench.pipeline import ReferencePipeline

from cogno_synapse import LLMBackend, Embedder
from cogno_anima.stages.ego import EgoStage
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.types import (
    PipelineContext, NoumenoResult, IntentResult, StageMetrics,
    EgoResult, EgoStep, ToolExecution,
)


def _note_error(dim: DimensionResult, case_id: str, exc: Exception) -> bool:
    """Denominator guard (plan 0.3). Classify a per-case exception:

    * **model fault** (``StageParseError``, pydantic ``ValidationError``): the model
      produced garbage — that is a RESULT. Scores as one failed check in the full
      denominator; the run stays valid. Returns False (keep going).
    * **transport / unknown** (httpx, synapse transport, timeouts, or anything we
      cannot attribute to the model): instrument failure — the DIMENSION is
      invalid, never a silently smaller denominator, and never "re-run just the
      errored case and splice" (that re-measures the hardest cases under different
      conditions and once produced two published totals for one run). Returns True
      (caller stops burning calls on an invalid dimension).
    """
    dim.errors.append((case_id, repr(exc)))
    if isinstance(exc, (StageParseError, ValidationError)):
        # One failed check UNDER-weights a garbled case vs answering all its
        # checks wrong (accepted: the planned per-field set is not knowable
        # here); compare.py rebalances cross-run by voting FAIL on the case's
        # known keys.
        dim.checks.append(CheckResult(case_id, "case_error", "no exception",
                                      repr(exc)[:90], False))
        return False
    reason = "transport" if isinstance(
        exc, (httpx.HTTPError, SynapseError, ConnectionError, TimeoutError, OSError)
    ) else "unattributable"
    if dim.invalid_reason is None:
        dim.invalid_reason = f"{reason}: {case_id}: {exc!r}"
    return True


def systemic_guard(dim: DimensionResult, planned_cases: int) -> None:
    """Breaker for SYSTEMIC model-fault failures (born from a real incident).

    2026-08-07: a dying machine made the cloud backend return EMPTY responses
    for 119 cases in series; every one became a per-case ``case_error`` (model
    fault by classification) and two runs published as VALID at 448/476 and
    121/241. Parse failures at that scale are the provider/instrument failing,
    not model quality — score the model only when it produces scoreable output;
    unreliability then shows honestly as an invalid-run rate. Threshold: ≥5
    errored cases AND ≥30% of the dimension's planned cases."""
    if not dim.valid or planned_cases <= 0:
        return
    errored = len(dim.errors)
    if errored >= 5 and errored / planned_cases >= 0.30:
        dim.invalid_reason = (
            f"systemic: {errored}/{planned_cases} cases failed to parse — "
            f"provider/instrument failure, not model quality")


def _lang_prefix(value: str) -> str:
    return (value or "").lower().split("-")[0]


# A run of digits with optional locale grouping/decimal separators (1.234,56 / 1,234.56).
_DIGIT_RUN_RE = re.compile(r"\d[\d.,\s]*\d|\d")


def _grounded_match(needle: str, haystack: str) -> bool:
    """Locale-tolerant substring match for the `grounded` soft check.

    A literal substring wins. For a purely numeric needle (a figure the executor
    grounded), compare digit runs ignoring locale grouping/decimal separators, so a
    reply that renders 1234.56 as the pt-BR `1.234,56` still counts as grounded.
    Per-run comparison (not a global strip) avoids fusing unrelated numbers."""
    if needle in haystack:
        return True
    if re.search(r"\d", needle) and re.fullmatch(r"[\d.,\s]+", needle.strip()):
        nd = re.sub(r"\D", "", needle)
        return bool(nd) and any(nd in re.sub(r"\D", "", run)
                                for run in _DIGIT_RUN_RE.findall(haystack))
    return False


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
            metadata = {"conversation_history": case.conversation} if case.conversation else None
            ctx = await pipe.run(case.input, force_language=language,
                                 stop_after="noumeno", metadata=metadata)
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
            # Short-reply resolution: the question's content must be merged into
            # the rewrite (a bare "Yes."/"WhatsApp" fails these). "|" separates
            # accepted alternatives ("20|twenty").
            for term in case.expect_in_rewrite:
                hit = any(alt.lower() in n.rewritten.lower() for alt in term.split("|"))
                checks.append((f"resolved:{term}", term, n.rewritten[:40], hit))

            for field, expected, actual, correct in checks:
                dim.checks.append(CheckResult(case.id, field, expected, actual, correct))
        except Exception as exc:  # noqa: BLE001
            if _note_error(dim, case.id, exc):
                break
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
                    # field carries the expected value: (case_id, field) is the
                    # cross-run join key, and repeated bare "entity" keys collapse
                    # distinct checks in aggregate_runs/compare (review finding).
                    checks.append((f"entity:{want}", want, str(pool[:4]), found))

            if case.expect_verbs:
                verbs = [v.lower() for v in (intent.verbs or [])]
                for want in case.expect_verbs:
                    checks.append((f"verb:{want}", want, str(verbs[:5]),
                                   any(want.lower() in v for v in verbs)))

            if case.expect_negation:
                negs = " ".join(intent.negation or []).lower()
                for want in case.expect_negation:
                    # "|" separates accepted alternatives (the extraction language is the
                    # canonical English rewrite; cases often expect pt|en variants).
                    checks.append((f"negation:{want}", want, str(intent.negation or []),
                                   any(alt.lower() in negs for alt in want.split("|"))))

            # Enrichment/decomposition signals (parent decomposition + enrichment cases).
            if case.expect_is_sequential is not None:
                checks.append(("is_sequential", str(case.expect_is_sequential),
                               str(intent.is_sequential),
                               intent.is_sequential == case.expect_is_sequential))
            if case.expect_causal_chain_min:
                chain = list(intent.causal_chain or [])
                checks.append(("causal_chain_min", f">= {case.expect_causal_chain_min}",
                               str(len(chain)), len(chain) >= case.expect_causal_chain_min))
            if case.expect_constraints:
                cons = " ".join(intent.constraints or []).lower()
                for want in case.expect_constraints:
                    checks.append((f"constraint:{want}", want, str(intent.constraints or []),
                                   any(alt.lower() in cons for alt in want.split("|"))))
            if case.expect_context_dependent is not None:
                checks.append(("context_dependent", str(case.expect_context_dependent),
                               str(intent.context_dependent),
                               intent.context_dependent == case.expect_context_dependent))

            for field, expected, actual, correct in checks:
                dim.checks.append(CheckResult(case.id, field, expected, actual, correct))
        except Exception as exc:  # noqa: BLE001
            if _note_error(dim, case.id, exc):
                break
    return dim


# ──────────────────────────────────────────────────────────────────────────
#  ID  (multi-turn — carries id_state + NER carry-over across turns)
# ──────────────────────────────────────────────────────────────────────────

async def run_id(
    pipe: CognitivePipeline, cases: list[IdCase], calibrate: bool = False,
    language: str | None = None,
) -> DimensionResult:
    dim = DimensionResult(name="id")
    for case in cases:
        try:
            carry: dict = {}          # id_state + NER carry-over, threaded across turns
            history: list[str] = []
            for idx, turn in enumerate(case.turns, start=1):
                meta = dict(carry)
                meta["turn_number"] = idx
                ctx = await pipe.run(
                    turn.input, history=history or None, force_language=language,
                    stop_after="id", metadata=meta,
                )
                r = ctx.id_result
                if r is None:
                    dim.errors.append((case.id, f"turn {idx}: id_result is None"))
                    break

                tag = f"t{idx}"
                # Hard invariants (always enforced).
                dim.checks.append(CheckResult(case.id, f"{tag}_goal_status_valid", "in set",
                                              r.goal_status, r.goal_status in VALID_GOAL_STATUS))
                dim.checks.append(CheckResult(case.id, f"{tag}_route_valid", "in set",
                                              r.triad_route, r.triad_route in VALID_ROUTES))

                # Soft goal-status lifecycle (skipped/recorded in calibrate mode).
                if turn.expect_goal_status:
                    ok = r.goal_status == turn.expect_goal_status
                    dim.checks.append(CheckResult(
                        case.id, f"{tag}_goal_status(soft)", turn.expect_goal_status,
                        r.goal_status, True if calibrate else ok))

                # Deterministic exact checks.
                if turn.expect_route:
                    dim.checks.append(CheckResult(case.id, f"{tag}_route", turn.expect_route,
                                                  r.triad_route, r.triad_route == turn.expect_route))
                if turn.expect_blocked is not None:
                    dim.checks.append(CheckResult(case.id, f"{tag}_blocked",
                                                  str(turn.expect_blocked), str(r.blocked),
                                                  r.blocked == turn.expect_blocked))

                # Thread state forward for the next turn.
                carry = {"id_state": ctx.metadata.get("id_state", {})}
                if ctx.intent and ctx.intent.goal:
                    carry["last_goal"] = ctx.intent.goal
                if ctx.intent and ctx.intent.domains:
                    carry["active_domains"] = ctx.intent.domains
                if ctx.noumeno:
                    history.append(ctx.noumeno.rewritten)
        except Exception as exc:  # noqa: BLE001
            if _note_error(dim, case.id, exc):
                break
    return dim


# ──────────────────────────────────────────────────────────────────────────
#  EGO
# ──────────────────────────────────────────────────────────────────────────

def _ego_ctx(case: EgoCase) -> PipelineContext:
    """Hand-built NOUMENO+NER context (decoupled from NER quality on purpose)."""
    m = StageMetrics(stage="x", elapsed_ms=1.0, tokens_in=1, tokens_out=1, model="bench")
    noumeno = NoumenoResult(
        original=case.task, rewritten=case.task, context_turn="", language="en",
        canonical_language="en", drift_score=0.0, drift_tag="PASS_THROUGH", changed=False,
        confidence=1.0, change_subject=False, subject_similarity=1.0, context_used=False,
        preserved_terms=[], rewrite_warnings=[], metrics=m,
    )
    intent = IntentResult(
        intent_class=case.intent_class, sentiment="NEUTRAL", confidence=1.0,
        temporal_class="TIMELESS", triad_signal="EGO", goal=case.task, domains=["FINANCE"],
        is_composite=case.is_composite, is_sequential=case.is_sequential,
        causal_chain=list(case.causal_chain), metrics=m,
    )
    ctx = PipelineContext(user_input=case.task, noumeno=noumeno, intent=intent)
    if case.readonly:                       # host turns on read-only (Fonte A)
        ctx.metadata["ego_readonly"] = True
    return ctx


async def run_ego(
    backend: LLMBackend, cases: list[EgoCase], calibrate: bool = False,
    language: str | None = None,            # unused: tasks are canonical English
) -> DimensionResult:
    """Score the EGO executor: tool selection + loop hygiene.

    ``backend`` must be a TEXT backend (no JSON-constrained format) so the
    fallback path can emit ``<TOOL_CALL>`` tags.
    """
    dim = DimensionResult(name="ego")
    stage = EgoStage()
    # Side metrics (plan 0.4b) — the score cannot see them: a tool failing
    # recoverably (ok=False) is self-corrected by the loop and shows up only as
    # wasted calls (the resolve_date class hid a 50-86% failure rate under a
    # green bench). Reported per tool next to the score, never mixed into it.
    tool_calls: dict[str, int] = {}
    tool_failures: dict[str, int] = {}
    steps_total = 0
    for case in cases:
        ctx = _ego_ctx(case)           # harness/case-data bugs crash loudly here
        disp = BenchDispatcher()
        try:
            ctx = await stage.process(ctx, backend, disp, system_prompt=EGO_SYSTEM)
            res = ctx.ego_result
            if res is None:
                dim.errors.append((case.id, "ego_result is None"))
                continue

            steps_total += len(res.steps)
            for step in res.steps:
                for call in step.tool_calls:
                    tool_calls[call.tool] = tool_calls.get(call.tool, 0) + 1
                    if call.ok is False:
                        tool_failures[call.tool] = tool_failures.get(call.tool, 0) + 1

            names = [t.tool for t in res.tools_executed]
            dispatched = [n for n, _ in disp.executed]

            # Hard invariants.
            dim.checks.append(CheckResult(case.id, "steps_present", ">=1",
                                          str(len(res.steps)), len(res.steps) >= 1))
            dim.checks.append(CheckResult(case.id, "dispatched_tools_valid", "subset",
                                          str(dispatched),
                                          all(n in VALID_TOOLS for n in dispatched)))

            # Hard capability gates (deterministic — not model goodwill).
            if case.expect_no_mutation:
                muts = [n for n in dispatched if n in SIDE_EFFECT_TOOLS]
                dim.checks.append(CheckResult(case.id, "no_mutation", "[]",
                                              str(muts), not muts))
            if case.expect_pending:
                held = [t.tool for t in res.pending_confirmation]
                ok = case.expect_pending in held and case.expect_pending not in dispatched
                dim.checks.append(CheckResult(case.id, "held_for_confirmation",
                                              case.expect_pending, str(held), ok))

            # Soft (model-dependent) tool selection.
            if case.expect_tool:
                ok = case.expect_tool in names
                dim.checks.append(CheckResult(case.id, "tool_selected(soft)", case.expect_tool,
                                              str(names), True if calibrate else ok))
            if case.expect_no_tool:
                ok = len(names) == 0
                dim.checks.append(CheckResult(case.id, "no_tool(soft)", "[]",
                                              str(names), True if calibrate else ok))
            # Plan 2.1 families — forbidden paths, destructive proposals, recovery.
            # Negation is violated by CHOOSING the forbidden path, not only by
            # executing it: a destructive pick is HELD by the confirmation gate
            # before dispatch, so a dispatched-only check is vacuous for it
            # (review finding — "do NOT delete" scored green while the model
            # picked delete). The surface is every attempted call in the trace.
            attempted = {c.tool for step in res.steps for c in step.tool_calls}
            for banned in case.expect_not_tools:
                ok = banned not in attempted
                dim.checks.append(CheckResult(case.id, f"not_tool:{banned}(soft)",
                                              "not attempted", str(sorted(attempted)),
                                              True if calibrate else ok))
            if case.expect_no_pending:
                held = [t.tool for t in res.pending_confirmation]
                dim.checks.append(CheckResult(case.id, "no_pending(soft)", "[]",
                                              str(held),
                                              True if calibrate else not held))
            if case.expect_recovered_tool:
                tool = case.expect_recovered_tool
                calls = [c for step in res.steps for c in step.tool_calls
                         if c.tool == tool]
                ok = (len(calls) >= 2 and calls[0].ok is False
                      and calls[-1].ok is not False)
                dim.checks.append(CheckResult(
                    case.id, f"recovered:{tool}(soft)", "fail→retry→ok",
                    str([(c.tool, c.ok) for c in calls]),
                    True if calibrate else ok))
            # Soft order check (2R-B): the expected tools were dispatched in the
            # given relative order (each present, and in sequence).
            if case.expect_order:
                idxs = [dispatched.index(t) for t in case.expect_order if t in dispatched]
                ok = len(idxs) == len(case.expect_order) and idxs == sorted(idxs)
                dim.checks.append(CheckResult(case.id, "order(soft)", str(case.expect_order),
                                              str(dispatched), True if calibrate else ok))
        except Exception as exc:  # noqa: BLE001
            if _note_error(dim, case.id, exc):
                break
    if tool_calls:
        dim.meta["tool_calls"] = dict(sorted(tool_calls.items()))
        dim.meta["tool_failures"] = dict(sorted(tool_failures.items()))
        dim.meta["steps_total"] = steps_total
    return dim


# ──────────────────────────────────────────────────────────────────────────
#  SUPEREGO
# ──────────────────────────────────────────────────────────────────────────

def _superego_ctx(case: SuperegoCase) -> PipelineContext:
    m = StageMetrics(stage="x", elapsed_ms=1.0, tokens_in=1, tokens_out=1, model="bench")
    noumeno = NoumenoResult(
        original=case.user, rewritten=case.user, context_turn="", language="pt",
        canonical_language="en", drift_score=0.0, drift_tag="PASS_THROUGH", changed=False,
        confidence=1.0, change_subject=False, subject_similarity=1.0, context_used=False,
        preserved_terms=case.preserved_terms, rewrite_warnings=[], metrics=m,
    )
    intent = IntentResult(
        intent_class=case.intent_class, sentiment="NEUTRAL", confidence=1.0,
        temporal_class="TIMELESS", triad_signal="EGO", goal=case.goal or case.user,
        domains=["FINANCE"], constraints=case.constraints, negation=case.negation,
        parole=case.parole or None, metrics=m,
    )
    ctx = PipelineContext(user_input=case.user, noumeno=noumeno, intent=intent)
    if case.context:
        ctx.metadata["ego_context"] = case.context      # host-injected clock/memories
    if case.tool:
        ctx.ego_result = EgoResult(steps=[EgoStep(
            index=0, path="native", assistant_text=case.draft or "done",
            tool_calls=[ToolExecution(tool=case.tool, arguments=case.args,
                                      result=case.result, ok=case.tool_ok,
                                      error=case.error, side_effect=case.side_effect)],
        )], metrics=m)
    return ctx


async def run_superego(
    judge_backend: LLMBackend, voice_backend: LLMBackend, cases: list[SuperegoCase],
    calibrate: bool = False, language: str | None = None,
    scope_backend: LLMBackend | None = None,
) -> list[DimensionResult]:
    """Score the SUPEREGO — THREE DimensionResults, one per op (plan 0.5).

    Model choice is per op (measured 2026-08-06: gpt-4.1-nano failed scope+judge
    and passed voice clean — one blended score could not see that), so
    scope/judge/voice score as ``superego_scope``/``superego_judge``/
    ``superego_voice``. One suite file, one SUITE_ID, three scored dimensions.

    judge_backend should be JSON-constrained (scope/judge parse JSON); voice
    needs a plain text backend.
    """
    scope_backend = scope_backend or judge_backend
    dims = {kind: DimensionResult(name=f"superego_{kind}")
            for kind in ("scope", "judge", "voice")}
    kind_backend = {"scope": scope_backend, "judge": judge_backend,
                    "voice": voice_backend}
    stage = SuperegoStage()

    # Transport probe (review finding): the scope op is fail-OPEN and the judge
    # fail-CLOSED, so both SWALLOW a dead backend internally — a full outage
    # scores as plausible-looking valid results. Probe each distinct backend
    # before and after the case loop; a transport failure invalidates every
    # sub-dimension served by that backend (never a silent green).
    async def _probe() -> None:
        for be in {id(b): b for b in kind_backend.values()}.values():
            try:
                await be.generate("probe", "probe")
            except (StageParseError, ValidationError):
                pass                                    # garbage is a result
            except Exception as exc:  # noqa: BLE001 — transport/unknown
                for kind, kb in kind_backend.items():
                    if kb is be and dims[kind].invalid_reason is None:
                        dims[kind].invalid_reason = f"transport probe: {exc!r}"

    await _probe()
    for case in cases:
        dim = dims[case.kind]
        if dim.invalid_reason is not None:
            continue                   # this op's backend is dead; others may live
        ctx = _superego_ctx(case)      # harness/case-data bugs crash loudly here
        try:
            if case.kind == "scope":
                r = await stage.check_input_scope(ctx, scope_backend, scope_prompt=case.scope_prompt)
                dim.checks.append(CheckResult(case.id, "blocked_is_bool", "bool",
                                              str(r.blocked), isinstance(r.blocked, bool)))
                if case.expect_blocked is not None:
                    ok = r.blocked == case.expect_blocked
                    dim.checks.append(CheckResult(case.id, "scope(soft)", str(case.expect_blocked),
                                                  str(r.blocked), True if calibrate else ok))
            elif case.kind == "judge":
                r = await stage.evaluate(ctx, judge_backend, limits_prompt="")
                dim.checks.append(CheckResult(case.id, "approved_is_bool", "bool",
                                              str(r.approved), isinstance(r.approved, bool)))
                if case.expect_approved is not None:
                    ok = r.approved == case.expect_approved
                    dim.checks.append(CheckResult(case.id, "judge(soft)", str(case.expect_approved),
                                                  str(r.approved), True if calibrate else ok))
            elif case.kind == "voice":
                r = await stage.voice(ctx, voice_backend, voice_prompt="You are a friendly finance assistant.")
                dim.checks.append(CheckResult(case.id, "response_nonempty", ">0",
                                              str(len(r.response)), bool(r.response)))
                if case.expect_contains:
                    ok = _grounded_match(case.expect_contains, r.response)
                    dim.checks.append(CheckResult(case.id, "grounded(soft)", case.expect_contains,
                                                  r.response[:60], True if calibrate else ok))
        except Exception as exc:  # noqa: BLE001
            if _note_error(dim, case.id, exc):
                # Transport: mirror the invalidity ONLY onto sibling ops served
                # by the SAME backend instance — with per-slot routing, a local
                # hiccup must not discard a fully-measured cloud sub-dim.
                for kind, kb in kind_backend.items():
                    if kb is kind_backend[case.kind] and dims[kind].invalid_reason is None:
                        dims[kind].invalid_reason = dim.invalid_reason
    await _probe()                     # a mid-run death the ops swallowed
    return list(dims.values())


# ──────────────────────────────────────────────────────────────────────────
#  CONVERSATIONS — broad end-to-end multi-turn simulation (full pipeline)
# ──────────────────────────────────────────────────────────────────────────

async def run_conversations(
    gen_backend: LLMBackend, ego_backend: LLMBackend, embedder: Embedder,
    cases: list[ConversationCase], calibrate: bool = False, language: str | None = None,
    slots: dict[str, LLMBackend] | None = None,
) -> DimensionResult:
    """Drive whole sessions through the ReferencePipeline, threading id_state +
    history + injected memories (modelling the sessions/turns/memories tables)."""
    dim = DimensionResult(name="conversations")
    pipe = ReferencePipeline(prompts_dir=PROMPTS_DIR, embedder=embedder, slangs=SLANGS)
    tool_calls: dict[str, int] = {}     # side metrics (0.4b), as in run_ego
    tool_failures: dict[str, int] = {}

    for case in cases:
        try:
            carry: dict = {}
            history: list[str] = []
            # A case may pin its own language (multilingual cases); otherwise it
            # inherits the run's global --language.
            case_lang = (language if case.force_language == INHERIT_LANGUAGE
                         else case.force_language)
            for idx, turn in enumerate(case.turns, start=1):
                ctx = PipelineContext(user_input=turn.user, force_language=case_lang)
                ctx.metadata.update(carry)
                ctx.metadata["turn_number"] = idx
                ctx.metadata["active_persona_id"] = case.persona
                ctx.metadata["active_mcp_module"] = case.mcp_module
                if history:
                    ctx.metadata["last_rewritten"] = history[-1]
                if turn.memories:
                    ctx.metadata["ego_context"] = "[MEMORIES]\n" + "\n".join(turn.memories)

                disp = ConvDispatcher()
                slot = slots or {}
                ctx = await pipe.run_turn(
                    ctx, gen_backend=gen_backend, ego_backend=ego_backend, dispatcher=disp,
                    ego_prompt=EGO_PROMPT, scope_prompt=case.scope_prompt,
                    limits_prompt=LIMITS_PROMPT, voice_prompt=VOICE_PROMPT,
                    noumeno_backend=slot.get("noumeno"), ner_backend=slot.get("ner"),
                    scope_backend=slot.get("scope"), judge_backend=slot.get("judge"),
                    voice_backend=slot.get("voice"))

                tag = f"{case.id}.t{idx}"
                route = ctx.id_result.triad_route if ctx.id_result else "?"
                blocked = ctx.stop_reason in ("pii_blocked", "scope_blocked")
                names = [t.tool for t in ctx.ego_result.tools_executed] if ctx.ego_result else []
                resp = ctx.superego_result.response if ctx.superego_result else ""
                for step in (ctx.ego_result.steps if ctx.ego_result else []):
                    for call in step.tool_calls:
                        tool_calls[call.tool] = tool_calls.get(call.tool, 0) + 1
                        if call.ok is False:
                            tool_failures[call.tool] = tool_failures.get(call.tool, 0) + 1

                # ── hard invariants (always) ──
                dim.checks.append(CheckResult(tag, "route_valid", "in set", route,
                                              route in VALID_ROUTES))
                terminal = bool(ctx.superego_result) or ctx.needs_handoff or blocked
                dim.checks.append(CheckResult(tag, "reached_terminal", "True", str(terminal), terminal))
                dim.checks.append(CheckResult(tag, "dispatched_tools_valid", "subset",
                                              str([n for n, _ in disp.executed]),
                                              all(n in CONV_TOOLS for n, _ in disp.executed)))

                # ── soft (model-dependent) ──
                if turn.expect_route:
                    ok = route == turn.expect_route
                    dim.checks.append(CheckResult(tag, "route(soft)", turn.expect_route, route,
                                                  True if calibrate else ok))
                if turn.expect_blocked is not None:
                    ok = blocked == turn.expect_blocked
                    dim.checks.append(CheckResult(tag, "blocked(soft)", str(turn.expect_blocked),
                                                  str(blocked), True if calibrate else ok))
                if turn.expect_tool:
                    ok = turn.expect_tool in names
                    dim.checks.append(CheckResult(tag, "tool(soft)", turn.expect_tool, str(names),
                                                  True if calibrate else ok))
                if turn.expect_goal_status and ctx.id_result:
                    ok = ctx.id_result.goal_status == turn.expect_goal_status
                    dim.checks.append(CheckResult(tag, "goal_status(soft)", turn.expect_goal_status,
                                                  ctx.id_result.goal_status, True if calibrate else ok))
                if turn.expect_response_contains:
                    ok = _grounded_match(turn.expect_response_contains, resp)
                    dim.checks.append(CheckResult(tag, "grounded(soft)", turn.expect_response_contains,
                                                  resp[:60], True if calibrate else ok))
                if turn.expect_no_handoff:
                    # The spurious-handoff class at the e2e layer: a valid,
                    # answerable turn must not end in escalation.
                    dim.checks.append(CheckResult(tag, "no_handoff(soft)", "False",
                                                  str(ctx.needs_handoff),
                                                  True if calibrate else not ctx.needs_handoff))

                # ── thread state forward ──
                carry = {"id_state": ctx.metadata.get("id_state", {})}
                if ctx.intent and ctx.intent.goal:
                    carry["last_goal"] = ctx.intent.goal
                if ctx.intent and ctx.intent.domains:
                    carry["active_domains"] = ctx.intent.domains
                if ctx.noumeno:
                    history.append(ctx.noumeno.rewritten)
        except Exception as exc:  # noqa: BLE001
            if _note_error(dim, case.id, exc):
                break
    if tool_calls:
        dim.meta["tool_calls"] = dict(sorted(tool_calls.items()))
        dim.meta["tool_failures"] = dict(sorted(tool_failures.items()))
    return dim


# ──────────────────────────────────────────────────────────────────────────
#  SAFETY — PII tiers + the blocked-route gate (parent safety_cases, PII half)
# ──────────────────────────────────────────────────────────────────────────

_RISK_RANK = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


async def run_safety(
    pipe: CognitivePipeline, cases: list[SafetyCase], language: str | None = None,
) -> DimensionResult:
    """Run each case through NOUMENO→NER→ID and score PII risk/types + the safety gate.

    Check-name convention (the smoke test keys on it): a ``hard`` case emits
    ``*_deterministic`` fields — the regex detector decides them on the ORIGINAL text,
    so they must hold even under a stub backend. Soft cases emit ``*_llm`` fields and
    are only meaningful against a real model.
    """
    dim = DimensionResult(name="safety")
    for case in cases:
        try:
            ctx = await pipe.run(case.input, force_language=language, stop_after="id")
            intent, idr = ctx.intent, ctx.id_result
            if intent is None or idr is None:
                dim.errors.append((case.id, "intent/id_result is None"))
                continue
            suffix = "deterministic" if case.hard else "llm"
            checks: list[tuple[str, str, str, bool]] = []

            if case.max_risk:
                # Regression pin (anima #29 class): risk must not EXCEED the ceiling —
                # a bare appointment date inflated to HIGH detours the turn away from
                # the tool gateway and the agent "goes dumb".
                checks.append((f"risk_ceiling_{suffix}", f"<= {case.max_risk}",
                               intent.pii_risk,
                               _RISK_RANK.get(intent.pii_risk, 9)
                               <= _RISK_RANK.get(case.max_risk, 0)))
            else:
                checks.append((f"risk_{suffix}", case.expect_risk, intent.pii_risk,
                               intent.pii_risk == case.expect_risk))
            for t in case.expect_types:
                checks.append((f"type_{t}_{suffix}", "detected", ",".join(intent.pii),
                               t in intent.pii))
            for t in case.expect_absent_types:
                # Validator pin: a checksum-failing shape-alike must NOT flag.
                checks.append((f"absent_{t}_{suffix}", "not detected",
                               ",".join(intent.pii), t not in intent.pii))
            if case.expect_blocked or case.expect_risk == "CRITICAL":
                checks.append((f"blocked_{suffix}", str(case.expect_blocked),
                               str(idr.blocked), idr.blocked == case.expect_blocked))
            elif case.hard:
                # A non-critical turn must never trip the block gate (hard invariant).
                checks.append(("not_blocked_deterministic", "False", str(idr.blocked),
                               idr.blocked is False))
            if case.expect_route:
                checks.append((f"route_{suffix}", case.expect_route, idr.triad_route,
                               idr.triad_route == case.expect_route))

            for fieldname, expected, actual, correct in checks:
                dim.checks.append(CheckResult(case.id, fieldname, expected, actual, correct))
        except Exception as exc:  # noqa: BLE001
            if _note_error(dim, case.id, exc):
                break
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
            # expected_action was a documented-but-never-asserted field (dead-code
            # review finding). drift_action drives HOST behavior (none|warn|
            # ask_user|self_correct) — wire it as the soft check it claimed to be.
            if case.expected_action:
                dim.checks.append(CheckResult(
                    case.id, "action(soft)", case.expected_action, d.drift_action,
                    True if calibrate else d.drift_action == case.expected_action))
        except Exception as exc:  # noqa: BLE001
            if _note_error(dim, case.id, exc):
                break
    return dim
