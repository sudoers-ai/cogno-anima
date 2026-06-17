"""
Dimension runners — execute cases through the reference pipeline and score them.

Adapted from the parent Cogno `eval_ner` mixin, but decoupled: no PipelineRunner,
no SkillRegistry, no infra. Scoring targets cogno-anima's `IntentResult`,
`NoumenoResult` and `DriftMetrics` contracts directly.
"""

from __future__ import annotations

from cognobench.harness import CognitivePipeline
from cognobench.types import CheckResult, DimensionResult
from cognobench.ner_cases import NERCase
from cognobench.drift_cases import DriftCase, VALID_ACTIONS
from cognobench.noumeno_cases import NoumenoCase, VALID_DRIFT_TAGS
from cognobench.id_cases import IdCase, VALID_GOAL_STATUS, VALID_ROUTES
from cognobench.ego_cases import (
    EgoCase, BenchDispatcher, EGO_SYSTEM, VALID_TOOLS, SIDE_EFFECT_TOOLS,
)
from cognobench.superego_cases import SuperegoCase
from cognobench.conversation_cases import (
    ConversationCase, BenchDispatcher as ConvDispatcher, INHERIT_LANGUAGE,
    EGO_PROMPT, LIMITS_PROMPT, VOICE_PROMPT, VALID_TOOLS as CONV_TOOLS,
)
from cognobench.harness import PROMPTS_DIR, SLANGS
from cognobench.pipeline import ReferencePipeline

from cogno_anima.llm import LLMBackend, Embedder
from cogno_anima.stages.ego import EgoStage
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.types import (
    PipelineContext, NoumenoResult, IntentResult, StageMetrics,
    EgoResult, EgoStep, ToolExecution,
)


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
            dim.errors.append((case.id, repr(exc)))
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
        metrics=m,
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
    for case in cases:
        try:
            ctx = _ego_ctx(case)
            disp = BenchDispatcher()
            ctx = await stage.process(ctx, backend, disp, system_prompt=EGO_SYSTEM)
            res = ctx.ego_result
            if res is None:
                dim.errors.append((case.id, "ego_result is None"))
                continue

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
        except Exception as exc:  # noqa: BLE001
            dim.errors.append((case.id, repr(exc)))
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
        preserved_terms=[], rewrite_warnings=[], metrics=m,
    )
    intent = IntentResult(
        intent_class=case.intent_class, sentiment="NEUTRAL", confidence=1.0,
        temporal_class="TIMELESS", triad_signal="EGO", goal=case.goal or case.user,
        domains=["FINANCE"], constraints=case.constraints, negation=case.negation,
        parole=case.parole or None, metrics=m,
    )
    ctx = PipelineContext(user_input=case.user, noumeno=noumeno, intent=intent)
    if case.tool:
        ctx.ego_result = EgoResult(steps=[EgoStep(
            index=0, path="native", assistant_text="done",
            tool_calls=[ToolExecution(tool=case.tool, arguments=case.args,
                                      result=case.result, ok=True)],
        )], metrics=m)
    return ctx


async def run_superego(
    judge_backend: LLMBackend, voice_backend: LLMBackend, cases: list[SuperegoCase],
    calibrate: bool = False, language: str | None = None,
) -> DimensionResult:
    """Score the SUPEREGO: scope guard + judge (goal↔execution) + voicer.

    judge_backend should be JSON-constrained (scope/judge parse JSON); voice
    needs a plain text backend.
    """
    dim = DimensionResult(name="superego")
    stage = SuperegoStage()
    for case in cases:
        try:
            ctx = _superego_ctx(case)
            if case.kind == "scope":
                r = await stage.check_input_scope(ctx, judge_backend, scope_prompt=case.scope_prompt)
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
                    ok = case.expect_contains in r.response
                    dim.checks.append(CheckResult(case.id, "grounded(soft)", case.expect_contains,
                                                  r.response[:60], True if calibrate else ok))
        except Exception as exc:  # noqa: BLE001
            dim.errors.append((case.id, repr(exc)))
    return dim


# ──────────────────────────────────────────────────────────────────────────
#  CONVERSATIONS — broad end-to-end multi-turn simulation (full pipeline)
# ──────────────────────────────────────────────────────────────────────────

async def run_conversations(
    gen_backend: LLMBackend, ego_backend: LLMBackend, embedder: Embedder,
    cases: list[ConversationCase], calibrate: bool = False, language: str | None = None,
) -> DimensionResult:
    """Drive whole sessions through the ReferencePipeline, threading id_state +
    history + injected memories (modelling the sessions/turns/memories tables)."""
    dim = DimensionResult(name="conversations")
    pipe = ReferencePipeline(prompts_dir=PROMPTS_DIR, embedder=embedder, slangs=SLANGS)

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
                ctx = await pipe.run_turn(
                    ctx, gen_backend=gen_backend, ego_backend=ego_backend, dispatcher=disp,
                    ego_prompt=EGO_PROMPT, scope_prompt=case.scope_prompt,
                    limits_prompt=LIMITS_PROMPT, voice_prompt=VOICE_PROMPT)

                tag = f"{case.id}.t{idx}"
                route = ctx.id_result.triad_route if ctx.id_result else "?"
                blocked = ctx.stop_reason in ("pii_blocked", "scope_blocked")
                names = [t.tool for t in ctx.ego_result.tools_executed] if ctx.ego_result else []
                resp = ctx.superego_result.response if ctx.superego_result else ""

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
                    ok = turn.expect_response_contains in resp
                    dim.checks.append(CheckResult(tag, "grounded(soft)", turn.expect_response_contains,
                                                  resp[:60], True if calibrate else ok))

                # ── thread state forward ──
                carry = {"id_state": ctx.metadata.get("id_state", {})}
                if ctx.intent and ctx.intent.goal:
                    carry["last_goal"] = ctx.intent.goal
                if ctx.intent and ctx.intent.domains:
                    carry["active_domains"] = ctx.intent.domains
                if ctx.noumeno:
                    history.append(ctx.noumeno.rewritten)
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
