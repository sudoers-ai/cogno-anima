"""
cogno_anima.stages.superego — SuperegoStage: guardrails, judge & voicer (Stage 5).

EGO=executor, SUPEREGO=locutor. The SUPEREGO has three LLM operations (A2 — the
host injects whichever backend it wants for each; they may differ):

  * ``check_input_scope`` (pre-EGO) — cheap ALLOW/BLOCK relevance guard; BLOCK
    skips the expensive EGO. Fail-OPEN (a cost guard must never refuse a
    legitimate user on error).
  * ``evaluate`` (post-EGO JUDGE) — approve the EGO's *execution* or send it back
    with a critique. Criterion #1 is goal↔execution ("asked X, did X not Y").
    Fail-CLOSED (never approve unverified — the cost of a false-pass is worse).
  * ``voice`` (post-EGO) — **writes** the final user response from the EGO's
    gathered data, in the persona's voice + limits; strips CoT, runs a
    deterministic PII backstop, and feeds synthesis drift.

Plus deterministic, dependency-free utilities (``strip_cot``,
``detect_adjustments``) and ``_blocked_response`` (PII-CRITICAL protection).

Host concerns (NOT here): the persona scope/limits/voice prompt text, the retry
LOOP orchestration + ``max_corrections``, billing, and the actual human handoff
(the core only signals it via ``stop_reason="human_handoff"`` / ``needs_handoff``).
"""

from __future__ import annotations

import re
import time
import json
import logging
from typing import Optional

from cogno_anima.types import (
    PipelineContext, StageMetrics, SuperegoResult, ScopeCheckResult,
)
from cogno_anima.llm import LLMBackend
from cogno_anima.stages.drift import DriftCalculator
from cogno_anima.security.detector import PiiDetector, default_detector

logger = logging.getLogger("cogno_anima.superego")

_COT_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_SCOPE_SYSTEM = (
    "You are a scope classifier for a business AI assistant. Detect ONLY clearly "
    "off-topic requests (recipes, trivia, homework, politics). Default stance: "
    "ALLOW. Block only when the input is obviously unrelated to the persona's "
    "domain. Respond with a single JSON object, no markdown, no explanation."
)

_JUDGE_SYSTEM = (
    "You are a strict quality judge for an AI assistant's execution. Respond with "
    "JSON only. Default to NOT approving when you cannot verify the criteria."
)

_BLOCKED_FALLBACK = (
    "I detected sensitive personal information in your message and can't process "
    "it as-is. Please rephrase without including personal data."
)


class SuperegoStage:
    """Stage 5 — guard, judge, voicer. LLM + deterministic utils; no Embedder."""

    name = "superego"

    def __init__(self, drift: Optional[DriftCalculator] = None,
                 pii_detector: Optional[PiiDetector] = None) -> None:
        self._drift = drift or DriftCalculator()
        self._pii = pii_detector or default_detector()

    # ── deterministic utilities ──────────────────────────────────────

    @staticmethod
    def strip_cot(text: str) -> tuple[str, bool]:
        """Remove <think>/<thinking> CoT blocks. Returns (clean, was_stripped)."""
        if not text:
            return text, False
        cleaned = _COT_RE.sub("", text).strip()
        return cleaned, cleaned != text.strip()

    @staticmethod
    def detect_adjustments(ctx: PipelineContext) -> list[str]:
        """Deterministic tone hints fed into the voice prompt (from NER/ID signals)."""
        adj: list[str] = []
        intent = ctx.intent
        if intent:
            adj += {
                "FRUSTRATED": ["tone:empathetic"], "CURIOUS": ["tone:engaging"],
                "PLAYFUL": ["tone:playful"], "URGENT": ["tone:direct"],
            }.get(intent.sentiment, [])
            adj += {
                "CREATIVE_TASK": ["style:creative"], "SOCIAL": ["style:warm"],
            }.get(intent.intent_class, [])
            if intent.pii_risk not in ("NONE", "LOW"):
                adj.append(f"pii:risk_{intent.pii_risk.lower()}")
        if ctx.id_result and ctx.id_result.emotional_override:
            adj.append(f"override:{ctx.id_result.emotional_override}")
        return adj or ["general:review"]

    # ── Early Input Scope Guard (pre-EGO) ────────────────────────────

    async def check_input_scope(
        self, ctx: PipelineContext, backend: LLMBackend, *, scope_prompt: str,
    ) -> ScopeCheckResult:
        t0 = time.perf_counter()
        model = getattr(backend, "model", "unknown")

        def _result(blocked: bool, msg: str, ti: int = 0, to: int = 0) -> ScopeCheckResult:
            return ScopeCheckResult(
                blocked=blocked, refusal_message=msg,
                metrics=StageMetrics(stage="superego_scope",
                                     elapsed_ms=(time.perf_counter() - t0) * 1000,
                                     tokens_in=ti, tokens_out=to, model=model),
            )

        # No rules to enforce → ALLOW.
        if not scope_prompt or not scope_prompt.strip():
            return _result(False, "")
        # NER-assisted bypass: greetings / follow-ups are always in-scope.
        if ctx.intent and ctx.intent.intent_class in ("SOCIAL", "CLARIFICATION"):
            return _result(False, "")

        prompt = self._build_scope_prompt(scope_prompt, ctx.user_input)
        try:
            raw, ti, to = await backend.generate(_SCOPE_SYSTEM, prompt)
            raw, _ = self.strip_cot(raw)
            data = self._parse_json(raw)
            blocked = bool(data.get("blocked", False))
            msg = str(data.get("refusal_message", "")) if blocked else ""
            logger.info("SUPEREGO scope blocked=%s", blocked)
            return _result(blocked, msg, ti, to)
        except Exception as exc:  # noqa: BLE001 — fail-open: never refuse on error
            logger.warning("scope guard failed (%s) — allowing by default", exc)
            return _result(False, "")

    @staticmethod
    def _build_scope_prompt(scope_prompt: str, user_input: str) -> str:
        return (
            f"# Scope Definition\n{scope_prompt}\n\n"
            f'# User Input\n"{user_input}"\n\n'
            "# Task\nIs the User Input IN-SCOPE or OUT-OF-SCOPE? Rules:\n"
            "- Block ONLY what is clearly, obviously unrelated to the scope.\n"
            "- When in doubt, ALLOW (false positives are NOT acceptable).\n"
            "- Greetings, follow-ups, clarifications and questions about the "
            "business/product are ALWAYS in-scope.\n\n"
            "# Examples\n"
            'User: "how do I bake a cake?" → blocked=true\n'
            'User: "who is the president?" → blocked=true\n'
            'User: "how much is the plan?" → blocked=false\n'
            'User: "thanks for the help" → blocked=false\n\n'
            'Respond ONLY with: {"blocked": true/false, "refusal_message": '
            '"...polite refusal in the user\'s language if blocked, else empty..."}'
        )

    # ── Quality gate / JUDGE (post-EGO) ──────────────────────────────

    async def evaluate(
        self, ctx: PipelineContext, backend: LLMBackend, *, limits_prompt: str,
    ) -> SuperegoResult:
        """Judge the EGO's execution. Fail-CLOSED (don't approve unverified).

        Criterion #1: goal↔execution — the user asked X and X (not Y) was done.
        """
        t0 = time.perf_counter()
        model = getattr(backend, "model", "unknown")

        def _result(approved: bool, critique: Optional[str], ti: int = 0, to: int = 0) -> SuperegoResult:
            return SuperegoResult(
                approved=approved, critique=critique,
                metrics=StageMetrics(stage="superego_judge",
                                     elapsed_ms=(time.perf_counter() - t0) * 1000,
                                     tokens_in=ti, tokens_out=to, model=model),
            )

        # Nothing executed → nothing to judge.
        if not ctx.ego_result:
            return _result(True, None)

        prompt = self._build_judge_prompt(ctx, limits_prompt)
        try:
            raw, ti, to = await backend.generate(_JUDGE_SYSTEM, prompt)
            raw, _ = self.strip_cot(raw)
            data = self._parse_json(raw)
            approved = bool(data.get("approved", False))
            critique = None if approved else str(data.get("critique", "")) or "execution rejected"
            logger.info("SUPEREGO judge approved=%s critique=%s", approved, (critique or "")[:80])
            return _result(approved, critique, ti, to)
        except Exception as exc:  # noqa: BLE001 — fail-CLOSED: don't pass unverified
            logger.warning("judge failed (%s) — not approving (fail-closed)", exc)
            return _result(False, "could not verify the execution; please retry")

    def _build_judge_prompt(self, ctx: PipelineContext, limits_prompt: str) -> str:
        ego = ctx.ego_result
        assert ego is not None  # evaluate() guarantees this before calling
        goal = (ctx.intent.goal if ctx.intent and ctx.intent.goal else "") or ctx.user_input
        executed = "\n".join(
            f"- {t.tool}({json.dumps(t.arguments, ensure_ascii=False)}) → "
            f"{'OK' if t.ok else 'ERROR'}: {t.result or t.error or ''}"
            for t in ego.tools_executed
        ) or "(no tools executed)"
        draft = ego.draft or "(none)"
        limits = f"\n# Persona limits\n{limits_prompt}\n" if limits_prompt and limits_prompt.strip() else ""
        return (
            f'# User request\n"{ctx.user_input}"\n\n'
            f"# Active goal\n{goal}\n"
            f"{limits}\n"
            f"# What the EGO executed\n{executed}\n\n"
            f"# EGO draft\n{draft}\n\n"
            "# Judge the EXECUTION against these criteria (most important first):\n"
            "1. GOAL↔EXECUTION: did it do exactly what was asked (X, not Y)?\n"
            "2. COMPLETENESS: was the goal fully met (not partial)?\n"
            "3. GROUNDING: is everything backed by the tool results (no invented data)?\n"
            "4. SAFETY/LIMITS: within the persona's limits, no policy violation?\n\n"
            'Respond ONLY with: {"approved": true/false, "critique": '
            '"...if not approved, what is wrong, to guide a retry..."}'
        )

    # ── Voicer (post-EGO) — writes the final response ────────────────

    async def voice(
        self, ctx: PipelineContext, backend: LLMBackend, *, voice_prompt: str,
    ) -> SuperegoResult:
        """Write the final user response from the EGO's data, in persona voice+limits.

        Applies deterministic tone hints, strips CoT, runs a PII backstop on the
        output, and feeds synthesis drift. Raises on LLM transport failure
        (errors propagate; the host decides fallback).
        """
        t0 = time.perf_counter()
        model = getattr(backend, "model", "unknown")
        adjustments = self.detect_adjustments(ctx)
        payload = self._tool_payload(ctx)

        prompt = self._build_voice_prompt(ctx, voice_prompt, payload, adjustments)
        raw, ti, to = await backend.generate(voice_prompt or "You are a helpful assistant.", prompt)
        response, cot_stripped = self.strip_cot(raw)

        # Deterministic PII backstop on the OUTPUT — flag involuntary leaks
        # (do NOT auto-redact: avoid over-redaction of intentionally-shared data;
        # the host's limits policy decides). Signal via adjustments.
        if response and self._pii.detect(response):
            adjustments.append("pii:flagged_in_output")

        # Feed synthesis drift (lexical grounding of response vs tool data).
        if ctx.drift is not None:
            self._drift.compute_synthesis(ctx.drift, payload, response)
            self._drift.compute_cumulative(ctx.drift)

        logger.info("SUPEREGO voice len=%d cot_stripped=%s adjustments=%s",
                    len(response), cot_stripped, adjustments)

        return SuperegoResult(
            response=response, approved=True, adjustments=adjustments,
            cot_stripped=cot_stripped,
            metrics=StageMetrics(stage="superego_voice",
                                 elapsed_ms=(time.perf_counter() - t0) * 1000,
                                 tokens_in=ti, tokens_out=to, model=model),
        )

    def _build_voice_prompt(self, ctx: PipelineContext, voice_prompt: str,
                            payload: str, adjustments: list[str]) -> str:
        signals = []
        if ctx.intent:
            signals.append(f"Sentiment: {ctx.intent.sentiment}")
        if ctx.noumeno and ctx.noumeno.language:
            signals.append(f"Reply language: {ctx.noumeno.language}")
        signals.append(f"Tone hints: {', '.join(adjustments)}")
        # Host-injected context (retrieved memories / history / clock) — the same
        # block the EGO sees; included so memories can ground the final reply.
        injected = ctx.metadata.get("ego_context")
        context_section = f"# Context (memories/history)\n{str(injected).strip()}\n\n" if injected else ""
        return (
            f'# User request\n"{ctx.user_input}"\n\n'
            f"{context_section}"
            f"# Data gathered by the executor (ground figures/dates ONLY in this)\n{payload}\n\n"
            f"# Signals\n" + "\n".join(signals) + "\n\n"
            "# Task\nWrite the final reply to the user in the persona's voice and "
            "within its limits. Use the context for background, but keep exact "
            "figures/dates verbatim from the executor data — do not invent or alter "
            "them. Reply with the message text only."
        )

    @staticmethod
    def _tool_payload(ctx: PipelineContext) -> str:
        if not ctx.ego_result:
            return "(no execution)"
        parts = [f"{t.tool}: {t.result or t.error or ''}"
                 for t in ctx.ego_result.tools_executed if t.ok]
        return "\n".join(parts) or (ctx.ego_result.draft or "(no data)")

    # ── PII-CRITICAL block ───────────────────────────────────────────

    def _blocked_response(
        self, ctx: PipelineContext, *, block_message: Optional[str] = None,
    ) -> SuperegoResult:
        return SuperegoResult(
            response=block_message or _BLOCKED_FALLBACK, blocked=True, approved=True,
            adjustments=["pii:blocked"],
            metrics=StageMetrics(stage="superego_blocked", elapsed_ms=0.0,
                                 tokens_in=0, tokens_out=0, model="none"),
        )

    # ── shared ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(raw: str) -> dict:
        match = _JSON_RE.search(raw or "")
        if not match:
            return {}
        try:
            data = json.loads(match.group())
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}
