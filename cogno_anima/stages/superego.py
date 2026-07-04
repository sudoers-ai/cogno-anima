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
from cogno_synapse import LLMBackend
from cogno_anima.stages.drift import DriftCalculator
from cogno_anima.security.detector import PiiDetector, default_detector

logger = logging.getLogger("cogno_anima.superego")

_COT_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
# A preserved term is "critical" (worth a grounding backstop) when it carries a
# figure or is an email/URL — altering one of these silently corrupts the answer.
_NUM_RE = re.compile(r"\d[\d.,]*\d|\d")
_CRITICAL_TERM_RE = re.compile(r"\d|@|https?://", re.IGNORECASE)

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
            register = SuperegoStage._parole_to_register(intent.parole)
            if register:
                adj.append(register)
        if ctx.id_result and ctx.id_result.emotional_override:
            adj.append(f"override:{ctx.id_result.emotional_override}")
        return adj or ["general:review"]

    @staticmethod
    def _parole_to_register(parole: Optional[str]) -> Optional[str]:
        """Collapse the user's NER ``parole`` onto a formality-accommodation hint.

        Distinct axis from sentiment (which carries *emotional* tone): this is
        *formality/lexical level* only. Soft signal — MIXED/None/unknown → no hint
        (degrade gracefully). GIRIA/POETICO are intentionally softened (the persona
        + limits clamp them; we never echo slang/poetic register verbatim).
        """
        return {
            "ACADEMICO": "register:formal",
            "FORMAL": "register:formal",
            "TECNICO": "register:technical",
            "COLOQUIAL": "register:casual",
            "GIRIA": "register:light",
            "POETICO": "register:expressive",
        }.get((parole or "").upper())

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
        # Continuation bypass: an ONGOING goal means the user already cleared the
        # scope guard on an earlier turn of this thread. A short follow-up ("at
        # 3pm", "with the cardiologist", a bare name) then carries little lexical
        # signal — NER often lands on UNKNOWN and the *contextless* scope
        # classifier wrongly blocks a legitimate continuation. Trust the goal:
        # once a conversation is in-scope, its follow-ups are too (fail-open, and
        # a genuine mid-thread topic change is caught by the NER/ID drift signals,
        # not this cheap gate).
        if ctx.id_result and ctx.id_result.goal_status == "ONGOING":
            return _result(False, "")

        language = ctx.noumeno.language if ctx.noumeno else ""
        prompt = self._build_scope_prompt(scope_prompt, ctx.user_input, language)
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
    def _build_scope_prompt(scope_prompt: str, user_input: str, language: str = "") -> str:
        # Pin the refusal language HARD (not a soft "in the user's language"): a
        # small model otherwise drifts to the wrong tongue (e.g. Spanish for a
        # pt-BR user) — same failure the voice/NOUMENO fixes addressed. Empty
        # language → no directive (let the model match the input).
        lang_name = language or "the user's language"
        lang_rule = (f"the refusal_message MUST be written in {language} "
                     "(the user's language), no other language") if language else \
                    "the refusal_message must be in the user's language"
        return (
            f"# Scope Definition\n{scope_prompt}\n\n"
            f'# User Input\n"{user_input}"\n\n'
            "# Task\nIs the User Input IN-SCOPE or OUT-OF-SCOPE? Rules:\n"
            "- Block ONLY what is clearly, obviously unrelated to the scope.\n"
            "- When in doubt, ALLOW (false positives are NOT acceptable).\n"
            "- Greetings, follow-ups, clarifications and questions about the "
            "business/product are ALWAYS in-scope.\n"
            f"- If blocked, {lang_rule}.\n\n"
            "# Examples\n"
            'User: "how do I bake a cake?" → blocked=true\n'
            'User: "who is the president?" → blocked=true\n'
            'User: "how much is the plan?" → blocked=false\n'
            'User: "thanks for the help" → blocked=false\n\n'
            'Respond ONLY with: {"blocked": true/false, "refusal_message": '
            f'"...polite refusal in {lang_name} if blocked, else empty..."}}'
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
            if approved:
                logger.info("stage=superego event=judge approved=true")
            else:
                # A rejection feeds the EGO↔SUPEREGO correction loop — surface it.
                logger.warning("stage=superego event=judge approved=false critique=%s",
                               (critique or "")[:80])
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
        # User-stated pragmatic restrictions (NER signals): the judge must verify
        # the execution honored them — including what the user forbade.
        restrictions = self._format_restrictions(ctx.intent)
        # Terms the NOUMENO preserved verbatim (names/URLs/emails/figures): the
        # judge uses them as concrete grounding evidence (2R-A).
        preserved = self._format_preserved(ctx)
        return (
            f'# User request\n"{ctx.user_input}"\n\n'
            f"# Active goal\n{goal}\n"
            f"{restrictions}"
            f"{preserved}"
            f"{limits}\n"
            f"# What the EGO executed\n{executed}\n\n"
            f"# EGO draft\n{draft}\n\n"
            "# Judge the EXECUTION against these criteria (most important first):\n"
            "1. GOAL↔EXECUTION: did it do exactly what was asked (X, not Y)?\n"
            "2. CONSTRAINTS: did it honor every user restriction (and NOT do what was forbidden)?\n"
            "3. COMPLETENESS: was the goal fully met (not partial)?\n"
            "4. GROUNDING: is everything backed by the tool results (no invented data), "
            "and are the preserved terms (if any) reproduced exactly?\n"
            "5. SAFETY/LIMITS: within the persona's limits, no policy violation?\n\n"
            "EXCEPTION — an honestly-relayed tool FAILURE is a VALID outcome: when a tool "
            "returned ERROR (a business refusal like a taken slot or a reached limit) and the "
            "draft truthfully reports that failure without fabricating success, APPROVE — "
            "a retry cannot fix a business refusal, and telling the user is the right action. "
            "Still REJECT a draft that claims success despite an ERROR result.\n\n"
            'Respond ONLY with: {"approved": true/false, "critique": '
            '"...if not approved, what is wrong, to guide a retry..."}'
        )

    @staticmethod
    def _format_restrictions(intent) -> str:
        """Render user constraints/negation for the judge prompt (empty if none)."""
        if not intent:
            return ""
        lines = []
        if intent.constraints:
            lines.append(f"Constraints (must respect): {', '.join(intent.constraints)}")
        if intent.negation:
            lines.append(f"Must NOT: {', '.join(intent.negation)}")
        return "# User constraints\n" + "\n".join(lines) + "\n" if lines else ""

    @staticmethod
    def _format_preserved(ctx: PipelineContext) -> str:
        """Render NOUMENO preserved terms as grounding evidence for the judge."""
        terms = [t for t in (ctx.noumeno.preserved_terms if ctx.noumeno else []) if (t or "").strip()]
        if not terms:
            return ""
        return "# Preserved terms (must be reproduced verbatim)\n" + ", ".join(terms) + "\n"

    @staticmethod
    def _preserved_mutated(preserved: list[str], payload: str, response: str) -> bool:
        """Flag-only grounding backstop: a CRITICAL preserved term (figure/email/
        URL) the executor grounded (present in ``payload``) shows up ALTERED in the
        response. Mutation-of-present only — a same-kind token must appear in the
        reply but differ; mere absence is NOT flagged (forcing every term in would
        be nonsense). See ``docs`` / 2R-A."""
        for term in preserved:
            term = (term or "").strip()
            if not term or not _CRITICAL_TERM_RE.search(term):
                continue
            if term not in payload or term in response:
                continue  # out of grounded scope, or reproduced verbatim → fine
            if SuperegoStage._same_kind_altered(term, response):
                return True
        return False

    @staticmethod
    def _same_kind_altered(term: str, response: str) -> bool:
        """Does a same-kind token appear in ``response`` but differ from ``term``?"""
        if "@" in term:
            return "@" in response and term not in response
        if re.match(r"https?://", term, re.IGNORECASE):
            return bool(re.search(r"https?://", response, re.IGNORECASE)) and term not in response
        # Numeric: a response figure is a digit-drop/add variant of the term's
        # figure (one digit-string is a prefix of the other but they differ).
        # Catches 1000→100 without flagging unrelated numbers (e.g. "2 items").
        td = re.sub(r"\D", "", term)
        if not td:
            return False
        for rn in _NUM_RE.findall(response):
            rd = re.sub(r"\D", "", rn)
            if rd and rd != td and (td.startswith(rd) or rd.startswith(td)):
                return True
        return False

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
            logger.warning("stage=superego event=pii_flagged_in_output")

        # Deterministic preserved-term backstop on the OUTPUT (2R-A) — flag-only,
        # never auto-inject. Fires only when a CRITICAL term (figure/email/URL)
        # that the executor grounded appears ALTERED in the reply (mutation-of-
        # present), not on mere absence (the reply may legitimately omit it).
        preserved = ctx.noumeno.preserved_terms if ctx.noumeno else []
        if response and self._preserved_mutated(preserved, payload, response):
            adjustments.append("preserved:mutated_in_output")
            logger.warning("stage=superego event=preserved_mutated_in_output")

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
        language = ctx.noumeno.language if ctx.noumeno else ""
        # Register accommodation (sibling of Reply language): match the user's
        # formality where it does not conflict with the persona — the persona's
        # voice/limits always win.
        register = next((a for a in adjustments if a.startswith("register:")), None)
        if register:
            signals.append(
                f"User register: {register.split(':', 1)[1]} — match it where it does "
                "not conflict with the persona voice/limits (persona takes precedence)"
            )
        signals.append(f"Tone hints: {', '.join(adjustments)}")
        # Host-injected context (retrieved memories / history / clock) — the same
        # block the EGO sees; included so memories can ground the final reply.
        injected = ctx.metadata.get("ego_context")
        context_section = f"# Context (memories/history)\n{str(injected).strip()}\n\n" if injected else ""
        # The reply language is a HARD instruction (leading the Task), not a soft signal —
        # a small model otherwise drifts into another language when the user's turn is short
        # (e.g. a bare "sim"). Empty language → no directive (let the model match the input).
        lang_rule = (f"Write the reply IN {language} (the user's language) — the ENTIRE reply, "
                     "with no other language. " if language else "")
        return (
            f'# User request\n"{ctx.user_input}"\n\n'
            f"{context_section}"
            f"# Data gathered by the executor (ground figures/dates ONLY in this)\n{payload}\n\n"
            f"# Signals\n" + "\n".join(signals) + "\n\n"
            f"# Task\n{lang_rule}Write the final reply to the user in the persona's voice and "
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
