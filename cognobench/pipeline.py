"""
cognobench.pipeline — a REFERENCE orchestrator wiring all stages end-to-end.

This is the "host glue" the cogno-anima library deliberately does NOT ship:
orchestration (control flow + the correction loop + signal handling + atomicity
hooks) is the host's job. It lives in cognobench (decoupled, not in the wheel) as
an **executable spec** of how a host composes the stages, and so the e2e tests
can exercise the seams (routing, EGO↔SUPEREGO correction loop, retry_metrics,
signals) that per-stage tests cannot.

Atomicity is represented by ``on_rollback``/``on_commit`` hooks (no-ops here);
a real host opens a DB tx / write-behind buffer / outbox there.
"""

from __future__ import annotations

from typing import Callable, Optional

from cogno_anima.stages.noumeno import Noumeno
from cogno_anima.stages.ner import IntentAnalyzer
from cogno_anima.stages.id import IDStage
from cogno_anima.stages.ego import EgoStage
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.tools import ToolDispatcher
from cogno_anima.llm import LLMBackend, Embedder
from cogno_anima.types import PipelineContext, StageMetrics, SuperegoResult


class ReferencePipeline:
    """NOUMENO → NER → ID → [guard] → EGO ⇄ SUPEREGO(judge) → SUPEREGO(voice) → Drift."""

    def __init__(self, *, prompts_dir, embedder: Embedder, slangs=None) -> None:
        self._embedder = embedder
        self._noumeno = Noumeno(embedder=embedder, prompts_dir=prompts_dir, slangs=slangs or {})
        self._ner = IntentAnalyzer(prompts_dir=prompts_dir)
        self._id = IDStage()
        self._ego = EgoStage()
        self._superego = SuperegoStage()

    async def run_turn(
        self,
        ctx: PipelineContext,
        *,
        gen_backend: LLMBackend,        # NOUMENO/NER/scope/judge (JSON-capable)
        ego_backend: LLMBackend,        # EGO executor (FC or text fallback)
        dispatcher: ToolDispatcher,
        ego_prompt: str,
        scope_prompt: str = "",
        limits_prompt: str = "",
        voice_prompt: str = "",
        voice_backend: Optional[LLMBackend] = None,
        max_corrections: int = 2,
        on_rollback: Optional[Callable[[PipelineContext], None]] = None,
        on_commit: Optional[Callable[[PipelineContext], None]] = None,
    ) -> PipelineContext:
        voice_backend = voice_backend or ego_backend

        # ── perception + routing ──────────────────────────────────────
        ctx = await self._noumeno.process(ctx, gen_backend)
        ctx = await self._ner.process(ctx, gen_backend)
        ctx = await self._id.process(ctx, self._embedder)

        # ── PII-CRITICAL gate (from ID) ───────────────────────────────
        if ctx.id_result and ctx.id_result.blocked:
            ctx.superego_result = self._superego._blocked_response(ctx)
            ctx.stop_reason = "pii_blocked"
            return ctx

        # ── Early scope guard (optional) ──────────────────────────────
        if scope_prompt:
            scope = await self._superego.check_input_scope(ctx, gen_backend, scope_prompt=scope_prompt)
            ctx.retry_metrics.append(scope.metrics)
            if scope.blocked:
                ctx.superego_result = SuperegoResult(
                    response=scope.refusal_message, blocked=True,
                    metrics=StageMetrics(stage="superego_voice", elapsed_ms=0.0,
                                         tokens_in=0, tokens_out=0, model="none"))
                ctx.stop_reason = "scope_blocked"
                return ctx

        # ── EGO route: execute + correction loop ──────────────────────
        if ctx.id_result and ctx.id_result.triad_route == "EGO":
            attempt = 1
            judge = None
            while True:
                ctx = await self._ego.process(ctx, ego_backend, dispatcher, system_prompt=ego_prompt)
                judge = await self._superego.evaluate(ctx, gen_backend, limits_prompt=limits_prompt)
                ctx.retry_metrics.append(judge.metrics)   # judge never the "main" superego (voice is)
                if judge.approved or attempt >= max_corrections:
                    break
                # rejected → this EGO attempt becomes retry history; feed critique back
                if ctx.ego_result:
                    ctx.retry_metrics.append(ctx.ego_result.metrics)
                if on_rollback:
                    on_rollback(ctx)
                ctx.metadata["ego_correction"] = {"reason": judge.critique, "attempt": attempt + 1}
                attempt += 1

            if judge is not None and not judge.approved:
                ctx.needs_handoff = True
                ctx.stop_reason = "human_handoff"
                return ctx
            if on_commit:
                on_commit(ctx)

        # ── voice (writes the final response; for EGO and non-task paths) ──
        ctx.superego_result = await self._superego.voice(ctx, voice_backend, voice_prompt=voice_prompt)
        return ctx
