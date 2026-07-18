import re
import time
import json
import asyncio
import logging
from pathlib import Path
from typing import Optional

from cogno_anima import metakeys as mk
from cogno_anima.types import PipelineContext, NoumenoResult, StageMetrics
from cogno_synapse import LLMBackend, Embedder
from cogno_anima.utils import expand_slangs, parse_json_object
from cogno_anima.prompts import load_prompt
from cogno_anima.errors import StageParseError

logger = logging.getLogger("cogno_anima.noumeno")

STAGE_NAME = "noumeno"



def classify_drift(score: float) -> str:
    """Classifies the drift score into corresponding tags."""
    if score == 0.0:
        return "PASS_THROUGH"
    if score <= 0.20:
        return "REWRITTEN"
    if score <= 0.40:
        return "COMPRESSED"
    if score <= 0.60:
        return "EXPANDED"
    return "DRIFT"


class Noumeno:
    """
    NOUMENO Stage — Perception and Normalization layer.
    """
    name = STAGE_NAME

    def __init__(
        self,
        embedder: Embedder,
        prompts_dir: Optional[Path] = None,
        slangs: Optional[dict[str, str]] = None,
        subject_threshold: float = 0.65,
        default_language: Optional[str] = None,
    ):
        self._embedder = embedder
        self._slangs = slangs or {}
        self._subject_threshold = subject_threshold
        # Host/tenant default language (e.g. the SaaS tenant setting). When set,
        # it is used whenever a request does not carry its own language, so the
        # tenant language is the default path and langdetect is only a last
        # resort. The library ships no business default (stays None).
        self._default_language = default_language

        # Load prompts
        self._system = load_prompt("noumeno", "system.txt", prompts_dir=prompts_dir)
        self._user_tpl = load_prompt("noumeno", "user.txt", prompts_dir=prompts_dir)

    async def process(self, ctx: PipelineContext, llm: LLMBackend) -> PipelineContext:
        """
        Runs the NOUMENO stage on the context.
        """
        t0 = time.perf_counter()
        user_input = ctx.user_input

        # 1. Normalized Input (Slang expansion)
        normalized_input = expand_slangs(user_input, self._slangs)

        # 2. Language resolution.
        #    Precedence: per-request tenant language (ctx.force_language)
        #    > stage default (host/tenant global config) > langdetect fallback.
        detected_lang = "und"
        if ctx.force_language:
            detected_lang = ctx.force_language
        elif self._default_language:
            detected_lang = self._default_language
        else:
            try:
                from langdetect import detect
                detected_lang = await asyncio.to_thread(detect, normalized_input)
            except Exception as le:
                logger.warning("Failed to detect language, defaulting to 'und': %s", le)

        # 3. Subject Continuity Check (pre-LLM)
        last_rewritten = ctx.metadata.get(mk.LAST_REWRITTEN)
        last_context_turn = ctx.metadata.get(mk.LAST_CONTEXT_TURN)

        subject_similarity = 1.0
        change_subject = False

        # Accumulate embedding cost (tokens + call count) across every similarity
        # call this stage makes, so it surfaces in StageMetrics just like the LLM
        # generate tokens do.
        emb_tokens = 0
        emb_calls = 0

        if last_rewritten:
            # Concurrent embed calls for input and history
            subject_similarity, t, c = await self._similarity(normalized_input, last_rewritten)
            emb_tokens += t
            emb_calls += c
            change_subject = subject_similarity < self._subject_threshold

        # 4. Formulate Prompt
        # The recent transcript (user + assistant) is injected UNCONDITIONALLY — a short reply
        # ("com o Vinicius Vale", "às 14h", "sim") is embedding-dissimilar to the last query, so
        # the subject-continuity gate would drop it; but it must still resolve against what the
        # assistant just asked. The single-query hint stays gated by continuity (it is a summary,
        # not the back-and-forth).
        conversation = (ctx.metadata.get(mk.CONVERSATION_HISTORY) or "").strip()
        history_parts = []
        if conversation:
            history_parts.append(f"Conversation so far:\n{conversation}")
        if not change_subject and last_rewritten:
            history_parts.append(
                f"Last Query (English): {last_rewritten}\n"
                f"Last Context Summary: {last_context_turn or ''}")
        history_str = ("\n\n".join(history_parts) + "\n\n") if history_parts else ""

        prompt = self._user_tpl.format(history=history_str, input=normalized_input)

        # 5. Call LLM
        raw_response, tokens_in, tokens_out = await llm.generate(self._system, prompt)

        # 6. Parse JSON Response
        data = self._parse_json(raw_response)
        rewritten = data.get("rewritten", "").strip() or user_input
        context_turn = data.get("context_turn", "").strip()
        confidence = float(data.get("confidence", 1.0))
        changed = bool(data.get("changed", False))
        preserved_terms = list(data.get("preserved_terms", []))
        rewrite_warnings = list(data.get("rewrite_warnings", []))

        # 7. Drift Computation (post-LLM)
        sim, t, c = await self._similarity(user_input, rewritten)
        emb_tokens += t
        emb_calls += c
        drift_score = round(1.0 - sim, 4)

        # Telemetry: build metrics after all embedding work so embedding cost and
        # elapsed time cover the whole stage.
        elapsed_ms = (time.perf_counter() - t0) * 1000
        metrics = StageMetrics(
            stage=STAGE_NAME,
            elapsed_ms=round(elapsed_ms, 2),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            embedding_tokens=emb_tokens,
            embedding_calls=emb_calls,
            model=llm.model,
        )

        # Reconciliation: if drift > 0.50, force changed = True & drift_tag = "DRIFT"
        if drift_score > 0.50:
            changed = True
            drift_tag = "DRIFT"
        else:
            drift_tag = classify_drift(drift_score)

        ctx.noumeno = NoumenoResult(
            original=user_input,
            rewritten=rewritten,
            context_turn=context_turn if not change_subject else "",
            language=detected_lang,
            canonical_language="en",
            drift_score=drift_score,
            drift_tag=drift_tag,
            changed=changed,
            confidence=confidence,
            change_subject=change_subject,
            subject_similarity=subject_similarity,
            context_used=bool(context_turn) and not change_subject,
            preserved_terms=preserved_terms,
            rewrite_warnings=rewrite_warnings,
            metrics=metrics
        )

        logger.info(
            "NOUMENO lang=%s drift=%.2f tag=%s changed=%s change_subject=%s",
            detected_lang, drift_score, drift_tag, changed, change_subject,
        )
        return ctx

    async def _similarity(self, a: str, b: str) -> tuple[float, int, int]:
        """Cosine similarity plus embedding cost ``(similarity, tokens, calls)``.

        Prefers a usage-aware embedder (``similarity_with_usage``, e.g.
        CachingEmbedder/OllamaEmbedder); falls back to the plain ``similarity``
        protocol method (0 tokens) so any Embedder implementation still works.
        Each similarity is counted as 2 embed operations.
        """
        usage_fn = getattr(self._embedder, "similarity_with_usage", None)
        if usage_fn is not None:
            sim, tokens = await usage_fn(a, b)
            return sim, tokens, 2
        sim = await self._embedder.similarity(a, b)
        return sim, 0, 2

    def _parse_json(self, raw: str) -> dict:
        cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            # Backends without format="json" (cloud) sometimes emit more than one top-level object
            # ("Extra data") — e.g. an empty/partial {} before the real one, or the object trailed
            # by prose. Pick the RICHEST object (not the first — a leading {} must not silently
            # win). No object at all → visible StageParseError.
            try:
                data = parse_json_object(cleaned)
            except ValueError:
                raise StageParseError(STAGE_NAME, raw, exc) from exc
        # Valid JSON that is not an object (e.g. "5", "[]", "true") would crash
        # the field reads downstream with a raw AttributeError — treat it as a
        # parse failure so the contract stays "valid dict OR StageParseError".
        if not isinstance(data, dict):
            raise StageParseError(STAGE_NAME, raw, TypeError("JSON is not an object"))
        return data
