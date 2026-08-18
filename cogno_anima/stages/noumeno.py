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
from cogno_anima.utils import (DEFAULT_CONFIDENCE, expand_slangs,
                               generate_json_resilient, parse_json_object)
from cogno_anima.prompts import load_prompt
from cogno_anima.errors import StageParseError

logger = logging.getLogger("cogno_anima.noumeno")

STAGE_NAME = "noumeno"

# Few-shot echo backstop. Measured live (CLOSER, 2026-08-18): on a bare "Sim" the model
# copied a few-shot OUTPUT byte-for-byte instead of resolving against the real conversation
# ("Yes, I would like to know more about how to configure it." — nothing in the conversation
# was about configuring), fabricating the task the whole downstream pipeline then executed.
# The stage therefore knows its own example outputs and treats a match as suspect.
FEW_SHOT_ECHO = "FEW_SHOT_ECHO"
# Expansion (and therefore parroting) risk exists only on short replies; a longer input
# matching an example is a legitimate coincidence ("qual o preço do bitcoin?").
_ECHO_MAX_INPUT_WORDS = 4
# A copy with a swapped head ("Yes, …" for the example's "Maybe, …") keeps the example's
# distinctive tail — compare the last N content words, not just the whole string.
_ECHO_TAIL_WORDS = 4


def _echo_norm(text: str) -> tuple[str, ...]:
    """Punctuation/case-insensitive word tuple for echo comparison."""
    return tuple(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _context_supported(rewritten: str, context_text: str) -> bool:
    """At least one distinctive word of the rewrite (4+ chars) appears in the given
    context (input + conversation). Names survive translation, so a rule-illustration
    match that is real resolution ("com o Vinicius Vale", or "sim" right after the
    assistant offered Vinicius Vale) passes; a fabricated one has no anchor."""
    ctx_words = set(_echo_norm(context_text))
    return any(w in ctx_words for w in _echo_norm(rewritten) if len(w) >= 4)


def _split_examples(system: str) -> tuple[str, tuple[tuple[str, ...], ...],
                                          tuple[tuple[str, ...], ...]]:
    """Split the system prompt at its ``Examples:`` heading and harvest every rewrite the
    model could parrot — ``"rewritten"`` values and inline ``… becomes "…"`` illustrations,
    from BOTH regions. Returns ``(prompt_without_examples, example_rewrites,
    rule_rewrites)``: the first set lives in the Examples block, absent from the retry
    prompt, so a retry answer is parrot-proof against it; the second is embedded in the
    RULES ("… becomes \"Book with Dr. Vinicius Vale\""), which the retry still sees, so
    those can never earn the retry's unconditional trust. The Examples block is assumed
    terminal — true for the shipped prompt; a custom ``prompts_dir`` that appends rules
    after it keeps its own risk (they would be absent from the retry prompt too)."""
    idx = system.find("\nExamples:")
    head, tail = (system, "") if idx < 0 else (system[:idx], system[idx:])

    def _harvest(text: str) -> tuple[tuple[str, ...], ...]:
        vals = re.findall(r'"rewritten"\s*:\s*"([^"]*)"', text)
        vals += re.findall(r'becomes\s+"([^"]+)"', text)
        return tuple(_echo_norm(v) for v in vals if v.strip())

    return head, _harvest(tail), _harvest(head)



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
        # Echo backstop material: the prompt minus its Examples block (the retry prompt)
        # and the normalized rewrites the model could copy, per region. Logged so an
        # inert backstop (custom prompt in a format the harvest misses) is visible.
        (self._system_sans_examples, self._example_rewrites,
         self._rule_rewrites) = _split_examples(self._system)
        logger.info("stage=noumeno echo_backstop examples=%d rule_illustrations=%d",
                    len(self._example_rewrites), len(self._rule_rewrites))

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

        # 5. Call LLM  +  6. Parse JSON Response
        # A response cut mid-stream is transient and buys exactly one more attempt; anything
        # else still raises on the first. Tokens are summed across attempts so a retry shows
        # up in metering instead of being billed invisibly.
        data, tokens_in, tokens_out = await generate_json_resilient(
            llm, self._system, prompt, self._parse_json, stage=STAGE_NAME)
        rewritten = data.get("rewritten", "").strip() or user_input

        # ── Few-shot echo backstop ────────────────────────────────────
        # The rewrite of a short reply matches one of the prompt's own example outputs —
        # the model may have copied the example instead of resolving against the real
        # conversation (observed live: "Sim" → the configure-it example, verbatim, in a
        # conversation about nothing of the sort). Retry ONCE without the Examples block:
        # the retry cannot parrot what is not in its context, so its answer is trusted
        # unconditionally — if it reproduces the same sentence, the match was genuine
        # resolution, not a copy. The backstop must never kill a turn the primary call
        # already served: an unparseable retry keeps the first answer but FLAGS it
        # (rewrite_warnings → the ID's clarification_suggested doubt signal).
        if self._matches_example(rewritten, normalized_input, self._example_rewrites):
            logger.warning(
                "stage=noumeno event=few_shot_echo rewritten=%r — retrying without the "
                "examples block", rewritten)
            try:
                data2, t2, o2 = await generate_json_resilient(
                    llm, self._system_sans_examples, prompt, self._parse_json,
                    stage=STAGE_NAME)
                tokens_in += t2
                tokens_out += o2
                retried = data2.get("rewritten", "").strip() or user_input
                if _echo_norm(retried) == _echo_norm(rewritten):
                    logger.info("stage=noumeno event=few_shot_echo_genuine — reproduced "
                                "without the examples in context")
                data = data2
                rewritten = retried
            except StageParseError:
                data = dict(data)
                data["rewrite_warnings"] = (
                    [FEW_SHOT_ECHO] + list(data.get("rewrite_warnings", [])))[:2]
                logger.warning("stage=noumeno event=few_shot_echo_retry_unparseable — "
                               "keeping the first answer, flagged")

        # Rule-illustration echo (checked AFTER the retry so it also covers the retry's
        # answer): the copied string lives in the RULES, which the retry prompt keeps, so
        # no retry can clear it — and it is also exactly what a legitimate resolution
        # looks like ("com o Vinicius Vale" → "Book with Dr. Vinicius Vale"). A match only
        # counts as fabrication when the rewrite has no lexical support in the input or
        # the conversation (none of its distinctive words appear there — the wrong-name
        # case: "com a Ana" rewritten as booking Vinicius Vale). Flag, never block.
        if (self._matches_example(rewritten, normalized_input, self._rule_rewrites)
                and not _context_supported(rewritten, normalized_input + " " + conversation)):
            data = dict(data)
            data["rewrite_warnings"] = (
                [FEW_SHOT_ECHO] + list(data.get("rewrite_warnings", [])))[:2]
            logger.warning("stage=noumeno event=few_shot_echo_rule_illustration "
                           "rewritten=%r — unsupported by input/conversation, flagged",
                           rewritten)

        context_turn = data.get("context_turn", "").strip()
        confidence = float(data.get("confidence", DEFAULT_CONFIDENCE))
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
            # KEPT on a first turn even though there is no history to summarise. With no
            # `last_rewritten` the model fills this with a restatement of the CURRENT turn
            # ("The user is asking for a comparison between…"), which is not prior context —
            # but the NER is the only consumer and it uses the field as a hint, measurably:
            # clearing it flipped `temporal` from MIXED to RECENT on
            # `compare a inflação atual com a de 2010`, 3 runs out of 3 either way.
            #
            # So the two halves are separated. The paraphrase is a useful signal and stays; the
            # LIE is `context_used` below, which claimed prior context had been used on a first
            # contact. Removing a working input to fix a mislabelled flag would have traded real
            # accuracy for tidiness.
            # `context_turn` survives a subject change; the THREAD's continuity does not.
            #
            # `change_subject` is a cosine against LAST_REWRITTEN, so when the previous turn was
            # a short reply its rewrite carries no content ("Claro" -> "Sure.") and everything
            # after it reads as a topic change: measured firing on 8 of 10 real CLOSER turns,
            # and 15 of 15 on the sequence that produced the live defect (cosine 0.522). The
            # transcript is already injected UNCONDITIONALLY for exactly that reason (anima
            # #52); this field was left gated on the same boolean that fix declared unreliable.
            #
            # And it is the only channel carrying the conversational frame to the NER. On the
            # turn that broke live, the model wrote the answer and the code dropped it one line
            # later: "The user is PROVIDING INFORMATION about the average volume of daily
            # customer service calls received." The NER, without it, read a lead ANSWERING a
            # discovery question as one ASKING for an arithmetic mean.
            #
            # The distinction that makes this safe: `context_turn` describes THIS utterance in
            # light of what was just said — even a topic-changing message is interpreted against
            # the assistant's last question. The thread's continuity (prior_goal,
            # active_domains) is a different thing and still drops; see ner.py.
            #
            # Measured, gpt-4o-mini, real embedder: goal naming its referent 2/15 -> 8/15
            # (p=0.050). SECRETARY suite 42/45 -> 41/45 — one check, on a holiday scenario,
            # inside the run-to-run spread; reschedule/cancel/long_meander/troll all identical.
            context_turn=context_turn if (not change_subject or conversation) else "",
            language=detected_lang,
            canonical_language="en",
            drift_score=drift_score,
            drift_tag=drift_tag,
            changed=changed,
            confidence=confidence,
            change_subject=change_subject,
            subject_similarity=subject_similarity,
            context_used=bool(context_turn) and not change_subject and bool(last_rewritten),
            preserved_terms=preserved_terms,
            rewrite_warnings=rewrite_warnings,
            metrics=metrics
        )

        logger.info(
            "NOUMENO lang=%s drift=%.2f tag=%s changed=%s change_subject=%s",
            detected_lang, drift_score, drift_tag, changed, change_subject,
        )
        return ctx

    @staticmethod
    def _matches_example(rewritten: str, normalized_input: str,
                         examples: tuple[tuple[str, ...], ...]) -> bool:
        """True when a SHORT reply's rewrite matches one of the given harvested
        rewrites — exactly, or by the example's distinctive tail (a copy with a swapped
        head: "Yes, …" for the example's "Maybe, …"). A rewrite that merely repeats the
        input is pass-through, never a parrot; a long input is out of expansion territory
        and a match there is coincidence."""
        if not examples:
            return False
        inp = _echo_norm(normalized_input)
        if len(inp) > _ECHO_MAX_INPUT_WORDS:
            return False
        rw = _echo_norm(rewritten)
        if not rw or rw == inp:
            return False
        for ex in examples:
            if rw == ex:
                return True
            if (len(ex) > _ECHO_TAIL_WORDS and len(rw) >= _ECHO_TAIL_WORDS
                    and rw[-_ECHO_TAIL_WORDS:] == ex[-_ECHO_TAIL_WORDS:]):
                return True
        return False

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
