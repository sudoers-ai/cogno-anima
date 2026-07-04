"""
Single source of truth for the NER closed vocabularies.

These sets are the canonical contract. The NER stage (`cogno_anima/stages/ner.py`)
validates LLM output against them, and the NER prompt
(`cogno_anima/prompt_templates/ner/system.txt`) enumerates the SAME values. The alignment between
this module and the prompt is enforced by `tests/unit/test_pipeline.py`, so the
two can never silently drift apart (the bug class that caused `GENERAL` to be
dropped and `LOGIC` to be accepted-but-never-taught).

Add or change a vocabulary value HERE and in the prompt together — the test
will fail until both agree.
"""

from __future__ import annotations

VALID_INTENTS: set[str] = {
    "INFORMATION_REQUEST", "ACTION_REQUEST", "CLARIFICATION",
    "CREATIVE_TASK", "SOCIAL", "UNKNOWN",
}

VALID_SENTIMENTS: set[str] = {
    "POSITIVE", "NEGATIVE", "NEUTRAL", "CURIOUS", "FRUSTRATED", "URGENT", "PLAYFUL",
}

VALID_TEMPORAL: set[str] = {"RECENT", "HISTORICAL", "TIMELESS", "MIXED"}

VALID_TRIAD: set[str] = {"ID", "EGO", "SUPEREGO", "BALANCED"}

# ID stage (Stage 3) vocabularies. The ID is heuristic (no LLM), so these are
# not enumerated in any prompt — they are the closed contract the stage
# sanitizes its own routing/continuity output against ("never trust" applies to
# carry-over state and host-injected hints just as it does to LLM output).
VALID_GOAL_STATUS: set[str] = {"NEW", "ONGOING", "COMPLETED", "ABANDONED"}

VALID_COMPLEXITY: set[str] = {"LOW", "MEDIUM", "HIGH", "EXPERT"}

# Closed vocabulary for `PipelineContext.stop_reason` — the terminal signal the
# core emits for the host to act on. "completed" is the happy path; the others
# are early-exits/escalations. The ACTION is always the host's (escalate to a
# human, serve the cache, send a refusal) — the core only sets the signal.
VALID_STOP_REASONS: set[str] = {
    "completed", "human_handoff", "semantic_cache", "scope_blocked", "pii_blocked",
    # The judge rejected the EGO execution but nothing was committed (the EGO only ran
    # READ tools — no mutating dispatch), so instead of dead-ending in a human handoff the
    # SUPEREGO voices a grounded continuation ("I found your appointment — change it to
    # 11:00?"). The turn is still terminal for the core; the HOST owns the escalation
    # policy (e.g. force a real handoff after N consecutive clarifications).
    "needs_clarification",
}

VALID_MODALITY: set[str] = {"CERTAIN", "PROBABLE", "POSSIBLE", "UNCERTAIN", "MIXED"}

VALID_SPEECH_ACTS: set[str] = {
    "DIRECTIVE", "EXPRESSIVE", "COMMISSIVE", "CONSTATIVE", "INTERROGATIVE", "MIXED",
}

VALID_PAROLE: set[str] = {
    "COLOQUIAL", "TECNICO", "ACADEMICO", "FORMAL", "GIRIA", "POETICO", "MIXED",
}

# Cognitive mode tags (returned short; prefixed "NER." by the code).
VALID_MANDATORY: set[str] = {
    "SYSTEM", "ANALYSIS", "MATH", "CREATIVE", "LINGUISTIC", "UNKNOWN",
}

# Aristotelian categories.
VALID_ARISTOTELIAN: set[str] = {
    "SUBSTANCE", "QUANTITY", "QUALITY", "RELATION", "PLACE",
    "TIME", "POSITION", "STATE", "ACTION", "PASSION",
}

# Knowledge-domain closed list — MUST match the `domains` list in the NER prompt.
NER_KNOWLEDGE_DOMAINS: set[str] = {
    "TECH", "SCIENCE", "HEALTH", "FINANCE", "LOGISTICS", "TRAVEL",
    "HISTORY", "LAW", "PHILOSOPHY", "EDUCATION", "CULTURE", "NEWS", "GENERAL",
}
