"""
Single source of truth for the NER closed vocabularies.

These sets are the canonical contract. The NER stage (`cogno_core/stages/ner.py`)
validates LLM output against them, and the NER prompt
(`prompts/ner/system.txt`) enumerates the SAME values. The alignment between
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
