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

# The sentiments that mean "this conversation is going badly WITH US". A streak is counted over
# the FAMILY, not over one repeated label: measured on a real WhatsApp conversation (2026-08), a
# user escalated FRUSTRATED → NEGATIVE ("IA burra") and the streak RESET on the escalating turn,
# because it demanded the same label twice. Escalation moves through labels — that is what
# escalation IS — so a counter keyed on one label never fires exactly when it matters most.
#
# URGENT is deliberately OUT. It is about the user's TASK, not about us: someone writing "preciso
# marcar hoje, é urgente" twice is in a hurry, not dissatisfied. And the cost of getting this
# wrong is not cosmetic — `emotional_override` outranks the ACTION/INFORMATION→EGO branch in
# `_resolve_route`, so two urgent turns would route away from the tool gateway and the booking
# would never be dispatched, for the rest of the session. A hurried user is the LAST one who
# should lose their tools. (`cogno_host.emotion._SOMBER_SENTIMENTS` groups the same three labels
# for a different purpose — picking a somber TTS delivery — where urgency does belong. Sharing
# the triple across the two would be a coincidence, not a shared meaning.)
NEGATIVE_SENTIMENTS: frozenset[str] = frozenset({"FRUSTRATED", "NEGATIVE"})

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

# SUPEREGO voice — the persona's DECLARED personality traits (host → `metakeys.VOICE_TRAITS`).
# Closed on purpose: a trait is a deterministic directive the voicer renders verbatim, not a
# free-text prompt fragment the tenant writes (that is `custom_rules`, which already exists).
# The vocabulary is small and each value maps to ONE rendered instruction in
# `SuperegoStage._TRAIT_DIRECTIVES`; the two must stay aligned (a unit test enforces it).
# Traits shape HOW the reply is said — never WHAT: figures, limits and refusals are untouched,
# and the persona's voice/limits prompt still outranks them. They are the persona's, so they
# outrank the CONTACT's `register:*` accommodation (which yields "where it does not conflict").
VALID_VOICE_TRAITS: set[str] = {
    "warm", "reserved",        # warmth axis
    "direct",                  # lead with the answer
    "formal", "casual",        # formality axis (the persona's own, vs. the user's register)
    "humorous",                # a light touch, never on bad news / refusals / PII
    "concise", "detailed",     # length axis
    "empathetic",              # name the user's situation before answering
}

# Mutually exclusive pairs. A configuration that declares BOTH sides of an axis is a
# contradiction the model would resolve by coin-flip; the sanitizer drops BOTH sides (no trait
# beats a self-contradicting instruction) and logs it. Order-independent.
VOICE_TRAIT_CONFLICTS: frozenset[frozenset[str]] = frozenset({
    frozenset({"warm", "reserved"}),
    frozenset({"formal", "casual"}),
    frozenset({"concise", "detailed"}),
})

# Cap on rendered traits. Beyond a handful the directives stop being a personality and become a
# second voice prompt — the tenant has `custom_rules` for that. Extra values are dropped in order.
MAX_VOICE_TRAITS: int = 4

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
