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

import re
import math
from typing import Any, Optional

from cogno_anima.utils import as_count

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

# ── Outgoing-PII provenance (SUPEREGO output backstop) ───────────────────────
# Why a personal datum found in the OUTGOING text was, or was not, allowed to leave. The rule the
# owner decided (2026-08-25): "PII may come IN, it must never go OUT" — except the contact's own,
# which they are entitled to hear back (confirming an e-mail they just typed is the ANSWER, not a
# leak). The vocabulary is closed because it is an audit alphabet: it lands in
# `SuperegoResult.adjustments` as `pii:provenance_<value>`, which the host counts, and a
# free-form reason would make "is this net working or getting in the way" uncountable. Which of
# these values means "withheld" is read off the vocabulary itself (`PII_PROVENANCE_UNKNOWN`), so
# there is no second token family that could drift out of step with this one.
#
# It is deliberately NOT a taxonomy of SOURCES (config / memory / graph / tool / third party).
# The SUPEREGO holds a string and an allowlist; it can tell "the contact said this" from "the
# contact did not", and nothing finer. Naming the source would be a guess, and the host's own
# bench (#549, scenarios 3 vs 3b) measured that source does not decide the question anyway —
# two values with the SAME provenance had opposite correct answers.
PII_PROVENANCE_TURN = "contact_turn"        # in THIS turn's message → may leave
PII_PROVENANCE_SESSION = "contact_session"  # said earlier in THIS session → may leave
# The reply is going to STAFF, not to a visitor. Provenance alone is under-determined and the
# merged bench (host #549, `a2aef70`) is what says so: its `tool_result_document_number` case is
# the tenant's own bookkeeper asking to re-read a ledger row HE wrote, and the answer key says
# withholding it is wrong. No rule that looks only at where the value came from can tell that
# from a patient asking for a doctor's CPF — the difference is WHO IS READING. `Identity.role`
# already carries it, so the bit is available rather than invented.
#
# Its scope is narrow on purpose, and the bench's own dissent is recorded with it: RBAC
# authorised a READ, not re-emission onto a chat transport, so this widens what may reach an
# employee's WhatsApp. It is a deliberate trade, not an oversight.
PII_PROVENANCE_READER_STAFF = "reader_is_staff"
PII_PROVENANCE_UNKNOWN = "not_from_contact"  # everything else → masked in place

VALID_PII_PROVENANCE: frozenset[str] = frozenset(
    {PII_PROVENANCE_TURN, PII_PROVENANCE_SESSION, PII_PROVENANCE_READER_STAFF,
     PII_PROVENANCE_UNKNOWN})

# ── How the outgoing-PII rule ACTS on what it decided ────────────────────────
# ONE switch, and the reasoning belongs with it rather than scattered across call sites.
#
# `observe` runs the entire rule — detection, provenance, the record — and then ships the text
# UNTOUCHED. It is the default, and the default was picked by arithmetic rather than by caution:
# on the merged bench the rule with the role bit still breaks one scenario in seven (all three of
# its decidable runs), and that scenario is the tenant's OWN reception number — the shape a
# scheduling persona says all day. Meanwhile production has shown ZERO leaks in 297 turns.
# Enforcing on those numbers trades a harm nobody has observed for one that would land on a real
# contact tomorrow. Observation converts the remaining uncertainty into production counts —
# `pii:would_redact_in_output` with the type — which is the evidence the missing second bit
# (tenant-declared quotable contacts) needs in order to be built for a real case.
#
# **The exit criterion, written down, because a mode with none is a switch nobody touches again.**
# Observation is only worth its name if someone can say what would end it, so:
#
#   Graduate a TENANT to `enforce` when, over at least PII_OBSERVATION_MIN_TURNS turns that
#   carried a SUPEREGO block, none of its `pii:withheld_<type>` stamps is that tenant's OWN
#   business data.
#
# The stamps carry the TYPE for exactly this reason: the classes have opposite verdicts and an
# undifferentiated count decides nothing. `ADDRESS` (a CEP) and `TAX_ID` (a CNPJ) are almost
# always the tenant's own — the frequent false positive. `NATIONAL_ID` is almost always a
# person's, and a person who is not the one reading — the leak. `PHONE` and `EMAIL` are the
# genuinely ambiguous pair (the reception's number and another patient's number are both PHONE),
# so they are the ones a human actually has to look at.
#
# **Why 200.** Rule of three: observing ZERO events in n trials bounds the true rate below 3/n at
# ~95% confidence, so zero stamps in 200 turns bounds the false-positive rate under 1.5%. It is
# also about an order of magnitude more than the sample that exists today — of the demo box's 297
# traces only 9 could carry a SUPEREGO block at all — which is what makes "we saw none" different
# from "we did not look". Fewer turns than this and a clean run is indistinguishable from an
# empty one.
#
# **Per tenant, and a DECISION, not an automatic flip.** Nothing in this package promotes anybody:
# a human reads the counts for one tenant and sets `mk.PII_OUTPUT_MODE`. A rule that graduated
# itself on a threshold would be enforcing on the strength of a quiet week.
PII_OBSERVATION_MIN_TURNS: int = 200

PII_MODE_OBSERVE = "observe"
PII_MODE_ENFORCE = "enforce"
VALID_PII_MODES: frozenset[str] = frozenset({PII_MODE_OBSERVE, PII_MODE_ENFORCE})
PII_MODE_DEFAULT = PII_MODE_OBSERVE

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
# Each value maps to ONE rendered instruction in `SuperegoStage._TRAIT_DIRECTIVES` (a unit test
# pins the alignment — the directives are prompt text and stay with the stage, the way the NER
# prompt stays a text file). Traits shape HOW the reply is said — never WHAT: figures, limits and
# refusals are untouched, and the persona's voice/limits prompt still outranks them. They are the
# persona's, so they outrank the CONTACT's per-turn signals (register, tone) — except PII and
# de-escalation, which the voice enforces in code (see `SuperegoStage._modulate_traits`).
#
# The AXES are the single datum: an axis is a pair whose two sides contradict each other, and a
# configuration that declares both is a contradiction the model would resolve by coin-flip — the
# sanitizer drops BOTH sides (no trait beats a self-contradicting instruction). The vocabulary and
# the conflict pairs are DERIVED from here, so a new axis cannot be added half-way.
VOICE_TRAIT_AXES: tuple[tuple[str, str], ...] = (
    ("warm", "reserved"),        # warmth
    ("formal", "casual"),        # formality (the persona's own, vs. the user's register)
    ("concise", "detailed"),     # length
    ("reserved", "humorous"),    # "no small talk" vs "one light remark" — the same coin-flip
)
# Traits with NO opposite (disjoint from every axis — a test pins it). `direct` and `empathetic`
# are worded to coexist (the answer comes first; the acknowledgment is woven in, never a
# preamble) — the directives, not this table, carry that guarantee.
VOICE_TRAIT_SINGLETONS: tuple[str, ...] = ("direct", "empathetic")

# Every side a trait contradicts (`reserved` sits on two axes). Derived, so it cannot drift.
VOICE_TRAIT_OPPOSITES: dict[str, frozenset[str]] = {
    t: frozenset(o for axis in VOICE_TRAIT_AXES if t in axis for o in axis if o != t)
    for axis in VOICE_TRAIT_AXES for t in axis
}

VALID_VOICE_TRAITS: frozenset[str] = frozenset(
    {t for axis in VOICE_TRAIT_AXES for t in axis} | set(VOICE_TRAIT_SINGLETONS))

# Ordered (a tuple, not a set of sets): the sanitizer reports drops in THIS order, so two
# workers — and the admin API's refusal message — never disagree on the order of a list.
VOICE_TRAIT_CONFLICTS: tuple[frozenset[str], ...] = tuple(frozenset(a) for a in VOICE_TRAIT_AXES)

# Cap on rendered traits. Beyond a handful the directives stop being a personality and become a
# second voice prompt — the tenant has `custom_rules` for that. Extra values are dropped in order.
MAX_VOICE_TRAITS: int = 4

# The contact's per-turn sentiment as a valence scalar (−1…+1) — the input of the emotional
# neutral (an EMA the host keeps, see `metakeys.CONTACT_STATE`) and of the turn's delta against
# it. Closed on the NER's own vocabulary; an unknown label reads as 0 (no evidence either way).
# ``NEGATIVE_SENTIMENTS`` (the streak family) is exactly the tail of this scale at
# ``CONTACT_ESCALATION_DELTA`` — a test pins the identity, so the two cannot drift into two
# different definitions of "upset".
SENTIMENT_VALENCE: dict[str, float] = {
    "POSITIVE": 1.0, "PLAYFUL": 0.6, "CURIOUS": 0.2, "NEUTRAL": 0.0,
    "URGENT": -0.3, "NEGATIVE": -0.7, "FRUSTRATED": -1.0,
}
# Below this many observed turns the neutral is noise: the voice ignores it and the contact
# receives the persona as declared.
CONTACT_STATE_MIN_TURNS: int = 5
# A turn this far BELOW the contact's own neutral is a real escalation (empathy, calm pace);
# the same sentiment inside the contact's normal range is not.
CONTACT_ESCALATION_DELTA: float = 0.5
# A neutral this warm / this guarded shapes a turn that carries no signal of its own.
CONTACT_WARM_NEUTRAL: float = 0.4
CONTACT_GUARDED_NEUTRAL: float = -0.4


def parse_contact_state_carrier(raw: Any) -> Optional[dict]:
    """The carrier as a dict, decoding the JSON text a raw driver hands back for a JSONB
    column — or ``None`` when it is not a mapping at all. Pure, never raises.

    Separate from :func:`sanitize_contact_state` because two callers need the same decode for
    different questions: "is this usable?" and "is this MALFORMED, or merely too young?" —
    and answering the second on the undecoded value called every well-formed JSON string
    malformed, which logged a warning on every turn of every new contact.
    """
    if isinstance(raw, (bytes, bytearray, str)):
        try:
            if isinstance(raw, (bytes, bytearray)):
                raw = bytes(raw).decode("utf-8")
            if len(raw) > MAX_TRAIT_CARRIER_CHARS:
                return None
            import json
            raw = json.loads(raw)
        except Exception:  # noqa: BLE001 — any parser failure is the carrier's problem
            return None
    return raw if isinstance(raw, dict) else None


def sanitize_contact_state(raw: Any) -> Optional[dict[str, float]]:
    """Validate a host-stamped emotional-neutral carrier → ``{valence_ema, n}`` or ``None``.

    Returns ONLY what the core reads. The host's state carries more (the habitual intensity,
    the mean message length, a schema version) — those are its series and its TTS delivery
    profile, and validating a field nothing here consumes would dress it as core semantics.

    Never FABRICATES: a carrier without ``valence_ema`` is not a neutral with a neutral value,
    it is a carrier without a neutral (a host that stamped only its counter would otherwise
    have the voice modulate against an invented 0.0). ``None`` also for a non-finite value, a
    bool where a number belongs, and for a state too young to trust
    (``n < CONTACT_STATE_MIN_TURNS`` — cold start: the persona as declared). Pure, never raises.
    """
    raw = parse_contact_state_carrier(raw)
    if raw is None or "valence_ema" not in raw:
        return None
    n = as_count(raw.get("n"))
    if n is None or n < CONTACT_STATE_MIN_TURNS:
        return None
    v = raw["valence_ema"]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return {"valence_ema": max(-1.0, min(1.0, float(v))), "n": float(n)}


_LOG_LABEL_WIDTH = 40
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]")
# Above these the carrier is not a configuration, it is an attack or a bug: refused whole.
MAX_TRAIT_CARRIER_CHARS = 4096
MAX_TRAIT_CARRIER_ITEMS = 64


def _label(x: Any) -> str:
    """A short, safe label for a dropped item — never echoes a tenant string past 40 chars, and
    never raises (a carrier element with a raising ``__repr__`` must not abort a turn). A
    label carries no newline, so a hostile value cannot forge a second log line."""
    try:
        text = x if isinstance(x, str) else repr(x)
    except Exception:  # noqa: BLE001 — the label is diagnostics, the turn is the product
        text = f"<{type(x).__name__}>"
    # slice FIRST (a 10 MB value must not be scanned whole), then neutralize every control and
    # line/paragraph separator (\n, \r, \x85, \u2028, \v, \f…) so a value cannot forge a log line
    return _CONTROL_RE.sub(" ", text[:_LOG_LABEL_WIDTH * 2])[:_LOG_LABEL_WIDTH]


def sanitize_voice_traits(raw: Any) -> tuple[list[str], list[str]]:
    """Sanitize a persona-traits carrier against the closed vocabulary → ``(kept, dropped)``.

    PURE: no logging, no I/O, never raises — the SUPEREGO logs what it drops, the host's admin
    API refuses at save time exactly what this would drop at voice time. One rule, two ends of
    the wire.

    Accepts a list/tuple (declaration order) or a comma-separated string (a plain text column;
    a JSON array string — a JSON column read raw — is decoded rather than split on its commas).
    A set/frozenset has no declared order, so it is SORTED first — otherwise the cap below would
    keep a different four per worker (hash randomization), a non-reproducible voice. Lowercases
    and strips; drops unknown values and non-strings; folds duplicates in order; drops BOTH sides
    of a contradicting axis (``VOICE_TRAIT_CONFLICTS``) BEFORE the cap, so a contradiction can
    never be hidden by truncation; caps at ``MAX_VOICE_TRAITS`` (first declared wins). Only ``,``
    separates: ``"warm; direct"`` is one unknown value, not two traits. Anything unusable →
    ``([], [label])``. ``dropped`` is in a stable order: unknowns as met, then each contradicting
    axis in ``VOICE_TRAIT_AXES`` order, then the overflow.
    """
    if raw is None:
        return [], []
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = bytes(raw).decode("utf-8")          # a Redis client without decode_responses
        except UnicodeDecodeError:
            return [], ["<bytes>"]
    if isinstance(raw, str):
        if len(raw) > MAX_TRAIT_CARRIER_CHARS:
            return [], [f"<{len(raw)} chars>"]
        text = raw.strip()
        if text.startswith("["):
            try:
                import json
                loaded = json.loads(text)
            except Exception:  # noqa: BLE001 — ValueError, RecursionError on '[' * 1e5, …
                return [], [_label(text)]
            if not isinstance(loaded, list):
                return [], [_label(text)]
            items: list[Any] = loaded
        else:
            # A Postgres ``text[]`` literal: {warm,direct} — and Postgres QUOTES an element
            # that needs it, so the quotes come off per item below, not with the braces.
            if text.startswith("{") and text.endswith("}"):
                text = text[1:-1]
            items = [i.strip().strip('"') for i in text.split(",")]
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    elif isinstance(raw, (set, frozenset)):
        try:
            items = sorted(raw, key=lambda x: _label(x).lower())   # case must not pick the four
        except Exception:  # noqa: BLE001
            return [], [_label(raw)]
    else:
        return [], [type(raw).__name__]
    if len(items) > MAX_TRAIT_CARRIER_ITEMS:
        return [], [f"<{len(items)} items>"]
    kept: list[str] = []
    dropped: list[str] = []
    for item in items:
        if not isinstance(item, str):
            dropped.append(_label(item))
            continue
        t = item.strip().lower()
        if not t:
            continue
        if t not in VALID_VOICE_TRAITS:
            dropped.append(_label(t))
        elif t not in kept:
            kept.append(t)
    kept_set = set(kept)
    conflicted = [t for pair in VOICE_TRAIT_CONFLICTS if pair <= kept_set
                  for t in kept if t in pair]
    if conflicted:
        dropped += conflicted
        kept = [t for t in kept if t not in conflicted]
    if len(kept) > MAX_VOICE_TRAITS:
        dropped += kept[MAX_VOICE_TRAITS:]
        kept = kept[:MAX_VOICE_TRAITS]
    # `dropped` is a REPORT, not a transcript: one entry per distinct value (a carrier of 5000
    # copies of "sassy" is one problem, not 5000), in first-seen order.
    return kept, list(dict.fromkeys(dropped))


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

# The member of the list above that is not a topic. The NER prompt instructs
# "Use GENERAL if none fits", so a turn tagged GENERAL is a turn where NO domain
# was identified — the value records an ABSENCE. It is a member of the closed
# list because the LLM must be able to answer something, not because it names a
# subject, and `carried_domains` (ner.py) is what keeps the difference from being
# re-derived by every caller that carries domains across turns.
FALLBACK_DOMAIN: str = "GENERAL"
