"""
Outgoing-PII redaction by PROVENANCE — "PII may come IN, it must not go OUT".

The rule the owner decided (2026-08-25): a personal datum that appeared in the CONTACT'S OWN
message may be said back to them; the same datum coming from tenant configuration, from memory,
from the tenant graph, from a third party's record or from a tool result must not leave. It is
enforced by REDACTING THE SPAN IN PLACE, never by refusing the reply: a refusal costs the contact
their answer and opens the re-voicing loop that once shipped a handoff instead of a booking
(``judge-conversational-and-failopen``). Mask the value, ship the sentence.

Three properties this module exists to hold:

**One detector, both ends.** The values that may leave are recognised by the SAME
:class:`~cogno_anima.security.detector.PiiDetector` that flags them — never by a second
definition. A write-side masking pass is the agreed follow-up and must reuse it too; two
detectors would disagree exactly where it costs, and the disagreement would be invisible.

**The allowlist carries DIGESTS, never values.** The host remembers what the contact said
earlier in the session and injects it through ``ctx.metadata`` (:data:`cogno_anima.metakeys.
PII_OUTPUT_ALLOWLIST`). ``metadata`` is read by whatever serialises the turn trace, by whatever
persists the session state, and by any best-effort guard that renders its kwargs into a log
line — so an allowlist of VALUES would open a fresh store of personal data in the clear to close
one leak, the very class being closed. Digests also keep the follow-up honest: once turns are
stored masked, an allowlist of values could no longer be rebuilt from history, and the choice
would be between keeping values in the clear and never masking on write.

**The ceiling, said out loud: this is de-identification of the flow, not encryption.** A phone
number, a CPF and most e-mail addresses have far too little entropy to survive an offline attack
on an unsalted SHA-256 — anyone holding the digests can confirm a guess in microseconds. What
the digest buys is that no personal datum travels in metadata, reaches a trace row, or lands in
a log line *because of this feature*. It is the same bargain, and the same ceiling, as the
host's ``scope_sha`` (host #545). Treat a persisted digest as pseudonymous personal data — it
belongs inside the identity purge, not outside it.

Comparison is on the DETECTED VALUE after normalisation (:func:`normalize_pii_value`), so
punctuation and spacing cannot defeat a match: ``529.982.247-25`` and ``52998224725`` are the
same CPF, ``(11) 99999-8888`` and ``11999998888`` the same phone.
"""

from __future__ import annotations

import hashlib
import re
from itertools import islice
from dataclasses import dataclass
from collections.abc import Iterable
from typing import Optional

from cogno_anima.security.detector import PiiDetector, PiiMatch
from cogno_anima.types import PiiFinding
from cogno_anima.vocab import (
    PII_MODE_DEFAULT,
    PII_MODE_ENFORCE,
    PII_PROVENANCE_READER_STAFF,
    PII_PROVENANCE_SESSION,
    PII_PROVENANCE_TURN,
    PII_PROVENANCE_UNKNOWN,
    VALID_PII_MODES,
)

__all__ = [
    "DEFAULT_REDACTION_MASK",
    "MAX_ALLOWLIST_DIGESTS",
    "PiiRedactionOutcome",
    "ProvenanceContext",
    "decide_provenance",
    "normalize_pii_value",
    "pii_digest",
    "pii_digests_in",
    "redact_pii",
    "sanitize_digests",
    "sanitize_pii_mode",
    "sanitize_reader_role",
]

# The mask that replaces a value that may not leave. Loud on purpose: a silent removal reads as
# the model forgetting the datum, and nobody can tell the net from a regression.
DEFAULT_REDACTION_MASK = "[{pii_type} REDACTED]"

# Everything that is not a word character. Removed before hashing so formatting cannot defeat a
# match. Unicode letters ARE word characters, so an accent survives normalisation — stripping it
# would map two different values onto one digest, and a digest collision here is a FALSE ALLOW.
_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)

# A digest is 64 lowercase hex characters. Anything else in the injected allowlist is host garbage
# and is dropped rather than trusted ("never trust host-injected hints" — the same discipline the
# ID applies to its carry-over state).
_DIGEST_RE = re.compile(r"\A[0-9a-f]{64}\Z")

# Bound on the injected allowlist. A session cannot plausibly need more, and an unbounded list
# from metadata would make the per-turn cost a function of whatever the host accumulated.
MAX_ALLOWLIST_DIGESTS = 512


def normalize_pii_value(value: str) -> str:
    """Canonical form used for comparison and hashing.

    Two forms, because one form collides. A value whose word characters are ALL DIGITS is
    reduced to those digits — that is the whole point, since ``"529.982.247-25"`` and
    ``"52998224725"`` are one CPF and ``"(11) 99999-8888"`` and ``"11999998888"`` one phone.
    Everything else is merely stripped and casefolded: ``"  Ana.Souza@Example.COM "`` →
    ``"ana.souza@example.com"``.

    **Deleting punctuation from a non-numeric value was a FALSE ALLOW**, which is the one failure
    direction this module cannot afford. Under the first version ``ana@example.com`` and
    ``an@aexample.com`` normalised to the same string, as did ``ana.souza@example.com`` and
    ``anasouza@example.com`` — so a contact who once typed their own address could allowlist a
    stranger's. An e-mail is never re-punctuated between the contact's message and the reply, so
    there was nothing to buy with it either.

    **A named limitation, so nobody reads more into it than it does:** an international dialling
    prefix is NOT normalised away — ``+55 11 99999-8888`` and ``(11) 99999-8888`` produce
    different forms, so a contact who wrote one and an agent that says the other will see the
    value masked. That fails toward masking (the safe direction) and it is a real false positive;
    it is left unhandled deliberately, because every fix for it either truncates the value (and
    two numbers sharing a tail then allowlist each other — a false ALLOW, which is a leak) or
    hardcodes a country into a module that has none.
    """
    stripped = _NON_WORD.sub("", value or "")
    if stripped.isdigit():
        return stripped
    return (value or "").strip().casefold()


def pii_digest(value: str, pii_type: str = "") -> str:
    """SHA-256 of ``TYPE:normalised``, or ``""`` when the value normalises to nothing.

    **The type is part of the digest, and it closes a measured false allow.** Numeric values are
    normalised to their digits, so a contact's own phone ``(52) 99822-4725`` and an unrelated
    person's CPF ``529.982.247-25`` reduce to the SAME eleven digits — and roughly one BR mobile
    in a hundred lands on a checksum-valid CPF. Without the prefix, a contact who once typed that
    phone allowlists that stranger's document for the rest of the session. With it the two
    digests differ and neither can stand in for the other. A value's detected TYPE is stable
    across renderings (the packs do not detect a bare eleven-digit run as a phone at all), so
    nothing legitimate is lost.

    The empty string is never digested: a single empty value in the allowlist would otherwise
    match every value that normalises to nothing, and "matches everything" is the one outcome an
    allowlist must not be able to reach by accident.

    NOT ENCRYPTION — see the module docstring. Unsalted, and the input space of a phone number is
    small enough to enumerate; this de-identifies the flow, it does not protect the value against
    someone who already holds the digest.
    """
    normalized = normalize_pii_value(value)
    if not normalized:
        return ""
    payload = f"{(pii_type or '').strip().upper()}:{normalized}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pii_digests_in(text: str, detector: PiiDetector) -> set[str]:
    """Digests of every PII value ``detector`` finds in ``text``.

    This is what the host accumulates per session and what this module compares against — the
    same function on both ends, so the two can never normalise differently.
    """
    return {d for d in (pii_digest(m.value, m.pii_type) for m in detector.find(text)) if d}


def sanitize_digests(raw: object) -> frozenset[str]:
    """Coerce a host-injected allowlist into digests. Never raises; garbage yields ``frozenset()``.

    Absence and garbage both mean "no earlier-session values", which makes the rule STRICTER
    (more redaction), never laxer. That direction is deliberate: a typo in a metadata key must
    not be able to silently switch the protection off.
    """
    if isinstance(raw, str) or not isinstance(raw, Iterable):
        return frozenset()
    out: set[str] = set()
    # `islice` bounds the WALK, not only the result. The `break` on `len(out)` alone left an
    # iterable of a million non-strings — or of malformed digests — fully traversed, which is
    # exactly the "per-turn cost as a function of whatever the host accumulated" this is here to
    # prevent. The headroom absorbs a list that is mostly duplicates.
    for item in islice(raw, MAX_ALLOWLIST_DIGESTS * 4):
        if not isinstance(item, str):
            continue
        candidate = item.strip().lower()
        if _DIGEST_RE.match(candidate):
            out.add(candidate)
        if len(out) >= MAX_ALLOWLIST_DIGESTS:
            break
    return frozenset(out)


def sanitize_reader_role(raw: object) -> str:
    """Coerce the host-injected reader role; anything that is not a ``str`` → ``""`` (= GUEST).

    The third host-injected key, and the only one whose failure direction is a LEAK, so it gets
    the same ``isinstance`` discipline as :func:`sanitize_digests` and :func:`sanitize_pii_mode`.
    A bare ``str(...)`` looked equivalent and was not: ``str(Role.GUEST)`` on a plain enum is
    ``"Role.GUEST"``, and ``"Role.GUEST"`` is neither empty nor ``"GUEST"``, so a host explicitly
    saying GUEST would have been read as STAFF and the value shipped. Same for a dict or a list.
    A ``str``-subclass enum passes through untouched, which is the common case that DOES work.
    """
    return raw if isinstance(raw, str) else ""


def sanitize_pii_mode(raw: object) -> str:
    """Coerce a host-supplied mode to `vocab.VALID_PII_MODES`; anything unknown → the default.

    Unknown falls to `observe`, not to `enforce`. A typo must not start masking a tenant's
    replies without anyone having decided to.
    """
    candidate = raw.strip().lower() if isinstance(raw, str) else ""
    return candidate if candidate in VALID_PII_MODES else PII_MODE_DEFAULT


@dataclass(frozen=True)
class ProvenanceContext:
    """Everything the provenance decision is allowed to look at — ONE place, by design.

    The rule is under-determined and the merged bench says so: two of its seven scenarios need a
    bit that provenance alone does not carry (host #549, ``a2aef70``). Both candidates are known
    and neither is invented — the READER's role (``Identity.role``: the tenant's own bookkeeper
    reading the ledger they wrote is not a stranger asking about a doctor) and a TENANT-DECLARED
    quotable set (the reception number the operator wrote down precisely so it would be given
    out). So the shape matters more than the current content: adding a bit must be adding a
    FIELD here and a branch in :func:`decide_provenance`, never re-deriving the rule at each of
    its call sites.

    Deliberately NOT a free-form dict. The decision lands in a closed audit alphabet
    (``vocab.VALID_PII_PROVENANCE``), and a bag of host-supplied keys would let a caller widen
    the allowlist with something nobody named.
    """
    turn_digests: frozenset[str] = frozenset()      # values in THIS turn's contact message
    session_digests: frozenset[str] = frozenset()   # values the contact said earlier (host)
    reader_role: str = ""                           # `Identity.role`; "" reads as GUEST

    @property
    def reader_is_staff(self) -> bool:
        """Is the reply going to somebody who works for the tenant?

        Anything that is not a role reads as GUEST — an unknown reader is the stricter
        assumption, and a host that forgets to pass this gets more masking rather than a silent
        widening. Derived from the same string as the host's own `(role or "GUEST").upper() !=
        "GUEST"`, rather than from a second list of role names, but NOT byte-identical to it:
        the strip happens BEFORE the default, because `"   "` is truthy and the host's ordering
        therefore reads a whitespace-only role as staff. Here that direction is a leak, so it is
        closed; a test pins it. (The same hole is open in the host's graph gate, where it decides
        whether a guest's prompt may carry the tenant-wide graph — reported, not fixed here.)
        """
        role = (self.reader_role or "").strip().upper()
        return bool(role) and role != "GUEST"


def decide_provenance(digest: str, context: ProvenanceContext) -> str:
    """May a value with this digest leave? Returns a ``vocab.VALID_PII_PROVENANCE`` member.

    The whole rule, in one pure function of (digest, context) — so the day a second bit is added,
    exactly one function changes and every consumer inherits it. An empty digest (a value that
    normalises to nothing) is never allowlisted: see :func:`pii_digest`.

    Order is deliberate, and it is about the RECORD rather than the outcome. The turn is checked
    first because it is the branch that must never fail — a contact hearing back the e-mail they
    just typed is the case that decides whether this feature survives — and because it is the
    only branch derived from data the stage HOLDS rather than from something a host had to
    remember to send. The reader's role is checked LAST although it is the broadest allow: an
    employee turn whose value the employee themselves typed should read `contact_turn`, so that
    `reader_is_staff` counts exactly the values that ONLY the role let out. That count is the
    one worth watching, since it is the widening this bit buys.
    """
    if not digest:
        return PII_PROVENANCE_UNKNOWN
    if digest in context.turn_digests:
        return PII_PROVENANCE_TURN
    if digest in context.session_digests:
        return PII_PROVENANCE_SESSION
    if context.reader_is_staff:
        return PII_PROVENANCE_READER_STAFF
    return PII_PROVENANCE_UNKNOWN


def _format_mask(template: Optional[str], pii_type: str) -> str:
    """Render the mask, falling back when a HOST template is malformed.

    ``redact_pii`` is exported at the package root and promises never to raise; ``str.format`` on
    a caller-supplied template does, three ways — ``"[REDACTED {name}]"`` (KeyError),
    ``"{"`` (ValueError), ``"100% {0}"`` (IndexError). A guard that ships a person's CPF because
    a mask string had a stray brace would be the worst possible trade.
    """
    try:
        return (template or DEFAULT_REDACTION_MASK).format(pii_type=pii_type)
    except (KeyError, IndexError, ValueError):
        return DEFAULT_REDACTION_MASK.format(pii_type=pii_type)


@dataclass(frozen=True)
class PiiRedactionOutcome:
    """What the rule did to one piece of outgoing text."""
    text: str                        # the text to ship (masked where the rule said no)
    findings: tuple[PiiFinding, ...]  # one per detected value: its type, provenance, verdict
    # No default on purpose: it is the opposite of `vocab.PII_MODE_DEFAULT`, so a hand-built
    # outcome defaulting to True would make `_pii_adjustments` stamp `pii:redacted_in_output` on
    # a turn that masked nothing. The mode is never implicit.
    enforced: bool                   # False → observation: decided everything, masked nothing

    @property
    def redacted(self) -> bool:
        """Did the shipped text actually change? Always False in observation mode."""
        return any(f.redacted for f in self.findings)

    @property
    def withheld(self) -> tuple[PiiFinding, ...]:
        """The values the rule said may not leave — masked under `enforce`, only counted under
        `observe`. This is the production signal the observation mode exists to produce."""
        return tuple(f for f in self.findings
                     if f.provenance == PII_PROVENANCE_UNKNOWN)

    @property
    def provenances(self) -> tuple[str, ...]:
        """Distinct provenances seen, in first-appearance order (the audit alphabet)."""
        seen: list[str] = []
        for f in self.findings:
            if f.provenance not in seen:
                seen.append(f.provenance)
        return tuple(seen)


def _clusters(matches: list[PiiMatch]) -> list[list[PiiMatch]]:
    r"""Group overlapping matches into maximal clusters — one splice per cluster, every member kept.

    Patterns overlap for real: ``credential_kv``'s ``\S{4,}`` stops at whitespace while the card
    and phone patterns span it, so ``"senha: 4539 1488 0343 6467"`` produces a CREDENTIAL over
    ``senha: 4539`` and a CREDIT_CARD over the sixteen digits — a PARTIAL overlap, neither
    containing the other.

    The first cut of this dropped any match starting before the previous one ended, on the
    reasoning that it "describes the same characters, which are leaving either way". That is true
    of a NESTED overlap and false of a partial one, and the difference was three separate harms,
    all measured on that string: 14 of the 19 card digits shipped in the clear, the mask was
    spliced into the middle of a value (the mangled sentence this function exists to prevent),
    and the CREDIT_CARD never reached ``findings`` at all — so under the shipped observation mode
    the counts that decide graduation would have under-reported it silently.

    Clustering fixes all three: the cluster's full span is replaced ONCE, and every member is
    recorded. A cluster leaves verbatim only if EVERY member may leave; a value that overlaps a
    withheld one goes with it, because there is no way to keep one without splitting the other.
    """
    ordered = sorted(matches, key=lambda m: (m.start, -(m.end - m.start)))
    out: list[list[PiiMatch]] = []
    end = -1
    for m in ordered:
        if out and m.start < end:
            out[-1].append(m)
        else:
            out.append([m])
        end = max(end, m.end)
    return out


def redact_pii(
    text: str,
    *,
    detector: PiiDetector,
    context: ProvenanceContext = ProvenanceContext(),
    mode: str = PII_MODE_DEFAULT,
    mask: Optional[str] = None,
) -> PiiRedactionOutcome:
    """Mask every PII value in ``text`` that the contact did not themselves supply.

    Both allowlist branches let the value out; which one did is recorded, not acted on, because
    it is the number that says whether the net is working or getting in the way.

    **Per SESSION, not per turn — measured, not assumed.** The merged host bench
    ``hostbench/pii_out_bench.py`` (#549, ``a2aef70``) has one scenario built to separate the two
    variants — ``own_email_recalled_three_turns_later``, an e-mail given at turn 1 and asked back
    at turn 3. Its answer key says withholding is WRONG; the per-session rule agrees and the
    per-turn rule does not, and no scenario in the suite is decided the other way. So dropping
    ``session_digests`` costs a correct answer to "qual email ficou no meu cadastro?" and buys
    nothing measured — a regression, not a simplification. (That bench's earlier per-RUN headline
    was WITHDRAWN by its own author as arithmetic: with every case decidable the per-run tally is
    just the answer key times the run count. Quote the SCENARIO unit, or the decidable-run one.)

    ``mode`` is the ONE switch (`vocab.VALID_PII_MODES`). Under `observe` — the default, and why
    it is the default is written where the constant lives — every step still runs and every
    finding is still recorded; only the masking is withheld, so the shipped text is byte-identical
    to the voicer's. There is exactly one conditional for it, on the replacement below: a mode
    that branched in several places would eventually mean different things in each.

    Never raises and never refuses: on any input it returns text to ship.
    """
    enforcing = sanitize_pii_mode(mode) == PII_MODE_ENFORCE
    if not text:
        return PiiRedactionOutcome(text=text, findings=(), enforced=enforcing)
    matches = detector.find(text)
    if not matches:
        return PiiRedactionOutcome(text=text, findings=(), enforced=enforcing)

    findings: list[PiiFinding] = []
    pieces: list[str] = []
    cursor = 0
    for cluster in _clusters(matches):
        # One finding per distinct VALUE, not per pattern hit. The phone packs overlap ON PURPOSE
        # — `(11) 98888-7777` matches `phone_ddd` whole and `phone_mobile` on its tail — and
        # counting one number as two withheld values would inflate exactly the production figure
        # that decides whether a tenant graduates out of observation. A same-type match NESTED in
        # a wider one is the same value seen through a narrower pattern: same characters, so this
        # is a span fact rather than a guess about digits. Different types are always kept (the
        # CREDENTIAL/CREDIT_CARD pair above is two genuinely different values).
        members = [m for m in cluster
                   if not any(w is not m and w.pii_type == m.pii_type
                              and w.start <= m.start and m.end <= w.end
                              and (w.end - w.start) > (m.end - m.start)
                              for w in cluster)]
        decided = [(m, decide_provenance(pii_digest(m.value, m.pii_type), context))
                   for m in members]
        withheld = any(p == PII_PROVENANCE_UNKNOWN for _, p in decided)
        widest = max(cluster, key=lambda m: m.end - m.start)
        start, end = min(m.start for m in cluster), max(m.end for m in cluster)
        mask_text = _format_mask(mask, widest.pii_type)
        for m, provenance in decided:
            findings.append(PiiFinding(
                pii_type=m.pii_type, provenance=provenance,
                # The cluster is masked as a whole, so a value the rule would have ALLOWED is
                # recorded as redacted when it overlaps one that was not. Its provenance still
                # says why it would have been let out — collateral, and visible as such.
                redacted=withheld and enforcing,
                mask=mask_text if withheld else ""))
        pieces.append(text[cursor:start])
        pieces.append(mask_text if (withheld and enforcing) else text[start:end])  # the one switch
        cursor = end
    pieces.append(text[cursor:])
    return PiiRedactionOutcome(text="".join(pieces), findings=tuple(findings),
                               enforced=enforcing)
