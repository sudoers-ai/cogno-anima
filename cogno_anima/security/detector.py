"""
Deterministic, pluggable PII detector.

Why: the LLM-reported `pii` list is unreliable (especially on small local
models) and PII is safety-critical. This module provides a high-precision
regex + validator detector whose output is *unioned* with the LLM's, so a CPF,
e-mail or credit card is caught even when the model misses it.

Design — pluggable by country:
    detector = PiiDetector(BRAZIL_PATTERNS + INTERNATIONAL_PATTERNS)
    detector = default_detector()                 # BR + international (default)
    detector = default_detector(extra=PORTUGAL_PATTERNS)   # add a country pack

Each `PiiPattern` maps a regex (plus an optional validator like a CPF/Luhn
check) to a canonical type from `VALID_PII_TYPES`. False positives are kept low
by validating check digits where the format allows it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from cogno_anima.security.pii import VALID_PII_TYPES


# ──────────────────────────────────────────────────────────────────────────
#  Validators (check digits / Luhn) — keep precision high
# ──────────────────────────────────────────────────────────────────────────

def _only_digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def luhn_valid(text: str) -> bool:
    """Luhn checksum for payment-card numbers (13–19 digits)."""
    digits = _only_digits(text)
    if not (13 <= len(digits) <= 19):
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def cpf_valid(text: str) -> bool:
    """Brazilian CPF check-digit validation."""
    d = _only_digits(text)
    if len(d) != 11 or d == d[0] * 11:
        return False
    for n in (9, 10):
        s = sum(int(d[i]) * ((n + 1) - i) for i in range(n))
        r = (s * 10) % 11
        r = 0 if r == 10 else r
        if r != int(d[n]):
            return False
    return True


def cnpj_valid(text: str) -> bool:
    """Brazilian CNPJ check-digit validation."""
    d = _only_digits(text)
    if len(d) != 14 or d == d[0] * 14:
        return False
    w1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    w2 = [6] + w1
    for weights, pos in ((w1, 12), (w2, 13)):
        s = sum(int(d[i]) * weights[i] for i in range(pos))
        r = s % 11
        dv = 0 if r < 2 else 11 - r
        if dv != int(d[pos]):
            return False
    return True


# ──────────────────────────────────────────────────────────────────────────
#  Pattern model
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class PiiPattern:
    """One detection rule: a regex (+ optional validator) → canonical PII type."""
    pii_type: str
    pattern: re.Pattern
    validator: Optional[Callable[[str], bool]] = None
    name: str = ""

    def __post_init__(self) -> None:
        if self.pii_type not in VALID_PII_TYPES:
            raise ValueError(f"Unknown PII type: {self.pii_type!r}")


def _p(pii_type: str, regex: str, validator=None, name: str = "", flags=re.I) -> PiiPattern:
    return PiiPattern(pii_type, re.compile(regex, flags), validator, name or pii_type)


# ──────────────────────────────────────────────────────────────────────────
#  Country / region packs
# ──────────────────────────────────────────────────────────────────────────

# International / universal — work regardless of country.
INTERNATIONAL_PATTERNS: list[PiiPattern] = [
    _p("EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", name="email"),
    _p("IP_ADDRESS",
       r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b",
       name="ipv4"),
    _p("CREDIT_CARD", r"\b\d(?:[ -]?\d){12,18}\b", validator=luhn_valid, name="credit_card"),
    # Common secret/token shapes + explicit credential context.
    _p("CREDENTIAL", r"\bAKIA[0-9A-Z]{16}\b", name="aws_key", flags=0),
    _p("CREDENTIAL", r"\b(?:sk|pk|rk|ghp|gho|xox[baprs])[-_][A-Za-z0-9]{16,}\b", name="api_token"),
    _p("CREDENTIAL",
       r"(?:senha|password|passwd|pwd|secret|token|api[_-]?key)\s*[:=]\s*\S{4,}",
       name="credential_kv"),
]

# Brazil-focused.
BRAZIL_PATTERNS: list[PiiPattern] = [
    _p("NATIONAL_ID", r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b", validator=cpf_valid, name="cpf"),
    _p("COMPANY_ID", r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b", validator=cnpj_valid, name="cnpj"),
    # BR phone: +55, DDD in parens, or mobile 9xxxx-xxxx with a separator.
    _p("PHONE", r"(?:\+55\s?)?\(\d{2}\)\s?9?\d{4}[-\s]?\d{4}\b", name="phone_ddd"),
    _p("PHONE", r"\+55\s?\d{2}\s?9?\d{4}[-\s]?\d{4}\b", name="phone_intl"),
    _p("PHONE", r"\b9\d{4}[-\s]\d{4}\b", name="phone_mobile"),
    _p("ADDRESS", r"\b\d{5}-\d{3}\b", name="cep"),  # Brazilian postal code
    # PIX random key (UUID v4 form).
    _p("BANK_ACCOUNT",
       r"\b[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
       name="pix_random"),
]

# US-focused (example of an extra pluggable pack).
US_PATTERNS: list[PiiPattern] = [
    _p("NATIONAL_ID", r"\b\d{3}-\d{2}-\d{4}\b", name="ssn"),
]


# ──────────────────────────────────────────────────────────────────────────
#  Detector
# ──────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PiiMatch:
    """One located PII value: its canonical type, the exact text, and its span.

    ``detect`` answers *what kinds* of personal data a text carries — enough to flag a turn, and
    all the risk computation ever needed. Redacting needs *where and which*, and deriving that
    from a second set of regexes elsewhere is the failure this type prevents: two definitions
    that agree today and diverge exactly where it costs. :meth:`PiiDetector.find` is the single
    definition; ``detect`` is now expressed over it.
    """
    pii_type: str
    value: str
    start: int
    end: int
    pattern_name: str


class PiiDetector:
    """Runs a set of PiiPatterns over text, returning canonical PII types found."""

    def __init__(self, patterns: list[PiiPattern]) -> None:
        self._patterns = list(patterns)

    def find(self, text: str) -> list[PiiMatch]:
        """Every validated PII value in ``text``, with its span — pattern order, then position.

        Unlike :meth:`detect` this does not stop at the first hit per pattern: a reply that names
        two different people's phone numbers has two spans, and masking one of them is not a
        protection. Matches that fail their check digit (CPF/CNPJ/Luhn) are dropped here, so a
        protocol number that merely looks like a document is never masked.

        The value is the WHOLE match (``group(0)``), which is what a span has to be. The
        superseded ``findall`` loop read ``match[0]`` — the first capture GROUP when a pattern
        has one. No shipped pattern does, so nothing here changes; a host adding a pack with a
        capturing group now validates and masks the whole match instead of the group.
        """
        if not text:
            return []
        found: list[PiiMatch] = []
        for pat in self._patterns:
            for m in pat.pattern.finditer(text):
                value = m.group(0)
                if pat.validator and not pat.validator(value):
                    continue
                found.append(PiiMatch(pii_type=pat.pii_type, value=value,
                                      start=m.start(), end=m.end(), pattern_name=pat.name))
        return found

    def detect(self, text: str) -> list[str]:
        """Return the deterministically-detected canonical PII types (deduped, ordered)."""
        found: list[str] = []
        for match in self.find(text):
            if match.pii_type not in found:
                found.append(match.pii_type)
        return found

    def checksum_rejected(self, text: str) -> set[str]:
        """Types the text POSITIVELY fails: a validated pattern (CPF/CNPJ checksum, card
        Luhn) matched the SHAPE but every candidate failed its checksum — so the number is
        definitely NOT that document (an order/protocol number that merely looks like one).

        Used to veto an LLM's claim of the same type: the NER unions the model's ``pii``
        list with this detector, and a model routinely tags any CPF-shaped number as
        NATIONAL_ID — checksum says otherwise, and checksum is authoritative. Without the
        veto the false claim inflates ``pii_risk`` to HIGH and the ID detours an actionable
        turn away from the tool gateway (the ``agent-dumb-pii-misroute`` failure class,
        caught by the safety bench). A type with ANY valid instance in the text is never
        vetoed; types without a validator (shape-only patterns) are never vetoed either —
        there is no counter-evidence, only absence."""
        if not text:
            return set()
        shaped: set[str] = set()
        for pat in self._patterns:
            if pat.validator is None:
                continue
            for match in pat.pattern.findall(text):
                shaped.add(pat.pii_type)
                break
        return shaped - set(self.detect(text))


def default_detector(extra: Optional[list[PiiPattern]] = None) -> PiiDetector:
    """Default detector: Brazil + international, with optional extra country packs.

    Example: ``default_detector(extra=US_PATTERNS)`` to also catch US SSNs.
    """
    patterns = [*BRAZIL_PATTERNS, *INTERNATIONAL_PATTERNS, *(extra or [])]
    return PiiDetector(patterns)
