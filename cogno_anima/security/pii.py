from __future__ import annotations

import re

VALID_PII_TYPES: set[str] = {
    "NATIONAL_ID",
    "TAX_ID",
    "EMAIL",
    "PHONE",
    "CREDIT_CARD",
    "BANK_ACCOUNT",
    "DATE_OF_BIRTH",
    "HEALTH_DATA",
    "ADDRESS",
    "IP_ADDRESS",
    "PASSPORT",
    "CREDENTIAL",
    "BIOMETRIC",
    "NAME",
}

PII_RISK_LEVELS: set[str] = {"NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"}

PII_RISK_MAP: dict[str, str] = {
    # A bare name is tracked (anaphora/session hints) but is NOT treated as a
    # privacy risk on its own — only in combination with other identifiers does
    # it matter, and that combination is already covered by the other types.
    "NAME":          "NONE",
    "ADDRESS":       "MEDIUM",
    "EMAIL":         "MEDIUM",
    "PHONE":         "MEDIUM",
    "IP_ADDRESS":    "MEDIUM",
    "DATE_OF_BIRTH": "HIGH",
    "NATIONAL_ID":   "HIGH",
    "TAX_ID":        "HIGH",
    "PASSPORT":      "HIGH",
    "CREDIT_CARD":   "HIGH",
    "BANK_ACCOUNT":  "HIGH",
    "HEALTH_DATA":   "CRITICAL",
    "CREDENTIAL":    "CRITICAL",
    "BIOMETRIC":     "CRITICAL",
}

_RISK_ORDER = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


def compute_pii_risk(pii_types: list[str]) -> str:
    """Deterministically compute the risk level (NONE, LOW, MEDIUM, HIGH, CRITICAL)."""
    if not pii_types:
        return "NONE"
    return max(
        (PII_RISK_MAP.get(t, "LOW") for t in pii_types),
        key=lambda r: _RISK_ORDER.get(r, 0),
    )


# Birth-context markers. A ``DATE_OF_BIRTH`` is a privacy risk only when the text actually
# frames a date as a BIRTH date. A bare date token ("dia 20/07", "reunião 15/03 às 10h") is
# NOT a DOB — but an LLM NER routinely mislabels one as such. That single false positive is
# expensive: DATE_OF_BIRTH ⇒ HIGH risk ⇒ the ID routes to SUPEREGO instead of the EGO, so a
# scheduling/booking turn ("marcar dia 20/07 às 10") loses its tool gateway and can never act.
# Domain-agnostic guard: keep the DOB only with birth vocabulary near it, else drop it (it is
# just an ordinary date). Falls back to keeping when in doubt is *not* what we want here — the
# absence of birth framing is exactly the signal that a date is not a birth date.
_BIRTH_CONTEXT_RE = re.compile(
    r"\bnasc\w*|\bborn\b|\bbirth\b|\bd\.?o\.?b\.?\b|anivers[áa]ri|birthday|"
    r"\bidade\b|data\s+de\s+nascimento",
    re.IGNORECASE)


def has_birth_context(text: str) -> bool:
    """True when ``text`` frames a date as a birth date (birth vocabulary present)."""
    return bool(text) and bool(_BIRTH_CONTEXT_RE.search(text))


def filter_uncontextualized_dob(pii_types: list[str], text: str) -> list[str]:
    """Drop a ``DATE_OF_BIRTH`` with no birth context in ``text`` — an ordinary/appointment date
    the NER misread as a DOB. Keeps a DOB framed by birth vocabulary; all other types pass
    through untouched. This prevents a plain date from inflating ``pii_risk`` to HIGH and
    detouring an actionable request away from the EGO tool gateway."""
    if "DATE_OF_BIRTH" not in pii_types or has_birth_context(text):
        return pii_types
    return [t for t in pii_types if t != "DATE_OF_BIRTH"]


def normalize_pii_types(raw: list[str]) -> list[str]:
    """Map aliases (e.g. CPF -> NATIONAL_ID, DOB -> DATE_OF_BIRTH) and drop invalid ones."""
    _PII_ALIASES: dict[str, str] = {
        "PHONE_NUMBER": "PHONE", "TELEPHONE": "PHONE", "MOBILE": "PHONE",
        "CELL_PHONE": "PHONE", "MOBILE_NUMBER": "PHONE",
        "CREDIT_CARD_NUMBER": "CREDIT_CARD", "CARD_NUMBER": "CREDIT_CARD",
        "PAYMENT_CARD": "CREDIT_CARD", "DEBIT_CARD": "CREDIT_CARD",
        "SSN": "NATIONAL_ID", "CPF": "NATIONAL_ID", "NI": "NATIONAL_ID",
        "DNI": "NATIONAL_ID", "RG": "NATIONAL_ID", "ID_NUMBER": "NATIONAL_ID",
        "NATIONAL_IDENTITY": "NATIONAL_ID", "IDENTITY_NUMBER": "NATIONAL_ID",
        "GOVERNMENT_ID": "NATIONAL_ID", "SOCIAL_SECURITY": "NATIONAL_ID",
        "SOCIAL_SECURITY_NUMBER": "NATIONAL_ID",
        "CNPJ": "TAX_ID", "EIN": "TAX_ID", "NIF": "TAX_ID",
        "TAX_NUMBER": "TAX_ID", "TAXPAYER_ID": "TAX_ID",
        "IBAN": "BANK_ACCOUNT", "ACCOUNT_NUMBER": "BANK_ACCOUNT",
        "BANK_DETAILS": "BANK_ACCOUNT", "ROUTING_NUMBER": "BANK_ACCOUNT",
        "PIX": "BANK_ACCOUNT",
        "PASSWORD": "CREDENTIAL", "API_KEY": "CREDENTIAL", "TOKEN": "CREDENTIAL",
        "SECRET": "CREDENTIAL", "PRIVATE_KEY": "CREDENTIAL",
        "DOB": "DATE_OF_BIRTH", "BIRTHDAY": "DATE_OF_BIRTH",
        "MEDICAL_DATA": "HEALTH_DATA", "MEDICAL_RECORD": "HEALTH_DATA",
        "FINGERPRINT": "BIOMETRIC", "FACE_ID": "BIOMETRIC",
        "FULL_NAME": "NAME", "PERSON_NAME": "NAME",
        "STREET_ADDRESS": "ADDRESS", "HOME_ADDRESS": "ADDRESS",
        "EMAIL_ADDRESS": "EMAIL",
    }
    normalized = []
    for x in raw:
        if not isinstance(x, str):
            continue
        canonical = str(x).upper().strip()[:30]
        canonical = _PII_ALIASES.get(canonical, canonical)
        if canonical in VALID_PII_TYPES:
            normalized.append(canonical)
    return list(dict.fromkeys(normalized))[:10]  # deduplicate, preserve order, cap 10
