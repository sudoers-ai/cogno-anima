from __future__ import annotations

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
    """Calcula deterministicamente o nível de risco (NONE, LOW, MEDIUM, HIGH, CRITICAL)."""
    if not pii_types:
        return "NONE"
    return max(
        (PII_RISK_MAP.get(t, "LOW") for t in pii_types),
        key=lambda r: _RISK_ORDER.get(r, 0),
    )


def normalize_pii_types(raw: list[str]) -> list[str]:
    """Mapeia aliases (ex: CPF -> NATIONAL_ID, DOB -> DATE_OF_BIRTH) e remove inválidos."""
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
