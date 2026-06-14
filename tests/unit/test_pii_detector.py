"""Unit tests for the deterministic, pluggable PII detector."""

import json
import pytest

from cogno_core.security.detector import (
    PiiDetector, PiiPattern, default_detector,
    BRAZIL_PATTERNS, INTERNATIONAL_PATTERNS, US_PATTERNS,
    cpf_valid, cnpj_valid, luhn_valid,
)
from cogno_core.stages.ner import IntentAnalyzer
from tests.conftest import StubBackend
from tests.unit.test_ner import make_noumeno_result, PERFECT_JSON, PROMPTS_DIR

import re  # noqa: E402


# ── Validators ────────────────────────────────────────────────────────────

def test_cpf_validator():
    assert cpf_valid("111.444.777-35")
    assert cpf_valid("11144477735")
    assert not cpf_valid("123.456.789-00")
    assert not cpf_valid("111.111.111-11")   # all-same rejected


def test_cnpj_validator():
    assert cnpj_valid("11.222.333/0001-81")
    assert not cnpj_valid("11.222.333/0001-00")


def test_luhn_validator():
    assert luhn_valid("4111 1111 1111 1111")
    assert not luhn_valid("4111 1111 1111 1112")


# ── Default detector (BR + international) ──────────────────────────────────

@pytest.fixture
def det():
    return default_detector()


@pytest.mark.parametrize("text, expected", [
    ("meu CPF é 111.444.777-35", "NATIONAL_ID"),
    ("CNPJ 11.222.333/0001-81", "TAX_ID"),
    ("escreve pra joao.silva@example.com", "EMAIL"),
    ("cartão 4111 1111 1111 1111", "CREDIT_CARD"),
    ("o servidor é 192.168.0.1", "IP_ADDRESS"),
    ("liga pra (11) 98765-4321", "PHONE"),
    ("meu CEP é 01310-100", "ADDRESS"),
    ("minha senha: Admin@2024xyz", "CREDENTIAL"),
])
def test_detects_each_type(det, text, expected):
    assert expected in det.detect(text)


def test_pix_random_key_is_bank_account(det):
    assert "BANK_ACCOUNT" in det.detect("minha chave pix é 123e4567-e89b-42d3-a456-426614174000")


def test_invalid_cpf_not_detected(det):
    # Wrong check digits → must NOT be flagged as NATIONAL_ID.
    assert "NATIONAL_ID" not in det.detect("número 123.456.789-00 aleatório")


def test_invalid_card_not_detected(det):
    assert "CREDIT_CARD" not in det.detect("código 4111 1111 1111 1112")


def test_no_pii_returns_empty(det):
    assert det.detect("qual a previsão do tempo para amanhã?") == []
    assert det.detect("") == []


def test_multiple_pii_in_one_text(det):
    found = det.detect("sou o joao@x.com, CPF 111.444.777-35, fone (11) 98765-4321")
    assert {"EMAIL", "NATIONAL_ID", "PHONE"} <= set(found)


# ── Pluggability ──────────────────────────────────────────────────────────

def test_us_ssn_not_detected_by_default_but_via_extra_pack(det):
    ssn_text = "SSN 123-45-6789"
    assert "NATIONAL_ID" not in det.detect(ssn_text)          # BR default ignores SSN shape
    us_det = default_detector(extra=US_PATTERNS)
    assert "NATIONAL_ID" in us_det.detect(ssn_text)           # opt-in country pack


def test_custom_country_pattern():
    # A consumer can register an arbitrary national pattern (e.g. a passport).
    custom = PiiPattern("PASSPORT", re.compile(r"\bP[A-Z]\d{7}\b"), name="custom_passport")
    detector = PiiDetector([custom])
    assert detector.detect("passaporte PB1234567") == ["PASSPORT"]


def test_pattern_rejects_unknown_type():
    with pytest.raises(ValueError):
        PiiPattern("NOT_A_TYPE", re.compile(r"x"))


def test_packs_are_composable():
    detector = PiiDetector([*BRAZIL_PATTERNS, *INTERNATIONAL_PATTERNS, *US_PATTERNS])
    assert "NATIONAL_ID" in detector.detect("SSN 123-45-6789")
    assert "NATIONAL_ID" in detector.detect("CPF 111.444.777-35")


# ── NER integration: deterministic detector unions with the LLM output ─────

@pytest.mark.asyncio
async def test_ner_pii_union_catches_what_llm_misses():
    """LLM returns pii=[] but the text has a CPF + email → detector catches both."""
    payload = dict(PERFECT_JSON)
    payload["pii"] = []
    payload["pii_risk"] = "NONE"
    backend = StubBackend(response=json.dumps(payload))
    analyzer = IntentAnalyzer(backend=backend, prompts_dir=PROMPTS_DIR)

    noumeno = make_noumeno_result(
        original="meu cpf é 111.444.777-35 e email teste@x.com",
        rewritten="my id and email follow",
    )
    result = await analyzer.analyze(noumeno)
    assert "NATIONAL_ID" in result.pii
    assert "EMAIL" in result.pii
    assert result.pii_risk == "HIGH"   # NATIONAL_ID → HIGH, recomputed in-core


@pytest.mark.asyncio
async def test_ner_pii_detector_runs_on_original_not_rewrite():
    """PII is detected on the ORIGINAL, even when the rewrite masks it."""
    payload = dict(PERFECT_JSON)
    payload["pii"] = []
    backend = StubBackend(response=json.dumps(payload))
    analyzer = IntentAnalyzer(backend=backend, prompts_dir=PROMPTS_DIR)

    noumeno = make_noumeno_result(
        original="cartão 4111 1111 1111 1111",
        rewritten="the user shared a payment card",   # masked, no digits
    )
    result = await analyzer.analyze(noumeno)
    assert "CREDIT_CARD" in result.pii
