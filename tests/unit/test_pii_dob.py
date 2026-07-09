"""Unit tests for the DATE_OF_BIRTH birth-context guard (pure functions).

A bare / appointment date must not be treated as a birth date: an LLM NER mislabelling
"dia 20/07 às 10" as DATE_OF_BIRTH inflates pii_risk to HIGH, which detours an actionable
request away from the EGO tool gateway (the "agent can't book" failure)."""

import pytest

from cogno_anima.security.pii import (
    compute_pii_risk,
    filter_uncontextualized_dob,
    has_birth_context,
)


@pytest.mark.parametrize("text", [
    "minha data de nascimento é 20/07/1985",
    "nasci em 20 de julho",
    "nascido em 1990",
    "my date of birth is 07/20/1985",
    "I was born on July 20",
    "DOB: 20/07/1985",
    "meu aniversário é 20/07",
    "tenho 40 anos de idade",
])
def test_has_birth_context_true(text):
    assert has_birth_context(text) is True


@pytest.mark.parametrize("text", [
    "quero marcar com o Dr. Jose Luiz Manzoli dia 20/07 as 10",
    "marca pra mim a reunião 15/03 às 14h",
    "pode ser dia 20/07?",
    "o prazo é 31/12/2026",
    "",
])
def test_has_birth_context_false(text):
    assert has_birth_context(text) is False


def test_filter_drops_uncontextualized_dob():
    pii = ["NAME", "DATE_OF_BIRTH"]
    out = filter_uncontextualized_dob(pii, "marcar dia 20/07 as 10")
    assert out == ["NAME"]
    assert compute_pii_risk(out) != "HIGH"


def test_filter_keeps_dob_with_birth_context():
    pii = ["DATE_OF_BIRTH"]
    out = filter_uncontextualized_dob(pii, "nasci em 20/07/1985")
    assert out == ["DATE_OF_BIRTH"]
    assert compute_pii_risk(out) == "HIGH"


def test_filter_leaves_other_types_untouched():
    # A real identifier still drives HIGH even with a bare date present — only the DOB is culled.
    pii = ["NATIONAL_ID", "DATE_OF_BIRTH"]
    out = filter_uncontextualized_dob(pii, "meu CPF é 123.456.789-00, marca dia 20/07")
    assert "NATIONAL_ID" in out and "DATE_OF_BIRTH" not in out
    assert compute_pii_risk(out) == "HIGH"


def test_filter_noop_without_dob():
    pii = ["EMAIL", "PHONE"]
    assert filter_uncontextualized_dob(pii, "qualquer texto") == pii
