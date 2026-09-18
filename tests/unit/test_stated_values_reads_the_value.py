"""The grounding assertion must read the FIGURE, not the locale it was written in.

`tests/integration/test_superego.py::test_voice_writes_grounded_response` asks whether the
voice reproduced the figure a tool returned (`Current balance: 1000 BRL`). It asserted
`"1000" in response`, and on the nightly canary of 2026-09-17 (run 35207468672, qwen3:8b) it
failed on:

    "Seu saldo atual é de R$ 1.000,00."

which is the figure, grounded, in the pt-BR the turn was conducted in. The test was pinning a
LOCALE while claiming to pin GROUNDING — a defect in the instrument, not in the measured.

`stated_values` is the replacement, and it lives under a model-gated file, so these are the
tests that actually run: the integration twin cannot execute without a backend, and an
instrument nothing checks is how the first one got this wrong.

The rule it must NOT be is "strip the separators": `1.000,00` flattens to `100000`, which
still CONTAINS `1000`, so the old assertion would have passed on a reply stating a hundred
thousand. That case is pinned below, because it is the one a careless fix reintroduces.
"""

from __future__ import annotations

import pytest

from tests.integration.test_superego import stated_values


@pytest.mark.parametrize("text, expected", [
    ("Seu saldo atual é de R$ 1.000,00.", 1000),          # pt-BR — the measured reply
    ("Your balance is $1,000.00.", 1000),                 # en-US
    ("Your balance is 1000 BRL.", 1000),                  # bare, as the tool returned it
    ("Saldo: R$ 1000,00", 1000),                          # decimal comma, no thousands mark
    ("Balance: 1000.00", 1000),                           # decimal point, no thousands mark
    ("O saldo é de R$ 1.000.000,00.", 1000000),           # two thousands marks
])
def test_the_figure_is_read_as_a_value_in_every_rendering(text, expected):
    assert expected in stated_values(text), (text, stated_values(text))


def test_it_does_not_flatten_separators_into_a_bigger_number():
    """The careless fix, pinned: `1.000,00` is one thousand, never one hundred thousand.

    Blind separator-stripping passes the old substring assertion too (`"1000" in "100000"`),
    so a reply stating the WRONG figure would be graded correct — the instrument failing in
    the expensive direction instead of the visible one.
    """
    values = stated_values("Seu saldo atual é de R$ 1.000,00.")
    assert 1000 in values
    assert 100000 not in values, values


def test_a_reply_stating_a_different_figure_is_NOT_accepted():
    """The teeth. A grounding assertion that accepts any number is not an assertion."""
    assert 1000 not in stated_values("Seu saldo atual é de R$ 1.500,00.")
    assert 1000 not in stated_values("Não consegui consultar o saldo.")


def test_it_survives_text_with_no_numbers_and_empty_input():
    """A prompt hint must never abort a turn, and neither must a test helper."""
    assert stated_values("") == set()
    assert stated_values("Não há saldo a mostrar.") == set()


def test_several_figures_are_all_read():
    """The voice often states more than one; the assertion asks about ONE of them."""
    values = stated_values("Entradas de R$ 1.000,00 e saídas de R$ 250,50.")
    assert {1000, 250.5} <= values, values
