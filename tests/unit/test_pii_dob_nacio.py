"""The Spanish preterite «nació» is the commonest way a record states a birth («María nació el
12/03/1990»), the twin of the Portuguese «nasceu» that `\\bnasc\\w*` already reads. A collision
control keeps Portuguese and English untouched."""
from cogno_anima.security.pii import has_birth_context


def test_nacio_is_a_birth_context():
    assert has_birth_context("María nació el 12/03/1990 en Sevilla")
    assert has_birth_context("NACIÓ EL 12/03/1990")


def test_control_no_portuguese_or_english_collision():
    for t in ("a nação comemora em 12/03", "o naciente projeto de 12/03",
              "the nation day is 12/03"):
        assert not has_birth_context(t), t
