"""A ``null``, a number or a boolean in the NER's ``mandatory_tags`` or ``pii`` reads as absent (0.1.13).

The NER iterated both fields as it found them: ``for t in data.get("mandatory_tags", [])`` and
``normalize_pii_types(data.get("pii", []))``. A JSON value that cannot be iterated — ``null``, a
number, ``true``, ``false`` — raised ``TypeError`` there, which is not the stage's
``StageParseError``, over a reply whose every other field was usable. The 0.1.9 sweep counted 14
such replies in 603; the schema sweep of 0.1.10 (38 dirty values) counts 16, the 8 values JSON
cannot iterate under each of the two fields. Those 16 are the twin here, in synthetic form.

Each now reads as the absent key always did (``mandatory_tags`` → ``NER.UNKNOWN``, ``pii`` → the
deterministic detector alone), and the record says so with a mark of the closed alphabet:
``null_field:<field>`` for a ``null``, ``coerced:<field>`` for the others. A string or an object
is read as it was before; that is the adverse half.

Every value here is invented.
"""

from __future__ import annotations

import copy

import pytest

from cogno_anima import repairs as rp
from cogno_anima.stages import ner as ner_stage
from cogno_anima.stages.ner import IntentAnalyzer
from cogno_anima.types import StageMetrics
from tests.unit.test_repairs_are_counted import _INTENT, _ner

#: The values JSON cannot iterate — the shapes of the schema sweep's dirty list that ended the
#: turn (``tests/unit/test_schemas_are_derived.py::DIRTY``).
NOT_ITERABLE = (None, 0, 1, 7.5, -1, 0.5, True, False)
FIELDS = ("mandatory_tags", "pii")
#: The contact's sentence of the end-to-end cases: it carries an e-mail the detector finds.
ORIGINAL = "manda a fatura para rosa.quintas@exemplo.com"


def _metrics() -> StageMetrics:
    return StageMetrics(stage="ner", elapsed_ms=0.0, tokens_in=0, tokens_out=0, model="t")


def _build(reply: dict, original: str = ""):
    return IntentAnalyzer()._build(copy.deepcopy(reply), "", _metrics(), original=original)


def _without(reply: dict, key: str) -> dict:
    return {k: v for k, v in reply.items() if k != key}


def _decided(result) -> dict:
    """Everything the stage decided — the result without its telemetry."""
    return result.model_dump(exclude={"repairs", "metrics"})


def _mark(field: str, value: object) -> str:
    return f"{rp.NULL_FIELD if value is None else rp.COERCED}:{field}"


# ── the twin: the 16 replies that ended the turn ─────────────────────────────────────────

@pytest.mark.parametrize("value", NOT_ITERABLE, ids=repr)
@pytest.mark.parametrize("field", FIELDS)
def test_a_value_json_cannot_iterate_reads_as_the_absent_key(field, value):
    """On ``6c90f67`` each of these raised ``TypeError``. Now the decision is the one the
    absent key gets, byte for byte, and only the record differs."""
    reply = dict(_INTENT, **{field: value})
    result = _build(reply, ORIGINAL)
    assert _decided(result) == _decided(_build(_without(_INTENT, field), ORIGINAL))
    assert result.repairs == [_mark(field, value)]


@pytest.mark.parametrize("value", NOT_ITERABLE, ids=repr)
def test_the_tag_fallback_reads_it_as_absent_too(value):
    """The second read of ``mandatory_tags``: with ``intent_class`` unknown the stage looks at
    the tags for a class. With nothing to read there it stays ``UNKNOWN`` and no fallback is
    marked — what an absent key gives."""
    reply = dict(_INTENT, intent_class="UNKNOWN", mandatory_tags=value)
    result = _build(reply)
    assert result.intent_class == "UNKNOWN"
    assert result.mandatory_tags == ["NER.UNKNOWN"]
    assert _mark("mandatory_tags", value) in result.repairs
    assert not any(m.startswith(rp.FALLBACK + ":") for m in result.repairs), result.repairs
    assert _decided(result) == _decided(_build(_without(reply, "mandatory_tags")))


@pytest.mark.parametrize("field", FIELDS)
async def test_through_the_stage_the_turn_goes_on_and_the_detector_still_counts(field):
    """End to end, through ``analyze`` and a scripted backend: no exception, an
    ``IntentResult``, and the deterministic detector's PII is still there — absent is "the
    model listed nothing", never "there is none"."""
    result = await _ner(dict(_INTENT, **{field: None}))
    assert result.repairs == [f"null_field:{field}"]
    assert result.mandatory_tags == (["NER.UNKNOWN"] if field == "mandatory_tags"
                                     else ["NER.SYSTEM"])


def test_a_null_pii_beside_an_email_in_the_message_keeps_the_detectors_finding():
    result = _build(dict(_INTENT, pii=None), ORIGINAL)
    assert result.pii == ["EMAIL"] and result.pii_risk == _build(
        dict(_INTENT, pii=[]), ORIGINAL).pii_risk


def test_the_two_marks_are_in_the_closed_alphabet():
    marks = ["null_field:mandatory_tags", "null_field:pii"]
    assert set(marks) <= rp.VALID_NER_REPAIRS
    assert rp.closed_repairs(marks, "ner") == marks


# ── the adverse half: what the stage could already iterate is read as it was ─────────────

@pytest.mark.parametrize("value,tags", [
    (["SYSTEM"], ["NER.SYSTEM"]),
    ("SYSTEM", ["NER.UNKNOWN"]),                 # a string iterates its characters: no tag
    ({"SYSTEM": "x"}, ["NER.SYSTEM"]),           # an object iterates its keys, as before
    ([], ["NER.UNKNOWN"]),
])
def test_an_iterable_value_is_read_as_it_was(value, tags):
    assert _build(dict(_INTENT, mandatory_tags=value)).mandatory_tags == tags


def test_the_helper_returns_nothing_only_for_what_json_cannot_iterate():
    helper = ner_stage._iterable_or_absent
    for value in NOT_ITERABLE:
        assert helper(value) == [], value
    for value in (["a"], "abc", {"k": 1}, [], "", {}):
        assert helper(value) is value, value


def test_control_without_the_helper_the_sixteen_end_the_turn(monkeypatch):
    """The other world, rebuilt in this one: with the helper as the identity, every one of the
    16 raises ``TypeError`` — the measured shape on ``6c90f67``."""
    monkeypatch.setattr(ner_stage, "_iterable_or_absent", lambda value: value)
    raised = 0
    for field in FIELDS:
        for value in NOT_ITERABLE:
            with pytest.raises(TypeError):
                _build(dict(_INTENT, **{field: value}), ORIGINAL)
            raised += 1
    assert raised == 16
