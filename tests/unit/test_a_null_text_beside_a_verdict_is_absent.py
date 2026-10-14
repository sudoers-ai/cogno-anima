"""A JSON ``null`` critique or refusal message is read as ABSENT — never as the text "None".

Both were read with ``str(data.get(key, ""))``, and ``str(None)`` is the four letters
``None``. Measured on ``32803f5``: a judge answering ``{"approved": false, "critique": null}``
produced ``SuperegoResult.critique == "None"`` — the reason handed to the executor's retry and
to the voice on exhaustion — and a guard answering ``{"blocked": true, "refusal_message":
null}`` produced ``refusal_message == "None"``, which is what the contact would be told. The
form was already COUNTED (``null_field:<field>`` on ``repairs``); it was not corrected.

A ``null`` now reads as the absent key always did, and the mark stays. Over 78 replies (each
of the two fields under 38 values, and the missing key) these two are the only results that
change.
"""

from __future__ import annotations

import json

import pytest

from cogno_anima.stages.superego import SuperegoStage
from tests.unit.test_superego import ScriptedBackend
from tests.unit.test_superego import _ctx


async def _judge(reply: dict):
    return await SuperegoStage().evaluate(
        _ctx(), ScriptedBackend([json.dumps(reply)]), limits_prompt="")


async def _guard(reply: dict):
    return await SuperegoStage().check_input_scope(
        _ctx(intent_class="INFORMATION_REQUEST"), ScriptedBackend([json.dumps(reply)]),
        scope_prompt="finance")


async def test_a_null_critique_is_the_judges_default_and_is_still_counted():
    null = await _judge({"approved": False, "critique": None})
    absent = await _judge({"approved": False})
    assert null.approved is False
    assert null.critique == absent.critique == "execution rejected"
    assert null.critique != "None"
    assert null.repairs == ["null_field:critique"]          # counted as what it was…
    assert absent.repairs == ["default:critique"]           # …and told apart from a missing key


async def test_a_null_refusal_message_is_no_message_and_is_still_counted():
    null = await _guard({"blocked": True, "refusal_message": None})
    absent = await _guard({"blocked": True})
    assert null.blocked is True
    assert null.refusal_message == absent.refusal_message == ""
    assert null.repairs == ["null_field:refusal_message"]
    assert absent.repairs == ["default:refusal_message"]


@pytest.mark.parametrize("value, text, mark", [
    ("did X, not Y", "did X, not Y", None),                 # a string is the string
    ("None", "None", None),                                 # …the WORD "None" included
    ("", "execution rejected", "default:critique"),         # blank: the default, as before
    (0, "0", "coerced:critique"),                           # any other value: rendered, counted
    (["a"], "['a']", "coerced:critique"),
    (False, "False", "coerced:critique"),
])
async def test_every_other_critique_reads_as_it_did(value, text, mark):
    result = await _judge({"approved": False, "critique": value})
    assert result.critique == text
    assert result.repairs == ([mark] if mark else [])


@pytest.mark.parametrize("value, text, mark", [
    ("I only handle finance.", "I only handle finance.", None),
    ("", "", "default:refusal_message"),
    (0, "0", "coerced:refusal_message"),
])
async def test_every_other_refusal_message_reads_as_it_did(value, text, mark):
    result = await _guard({"blocked": True, "refusal_message": value})
    assert result.refusal_message == text
    assert result.repairs == ([mark] if mark else [])


async def test_the_text_is_not_read_when_the_verdict_does_not_ask_for_it():
    approved = await _judge({"approved": True, "critique": None})
    allowed = await _guard({"blocked": False, "refusal_message": None})
    assert approved.critique is None and approved.repairs == []
    assert allowed.refusal_message == "" and allowed.repairs == []
