"""The voicer sometimes answers in JSON, and nothing between it and the contact unwrapped it.

Measured live on 2026-08-24: `{"message": "Oi, Vinicius! …"}` was persisted as the turn's
response — the person would have been shown the JSON. One turn in 283, and **deterministic on
the input rather than random**: the same sentence reproduced it 2/2 while another sentence gave
plain text 0/1, same process, same persona, same voicer.

This is the net, not the fix — a voicer answering in JSON is a prompt problem. The net is narrow
on purpose: it opens exactly one shape and leaves everything else alone, because unwrapping a
shape it cannot read means shipping the wrong text to a person.
"""

from __future__ import annotations

import json

import pytest

from cogno_anima.stages.superego import SuperegoStage
from cogno_anima import metakeys as mk
from cogno_anima import vocab
from tests.unit.test_superego import ScriptedBackend, _ctx

unwrap = SuperegoStage.unwrap_envelope


@pytest.mark.parametrize("key", ["message", "reply", "response", "text",
                                 "Message", "  TEXT  "])
def test_the_one_shape_it_opens(key):
    assert unwrap(json.dumps({key: "Oi, Vinicius! 😊"})) == "Oi, Vinicius! 😊"


def test_the_turn_that_actually_leaked():
    leaked = '{"message":"Oi, Vinicius! Tudo bem? \\ud83d\\ude0a\\n\\nO **Cogno** é ..."}'
    out = unwrap(leaked)
    assert out is not None and out.startswith("Oi, Vinicius!") and "**Cogno**" in out


@pytest.mark.parametrize("text,why", [
    ('{"message": "a", "reply": "b"}', "two keys: which one is the reply?"),
    ('{"data": "x"}', "not an envelope key"),
    ('{"message": 42}', "value is not a string"),
    ('{"message": {"text": "a"}}', "value is not a string"),
    ('{"message": null}', "value is not a string"),
    ("[1, 2]", "a list is not an envelope"),
    ('"just a string"', "a bare string is not an envelope"),
    ("{}", "nothing to unwrap"),
    ("{not json at all}", "does not parse"),
])
def test_everything_else_is_left_alone(text, why):
    """Picking a field out of a shape it cannot read is guessing, and a wrong guess ships the
    wrong text to a person. Silence is the safe answer."""
    assert unwrap(text) is None, why


@pytest.mark.parametrize("reply", [
    "use {tenant} no template",
    "Confirmado! {não é JSON}",
    "{",
    "",
    "   ",
])
def test_a_reply_that_merely_CONTAINS_braces_is_untouched(reply):
    """The realistic false positive: a legitimate reply mentioning braces. It is not JSON, so
    it never reaches the parse."""
    assert unwrap(reply) is None


def test_a_deeply_nested_payload_does_not_blow_the_stack():
    """`json.loads` raises `RecursionError` — a `ValueError` handler alone would let it through
    and a malformed reply would cost the turn."""
    assert unwrap("[" * 5000 + "]" * 5000) is None


# ── the wiring: a pure helper nobody calls is a pure helper that ships nothing ──

@pytest.mark.asyncio
async def test_voice_unwraps_and_flags_it():
    ctx = _ctx()
    backend = ScriptedBackend(['{"message": "Confirmado, Vinicius! 😊"}'])
    res = await SuperegoStage().voice(ctx, backend, voice_prompt="persona")
    assert res.response == "Confirmado, Vinicius! 😊"      # the person sees the reply...
    assert "voice:json_unwrapped" in res.adjustments        # ...and the trace sees the net work


@pytest.mark.asyncio
async def test_voice_leaves_a_plain_reply_alone_and_does_not_flag():
    ctx = _ctx()
    backend = ScriptedBackend(["Confirmado, Vinicius! Registrei os R$ 50."])
    res = await SuperegoStage().voice(ctx, backend, voice_prompt="persona")
    assert res.response == "Confirmado, Vinicius! Registrei os R$ 50."
    assert "voice:json_unwrapped" not in res.adjustments


@pytest.mark.asyncio
async def test_the_envelope_is_opened_BEFORE_the_pii_backstop_reads_it():
    """Order is the whole point, and an envelope can hide PII from a deterministic detector.

    JSON escapes are legal and a model emits them: `\\u0031\\u0032\\u0033.456.789-09` carries a
    CPF that the detector cannot see in the raw envelope (measured: `[]`) and sees plainly once
    unwrapped (`['NATIONAL_ID']`). Run the backstops on the envelope and the leak ships unflagged
    — so the unwrap has to come first, and this test fails if it is ever moved after.

    Since the PII backstop decides by provenance, the order is now visible in the shipped text
    rather than only in a flag: this CPF is in no allowlist, so opening the envelope first is
    what turns an unflagged leak into a mask. Read the envelope late and the contact receives
    the document number. Asserted under `enforce` because that is where the consequence is
    visible in the TEXT — under the shipped `observe` default the same ordering decides the same
    way, and only the count would show it.
    """
    ctx = _ctx()
    ctx.metadata[mk.PII_OUTPUT_MODE] = vocab.PII_MODE_ENFORCE
    backend = ScriptedBackend(['{"message": "CPF \\u0031\\u0032\\u0033.456.789-09"}'])
    res = await SuperegoStage().voice(ctx, backend, voice_prompt="persona")
    assert "123.456.789-09" not in res.response
    assert res.response == "CPF [NATIONAL_ID REDACTED]"
    assert "pii:flagged_in_output" in res.adjustments
    assert "pii:redacted_in_output" in res.adjustments


@pytest.mark.parametrize("key", ["message", "text", "reply", "response", "content"])
@pytest.mark.asyncio
async def test_the_backstop_covers_the_CORRECTION_entry_point_too(key):
    """`voice()` is called on two paths — the normal one and the judge's re-voice — and the
    backstop lives in `voice()` precisely so one insertion covers both. The key varies between
    turns (measured: `message` and `text` on the same process), so every accepted key is
    exercised here rather than the one that happened to leak first."""
    ctx = _ctx()
    ctx.metadata[mk.VOICE_CORRECTION] = {"kind": "unverified_claim", "reason": "no tool ran"}
    backend = ScriptedBackend([json.dumps({key: "Certo — não confirmei nada ainda."})])
    res = await SuperegoStage().voice(ctx, backend, voice_prompt="persona")
    assert res.response == "Certo — não confirmei nada ainda."
    assert "voice:json_unwrapped" in res.adjustments


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
def test_a_blank_envelope_is_left_alone_because_silence_is_a_different_bug(blank):
    """Nothing downstream guards an empty response — measured across `cogno_soma.pipeline` and
    the host: neither has a fallback for one, so it would reach the channel as silence.
    Unwrapping a blank value trades visible garbage for an unhandled state; the envelope stays,
    and the voicer producing nothing shows up as its own failure."""
    assert unwrap(json.dumps({"message": blank})) is None
