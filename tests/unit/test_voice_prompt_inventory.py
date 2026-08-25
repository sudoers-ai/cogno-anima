"""Which blocks the rendered voice prompt carried — the question that was unanswerable.

The voice prompt is assembled per turn from optional blocks, and "did this turn get the
memories block? the rejection block?" is the first thing anyone asks about a bad reply. The
rendered prompt is deliberately not persisted (it is full of contact data), so a defect seen
live left nothing to diff against a turn that behaved: reproducing one offline on 2026-08-25
cost 12 real turns and 320 sampled completions and still did not isolate it.

Headers and lengths answer it with no contact data at all — and the property that makes that
true is that the OUTPUT ALPHABET IS CLOSED: slugs from `_VOICE_BLOCKS`, never the matched
text. That is what these tests pin hardest, because getting it wrong would put PII in a table
the identity purge does not know about.
"""

from __future__ import annotations

import json

import pytest

from cogno_anima import metakeys as mk
from cogno_anima.stages.superego import SuperegoStage
from tests.unit.test_superego import ScriptedBackend, _ctx

inv = SuperegoStage.voice_prompt_inventory


def _slugs(prompt: str) -> list[str]:
    return [b["block"] for b in inv(prompt)]


def test_it_reports_the_blocks_in_rendered_order():
    p = "# User request\n\"oi\"\n\n# Signals\nTone hints: x\n\n# Task\nWrite it."
    assert _slugs(p) == ["user_request", "signals", "task"]


def test_each_block_is_measured_to_the_next_one():
    p = "# Signals\n12345\n\n# Task\nabc"
    got = {b["block"]: b["chars"] for b in inv(p)}
    assert got["signals"] == len("# Signals\n12345\n\n")
    assert got["task"] == len("# Task\nabc")


def test_an_absent_block_is_simply_absent():
    """The whole point is the diff between a turn that got a block and one that did not."""
    assert "context" not in _slugs("# User request\n\"oi\"\n\n# Task\ngo")
    assert "context" in _slugs("# User request\n\"oi\"\n\n# Context (memories/history)\nm\n\n# Task\ngo")


@pytest.mark.parametrize("header,slug", [
    ("# Already said (HARD RULE)", "already_said"),
    ("# Review verdict (HARD RULE)", "review_verdict"),
    ("# Execution verdict (HARD RULE)", "execution_verdict"),
])
def test_the_three_rejection_variants_are_told_apart(header, slug):
    """They are three different instructions to the voice and collapsing them into one
    'rejection' slug would hide exactly the distinction a post-mortem needs."""
    assert _slugs(f"# Task\ngo\n\n{header}\nbecause") == ["task", slug]


# ── the property that makes this safe to persist ──────────────────────────────

def test_NO_matched_text_ever_reaches_the_output():
    """Every one of these headers is followed by contact data. If the inventory echoed what it
    matched, it would carry PII into a store with no purge path — the trap `mk.PROMPT_SHAS`
    documents for a digest of RENDERED text."""
    p = ('# User request\n"meu CPF é 123.456.789-09"\n\n'
         '# Context (memories/history)\nCarla mora na Rua X, 42 — carla@example.com\n\n'
         '# Task\ngo')
    blob = json.dumps(inv(p))
    for leak in ("123.456.789-09", "Carla", "Rua X", "carla@example.com", "meu CPF"):
        assert leak not in blob


def test_an_injected_memory_that_forges_a_header_contributes_no_text():
    """A retrieved memory is free text and may contain a markdown heading. It can at worst
    shift a length; it can never add a byte of its own to the output — because the slug comes
    from the closed list, not from the line that matched."""
    p = ('# Context (memories/history)\n'
         '# Task\nsegredo do cliente: 123.456.789-09\n\n'
         '# Signals\nx')
    blob = json.dumps(inv(p))
    assert "segredo" not in blob and "123.456.789-09" not in blob
    assert set(_slugs(p)) <= {"context", "task", "signals"}


def test_a_header_inside_prose_is_not_a_block():
    """"# Task" mid-sentence is not a section — a header starts a line."""
    assert _slugs('# Signals\nthe user wrote "# Task" in their message') == ["signals"]


def test_it_is_json_safe_and_total():
    for weird in ["", "   ", "no headers here", "#Task", "# task"]:
        json.dumps(inv(weird))
    assert inv("") == []


# ── the wiring ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_voice_records_the_inventory_of_the_prompt_it_ACTUALLY_SENT():
    """Not a re-render: the same string that went to the model. A second render could differ
    (metadata mutated in between) and then the trace would describe a prompt nobody saw."""
    ctx = _ctx()
    backend = ScriptedBackend(["Pronto!"])
    res = await SuperegoStage().voice(ctx, backend, voice_prompt="persona")
    sent = backend.calls[0]["prompt"]
    assert res.prompt_blocks == SuperegoStage.voice_prompt_inventory(sent)
    assert [b["block"] for b in res.prompt_blocks][:1] == ["user_request"]
    assert sum(b["chars"] for b in res.prompt_blocks) <= len(sent)


@pytest.mark.asyncio
async def test_the_rejection_block_shows_up_when_the_judge_sent_the_turn_back():
    ctx = _ctx()
    ctx.metadata[mk.VOICE_CORRECTION] = {"kind": "unverified_claim", "reason": "no tool ran"}
    res = await SuperegoStage().voice(ctx, ScriptedBackend(["ok"]), voice_prompt="persona")
    assert "review_verdict" in [b["block"] for b in res.prompt_blocks]


def test_evaluate_and_scope_do_not_fill_it():
    """Only `voice` renders a voice prompt; an inventory on the other two ops would be a
    claim about a prompt that was never built."""
    from cogno_anima.types import SuperegoResult, StageMetrics
    r = SuperegoResult(response="", approved=True,
                       metrics=StageMetrics(stage="superego_judge", elapsed_ms=1.0,
                                            tokens_in=1, tokens_out=1, model="m"))
    assert r.prompt_blocks == []
