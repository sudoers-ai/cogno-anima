"""The defect, in sentences that are RED on the tree before the change for the reason they name.

This module imports NOTHING the change added (no ``cogno_anima.repairs``, no ``repairs`` field
read by name, no new metakey), on purpose: a twin that fails with an ``ImportError`` or an
``AttributeError`` on the old tree has not shown the defect, it has shown that the test is newer
than the code. Every assertion is about a turn that must not die, or about two RECORDS a stage
leaves — and a record exists in either world.

The defect is one sentence per stage: **a reply the stage had to repair left the same record as
a reply it read as it came.**

1. the NOUMENO: a fenced reply and a bare one;
2. the NER: a reply with a word outside the vocabulary and the reply the stage turned it into;
3. the scope guard: a reply wrapped in prose and a bare one;
4. the judge: a rejection with no critique and one that wrote the default critique itself;
5. the selector: an ``error`` over prose and an ``error`` over an object without its lists;

and one more, of another kind — the repair that was a dead turn:

6. the NOUMENO: a ``null`` in a text field raised ``AttributeError``, which is not a
   ``StageParseError``, over a reply every other field of which was usable.

Measured on the tree before the change: 7 failed. The alphabet, the per-mark twins, the adverse
shapes and the controls are in ``test_repairs_are_counted.py``.
"""

from __future__ import annotations

import json

from cogno_anima import metakeys as mk
from cogno_anima.stages import scope_options
from cogno_anima.stages.ner import IntentAnalyzer
from cogno_anima.stages.noumeno import Noumeno
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.types import PipelineContext
from tests.conftest import StubEmbedder
from tests.unit.test_superego import ScriptedBackend, _ctx

_SCOPE = "You handle scheduling and study support for the school."
_USER = "what can I use to study for the exam?"

_REWRITE = {"rewritten": "Send the invoice.", "context_turn": "", "confidence": 0.9,
            "changed": False, "preserved_terms": [], "rewrite_warnings": []}

_INTENT = {
    "intent_class": "ACTION_REQUEST", "sentiment": "NEUTRAL", "confidence": 0.9,
    "temporal_class": "TIMELESS", "triad_signal": "EGO",
    "entities": {"people": [], "objects": ["invoice"], "concepts": ["billing"]},
    "location": None, "mandatory_tags": ["SYSTEM"],
    "aristotelian": {"ACTION": "SEND_INVOICE | send the invoice"},
    "domains": ["FINANCE"], "goal": "send the invoice",
    "causal_chain": ["the invoice is due"], "parole": "FORMAL", "negation": [],
    "constraints": [], "modality": "CERTAIN", "speech_act": "DIRECTIVE", "verbs": ["send"],
    "context_dependent": False, "is_composite": False, "is_sequential": False, "pii": [],
}


def _record(result) -> dict:
    """Everything a result says, minus what differs between any two calls (the clock, the raw
    reply). What is left is what a host could persist about the stage's READ."""
    dumped = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    dumped.pop("metrics", None)
    dumped.pop("raw_response", None)
    return dumped


async def _noumeno(reply: str):
    ctx = PipelineContext(user_input="Manda a fatura.", force_language="pt")
    ctx = await Noumeno(embedder=StubEmbedder()).process(ctx, ScriptedBackend([reply]))
    return ctx.noumeno


async def _ner(reply: str):
    noumeno = _ctx(user="Manda a fatura.").noumeno
    return await IntentAnalyzer().analyze(noumeno, llm=ScriptedBackend([reply]))


async def test_red_twin_noumeno_a_fenced_reply_is_not_the_record_of_a_bare_one():
    bare = json.dumps(_REWRITE)
    clean, fenced = await _noumeno(bare), await _noumeno(f"```json\n{bare}\n```")
    assert clean.rewritten == fenced.rewritten == "Send the invoice.", "the premise: same read"
    assert _record(clean) != _record(fenced)


async def test_red_twin_ner_a_coerced_word_is_not_the_record_of_the_word_it_became():
    said, became = dict(_INTENT, sentiment="ANGRY-ISH"), dict(_INTENT, sentiment="NEUTRAL")
    dirty, clean = await _ner(json.dumps(said)), await _ner(json.dumps(became))
    assert dirty.sentiment == clean.sentiment == "NEUTRAL", "the premise: the word was coerced"
    assert _record(dirty) != _record(clean)


async def _guard(reply: str) -> "tuple[bool, dict]":
    ctx = _ctx(user=_USER, intent_class="INFORMATION_REQUEST")
    result = await SuperegoStage().check_input_scope(ctx, ScriptedBackend([reply]),
                                                     scope_prompt=_SCOPE)
    left = {k: v for k, v in ctx.metadata.items() if k.startswith("scope_")}
    return result.blocked, {"result": _record(result), "metadata": left}


async def test_red_twin_guard_a_reply_in_prose_is_not_the_record_of_a_bare_one():
    bare = '{"blocked": false, "refusal_message": ""}'
    (b1, clean), (b2, wrapped) = await _guard(bare), await _guard(f"Here you go: {bare} Done.")
    assert b1 is b2 is False, "the premise: both were read, and both allowed"
    assert clean != wrapped


async def test_red_twin_judge_a_missing_critique_is_not_the_record_of_a_written_one():
    stage = SuperegoStage()
    without = await stage.evaluate(_ctx(), ScriptedBackend(['{"approved": false}']),
                                   limits_prompt="")
    written = await stage.evaluate(
        _ctx(), ScriptedBackend(['{"approved": false, "critique": "execution rejected"}']),
        limits_prompt="")
    assert without.critique == written.critique == "execution rejected", "the premise"
    assert _record(without) != _record(written)


async def test_red_twin_selector_an_error_over_prose_is_not_an_error_over_a_wrong_object():
    options = ["Enrolment", "Timetable"]
    prose = await scope_options.select_options(
        "is there a gym?", options, ScriptedBackend(["I could not decide."]))
    wrong = await scope_options.select_options(
        "is there a gym?", options, ScriptedBackend(['{"picked": ["Timetable"]}']))
    assert prose.outcome == wrong.outcome == scope_options.OUTCOME_ERROR, "the premise"
    assert prose.record() != wrong.record()


async def test_red_twin_noumeno_a_null_rewrite_does_not_kill_the_turn():
    """``data.get("rewritten", "").strip()`` on a ``null``: ``AttributeError``, a dead turn.
    An ABSENT key has always read as the contact's own text; a ``null`` is that same reply."""
    result = await _noumeno(json.dumps(dict(_REWRITE, rewritten=None)))
    assert result.rewritten == "Manda a fatura."


async def test_red_twin_noumeno_a_null_context_does_not_kill_the_turn():
    result = await _noumeno(json.dumps(dict(_REWRITE, context_turn=None)))
    assert result.rewritten == "Send the invoice." and result.context_turn == ""


def test_the_guards_record_keys_are_the_ones_this_module_reads():
    """The guard twin filters the metadata by prefix; if the guard's keys ever stop sharing it,
    the twin would compare less than it says."""
    assert all(key.startswith("scope_") for key in (
        mk.SCOPE_VERDICT_READ, mk.SCOPE_PROMPT_SHA, mk.SCOPE_PROMPT_BLOCKS))
