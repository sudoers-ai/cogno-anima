"""What a stage REPAIRS in a model's JSON reply is counted — and counting decides nothing.

Five stages ask for a JSON object and make the reply usable by hand: the NOUMENO, the NER, the
scope guard, the judge and the scope-option selector. ``cogno_anima.repairs`` is the closed
alphabet of those repairs, and each stage's result carries the marks of its own call.

The RED TWINS — one sentence per stage, each failing on the tree before the change — live in
``test_repairs_red_twins.py``, which imports nothing the change added. This file is everything
around them:

* the alphabet is closed, derived from one table per stage, and a mark is never a byte of the
  reply (a canary written into every key and value of a reply reaches no mark);
* EVERY member of every alphabet is produced by a dirty reply, and the clean reply of that
  stage produces none — both directions, enumerated from the alphabet, so a mark nobody can
  produce and a mark nobody declared both fail here;
* the marks decide nothing: with the marking function switched off, every result is the same
  except for the list itself;
* the guard's and the judge's lists never repeat ``verdict_read``;
* the carriers: ``mk.SCOPE_REPAIRS`` is stamped when a verdict was asked and removed on a
  bypass, and the selector's record gains the list.
"""

from __future__ import annotations

import json
import re

import pytest
from hypothesis import given, settings, strategies as st

import cogno_anima
from cogno_anima import metakeys as mk
from cogno_anima import repairs as rp
from cogno_anima.errors import StageParseError
from cogno_anima.stages import ner as ner_stage
from cogno_anima.stages import noumeno as noumeno_stage
from cogno_anima.stages import scope_options
from cogno_anima.stages.ner import IntentAnalyzer
from cogno_anima.stages.noumeno import Noumeno
from cogno_anima.stages.superego import UNREADABLE_VERDICT_CRITIQUE, SuperegoStage
from cogno_anima.types import PipelineContext, StageMetrics
from cogno_anima.utils import generate_json_resilient
from cogno_anima.verdict import (VALID_VERDICT_READS, VERDICT_BOOLEAN, VERDICT_CALL_FAILED,
                                 VERDICT_STRING_BOOL, VERDICT_UNPARSEABLE)
from tests.conftest import StubEmbedder
from tests.unit.test_superego import RaisingBackend, ScriptedBackend, _ctx

_SCOPE = "You handle scheduling and study support for the school."
_USER = "what can I use to study for the exam?"
_CANARY = "zxqcanary"


def _fence(body: str) -> str:
    return f"```json\n{body}\n```"


def _think(body: str) -> str:
    return f"<think>weighing it</think>\n{body}"


# ── the alphabet ─────────────────────────────────────────────────────────────────────────

def test_every_alphabet_is_derived_from_its_table_and_nothing_else():
    tables = {
        "noumeno": ((rp.SALVAGED_OBJECT, rp.FENCE_OR_THINK_STRIPPED, rp.TRUNCATED_RETRY),
                    rp.NOUMENO_REPAIR_FIELDS),
        "ner": ((rp.SALVAGED_OBJECT, rp.FENCE_OR_THINK_STRIPPED, rp.TRUNCATED_RETRY),
                rp.NER_REPAIR_FIELDS),
        "superego_scope": ((rp.SALVAGED_OBJECT, rp.FENCE_OR_THINK_STRIPPED),
                           rp.SCOPE_REPAIR_FIELDS),
        "superego_judge": ((rp.SALVAGED_OBJECT, rp.FENCE_OR_THINK_STRIPPED),
                           rp.JUDGE_REPAIR_FIELDS),
        "superego_select": ((rp.SALVAGED_OBJECT, rp.FENCE_OR_THINK_STRIPPED, rp.UNPARSEABLE),
                            rp.SELECT_REPAIR_FIELDS),
    }
    assert set(tables) == set(rp.REPAIRS_BY_STAGE)
    for stage, (singles, fields) in tables.items():
        derived = set(singles) | {f"{family}:{name}" for family, names in fields.items()
                                  for name in names}
        assert rp.REPAIRS_BY_STAGE[stage] == derived, stage
        assert set(fields) <= set(rp.REPAIR_FAMILIES), stage
    assert rp.VALID_REPAIRS == set().union(*rp.REPAIRS_BY_STAGE.values())


def test_the_stage_keys_are_the_stages_own_metrics_labels():
    """A reader closes a list with the row of ITS stage — so the keys are the labels the
    stages put on their ``StageMetrics``, read from the code and not retyped."""
    assert noumeno_stage.STAGE_NAME in rp.REPAIRS_BY_STAGE
    assert ner_stage.STAGE_NAME in rp.REPAIRS_BY_STAGE
    assert scope_options.SELECT_STAGE in rp.REPAIRS_BY_STAGE
    assert {"superego_scope", "superego_judge"} <= set(rp.REPAIRS_BY_STAGE)


def test_a_mark_is_a_constant_or_a_family_and_a_field_name():
    shape = re.compile(r"^[a-z_]+(:[a-z_]+)?$")
    for mark in rp.VALID_REPAIRS:
        assert shape.match(mark), mark
        if ":" in mark:
            assert mark.split(":")[0] in rp.REPAIR_FAMILIES, mark


def test_the_verdict_keys_are_in_no_alphabet_and_unparseable_is_the_selectors_alone():
    """``verdict_read`` already says how ``approved``/``blocked`` was read. A second alphabet
    for that fact would let the two disagree."""
    for stage in ("superego_scope", "superego_judge"):
        marks = rp.REPAIRS_BY_STAGE[stage]
        assert not any(m.endswith((":approved", ":blocked")) for m in marks)
        assert not (marks & set(VALID_VERDICT_READS)), stage
    assert rp.UNPARSEABLE is VERDICT_UNPARSEABLE or rp.UNPARSEABLE == VERDICT_UNPARSEABLE
    holders = [s for s, marks in rp.REPAIRS_BY_STAGE.items() if rp.UNPARSEABLE in marks]
    assert holders == ["superego_select"]


def test_the_package_exports_what_a_host_needs_to_close_a_list():
    assert cogno_anima.REPAIRS_BY_STAGE is rp.REPAIRS_BY_STAGE
    assert cogno_anima.VALID_REPAIRS is rp.VALID_REPAIRS
    assert cogno_anima.closed_repairs is rp.closed_repairs


# ── the log and the reader on the far side of a carrier ─────────────────────────────────

def test_the_log_drops_what_its_stage_does_not_declare():
    log = rp.RepairLog("superego_judge")
    log.add("free text a model wrote")
    log.add(rp.TRUNCATED_RETRY)                 # a real mark — of another stage
    log.field(rp.CUT, "critique")               # a real family this stage has no row for
    log.field(rp.DEFAULT, "goal")               # a real mark of the NER
    log.add(None)                               # type: ignore[arg-type]
    assert log.marks() == []
    log.add(rp.SALVAGED_OBJECT)
    log.add(rp.SALVAGED_OBJECT)
    assert log.marks() == [rp.SALVAGED_OBJECT] and len(log) == 1


def test_an_unknown_stage_records_nothing():
    log = rp.RepairLog("voice")
    log.add(rp.SALVAGED_OBJECT)
    assert log.marks() == []


def test_one_form_mark_per_field_and_the_other_families_sit_beside_it():
    log = rp.RepairLog("ner")
    log.field(rp.NULL_FIELD, "intent_class")
    log.field(rp.COERCED, "intent_class")       # the same question, already answered
    log.field(rp.FALLBACK, "intent_class")      # another question
    log.field(rp.COERCED, "goal")
    log.field(rp.CUT, "goal")
    assert log.marks() == ["null_field:intent_class", "fallback:intent_class",
                           "coerced:goal", "cut:goal"]


def test_a_refused_form_mark_does_not_take_the_fields_one_slot():
    """``null_field:goal`` is not in the NER's alphabet (the prompt allows it). Refusing it must
    not stop the ``coerced:goal`` that follows."""
    log = rp.RepairLog("ner")
    log.field(rp.NULL_FIELD, "goal")
    log.field(rp.COERCED, "goal")
    assert log.marks() == ["coerced:goal"]


def test_the_log_hands_out_a_copy():
    log = rp.RepairLog("ner")
    log.add(rp.SALVAGED_OBJECT)
    log.marks().append("mine")
    assert log.marks() == [rp.SALVAGED_OBJECT]


@pytest.mark.parametrize("raw,stage,expected", [
    (["cut:goal", "cut:goal", "default:critique", 7, None, "x"], "ner", ["cut:goal"]),
    (("salvaged_object",), "superego_judge", ["salvaged_object"]),
    (["cut:goal"], "superego_judge", []),            # a real mark, the wrong stage
    ("salvaged_object", "ner", []),                  # a string is not a list of marks
    ({"salvaged_object": 1}, "ner", []),
    (None, "ner", []),
    (["salvaged_object"], "voice", []),
    (["salvaged_object"], None, []),
])
def test_closed_repairs_lets_only_members_through(raw, stage, expected):
    assert rp.closed_repairs(raw, stage) == expected


def test_closed_repairs_never_raises_on_a_carrier_gone_wrong():
    class Hostile(list):
        def __iter__(self):
            raise RuntimeError("no")
    with pytest.raises(RuntimeError):
        list(Hostile())                           # the premise: iterating it does raise
    assert rp.closed_repairs({"a"}, "ner") == []
    assert rp.closed_repairs(object(), "ner") == []


# ── the reply as a whole (guard, judge, selector) ───────────────────────────────────────

_OBJ = '{"approved": true}'


@pytest.mark.parametrize("text,stripped,expected", [
    (_OBJ, False, []),
    ("  " + _OBJ + "\n", False, []),
    (_OBJ, True, [rp.FENCE_OR_THINK_STRIPPED]),
    (_fence(_OBJ), False, [rp.FENCE_OR_THINK_STRIPPED]),
    ("```\n" + _OBJ + "\n```", False, [rp.FENCE_OR_THINK_STRIPPED]),
    ("Sure: " + _OBJ, False, [rp.SALVAGED_OBJECT]),
    (_OBJ + " — done", False, [rp.SALVAGED_OBJECT]),
    ("[" + _OBJ + "]", False, [rp.SALVAGED_OBJECT]),
    ("Here it is:\n" + _fence(_OBJ), False, [rp.SALVAGED_OBJECT]),
    (_fence("Sure: " + _OBJ), False, [rp.FENCE_OR_THINK_STRIPPED, rp.SALVAGED_OBJECT]),
    # Nothing was read, so nothing was salvaged: saying so is the verdict reader's.
    ("no json here", False, []),
    ("no json here", True, [rp.FENCE_OR_THINK_STRIPPED]),
    (_OBJ + " " + _OBJ, False, []),
    ("", False, []),
    (None, False, []),
])
def test_reply_form_repairs(text, stripped, expected):
    assert rp.reply_form_repairs(text, cot_stripped=stripped) == expected


@pytest.mark.parametrize("value,expected", [
    ("true", True), ("False", True), ("  YES ", True), ("0", True), ("no", True),
    ("maybe", False), ("", False), (True, False), (1, False), (None, False),
])
def test_spelled_boolean(value, expected):
    assert rp.spelled_boolean(value) is expected


# ── NOUMENO ─────────────────────────────────────────────────────────────────────────────

_REWRITE = {"rewritten": "Send the invoice.", "context_turn": "", "confidence": 0.9,
            "changed": False, "preserved_terms": [], "rewrite_warnings": []}
_CUT_REPLY = '{"rewritten": "Send the inv'


def _without(obj: dict, key: str) -> dict:
    return {k: v for k, v in obj.items() if k != key}


#: One dirty reply per mark of the NOUMENO's alphabet. A LIST of replies is a scripted
#: sequence (the truncation retry needs two).
_NOUMENO_DIRTY: "dict[str, object]" = {
    rp.SALVAGED_OBJECT: "{} " + json.dumps(_REWRITE),
    rp.FENCE_OR_THINK_STRIPPED: _fence(json.dumps(_REWRITE)),
    rp.TRUNCATED_RETRY: [_CUT_REPLY, json.dumps(_REWRITE)],
    "default:rewritten": _without(_REWRITE, "rewritten"),
    "default:context_turn": _without(_REWRITE, "context_turn"),
    "default:confidence": _without(_REWRITE, "confidence"),
    "default:changed": _without(_REWRITE, "changed"),
    "default:preserved_terms": _without(_REWRITE, "preserved_terms"),
    "default:rewrite_warnings": _without(_REWRITE, "rewrite_warnings"),
    "null_field:rewritten": dict(_REWRITE, rewritten=None),
    "null_field:context_turn": dict(_REWRITE, context_turn=None),
    "null_field:changed": dict(_REWRITE, changed=None),
    "null_field:preserved_terms": dict(_REWRITE, preserved_terms=None),
    "null_field:rewrite_warnings": dict(_REWRITE, rewrite_warnings=None),
    "string_bool:changed": dict(_REWRITE, changed="false"),
    "coerced:confidence": dict(_REWRITE, confidence="0.9"),
    "coerced:changed": dict(_REWRITE, changed=1),
    "coerced:preserved_terms": dict(_REWRITE, preserved_terms="invoice"),
    "coerced:rewrite_warnings": dict(_REWRITE, rewrite_warnings="LOW_CONFIDENCE"),
}


def _script(reply: object) -> "list[str]":
    if isinstance(reply, list):
        return list(reply)
    return [reply if isinstance(reply, str) else json.dumps(reply)]


async def _noumeno(reply: object, user: str = "Manda a fatura."):
    ctx = PipelineContext(user_input=user, force_language="pt")
    ctx = await Noumeno(embedder=StubEmbedder()).process(ctx, ScriptedBackend(_script(reply)))
    return ctx.noumeno


def test_the_noumeno_table_covers_its_alphabet_exactly():
    assert set(_NOUMENO_DIRTY) == rp.VALID_NOUMENO_REPAIRS


async def test_noumeno_a_clean_reply_records_nothing():
    result = await _noumeno(_REWRITE)
    assert result.repairs == [] and result.degradations == []


@pytest.mark.parametrize("mark", sorted(rp.VALID_NOUMENO_REPAIRS))
async def test_noumeno_every_mark_has_a_reply_that_produces_it_alone(mark):
    result = await _noumeno(_NOUMENO_DIRTY[mark])
    assert result.repairs == [mark]
    assert result.degradations == [], "a repair is not a degradation"


async def test_noumeno_an_empty_rewrite_is_the_default_an_absent_one_is():
    for value in ("", "   "):
        result = await _noumeno(dict(_REWRITE, rewritten=value))
        assert result.rewritten == "Manda a fatura."
        assert result.repairs == ["default:rewritten"]


@pytest.mark.parametrize("name,value", [
    ("confidence", True), ("confidence", "1"), ("changed", "maybe"), ("changed", 0),
    ("changed", []), ("preserved_terms", {"a": 1}), ("rewrite_warnings", {"a": 1}),
])
async def test_noumeno_other_shapes_of_a_coerced_value(name, value):
    assert (await _noumeno(dict(_REWRITE, **{name: value}))).repairs == [f"coerced:{name}"]


@pytest.mark.parametrize("name,value", [
    ("confidence", 0), ("confidence", 1), ("confidence", 0.0), ("changed", True),
    ("preserved_terms", ["invoice"]), ("rewrite_warnings", ["LOW_CONFIDENCE"]),
    ("context_turn", "The user wants the invoice sent."), ("rewritten", "  Send it.  "),
])
async def test_noumeno_what_the_contract_allows_is_no_mark(name, value):
    assert (await _noumeno(dict(_REWRITE, **{name: value}))).repairs == []


@pytest.mark.parametrize("value", ["TRUE", " False ", "yes", "0"])
async def test_noumeno_a_spelled_changed_is_a_string_bool_and_reads_false(value):
    result = await _noumeno(dict(_REWRITE, changed=value))
    assert result.changed is False and result.repairs == ["string_bool:changed"]


async def test_noumeno_an_empty_context_is_an_answer_not_a_repair():
    assert (await _noumeno(dict(_REWRITE, context_turn=""))).repairs == []


async def test_noumeno_marks_accumulate_in_the_order_they_happened():
    reply = _think(_fence("{} " + json.dumps(_without(dict(_REWRITE, changed="true"),
                                                      "confidence"))))
    result = await _noumeno(reply)
    assert result.repairs == [rp.FENCE_OR_THINK_STRIPPED, rp.SALVAGED_OBJECT,
                              "default:confidence", "string_bool:changed"]


async def test_noumeno_a_reply_that_raises_recorded_nothing_on_the_way():
    """The fence came off and THEN the read failed: the turn has no result to carry a mark, and
    the log handed to the parser must not have one either."""
    log = rp.RepairLog("noumeno")
    with pytest.raises(StageParseError):
        Noumeno(embedder=StubEmbedder())._parse_json(_fence("not json at all"), log)
    assert log.marks() == []


async def test_noumeno_what_still_raises_still_raises():
    """The one read that changed is ``null`` in a TEXT field. A value that is neither text nor
    ``null``, and a ``null`` confidence, end the turn exactly as before."""
    with pytest.raises(AttributeError):
        await _noumeno(dict(_REWRITE, rewritten=5))
    with pytest.raises(AttributeError):
        await _noumeno(dict(_REWRITE, context_turn=["x"]))
    with pytest.raises(TypeError):
        await _noumeno(dict(_REWRITE, confidence=None))


def test_noumeno_parse_json_is_callable_as_it_always_was():
    stage = Noumeno(embedder=StubEmbedder())
    assert stage._parse_json(json.dumps(_REWRITE)) == _REWRITE
    assert stage._parse_json(_fence(json.dumps(_REWRITE))) == _REWRITE


# ── NER ─────────────────────────────────────────────────────────────────────────────────

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

#: A value of the wrong type or outside the vocabulary, per field. The strict question —
#: would a closed enumeration have refused it? — so the lower-case and the aliased member are
#: here although the stage recovers both.
_NER_COERCED = {
    "intent_class": "action_request", "sentiment": "ANGRY-ISH", "confidence": "0.9",
    "temporal_class": "SOON", "triad_signal": "ALL", "entities": ["invoice"], "location": 5,
    "mandatory_tags": ["NER.SYSTEM"], "aristotelian": {"MOOD": "CALM | at ease"},
    "domains": ["ECONOMICS"], "goal": 12, "causal_chain": "because", "parole": "SLANG",
    "negation": [3], "constraints": {"a": 1}, "modality": "SURE", "speech_act": "",
    "verbs": [None, "send"], "context_dependent": "maybe", "is_composite": 1,
    "is_sequential": [True], "pii": ["CPF"],
}
_NER_CUT = {
    "mandatory_tags": ["SYSTEM", "MATH", "ANALYSIS", "CREATIVE"],
    "aristotelian": {"ACTION": "SEND_INVOICE | " + "d" * 41},
    "goal": "g" * 81, "causal_chain": ["cause"] * 5, "negation": ["n" * 41],
    "constraints": ["a", "b", "c", "d", "e"], "verbs": ["v"] * 6,
}
_NER_FALLBACK = {
    "intent_class": dict(_INTENT, intent_class="UNKNOWN"),
    "domains": dict(_INTENT, domains=[], mandatory_tags=["MATH"]),
}


def _ner_dirty(mark: str) -> object:
    if mark == rp.SALVAGED_OBJECT:
        return "{} " + json.dumps(_INTENT)
    if mark == rp.FENCE_OR_THINK_STRIPPED:
        return _fence(json.dumps(_INTENT))
    if mark == rp.TRUNCATED_RETRY:
        return ['{"intent_class": "ACTION_REQ', json.dumps(_INTENT)]
    family, name = mark.split(":")
    if family == rp.DEFAULT:
        return _without(_INTENT, name)
    if family == rp.NULL_FIELD:
        return dict(_INTENT, **{name: None})
    if family == rp.STRING_BOOL:
        return dict(_INTENT, **{name: "false"})
    if family == rp.COERCED:
        return dict(_INTENT, **{name: _NER_COERCED[name]})
    if family == rp.CUT:
        return dict(_INTENT, **{name: _NER_CUT[name]})
    if family == rp.FALLBACK:
        return _NER_FALLBACK[name]
    raise AssertionError(mark)


async def _ner(reply: object):
    noumeno = _ctx(user="Manda a fatura.").noumeno
    return await IntentAnalyzer().analyze(noumeno, llm=ScriptedBackend(_script(reply)))


def test_the_ner_tables_cover_their_families_exactly():
    assert set(_NER_COERCED) == set(rp.NER_REPAIR_FIELDS[rp.COERCED])
    assert set(_NER_CUT) == set(rp.NER_REPAIR_FIELDS[rp.CUT])
    assert set(_NER_FALLBACK) == set(rp.NER_REPAIR_FIELDS[rp.FALLBACK])


def test_the_ner_table_names_every_field_the_stage_reads():
    """The NER's ``default`` row is the list of keys ``_build`` reads from the model's object.
    Read from the SOURCE, so a field ``_build`` starts reading tomorrow fails here until the
    table names it (or this test names why it is exempt)."""
    import inspect
    source = inspect.getsource(IntentAnalyzer._build)
    read = set(re.findall(r'data\.get\("([a-z_]+)"', source))
    assert read, "the premise: the reads were found"
    assert read == set(rp.NER_REPAIR_FIELDS[rp.DEFAULT])


def test_the_noumeno_table_names_every_field_the_stage_reads():
    import inspect
    source = inspect.getsource(Noumeno.process)
    read = set(re.findall(r'data2?\.get\("([a-z_]+)"', source))
    read |= set(re.findall(r'_text\(data, "([a-z_]+)"\)', source))
    read |= set(re.findall(r'read_boolean\(data, "([a-z_]+)"\)', source))
    assert read == set(rp.NOUMENO_REPAIR_FIELDS[rp.DEFAULT])


async def test_ner_a_clean_reply_records_nothing():
    assert (await _ner(_INTENT)).repairs == []


@pytest.mark.parametrize("mark", sorted(rp.VALID_NER_REPAIRS))
async def test_ner_every_mark_has_a_reply_that_produces_it(mark):
    """One dirty field gives ONE form mark and nothing else — except the fallback that the
    dirty field itself triggers (a missing ``intent_class`` is then filled from the tags)."""
    result = await _ner(_ner_dirty(mark))
    assert mark in result.repairs
    beside = set(result.repairs) - {mark}
    assert all(m.startswith(rp.FALLBACK + ":") for m in beside), result.repairs


@pytest.mark.parametrize("value", ["action_request", " ACTION_REQUEST", "Action_Request"])
async def test_ner_a_member_in_another_spelling_is_a_mark_whatever_the_stage_recovers(value):
    result = await _ner(dict(_INTENT, intent_class=value))
    assert "coerced:intent_class" in result.repairs


@pytest.mark.parametrize("name", ["location", "goal", "parole", "modality", "speech_act"])
async def test_ner_a_null_the_prompt_allows_is_no_mark(name):
    assert (await _ner(dict(_INTENT, **{name: None}))).repairs == []


#: Shapes of the SAME defect the table above does not hold: one more dirty value per rule.
@pytest.mark.parametrize("name,value", [
    ("confidence", 1.5), ("confidence", -0.1), ("confidence", True),
    ("domains", ["FINANCE", "FINANCE"]), ("domains", ["finance"]), ("domains", "FINANCE"),
    ("domains", [7]),
    ("entities", {"people": "nobody", "objects": [], "concepts": []}),
    ("entities", {"people": [], "objects": [3], "concepts": []}),
    ("entities", {"people": [], "objects": [""], "concepts": []}),
    ("mandatory_tags", ["system"]), ("mandatory_tags", "SYSTEM"), ("mandatory_tags", [1]),
    ("aristotelian", {"ACTION": 5}), ("aristotelian", {"action": "SEND | it"}),
    ("aristotelian", ["ACTION"]), ("aristotelian", {"ACTION": "  "}),
    ("location", ["Lisbon"]), ("goal", ["send"]), ("goal", False),
    ("parole", "formal"), ("modality", 3), ("speech_act", ["DIRECTIVE"]),
    ("verbs", "send"), ("causal_chain", [""]), ("negation", [None]),
    ("is_composite", "perhaps"), ("is_sequential", 0), ("context_dependent", []),
    ("pii", "EMAIL"), ("pii", ["email"]), ("pii", [4]),
])
async def test_ner_other_shapes_of_a_coerced_value(name, value):
    result = await _ner(dict(_INTENT, **{name: value}))
    assert f"coerced:{name}" in result.repairs


#: The ADVERSE half: values that look unusual and are what the contract allows. None marks.
@pytest.mark.parametrize("name,value", [
    ("confidence", 0), ("confidence", 1), ("confidence", 0.0), ("confidence", 1.0),
    ("entities", {}), ("entities", {"people": []}),
    ("entities", {"people": [], "objects": [], "concepts": [], "pronouns": 7}),
    ("location", ""), ("location", "Lisbon"), ("goal", ""),
    ("mandatory_tags", ["UNKNOWN"]), ("mandatory_tags", ["SYSTEM", "MATH", "ANALYSIS"]),
    ("aristotelian", {}), ("aristotelian", {"ACTION": "SEND | " + "d" * 40}),
    ("aristotelian", {"TIME": "x" * 40}),
    ("goal", "g" * 80), ("goal", "  " + "g" * 80 + "  "),
    ("causal_chain", []), ("causal_chain", ["c" * 60] * 4),
    ("negation", ["n" * 40] * 4), ("constraints", ["c" * 40] * 4), ("verbs", ["v" * 40] * 5),
    ("pii", ["EMAIL", "PHONE"]), ("domains", ["FINANCE", "GENERAL"]),
    ("context_dependent", True), ("is_composite", True),
])
async def test_ner_what_the_contract_allows_is_no_mark(name, value):
    assert (await _ner(dict(_INTENT, **{name: value}))).repairs == []


@pytest.mark.parametrize("name,value", [
    ("mandatory_tags", ["SYSTEM"] * 4), ("goal", "g" * 81),
    ("aristotelian", {"TIME": "x" * 41}), ("causal_chain", ["c" * 61]),
    ("causal_chain", ["c"] * 5), ("negation", ["n"] * 5), ("constraints", ["c" * 41]),
    ("verbs", ["v" * 41]), ("verbs", ["v"] * 6),
])
async def test_ner_one_past_the_cap_is_cut(name, value):
    assert (await _ner(dict(_INTENT, **{name: value}))).repairs == [f"cut:{name}"]


async def test_ner_only_an_entry_the_stage_keeps_can_be_cut():
    """A long description under a key the stage DROPS is a coerced value, not a cut one."""
    result = await _ner(dict(_INTENT, aristotelian={"MOOD": "CALM | " + "d" * 60}))
    assert result.aristotelian == {} and result.repairs == ["coerced:aristotelian"]


async def test_ner_an_empty_domain_list_is_an_answer_and_the_fallback_is_the_mark():
    """``[]`` is what the prompt asks for when no domain fits. What the tags then filled in is
    the fallback — and with a tag that maps to no domain there is nothing to mark."""
    filled = await _ner(dict(_INTENT, domains=[]))
    assert filled.domains == ["TECH"] and filled.repairs == ["fallback:domains"]
    left = await _ner(dict(_INTENT, domains=[], mandatory_tags=["CREATIVE"]))
    assert left.domains == [] and left.repairs == []


async def test_ner_an_unknown_the_model_wrote_is_an_answer_and_the_fallback_is_the_mark():
    """``UNKNOWN`` is a member of the vocabulary: the model answered. What the tags then did is
    the fallback, and without usable tags there is nothing to mark at all."""
    assert (await _ner(dict(_INTENT, intent_class="UNKNOWN"))).repairs == [
        "fallback:intent_class"]
    left = await _ner(dict(_INTENT, intent_class="UNKNOWN", mandatory_tags=["LINGUISTIC"]))
    assert left.intent_class == "UNKNOWN" and left.repairs == []


async def test_ner_what_the_stage_decides_from_readable_values_is_not_a_repair():
    """Grounding, the anaphora flip and the composite/sequential reconciliation are the stage's
    judgement over a well-formed reply — a strict schema would remove none of them."""
    invented = dict(_INTENT, entities={"people": ["Zebedeu Quintanilha"], "objects": [],
                                       "concepts": []})
    result = await _ner(invented)
    assert result.entities_people == [] and result.repairs == []
    sequential = await _ner(dict(_INTENT, is_composite=False, is_sequential=True))
    assert sequential.is_sequential is False and sequential.repairs == []


async def test_ner_the_marks_come_through_the_one_entry_point_too():
    stage = IntentAnalyzer()
    metrics = StageMetrics(stage="ner", elapsed_ms=0.0, tokens_in=0, tokens_out=0, model="t")
    result = stage._parse(_fence(json.dumps(_without(_INTENT, "goal"))), metrics)
    assert result.repairs == [rp.FENCE_OR_THINK_STRIPPED, "default:goal"]
    assert stage._build(dict(_INTENT), "", metrics).repairs == []


# ── the marks DECIDE NOTHING ────────────────────────────────────────────────────────────

def _decided(result) -> dict:
    dumped = result.model_dump()
    for key in ("repairs", "metrics"):
        dumped.pop(key)
    return dumped


async def test_ner_switching_the_marking_off_changes_nothing_but_the_list(monkeypatch):
    """Every dirty reply of the table, read twice: with ``_reply_repairs`` as it is, and with
    it replaced by nothing. The two results differ in ``repairs`` and in nothing else."""
    marks = [m for m in sorted(rp.VALID_NER_REPAIRS) if m != rp.TRUNCATED_RETRY]
    with_marks = [await _ner(_ner_dirty(m)) for m in marks]
    monkeypatch.setattr(ner_stage, "_reply_repairs", lambda data, log: None)
    without = [await _ner(_ner_dirty(m)) for m in marks]
    moved = 0
    for mark, a, b in zip(marks, with_marks, without):
        assert _decided(a) == _decided(b), mark
        moved += a.repairs != b.repairs
    assert moved > 50, "the control: the switch did take the marks away"


async def test_noumeno_switching_the_marking_off_changes_nothing_but_the_list(monkeypatch):
    marks = [m for m in sorted(rp.VALID_NOUMENO_REPAIRS) if m != rp.TRUNCATED_RETRY]
    with_marks = [await _noumeno(_NOUMENO_DIRTY[m]) for m in marks]
    monkeypatch.setattr(noumeno_stage, "_reply_repairs", lambda data, log: None)
    without = [await _noumeno(_NOUMENO_DIRTY[m]) for m in marks]
    moved = 0
    for mark, a, b in zip(marks, with_marks, without):
        assert _decided(a) == _decided(b), mark
        moved += a.repairs != b.repairs
    assert moved > 10, "the control: the switch did take the marks away"


async def test_a_marking_function_that_blows_up_costs_no_turn(monkeypatch, caplog):
    """A count must not cost a turn. The marking is fenced; what it raises is a WARNING."""
    def boom(*_a, **_k):
        raise RuntimeError("marking is broken")
    monkeypatch.setattr(ner_stage, "spelled_boolean", boom)
    monkeypatch.setattr(noumeno_stage, "spelled_boolean", boom)
    with caplog.at_level("WARNING"):
        intent = await _ner(dict(_INTENT, is_composite="true"))
        rewrite = await _noumeno(dict(_REWRITE, changed="true"))
    assert intent.is_composite is True and rewrite.rewritten == "Send the invoice."
    assert caplog.text.count("event=repair_marks_failed") == 2


# ── the truncation retry ────────────────────────────────────────────────────────────────

async def test_the_retry_is_marked_and_the_helper_returns_what_it_always_returned():
    def parse(raw: str) -> dict:
        return json.loads(raw)
    plain = await generate_json_resilient(
        ScriptedBackend([_CUT_REPLY, json.dumps(_REWRITE)]), "s", "p", parse, stage="noumeno")
    log = rp.RepairLog("noumeno")
    marked = await generate_json_resilient(
        ScriptedBackend([_CUT_REPLY, json.dumps(_REWRITE)]), "s", "p", parse, stage="noumeno",
        repairs=log)
    assert plain == marked and plain[0] == _REWRITE
    assert log.marks() == [rp.TRUNCATED_RETRY]


async def test_no_retry_no_mark_and_a_malformed_reply_still_raises_on_the_first_attempt():
    def parse(raw: str) -> dict:
        return json.loads(raw)
    log = rp.RepairLog("noumeno")
    await generate_json_resilient(ScriptedBackend([json.dumps(_REWRITE)]), "s", "p", parse,
                                  stage="noumeno", repairs=log)
    assert log.marks() == []
    backend = ScriptedBackend(["I would rather answer in prose.", json.dumps(_REWRITE)])
    with pytest.raises(ValueError):
        await generate_json_resilient(backend, "s", "p", parse, stage="noumeno", repairs=log)
    assert len(backend.calls) == 1 and log.marks() == []


# ── the scope guard ─────────────────────────────────────────────────────────────────────

async def _guard(reply: str, backend=None):
    ctx = _ctx(user=_USER, intent_class="INFORMATION_REQUEST")
    backend = backend or ScriptedBackend([reply])
    result = await SuperegoStage().check_input_scope(ctx, backend, scope_prompt=_SCOPE)
    return result, ctx


_ALLOW = '{"blocked": false, "refusal_message": ""}'
_BLOCK = '{"blocked": true, "refusal_message": "Not something I can help with."}'

_GUARD_DIRTY = {
    rp.FENCE_OR_THINK_STRIPPED: _fence(_ALLOW),
    rp.SALVAGED_OBJECT: "Verdict: " + _ALLOW,
    "default:refusal_message": '{"blocked": true}',
    "null_field:refusal_message": '{"blocked": true, "refusal_message": null}',
    "coerced:refusal_message": '{"blocked": true, "refusal_message": ["no"]}',
}


def test_the_guard_table_covers_its_alphabet_exactly():
    assert set(_GUARD_DIRTY) == rp.VALID_SCOPE_REPAIRS


@pytest.mark.parametrize("reply", [_ALLOW, _BLOCK, '{"blocked": false}'])
async def test_guard_a_clean_reply_records_nothing_and_says_so(reply):
    """``{"blocked": false}`` with no message is clean: the message is read only on a block."""
    result, ctx = await _guard(reply)
    assert result.verdict_read == VERDICT_BOOLEAN
    assert result.repairs == [] and ctx.metadata[mk.SCOPE_REPAIRS] == []


@pytest.mark.parametrize("mark", sorted(rp.VALID_SCOPE_REPAIRS))
async def test_guard_every_mark_has_a_reply_that_produces_it_alone(mark):
    result, ctx = await _guard(_GUARD_DIRTY[mark])
    assert result.repairs == [mark]
    assert ctx.metadata[mk.SCOPE_REPAIRS] == [mark]
    assert result.verdict_read == VERDICT_BOOLEAN, "the verdict was read; the reply was repaired"


async def test_guard_a_think_block_is_the_same_mark_as_a_fence():
    result, _ = await _guard(_think(_ALLOW))
    assert result.repairs == [rp.FENCE_OR_THINK_STRIPPED]


async def test_guard_an_empty_refusal_on_a_block_is_the_default():
    result, _ = await _guard('{"blocked": true, "refusal_message": "  "}')
    assert result.blocked is True and result.repairs == ["default:refusal_message"]


async def test_guard_an_unread_verdict_is_verdict_reads_to_say_and_only_its():
    """A string ``"false"`` and a reply with no JSON: ``verdict_read`` names each, and the
    repairs list repeats neither. What came OFF the reply is still said."""
    spelled, _ = await _guard('{"blocked": "false"}')
    assert spelled.verdict_read == VERDICT_STRING_BOOL and spelled.repairs == []
    prose, _ = await _guard("I cannot decide.")
    assert prose.verdict_read == VERDICT_UNPARSEABLE and prose.repairs == []
    thought, ctx = await _guard(_think("I cannot decide."))
    assert thought.verdict_read == VERDICT_UNPARSEABLE
    assert thought.repairs == ctx.metadata[mk.SCOPE_REPAIRS] == [rp.FENCE_OR_THINK_STRIPPED]


async def test_guard_a_call_that_raised_has_no_reply_to_repair():
    result, ctx = await _guard("", backend=RaisingBackend())
    assert result.verdict_read == VERDICT_CALL_FAILED
    assert result.repairs == [] and ctx.metadata[mk.SCOPE_REPAIRS] == []


async def test_guard_a_bypass_removes_the_key_a_carrier_left():
    """Absent means NOTHING WAS ASKED — so a bypassed turn must not wear an earlier turn's
    list. The rule ``SCOPE_VERDICT_READ`` follows, for the same reader."""
    ctx = _ctx(user="oi", intent_class="SOCIAL")
    ctx.metadata[mk.SCOPE_REPAIRS] = [rp.SALVAGED_OBJECT]
    backend = ScriptedBackend([_ALLOW])
    result = await SuperegoStage().check_input_scope(ctx, backend, scope_prompt=_SCOPE)
    assert backend.calls == [], "the premise: the classifier was not asked"
    assert mk.SCOPE_REPAIRS not in ctx.metadata and result.repairs == []
    assert mk.SCOPE_VERDICT_READ not in ctx.metadata


async def test_guard_the_metadata_list_is_not_the_results_own():
    result, ctx = await _guard(_fence(_ALLOW))
    ctx.metadata[mk.SCOPE_REPAIRS].append("mine")
    assert result.repairs == [rp.FENCE_OR_THINK_STRIPPED]


# ── the judge ───────────────────────────────────────────────────────────────────────────

async def _judge(reply: str, backend=None):
    backend = backend or ScriptedBackend([reply])
    return await SuperegoStage().evaluate(_ctx(), backend, limits_prompt="")


_APPROVE = '{"approved": true}'
_REJECT = '{"approved": false, "critique": "booked the wrong day"}'

_JUDGE_DIRTY = {
    rp.FENCE_OR_THINK_STRIPPED: _think(_APPROVE),
    rp.SALVAGED_OBJECT: _APPROVE + "\nThe execution matches the request.",
    "default:critique": '{"approved": false}',
    "null_field:critique": '{"approved": false, "critique": null}',
    "coerced:critique": '{"approved": false, "critique": {"why": "wrong day"}}',
}


def test_the_judge_table_covers_its_alphabet_exactly():
    assert set(_JUDGE_DIRTY) == rp.VALID_JUDGE_REPAIRS


@pytest.mark.parametrize("reply", [_APPROVE, _REJECT, '{"approved": true, "critique": null}'])
async def test_judge_a_clean_reply_records_nothing(reply):
    """An approval with a ``null`` critique is clean: the critique is read only on a rejection."""
    result = await _judge(reply)
    assert result.verdict_read == VERDICT_BOOLEAN and result.repairs == []


@pytest.mark.parametrize("mark", sorted(rp.VALID_JUDGE_REPAIRS))
async def test_judge_every_mark_has_a_reply_that_produces_it_alone(mark):
    result = await _judge(_JUDGE_DIRTY[mark])
    assert result.repairs == [mark]
    assert result.verdict_read == VERDICT_BOOLEAN


async def test_judge_an_unread_verdict_is_verdict_reads_to_say_and_only_its():
    spelled = await _judge('{"approved": "false", "critique": "x"}')
    assert spelled.verdict_read == VERDICT_STRING_BOOL and spelled.repairs == []
    assert spelled.critique == UNREADABLE_VERDICT_CRITIQUE
    fenced = await _judge(_fence('{"critique": "x"}'))
    assert fenced.approved is False and fenced.repairs == [rp.FENCE_OR_THINK_STRIPPED]


async def test_judge_a_call_that_raised_and_a_turn_with_nothing_to_judge_record_nothing():
    failed = await _judge("", backend=RaisingBackend())
    assert failed.verdict_read == VERDICT_CALL_FAILED and failed.repairs == []
    nothing = await SuperegoStage().evaluate(_ctx(with_ego=False), ScriptedBackend([_APPROVE]),
                                            limits_prompt="")
    assert nothing.verdict_read == "" and nothing.repairs == []


async def test_the_voice_carries_no_repairs():
    result = await SuperegoStage().voice(_ctx(), ScriptedBackend(['{"message": "Done."}']),
                                         voice_prompt="Be brief.")
    assert result.response == "Done." and result.repairs == []
    assert "voice:json_unwrapped" in result.adjustments, "the voice's own net keeps its count"


# ── the selector ────────────────────────────────────────────────────────────────────────

_OPTIONS = ["Enrolment", "Timetable"]
_PICK = {"answers": [], "maybe": ["Timetable"], "asked": ""}

_SELECT_DIRTY = {
    rp.FENCE_OR_THINK_STRIPPED: _fence(json.dumps(_PICK)),
    rp.SALVAGED_OBJECT: "My pick: " + json.dumps(_PICK),
    rp.UNPARSEABLE: "I could not decide.",
    "default:answers": _without(_PICK, "answers"),
    "default:maybe": _without(_PICK, "maybe"),
    "default:asked": _without(_PICK, "asked"),
    "null_field:answers": dict(_PICK, answers=None),
    "null_field:maybe": dict(_PICK, maybe=None),
    "null_field:asked": dict(_PICK, asked=None),
    "coerced:answers": dict(_PICK, answers="Timetable"),
    "coerced:maybe": dict(_PICK, maybe={"Timetable": 1}),
    "coerced:asked": dict(_PICK, asked=["gym"]),
}


async def _select(reply: object, backend=None):
    backend = backend or ScriptedBackend(_script(reply))
    return await scope_options.select_options("is there a timetable?", _OPTIONS, backend)


def test_the_selector_table_covers_its_alphabet_exactly():
    assert set(_SELECT_DIRTY) == rp.VALID_SELECT_REPAIRS


async def test_selector_a_clean_reply_records_nothing():
    got = await _select(_PICK)
    assert got.outcome == scope_options.OUTCOME_SUGGESTED
    assert got.repairs == () and got.record()["repairs"] == []


@pytest.mark.parametrize("mark", sorted(rp.VALID_SELECT_REPAIRS))
async def test_selector_every_mark_has_a_reply_that_produces_it_alone(mark):
    got = await _select(_SELECT_DIRTY[mark])
    assert got.repairs == (mark,)
    assert got.record()["repairs"] == [mark]


async def test_selector_the_three_errors_are_three_records():
    """``error`` used to be one outcome for three facts. The marks tell them apart."""
    prose = await _select("I could not decide.")
    wrong = await _select({"picked": ["Timetable"]})
    not_lists = await _select({"answers": "Timetable", "maybe": []})
    failed = await _select("", backend=RaisingBackend())
    assert {s.outcome for s in (prose, wrong, not_lists, failed)} == {
        scope_options.OUTCOME_ERROR}
    assert prose.repairs == (rp.UNPARSEABLE,)
    assert wrong.repairs == ("default:answers", "default:maybe", "default:asked")
    assert not_lists.repairs == ("coerced:answers", "default:asked")
    assert failed.repairs == ()


async def test_selector_an_option_off_the_list_is_discarded_not_marked():
    got = await _select(dict(_PICK, maybe=["Timetable", "Invented"]))
    assert got.discarded == 1 and got.repairs == ()


async def test_selector_no_call_no_marks():
    got = await scope_options.select_options("", _OPTIONS, ScriptedBackend([_fence("{}")]))
    assert got.called is False and got.repairs == ()


async def test_selector_the_stamp_carries_the_list():
    ctx = PipelineContext(user_input="is there a timetable?")
    await scope_options.select_scope_options(
        ctx, ScriptedBackend([_fence(json.dumps(_PICK))]), options=_OPTIONS)
    assert ctx.metadata[mk.SCOPE_OPTIONS_SELECTION]["repairs"] == [rp.FENCE_OR_THINK_STRIPPED]


# ── never a byte of the reply ───────────────────────────────────────────────────────────

def _poisoned(obj: object) -> object:
    """``obj`` with the canary in every string value and one invented canary key."""
    if isinstance(obj, dict):
        out = {k: _poisoned(v) for k, v in obj.items()}
        out[_CANARY + "_key"] = _CANARY
        return out
    if isinstance(obj, list):
        return [_poisoned(v) for v in obj] + [_CANARY]
    if isinstance(obj, str):
        return f"{obj} {_CANARY}"
    return obj


async def test_no_mark_ever_carries_what_the_model_wrote():
    intent = await _ner("Thinking " + _CANARY + " " + json.dumps(_poisoned(_INTENT)))
    rewrite = await _noumeno(_CANARY + json.dumps(_poisoned(dict(_REWRITE, changed="no"))))
    guard, ctx = await _guard(_CANARY + ' {"blocked": true, "refusal_message": 7, "'
                              + _CANARY + '": "' + _CANARY + '"}')
    judge = await _judge('{"approved": false, "critique": ["' + _CANARY + '"]} ' + _CANARY)
    pick = await _select(_CANARY + " " + json.dumps(_poisoned(dict(_PICK, asked=3))))
    lists = [intent.repairs, rewrite.repairs, guard.repairs, ctx.metadata[mk.SCOPE_REPAIRS],
             judge.repairs, list(pick.repairs)]
    assert all(lists), "the premise: every stage repaired something"
    for marks in lists:
        assert set(marks) <= rp.VALID_REPAIRS
        assert _CANARY not in " ".join(marks)


# ── arbitrary replies ───────────────────────────────────────────────────────────────────

_JSON = st.recursive(
    st.none() | st.booleans() | st.integers(-5, 5) | st.floats(allow_nan=False, width=16)
    | st.text(max_size=6) | st.sampled_from(["true", "ACTION_REQUEST", "FINANCE", "SYSTEM"]),
    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=4), inner,
                                                                max_size=3),
    max_leaves=6)


@settings(max_examples=300, deadline=None)
@given(data=st.dictionaries(st.sampled_from(sorted(rp.NER_REPAIR_FIELDS[rp.DEFAULT])), _JSON))
def test_ner_marking_never_raises_and_never_leaves_the_alphabet(data):
    log = rp.RepairLog("ner")
    ner_stage._reply_repairs(data, log)
    assert set(log.marks()) <= rp.VALID_NER_REPAIRS
    missing = set(rp.NER_REPAIR_FIELDS[rp.DEFAULT]) - set(data)
    assert {f"default:{name}" for name in missing} <= set(log.marks())


@settings(max_examples=300, deadline=None)
@given(data=st.dictionaries(st.sampled_from(sorted(rp.NOUMENO_REPAIR_FIELDS[rp.DEFAULT])),
                            _JSON))
def test_noumeno_marking_never_raises_and_never_leaves_the_alphabet(data):
    log = rp.RepairLog("noumeno")
    noumeno_stage._reply_repairs(data, log)
    assert set(log.marks()) <= rp.VALID_NOUMENO_REPAIRS


@settings(max_examples=300, deadline=None)
@given(text=st.text(max_size=60), stripped=st.booleans())
def test_reply_form_repairs_never_raises_on_arbitrary_text(text, stripped):
    assert set(rp.reply_form_repairs(text, cot_stripped=stripped)) <= {
        rp.FENCE_OR_THINK_STRIPPED, rp.SALVAGED_OBJECT}


@settings(max_examples=200, deadline=None)
@given(data=st.dictionaries(st.sampled_from(sorted(rp.NER_REPAIR_FIELDS[rp.DEFAULT])), _JSON))
def test_ner_build_ends_the_same_way_with_the_marking_on_or_off(data):
    """Whatever ``_build`` does with an arbitrary object — a result, or the exception it has
    always raised — it does the same with the marking replaced by nothing."""
    stage = IntentAnalyzer()
    metrics = StageMetrics(stage="ner", elapsed_ms=0.0, tokens_in=0, tokens_out=0, model="t")

    def run() -> object:
        try:
            return _decided(stage._build(json.loads(json.dumps(data)), "", metrics))
        except Exception as exc:  # noqa: BLE001 — the type is the outcome being compared
            return type(exc).__name__

    marked = run()
    original = ner_stage._reply_repairs
    ner_stage._reply_repairs = lambda d, log: None
    try:
        assert run() == marked
    finally:
        ner_stage._reply_repairs = original
