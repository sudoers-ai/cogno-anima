"""The schema of each stage's reply is DERIVED from the vocabularies — and has ONE reader, behind a
switch that is off by default.

``cogno_anima.schemas`` holds one JSON Schema per JSON stage (NOUMENO, NER, scope guard, judge).
This file holds it to three things:

* **derived, in both directions.** Every closed vocabulary a stage validates against is the
  ``enum`` of its field, and every ``enum`` in a schema is one of those vocabularies — read from
  the objects, so a word added to a vocabulary shows up in the schema and an enumeration typed
  by hand has nowhere to hide. The properties of each object are the fields the stage reads,
  from the stage's own table in ``cogno_anima.repairs``.
* **the bridge to the repair marks.** A reply the schema ACCEPTS needs no form repair
  (``default`` / ``null_field`` / ``string_bool`` / ``coerced``) and never ends the turn — over
  every field under 38 dirty values, through the real stages. The values it accepts and the
  stage still marks are the declared ones: the right type, the wrong CONTENT. For the NOUMENO
  and the NER the converse holds as well: what the stage reads without a form mark is accepted.
* **one reader, off by default.** Exactly one module of the package imports it —
  ``cogno_anima.structured``, inside the function that runs only when the host switched a stage
  on — and the package root does not export it. With the switch off every stage's call is
  ``generate(system, prompt)``; the switch itself is held in ``test_stage_schema_switch.py``.
* **two forms.** ``schema_for(stage, "open")`` is the derived schema byte for byte (digests
  pinned); ``"strict"`` adds the keys the stage's PROMPT asks for and the reader ignores, from a
  table tied to the prompt's own examples in both directions. Both forms convert to every dialect
  the transport has, and no enumeration is copied on the way.

No model is involved anywhere: the schema is data, and the stages run over scripted replies.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest
from cogno_synapse import SCHEMA_DIALECTS, SCHEMA_FORMS, schema_digest, to_dialect

import cogno_anima
from cogno_anima import repairs as rp
from cogno_anima import schemas
from cogno_anima import vocab
from cogno_anima.security import pii
from cogno_anima.stages import ner as ner_stage
from cogno_anima.stages.ner import IntentAnalyzer
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.types import StageMetrics
from cogno_anima.verdict import VERDICT_BOOLEAN
from tests.unit.test_repairs_are_counted import (_ALLOW, _APPROVE, _BLOCK, _GUARD_DIRTY,
                                                 _INTENT, _JUDGE_DIRTY, _NOUMENO_DIRTY, _REJECT,
                                                 _REWRITE, _guard, _judge, _ner_dirty, _noumeno,
                                                 _without)

FORM = tuple(f + ":" for f in rp.FORM_FAMILIES)
PACKAGE = Path(cogno_anima.__file__).parent


# ── a validator for the subset of JSON Schema this module uses ──────────────────────────────

_TYPES = {
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    # JSON has no "bool is a number": `true` is refused where a number is asked.
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}
_KEYWORDS = {"type", "enum", "items", "properties", "required", "additionalProperties"}


def accepts(schema: dict, value: object) -> bool:
    assert set(schema) <= _KEYWORDS, f"a keyword this validator does not know: {set(schema)}"
    kinds = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
    if not any(_TYPES[k](value) for k in kinds):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if isinstance(value, list) and "items" in schema:
        return all(accepts(schema["items"], item) for item in value)
    if isinstance(value, dict):
        props = schema.get("properties", {})
        if any(key not in value for key in schema.get("required", ())):
            return False
        if schema.get("additionalProperties") is False and set(value) - set(props):
            return False
        return all(accepts(props[key], item) for key, item in value.items() if key in props)
    return True


def _nodes(schema: dict):
    yield schema
    for child in schema.get("properties", {}).values():
        yield from _nodes(child)
    if "items" in schema:
        yield from _nodes(schema["items"])


def test_the_validator_itself_refuses_what_it_should():
    """The control of everything below: a validator that accepts everything proves nothing."""
    s = {"type": "object", "required": ["a"], "additionalProperties": False,
         "properties": {"a": {"type": ["string", "null"], "enum": ["x", None]},
                        "b": {"type": "array", "items": {"type": "number"}}}}
    assert accepts(s, {"a": "x"}) and accepts(s, {"a": None, "b": [1, 2.5]})
    for bad in ({}, {"a": "y"}, {"a": "x", "c": 1}, {"a": "x", "b": [True]}, {"a": "x", "b": 3},
                ["a"], None):
        assert not accepts(s, bad), bad


# ── the stages that have a schema, and their fields ─────────────────────────────────────────

def test_one_schema_per_json_stage_and_none_for_the_selector():
    assert set(schemas.STAGE_SCHEMAS) == set(rp.REPAIRS_BY_STAGE) - {"superego_select"}
    for form in SCHEMA_FORMS:
        assert schemas.schema_for("superego_select", form) is None
        for stage in ("voice", "ego", "", None, 7):
            assert schemas.schema_for(stage, form) is None
    for form in ("", "OPEN", "Strict", "closed", None, True, 1, ["open"]):
        assert schemas.schema_for("ner", form) is None, form
    assert schemas.schema_for("ner", "open") is not None                  # control


@pytest.mark.parametrize("stage,table,verdict_key", [
    ("noumeno", rp.NOUMENO_REPAIR_FIELDS, None),
    ("ner", rp.NER_REPAIR_FIELDS, None),
    ("superego_scope", rp.SCOPE_REPAIR_FIELDS, schemas.SCOPE_VERDICT_KEY),
    ("superego_judge", rp.JUDGE_REPAIR_FIELDS, schemas.JUDGE_VERDICT_KEY),
])
def test_the_properties_are_the_fields_the_stage_reads(stage, table, verdict_key):
    """The stage's own table says which keys it reads; the schema has exactly those, all
    required. (That the table IS what the stage reads is pinned in the repairs suite, from the
    stage's source.)"""
    schema = schemas.STAGE_SCHEMAS[stage]
    expected = set(table[rp.DEFAULT]) | ({verdict_key} if verdict_key else set())
    assert set(schema["properties"]) == expected
    assert schema["required"] == sorted(expected)
    assert schema["type"] == "object"


def test_the_verdict_keys_are_the_ones_the_stages_ask_the_reader_for():
    """Read from the source: the key each verdict stage hands to ``read_verdict``."""
    import inspect
    scope = inspect.getsource(SuperegoStage.check_input_scope)
    judge = inspect.getsource(SuperegoStage.evaluate)
    assert f'read_verdict(raw, "{schemas.SCOPE_VERDICT_KEY}")' in scope
    assert f'read_verdict(raw, "{schemas.JUDGE_VERDICT_KEY}")' in judge


# ── derived: vocabulary → schema ────────────────────────────────────────────────────────────

def _ner_prop(name: str) -> dict:
    return schemas.STAGE_SCHEMAS["ner"]["properties"][name]


@pytest.mark.parametrize("name", sorted(ner_stage._ENUM_FIELDS))
def test_every_closed_vocabulary_of_the_ner_is_the_enum_of_its_field(name):
    vocabulary, nullable = ner_stage._ENUM_FIELDS[name]
    prop = _ner_prop(name)
    assert [v for v in prop["enum"] if v is not None] == sorted(vocabulary)
    assert (None in prop["enum"]) is nullable
    assert prop["type"] == (["string", "null"] if nullable else "string")


@pytest.mark.parametrize("name", sorted(schemas.NER_LIST_VOCABULARIES))
def test_every_closed_vocabulary_of_a_list_is_the_enum_of_its_items(name):
    prop = _ner_prop(name)
    assert prop["type"] == "array"
    assert prop["items"]["enum"] == sorted(schemas.NER_LIST_VOCABULARIES[name])


def test_the_aristotelian_keys_are_the_closed_set_and_the_object_is_closed():
    prop = _ner_prop("aristotelian")
    assert set(prop["properties"]) == vocab.VALID_ARISTOTELIAN
    assert prop["additionalProperties"] is False and "required" not in prop


def test_the_nullable_fields_are_the_ones_the_marks_treat_as_nullable():
    """One rule, two modules: a field is a union with ``null`` here exactly when a ``null``
    there is no ``null_field`` mark (the prompt allows it)."""
    props = schemas.STAGE_SCHEMAS["ner"]["properties"]
    nullable = {name for name, p in props.items()
                if isinstance(p["type"], list) and "null" in p["type"]}
    read = set(rp.NER_REPAIR_FIELDS[rp.DEFAULT])
    never_repaired = {"mandatory_tags", "pii"}        # a `null` there raises: see the repairs table
    assert nullable == read - set(rp.NER_REPAIR_FIELDS[rp.NULL_FIELD]) - never_repaired
    assert nullable == {"location", "goal", "parole", "modality", "speech_act"}


@pytest.mark.parametrize("target,word", [
    (vocab.VALID_INTENTS, "ZZ_NEW_INTENT"), (vocab.NER_KNOWLEDGE_DOMAINS, "ZZ_NEW_DOMAIN"),
    (pii.VALID_PII_TYPES, "ZZ_NEW_PII"), (vocab.VALID_PAROLE, "ZZ_NEW_PAROLE"),
    (vocab.VALID_ARISTOTELIAN, "ZZ_NEW_CATEGORY"),
])
def test_a_word_added_to_a_vocabulary_is_in_the_schema_built_next(target, word):
    """Derived, not copied: the builder reads the vocabulary OBJECT. The word is put in, the
    schemas are built again, and the word is there; taken out, it is gone."""
    def words() -> str:
        return json.dumps(schemas.build_stage_schemas())
    assert word not in words(), "the premise"
    target.add(word)
    try:
        assert word in words()
    finally:
        target.discard(word)
    assert word not in words()
    assert schemas.build_stage_schemas() == schemas.STAGE_SCHEMAS


# ── derived: schema → vocabulary ────────────────────────────────────────────────────────────

def _vocabularies() -> "list[frozenset]":
    """Every closed set of strings the package declares as a vocabulary."""
    found = [frozenset(v) for name, v in vars(vocab).items()
             if isinstance(v, (set, frozenset)) and name.isupper() and v
             and all(isinstance(x, str) for x in v)]
    found.append(frozenset(pii.VALID_PII_TYPES))
    return found


def test_every_enum_in_every_schema_is_a_vocabulary_of_the_package():
    """The other direction: no enumeration was typed here. Each one, ``null`` aside, is equal
    to a set the package already declares."""
    known = _vocabularies()
    assert len(known) >= 10, "the premise: the vocabularies were found"
    enums = [frozenset(v for v in node["enum"] if v is not None)
             for schema in schemas.STAGE_SCHEMAS.values() for node in _nodes(schema)
             if "enum" in node]
    assert len(enums) == len(ner_stage._ENUM_FIELDS) + len(schemas.NER_LIST_VOCABULARIES)
    for enum in enums:
        assert enum in known, sorted(enum)


def test_an_enum_is_sorted_so_every_worker_builds_the_same_bytes():
    for schema in schemas.STAGE_SCHEMAS.values():
        for node in _nodes(schema):
            if "enum" in node:
                words = [v for v in node["enum"] if v is not None]
                assert words == sorted(words)
    assert json.dumps(schemas.build_stage_schemas(), sort_keys=True) == json.dumps(
        schemas.STAGE_SCHEMAS, sort_keys=True)


def test_no_keyword_outside_the_plain_subset():
    """No range, length or uniqueness keyword: which of those a provider's strict mode takes is
    the transport's question, and the marks keep counting what they would have refused."""
    for schema in schemas.STAGE_SCHEMAS.values():
        for node in _nodes(schema):
            assert set(node) <= _KEYWORDS, set(node) - _KEYWORDS


@pytest.mark.parametrize("form", SCHEMA_FORMS)
def test_schema_for_hands_out_a_copy(form):
    mine = schemas.schema_for("ner", form)
    mine["properties"]["intent_class"]["enum"].append("MINE")
    mine["properties"]["entities"]["properties"]["people"]["items"]["type"] = "number"
    mine["required"].clear()
    assert schemas.schema_for("ner", "open") == schemas.STAGE_SCHEMAS["ner"]
    assert "MINE" not in json.dumps(schemas.STAGE_SCHEMAS)
    assert "MINE" not in json.dumps(schemas.schema_for("ner", form))


# ── the two forms ───────────────────────────────────────────────────────────────────────────

#: The derived schemas as they were before any form existed (``cogno-anima`` 0.1.10), named two
#: ways: the canonical digest a log line carries, and a digest of the object in the order it is
#: built — the "byte for byte" of the open form. A vocabulary that gains a word moves both, on
#: purpose; an edit to this module that moves them by accident is what this catches.
_OPEN_DIGESTS = {
    "noumeno": ("3b646bea34b4", "8e05543ef3c11d63"),
    "ner": ("04108d4bcbc9", "8a23cf917797b76a"),
    "superego_scope": ("7669feae8814", "3bf32b4cd6da7d74"),
    "superego_judge": ("230d8b4e0dbd", "fc4a1b9a761e2f57"),
}


def _built(schema: dict) -> str:
    return hashlib.sha256(json.dumps(schema).encode()).hexdigest()[:16]


def test_the_open_form_is_the_derived_schema_byte_for_byte():
    assert set(_OPEN_DIGESTS) == set(schemas.STAGE_SCHEMAS)
    for stage, (canonical, built) in _OPEN_DIGESTS.items():
        schema = schemas.schema_for(stage, "open")
        assert schema == schemas.STAGE_SCHEMAS[stage]
        assert (schema_digest(schema), _built(schema)) == (canonical, built), stage
    # the control: the digest does move when a schema does
    moved = schemas.schema_for("ner", "open")
    moved["properties"]["goal"]["type"] = "string"
    assert schema_digest(moved) != _OPEN_DIGESTS["ner"][0]


def test_the_strict_form_adds_only_permitted_keys_and_only_to_the_ner():
    for stage in schemas.STAGE_SCHEMAS:
        opened, strict = schemas.schema_for(stage, "open"), schemas.schema_for(stage, "strict")
        if stage != "ner":
            assert strict == opened, stage
            continue
        added = set(strict["properties"]) - set(opened["properties"])
        assert added == set(schemas.NER_PROMPT_ONLY_KEYS) and len(added) == 6
        added_entities = (set(strict["properties"]["entities"]["properties"])
                          - set(opened["properties"]["entities"]["properties"]))
        assert added_entities == set(schemas.NER_PROMPT_ONLY_ENTITY_KEYS) and len(added_entities) == 2
        # PERMITTED, never required: nothing the stage reads became optional or mandatory here
        assert strict["required"] == opened["required"]
        assert "required" not in strict["properties"]["entities"]
        # …and everything that was there is there, unchanged
        for name, prop in opened["properties"].items():
            if name != "entities":
                assert strict["properties"][name] == prop, name
        for name, prop in opened["properties"]["entities"]["properties"].items():
            assert strict["properties"]["entities"]["properties"][name] == prop, name
        # this module closes nothing: that is the provider's rule and the transport's job
        assert "additionalProperties" not in strict
        assert "additionalProperties" not in strict["properties"]["entities"]


def _prompt_examples() -> "list[dict]":
    """Every example object in the NER's system prompt — what the model is SHOWN to write."""
    text = (PACKAGE / "prompt_templates" / "ner" / "system.txt").read_text(encoding="utf-8")
    decoder, found, at = json.JSONDecoder(), [], 0
    while True:
        at = text.find("Output:\n{", at)
        if at < 0:
            return found
        obj, end = decoder.raw_decode(text[at + len("Output:\n"):])
        found.append(obj)
        at += len("Output:\n") + end


def test_the_prompt_only_keys_are_the_ones_the_prompt_asks_for_and_the_reader_ignores():
    """Tied to the prompt's own text in BOTH directions: a key of the table that leaves the
    prompt fails, and so does a key the prompt starts asking for that the table does not hold
    (under a closed schema the model could no longer write it)."""
    examples = _prompt_examples()
    assert len(examples) >= 3, "the examples were not found — the test would agree with anything"
    read = set(rp.NER_REPAIR_FIELDS[rp.DEFAULT])
    read_entities = set(schemas._ENTITY_KEYS)
    for example in examples:
        assert set(example) - read == set(schemas.NER_PROMPT_ONLY_KEYS)
        assert set(example["entities"]) - read_entities == set(schemas.NER_PROMPT_ONLY_ENTITY_KEYS)
        # the declared type is the one the prompt shows
        for name, kind in {**schemas.NER_PROMPT_ONLY_KEYS,
                           **{f"entities.{k}": v
                              for k, v in schemas.NER_PROMPT_ONLY_ENTITY_KEYS.items()}}.items():
            value = example["entities"][name.split(".")[1]] if "." in name else example[name]
            assert isinstance(value, list if kind == "array" else str), name
    # no key is in both tables: a key the reader reads is not "prompt only"
    assert not set(schemas.NER_PROMPT_ONLY_KEYS) & read
    assert not set(schemas.NER_PROMPT_ONLY_ENTITY_KEYS) & read_entities
    # `langue` is deliberately out: the prompt tells the model to omit it and no example has it
    assert "langue" not in schemas.NER_PROMPT_ONLY_KEYS
    assert not any("langue" in example for example in examples)


def test_the_strict_form_accepts_what_the_prompt_shows():
    """The examples the prompt itself gives are replies the converted strict schema must be able
    to take once every key is written — or the prompt and the schema ask for different things."""
    strict = to_dialect(schemas.schema_for("ner", "strict"), "openai_strict")

    def filled(value, schema):
        if isinstance(value, dict) and "properties" in schema:
            return {key: filled(value.get(key), sub) for key, sub in schema["properties"].items()}
        return value

    examples = _prompt_examples()
    for example in examples:
        assert accepts(strict, filled(example, strict))
        # the control: as written (keys left out) the closed, all-required schema refuses it
        assert not accepts(strict, example)
    # and the OPEN form, sent open, takes the examples as written
    opened = schemas.schema_for("ner", "open")
    assert all(accepts(opened, example) for example in examples)


@pytest.mark.parametrize("stage", sorted(schemas.STAGE_SCHEMAS))
@pytest.mark.parametrize("form", SCHEMA_FORMS)
@pytest.mark.parametrize("dialect", sorted(SCHEMA_DIALECTS))
def test_every_form_converts_to_every_dialect_and_no_enumeration_is_copied(stage, form, dialect):
    """The transport converts what this module hands over, without a ``ValueError``, and every
    ``enum`` that comes out is a vocabulary of the package — plus ``null`` only where the strict
    dialect widened a key. Equality by enumeration, against the objects, not against a list."""
    converted = to_dialect(schemas.schema_for(stage, form), dialect)
    vocabularies = [sorted(v) for v in (
        *(vocabulary for vocabulary, _ in ner_stage._ENUM_FIELDS.values()),
        *schemas.NER_LIST_VOCABULARIES.values())]
    seen = 0
    for node in _nodes(converted):
        assert set(node) <= _KEYWORDS, set(node) - _KEYWORDS
        if "enum" in node:
            seen += 1
            words = [w for w in node["enum"] if w is not None]
            assert words in vocabularies, words
    assert seen == (10 if stage == "ner" else 0)
    if dialect == "openai_strict":
        for node in _nodes(converted):
            if node.get("type") == "object" or "properties" in node:
                assert node["additionalProperties"] is False
                assert node["required"] == list(node["properties"])
    else:
        assert converted == schemas.schema_for(stage, form)


# ── the bridge: what the schema accepts needs no form repair ────────────────────────────────

#: Every shape a model's value can take that the stages have a branch for — the dirty values
#: of the repairs suite's tables and of the cross-tree sweep, in one list.
DIRTY = [None, "", "  ", "true", "false", "yes", "x" * 90, 0, 1, 7.5, -1, 0.5, True, False,
         [], ["a"], ["a"] * 7, [None, 3, "ok", ""], [""], ["ok", "ok"], {}, {"k": "v"},
         "action_request", "ACTION_REQUEST", "MEDICINE", "FINANCE", ["FINANCE", "FINANCE"],
         ["NER.SYSTEM", "MATH", "ANALYSIS", "CREATIVE"], ["SYSTEM"],
         {"ACTION": "X | " + "d" * 60, "MOOD": 3}, {"ACTION": "  "}, {"TIME": "NOW | soon"},
         ["CPF", "EMAIL", 4], ["EMAIL"], "0.9", {"people": "x", "objects": [1, None, "a"]},
         {"people": ["Ana"], "objects": [""]}, {"people": [], "objects": [3]}]


def _variants(clean: dict):
    yield "clean", clean
    for key in clean:
        yield f"missing:{key}", _without(clean, key)
        for n, value in enumerate(DIRTY):
            yield f"{key}:{n}", dict(clean, **{key: value})


def _form(marks) -> "list[str]":
    return [m for m in marks if m.startswith(FORM)]


def _blank(value: object) -> bool:
    return isinstance(value, str) and not value.strip()


def _content_shaped(stage: str, mark: str, obj: dict) -> bool:
    """Is ``mark`` one of the DECLARED repairs of a value of the right type and the wrong
    content — the ones a schema of types and enumerations cannot refuse?"""
    family, name = mark.split(":")
    value = obj.get(name)
    if stage == "noumeno":
        return mark == "default:rewritten" and _blank(value)
    if family != rp.COERCED:
        return False
    if name == "confidence":                                  # a number outside [0, 1]
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "domains":                                     # a domain listed twice
        return isinstance(value, list) and len(set(value)) != len(value)
    if name in ner_stage._PHRASE_LISTS:                       # a blank phrase in a list
        return isinstance(value, list) and any(_blank(i) for i in value)
    if name == "entities":
        return isinstance(value, dict) and any(
            isinstance(v, list) and any(_blank(i) for i in v) for v in value.values())
    if name == "aristotelian":
        return isinstance(value, dict) and any(_blank(v) for v in value.values())
    if name in ner_stage._ENUM_FIELDS:                        # "" where `null` means "none"
        return False
    return False


def _ner_outcome(obj: dict) -> "tuple[bool, list[str]]":
    """``(raised, marks)`` of the real ``_build`` over a decoded object."""
    metrics = StageMetrics(stage="ner", elapsed_ms=0.0, tokens_in=0, tokens_out=0, model="t")
    try:
        return False, IntentAnalyzer()._build(copy.deepcopy(obj), "", metrics).repairs
    except Exception:  # noqa: BLE001 — "the turn ended" is the outcome under test
        return True, []


async def _noumeno_outcome(obj: dict) -> "tuple[bool, list[str]]":
    try:
        return False, (await _noumeno(obj)).repairs
    except Exception:  # noqa: BLE001
        return True, []


def test_ner_what_the_schema_accepts_needs_no_form_repair_and_ends_no_turn():
    schema = schemas.STAGE_SCHEMAS["ner"]
    accepted = declared = refused_and_raised = 0
    for label, obj in _variants(_INTENT):
        raised, marks = _ner_outcome(obj)
        ok = accepts(schema, obj)
        if not ok:
            refused_and_raised += raised
            # the converse: what the stage reads with NO form mark, the schema accepts
            assert raised or _form(marks), f"refused, and the stage read it clean: {label}"
            continue
        accepted += 1
        assert not raised, f"accepted by the schema, and the turn ended: {label}"
        for mark in _form(marks):
            assert _content_shaped("ner", mark, obj), f"{label}: {mark}"
            declared += 1
    # 859 replies (the clean one, and each of 22 fields missing or under 38 dirty values). The
    # numbers are pinned so that a change in either module is SEEN: 91 accepted, 9 of them with
    # a declared content-shaped mark; and of the 768 refused, 16 end the turn today.
    assert (accepted, declared, refused_and_raised) == (91, 9, 16)


async def test_noumeno_what_the_schema_accepts_needs_no_form_repair_and_ends_no_turn():
    schema = schemas.STAGE_SCHEMAS["noumeno"]
    accepted = declared = refused_and_raised = 0
    for label, obj in _variants(_REWRITE):
        raised, marks = await _noumeno_outcome(obj)
        ok = accepts(schema, obj)
        if not ok:
            refused_and_raised += raised
            assert raised or _form(marks), f"refused, and the stage read it clean: {label}"
            continue
        accepted += 1
        assert not raised, f"accepted by the schema, and the turn ended: {label}"
        for mark in _form(marks):
            assert _content_shaped("noumeno", mark, obj), f"{label}: {mark}"
            declared += 1
    # 235 replies (6 fields): 48 accepted, 2 with the declared mark (a blank rewrite); of the
    # 187 refused, 96 end the turn today — none of which the schema lets through.
    assert (accepted, declared, refused_and_raised) == (48, 2, 96)


@pytest.mark.parametrize("mark", sorted(m for m in rp.VALID_NER_REPAIRS if m.startswith(FORM)))
def test_ner_every_form_mark_of_the_alphabet_is_a_reply_the_schema_refuses(mark):
    """Enumerated from the ALPHABET, with the repairs suite's own dirty reply per mark."""
    assert not accepts(schemas.STAGE_SCHEMAS["ner"], _ner_dirty(mark))


@pytest.mark.parametrize("mark", sorted(m for m in rp.VALID_NOUMENO_REPAIRS
                                        if m.startswith(FORM)))
def test_noumeno_every_form_mark_of_the_alphabet_is_a_reply_the_schema_refuses(mark):
    assert not accepts(schemas.STAGE_SCHEMAS["noumeno"], _NOUMENO_DIRTY[mark])


@pytest.mark.parametrize("mark", sorted(m for m in rp.VALID_NER_REPAIRS
                                        if m.startswith(("cut:", "fallback:"))))
def test_ner_a_cut_or_a_fallback_is_not_the_schemas_to_refuse(mark):
    """The declared limit, pinned so it cannot quietly change: a value that is too LONG, and a
    value the stage fills from other fields, are accepted."""
    assert accepts(schemas.STAGE_SCHEMAS["ner"], _ner_dirty(mark))


def test_the_clean_replies_are_accepted():
    assert accepts(schemas.STAGE_SCHEMAS["noumeno"], _REWRITE)
    assert accepts(schemas.STAGE_SCHEMAS["ner"], _INTENT)
    # what the stage does not read does not make a reply unacceptable: the prompt's own extras
    extras = dict(_INTENT, abstract_tags=["BILLING"], langue="pt", pii_risk="NONE",
                  entities=dict(_INTENT["entities"], pronouns=[], possessives=[]))
    assert accepts(schemas.STAGE_SCHEMAS["ner"], extras)


# ── the verdict stages: accepted ⇒ a boolean verdict and nothing repaired ───────────────────

_VERDICT_VALUES = [None, "", "  ", "false", "true", 0, 1, True, False, [], ["no"], {}, "text"]


def _verdict_variants(key: str, text: str):
    for verdict in _VERDICT_VALUES:
        for message in _VERDICT_VALUES:
            yield {key: verdict, text: message}
        yield {key: verdict}
    yield {text: "only the text"}
    yield {}


async def test_guard_what_the_schema_accepts_is_a_boolean_verdict_with_nothing_repaired():
    schema = schemas.STAGE_SCHEMAS["superego_scope"]
    accepted = 0
    for obj in _verdict_variants("blocked", "refusal_message"):
        if not accepts(schema, obj):
            continue
        accepted += 1
        result, _ = await _guard(json.dumps(obj))
        assert result.verdict_read == VERDICT_BOOLEAN, obj
        assert not _form(result.repairs) or _blank(obj["refusal_message"]), obj
    assert accepted == 10, "two booleans × the five strings of the table"
    assert accepts(schema, json.loads(_ALLOW)) and accepts(schema, json.loads(_BLOCK))


async def test_judge_what_the_schema_accepts_is_a_boolean_verdict_with_nothing_repaired():
    schema = schemas.STAGE_SCHEMAS["superego_judge"]
    accepted = 0
    for obj in _verdict_variants("approved", "critique"):
        if not accepts(schema, obj):
            continue
        accepted += 1
        result = await _judge(json.dumps(obj))
        assert result.verdict_read == VERDICT_BOOLEAN, obj
        assert not _form(result.repairs) or _blank(obj["critique"]), obj
    assert accepted == 10, "two booleans × the five strings of the table"
    assert accepts(schema, json.loads(_REJECT))


def test_the_verdict_schemas_are_what_the_prompt_asks_for_not_what_the_stage_tolerates():
    """The declared limit of the converse: the text beside the verdict is read only when the
    verdict asks for it, so the stage reads ``{"approved": true}`` clean and the schema — both
    keys, always — refuses it."""
    assert not accepts(schemas.STAGE_SCHEMAS["superego_judge"], json.loads(_APPROVE))
    assert not accepts(schemas.STAGE_SCHEMAS["superego_scope"], {"blocked": False})


@pytest.mark.parametrize("stage,table", [("superego_scope", _GUARD_DIRTY),
                                         ("superego_judge", _JUDGE_DIRTY)])
def test_every_form_mark_of_a_verdict_stage_is_a_reply_the_schema_refuses(stage, table):
    for mark, reply in table.items():
        if mark.startswith(FORM):
            assert not accepts(schemas.STAGE_SCHEMAS[stage], json.loads(reply)), mark


# ── ONE reader, behind the switch ───────────────────────────────────────────────────────────

def _imports(path: Path) -> "set[str]":
    found: "set[str]" = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found |= {f"{node.module}.{alias.name}" for alias in node.names}
    return found


def test_exactly_one_module_of_the_package_imports_the_schemas_and_not_at_import_time():
    """One reader: ``structured.schema_ask``. And the import sits INSIDE that function — the
    schemas module imports the stages, so a stage could not import it at module level, and a
    turn with the switch off has no business importing it at all."""
    readers = [path.relative_to(PACKAGE).as_posix() for path in sorted(PACKAGE.rglob("*.py"))
               if path.name != "schemas.py" and "cogno_anima.schemas" in _imports(path)]
    assert readers == ["structured.py"]
    tree = ast.parse((PACKAGE / "structured.py").read_text(encoding="utf-8"))
    top_level = {node.module for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert "cogno_anima.schemas" not in top_level
    inside = [fn.name for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef)
              and any(isinstance(n, ast.ImportFrom) and n.module == "cogno_anima.schemas"
                      for n in ast.walk(fn))]
    assert inside == ["schema_ask"]
    # the control: the same walk does see an import that exists
    assert "cogno_anima.repairs" in _imports(PACKAGE / "stages" / "ner.py")
    assert "cogno_anima.repairs" in _imports(PACKAGE / "schemas.py")


def test_the_package_root_does_not_export_it():
    assert "schemas" not in cogno_anima.__all__
    assert not any("SCHEMA" in name.upper() for name in cogno_anima.__all__)


async def test_no_stage_hands_a_backend_anything_but_the_two_arguments():
    """What "off by default" means at the wire: with no switch stamped, every call is
    ``generate(system, prompt)`` — two positional strings, no keyword — whatever this module
    holds."""
    calls = []

    class Recording:
        model = "t"

        def __init__(self, reply):
            self.reply = reply

        async def generate(self, *args, **kwargs):
            calls.append((len(args), tuple(sorted(kwargs)),
                          all(isinstance(a, str) for a in args)))
            return self.reply, 5, 3

    from cogno_anima.stages import scope_options
    from cogno_anima.stages.noumeno import Noumeno
    from cogno_anima.types import PipelineContext
    from tests.conftest import StubEmbedder
    from tests.unit.test_superego import _ctx

    ctx = PipelineContext(user_input="Manda a fatura.", force_language="pt")
    await Noumeno(embedder=StubEmbedder()).process(ctx, Recording(json.dumps(_REWRITE)))
    await IntentAnalyzer().analyze(_ctx().noumeno, llm=Recording(json.dumps(_INTENT)))
    stage = SuperegoStage()
    await stage.check_input_scope(_ctx(user="what can I study?",
                                       intent_class="INFORMATION_REQUEST"),
                                  Recording(_ALLOW), scope_prompt="Only school things.")
    await stage.evaluate(_ctx(), Recording(_APPROVE), limits_prompt="")
    await stage.voice(_ctx(), Recording("Done."), voice_prompt="Be brief.")
    await scope_options.select_options("is there a gym?", ["Timetable"], Recording("{}"))
    assert len(calls) == 6, "the premise: every stage made its call"
    assert set(calls) == {(2, (), True)}
