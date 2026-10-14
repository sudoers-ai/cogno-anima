"""A reply written under a schema that makes EVERY key mandatory spells an optional key as
``null`` — and the stage reads that ``null`` as the absent key, with no mark. Only then.

A provider's strict mode closes every object and requires every key, so a key the stage treats
as optional (the sub-keys of the NER's ``entities``, the ten categories of ``aristotelian``) can
no longer be left out: the model writes ``null`` for the ones that do not apply. Read as it
stands, that ``null`` is a ``coerced`` mark on every such reply — and a comparison of schema'd
and unschema'd replies would count the schema's own spelling as its defect, and read backwards.

The rule (``cogno_anima.repairs.nulls_as_absent``) applies ONLY under the receipt of such a
schema. With no schema, an open one or a refused one, a ``null`` is still what it always was.

Tied to its two sources so neither table can rot alone:

* ``NULL_IS_ABSENT_RECEIPTS`` is exactly the dialects whose conversion (``cogno_synapse``) turns
  an optional key into a required, nullable one;
* ``OPTIONAL_KEY_FIELDS`` is exactly the fields whose schema is an object with a key that is
  not required.

The twin runs the REAL stage in both worlds: the same strict-shaped reply, read with the strict
receipt (no mark) and without it (the marks the rule exists to remove), and the DECISIONS are the
same in both — the rule changes what is counted, never what is decided.
"""

from __future__ import annotations

import copy
import json

import pytest
from cogno_synapse import SCHEMA_DIALECTS, SCHEMA_RECEIPTS, to_dialect

from cogno_anima import metakeys as mk
from cogno_anima import repairs as rp
from cogno_anima import schemas
from cogno_anima.stages.ner import IntentAnalyzer
from tests.unit.test_repairs_are_counted import _INTENT, _decided
from tests.unit.test_schemas_are_derived import accepts
from tests.unit.test_stage_schema_switch import JSON_STAGES, Plain, Structured
from tests.unit.test_superego import _ctx

STRICT = "openai_strict"
NER_STRICT_SCHEMA = to_dialect(schemas.schema_for("ner", "strict"), STRICT)


def strict_shaped(reply: dict) -> dict:
    """``reply`` as a mandatory-keys schema makes a model write it: every key of the converted
    schema present, ``null`` under each one the reply did not carry."""
    def fill(value, schema):
        if isinstance(value, dict) and "properties" in schema:
            return {key: fill(value.get(key), sub) for key, sub in schema["properties"].items()}
        return value
    return fill(copy.deepcopy(reply), NER_STRICT_SCHEMA)


#: A clean reply that leaves OUT what does not apply — two of the three entity lists and nine of
#: the ten categories — as the prompt allows. The reader has always read it with no mark.
SPARSE = dict(_INTENT, entities={"objects": ["invoice"]})
SHAPED = strict_shaped(SPARSE)


async def read(reply: dict, receipt):
    """The REAL NER over ``reply``. ``receipt`` is what the serving backend says it carried;
    ``"off"`` runs the stage with the switch off on a plain backend."""
    ctx = _ctx(user="Manda a fatura.")
    if receipt == "off":
        await IntentAnalyzer().process(ctx, Plain(json.dumps(reply)))
    else:
        ctx.metadata[mk.STAGE_SCHEMA] = {"ner": "strict"}
        await IntentAnalyzer().process(ctx, Structured(json.dumps(reply), receipt=receipt))
    return ctx.intent


# ── the premise: the shaped reply is what the strict schema asks for ──────────────────────────

def test_the_shaped_reply_is_accepted_by_the_converted_schema_and_the_clean_one_is_not():
    assert accepts(NER_STRICT_SCHEMA, SHAPED)
    assert not accepts(NER_STRICT_SCHEMA, SPARSE), "the clean reply leaves keys out"
    assert accepts(schemas.schema_for("ner", "open"), SPARSE), "…which the open schema allows"
    # what the shaping added is nulls, under the optional keys and the prompt-only ones
    assert SHAPED["entities"] == {"people": None, "objects": ["invoice"], "concepts": None,
                                  "pronouns": None, "possessives": None}
    nulls = sorted(k for k, v in SHAPED["aristotelian"].items() if v is None)
    assert len(nulls) == 9 and SHAPED["aristotelian"]["ACTION"] == _INTENT["aristotelian"]["ACTION"]
    assert all(SHAPED[key] is None for key in schemas.NER_PROMPT_ONLY_KEYS)


# ── the two tables, tied to their sources ─────────────────────────────────────────────────────

def test_the_receipts_are_exactly_the_dialects_that_make_an_optional_key_mandatory():
    probe = {"type": "object",
             "properties": {"kept": {"type": "string"}, "optional": {"type": "string"}},
             "required": ["kept"]}
    mandatory = set()
    for dialect in SCHEMA_DIALECTS:
        converted = to_dialect(probe, dialect)
        if "optional" in converted["required"]:
            assert "null" in converted["properties"]["optional"]["type"], dialect
            mandatory.add(dialect)
    assert mandatory == rp.NULL_IS_ABSENT_RECEIPTS
    assert rp.NULL_IS_ABSENT_RECEIPTS < SCHEMA_RECEIPTS
    assert len(SCHEMA_DIALECTS - mandatory) >= 1, "the harness can tell the two kinds apart"


@pytest.mark.parametrize("form", ["open", "strict"])
def test_the_fields_are_exactly_the_objects_with_a_key_that_is_not_required(form):
    for stage in JSON_STAGES:
        schema = schemas.schema_for(stage, form)
        optional = tuple(
            name for name, prop in schema["properties"].items()
            if prop.get("type") == "object"
            and set(prop.get("properties", {})) - set(prop.get("required", ())))
        assert optional == rp.OPTIONAL_KEY_FIELDS.get(stage, ()), stage
        # and no key is optional at the TOP level of what the stage itself reads: a top-level
        # null means what it says in every form
        read_keys = set(schemas.STAGE_SCHEMAS[stage]["properties"])
        assert read_keys <= set(schema["required"]), stage
    assert set(rp.OPTIONAL_KEY_FIELDS) == {"ner"}


# ── the function ──────────────────────────────────────────────────────────────────────────────

def test_under_the_strict_receipt_a_null_under_an_optional_key_is_left_out():
    before = copy.deepcopy(SHAPED)
    out = rp.nulls_as_absent(SHAPED, "ner", STRICT)
    assert out["entities"] == SPARSE["entities"] == {"objects": ["invoice"]}
    assert out["aristotelian"] == SPARSE["aristotelian"]
    assert SHAPED == before, "the input was mutated"
    assert out is not SHAPED and out["entities"] is not SHAPED["entities"]
    # the top level is never touched: the prompt-only keys stay (ignored by the reader), and
    # so does a null the schema let the model write
    assert out["location"] is None and "location" in out
    assert all(key in out for key in schemas.NER_PROMPT_ONLY_KEYS)


@pytest.mark.parametrize("applied", [None, "openai_json_schema", "ollama_format", "refused",
                                     "chain", "OPENAI_STRICT", "", True, 1])
def test_under_any_other_receipt_the_reply_is_returned_as_it_is(applied):
    assert rp.nulls_as_absent(SHAPED, "ner", applied) is SHAPED


def test_it_touches_only_the_stages_own_fields_and_never_raises():
    for stage in ("noumeno", "superego_scope", "superego_judge", "superego_select", "", None, 3):
        assert rp.nulls_as_absent(SHAPED, stage, STRICT) is SHAPED, stage
    for data in (None, [], "x", 3, {"entities": None}, {"entities": []}, {"entities": "x"},
                 {"aristotelian": {"ACTION": "A | b"}}, {}):
        assert rp.nulls_as_absent(data, "ner", STRICT) is data
    assert rp.nulls_as_absent({"entities": {"people": None}}, "ner", STRICT) == {"entities": {}}


# ── the twin, through the REAL stage, in both worlds ──────────────────────────────────────────

async def test_the_same_reply_is_marked_without_the_receipt_and_clean_with_it():
    clean = await read(SPARSE, "off")
    assert clean.repairs == []

    broken = await read(SHAPED, "off")                      # the world without the rule
    assert broken.repairs == ["coerced:entities", "coerced:aristotelian"]

    fixed = await read(SHAPED, STRICT)                      # the strict receipt
    assert fixed.repairs == []
    assert fixed.metrics.schema_use == {"requested": "strict", "applied": STRICT}

    # the rule changes what is COUNTED, never what is DECIDED
    assert _decided(fixed) == _decided(broken) == _decided(clean) | {
        "raw_response": fixed.raw_response}
    assert fixed.raw_response == json.dumps(SHAPED), "the raw reply is kept as the model wrote it"


@pytest.mark.parametrize("receipt", ["openai_json_schema", "ollama_format", "refused"])
async def test_only_the_strict_receipt_reads_null_as_absent(receipt):
    """A schema that travelled OPEN, or did not travel, forces no key: a ``null`` there is the
    model's own, and it is still counted."""
    result = await read(SHAPED, receipt)
    assert result.repairs == ["coerced:entities", "coerced:aristotelian"]
    assert result.metrics.schema_use == {"requested": "strict", "applied": receipt}


async def test_a_strict_request_that_no_backend_carried_reads_null_as_it_always_did():
    """Requested strict, applied ``None``: the rule keys on what was APPLIED."""
    ctx = _ctx(user="Manda a fatura.")
    ctx.metadata[mk.STAGE_SCHEMA] = {"ner": "strict"}
    await IntentAnalyzer().process(ctx, Plain(json.dumps(SHAPED)))
    assert ctx.intent.metrics.schema_use == {"requested": "strict", "applied": None}
    assert ctx.intent.repairs == ["coerced:entities", "coerced:aristotelian"]


async def test_what_is_still_a_repair_under_the_strict_receipt():
    """The control of the rule: it removes the schema's own spelling and nothing else."""
    cases = {
        "null_field:sentiment": dict(SHAPED, sentiment=None),
        "null_field:entities": dict(SHAPED, entities=None),
        "null_field:aristotelian": dict(SHAPED, aristotelian=None),
        "coerced:entities": dict(SHAPED, entities=dict(SHAPED["entities"], people="nobody")),
        "coerced:aristotelian": dict(SHAPED, aristotelian=dict(SHAPED["aristotelian"], ACTION=7)),
        "coerced:sentiment": dict(SHAPED, sentiment="ANGRY-ISH"),
        "string_bool:is_composite": dict(SHAPED, is_composite="false"),
        "default:goal": {k: v for k, v in SHAPED.items() if k != "goal"},
    }
    for mark, reply in cases.items():
        result = await read(reply, STRICT)
        assert result.repairs == [mark], mark


# ── the keys the prompt asks for and the reader ignores ───────────────────────────────────────

@pytest.mark.parametrize("receipt", ["off", STRICT])
async def test_the_prompt_only_keys_are_ignored_with_no_mark(receipt):
    """Present with values, present as ``null`` or absent: the same result, nothing counted."""
    base = SPARSE if receipt == "off" else SHAPED
    valued = dict(base, abstract_tags=["BILLING"], comparatives=["cheaper"], pii_risk="HIGH",
                  raw_intent_class="SOCIAL", raw_domains=["HEALTH"], raw_goal="another goal",
                  entities=dict(base["entities"], pronouns=["she"], possessives=["her"]))
    nulled = dict(base, **{key: None for key in schemas.NER_PROMPT_ONLY_KEYS})
    plain = await read(base, receipt)
    for reply in (valued, nulled):
        result = await read(reply, receipt)
        assert result.repairs == plain.repairs == []
        assert {k: v for k, v in _decided(result).items() if k != "raw_response"} == \
               {k: v for k, v in _decided(plain).items() if k != "raw_response"}
    # the reader really does ignore them: the values above would have moved these fields
    assert plain.pii_risk == "NONE" and plain.intent_class == "ACTION_REQUEST"
    assert plain.domains == ["FINANCE"] and plain.goal == "send the invoice"
