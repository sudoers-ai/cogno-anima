"""
cogno_anima.schemas — the JSON Schema of each stage's reply, DERIVED from the closed vocabularies.

Four stages ask a model for a JSON object whose fields this package then validates by hand
against closed vocabularies: the NOUMENO, the NER, the scope guard and the judge. A provider that
accepts a schema with the request can refuse most of what that validation repairs — a word
outside the vocabulary, a key left out, a ``null`` where a value belongs, a boolean spelled as a
string. This module is that schema, one per stage.

**INERT. Nothing in this package reads it.** No stage passes a schema to a backend, no prompt
mentions one and no result changes: with this module present or absent, every call a stage makes
is the same ``backend.generate(system, prompt)``. It exists so that the day a transport can carry
a schema, the schema it carries is not a second copy of the vocabularies.

**Derived, never copied.** Every ``enum`` here is built FROM the object the stage validates
against — ``vocab.VALID_INTENTS``, ``vocab.NER_KNOWLEDGE_DOMAINS``, ``security.pii.VALID_PII_TYPES``
… — sorted. The fields of each object are the fields the stage READS, taken from the ``default``
row of the stage's own table in :mod:`cogno_anima.repairs` (a key the stage reads and the model
left out is a repair there, and a required property here). A vocabulary that gains a word, or a
stage that starts reading a field, moves the schema with it; a field this module has no type for
raises at import.

**It is the reply a stage reads WITHOUT A FORM REPAIR — and that is the bridge to the marks.**
A reply this schema accepts needs none of ``default:<field>`` (``required``),
``null_field:<field>`` (the type, unless the prompt allows ``null``), ``string_bool:<field>``
(``boolean``) or ``coerced:<field>`` (the type, the ``enum``).
``tests/unit/test_schemas_are_derived.py`` walks that claim over the same dirty replies the marks
are tested with. What the schema does NOT refuse, by design:

* a value of the right type and the wrong CONTENT — a ``confidence`` of 1.5, a blank string, a
  domain listed twice. Ranges, lengths and uniqueness are left out on purpose: which of those
  keywords a provider's strict mode accepts is the transport's question, and a schema that
  carries a keyword its provider refuses is a request that fails;
* a value longer than the stage keeps (``cut:<field>``), for the same reason;
* the two fallbacks (``fallback:<field>``) and the reply's envelope (a fence, prose around the
  object), which are not properties of the object at all.

**It is NOT yet the request a strict provider takes.** The objects here are as open as the
stages are: the NER ignores keys it does not read (the prompt itself asks for three more), its
``entities`` and ``aristotelian`` maps may leave a key out, and a field the prompt lets the model
answer with ``null`` is a union with ``null``. A provider whose strict mode wants every object
closed and every key required needs that applied on top — a mechanical step, and one that
changes what the model is asked to emit, which is why it belongs with the measurement and not
here. The one object that IS closed is ``aristotelian``: its keys are the closed
``VALID_ARISTOTELIAN``, and a key outside it is a repair today.

For the NOUMENO and the NER the converse holds too: a reply the stage reads without a form
repair is accepted. For the guard and the judge it does not, and cannot with a schema this plain:
the text beside the verdict (a refusal message, a critique) is read only when the verdict asks
for it, so ``{"approved": true}`` is a clean read and is refused here. Their schemas are what
the PROMPT asks for — both keys, always.

The scope-option selector and the pre-judge have no schema here. The pre-judge is already
strict; whether the selector gains from one is a measurement that has not been made.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable, Optional

from cogno_anima.repairs import (DEFAULT, JUDGE_REPAIR_FIELDS, NER_REPAIR_FIELDS,
                                 NOUMENO_REPAIR_FIELDS, SCOPE_REPAIR_FIELDS)
from cogno_anima.security.pii import VALID_PII_TYPES
from cogno_anima.stages.ner import _ENUM_FIELDS, _FLAGS, _PHRASE_LISTS
from cogno_anima.vocab import NER_KNOWLEDGE_DOMAINS, VALID_ARISTOTELIAN, VALID_MANDATORY

__all__ = [
    "JUDGE_VERDICT_KEY",
    "SCOPE_VERDICT_KEY",
    "NER_LIST_VOCABULARIES",
    "STAGE_SCHEMAS",
    "build_stage_schemas",
    "schema_for",
]

#: The boolean each verdict stage decides on — the key ``cogno_anima.verdict.read_verdict`` is
#: asked for in ``SuperegoStage.check_input_scope`` and ``SuperegoStage.evaluate``. Named here
#: because the repairs tables deliberately leave the verdict key out (how it was read is
#: ``verdict_read``), and a schema of the reply has to hold it.
SCOPE_VERDICT_KEY = "blocked"
JUDGE_VERDICT_KEY = "approved"

#: The NER's list fields whose ITEMS come from a closed vocabulary.
NER_LIST_VOCABULARIES: "dict[str, set[str]]" = {
    "mandatory_tags": VALID_MANDATORY,
    "domains": NER_KNOWLEDGE_DOMAINS,
    "pii": VALID_PII_TYPES,
}

#: The sub-keys of the NER's ``entities`` object the stage reads.
_ENTITY_KEYS: "tuple[str, ...]" = ("people", "objects", "concepts")


def _string() -> "dict[str, Any]":
    return {"type": "string"}


def _nullable(schema: "dict[str, Any]") -> "dict[str, Any]":
    """``schema`` or ``null`` — a union on the type, and on the ``enum`` when there is one."""
    out = dict(schema)
    out["type"] = [schema["type"], "null"]
    if "enum" in schema:
        out["enum"] = [*schema["enum"], None]
    return out


def _enum(vocabulary: "Iterable[str]") -> "dict[str, Any]":
    """A string from ``vocabulary`` — sorted, so the schema is the same bytes on every worker."""
    return {"type": "string", "enum": sorted(vocabulary)}


def _array(items: "dict[str, Any]") -> "dict[str, Any]":
    return {"type": "array", "items": items}


def _required(fields: "tuple[str, ...]",
              properties: "dict[str, dict[str, Any]]") -> "dict[str, Any]":
    """An object that must carry every one of ``fields`` — in the table's order, and ONLY those
    as properties: a field of the stage's table this module has no type for raises here, which
    is the point of deriving. Open to keys the stage does not read."""
    return {"type": "object", "properties": {name: properties[name] for name in fields},
            "required": sorted(fields)}


def _noumeno() -> "dict[str, Any]":
    return _required(NOUMENO_REPAIR_FIELDS[DEFAULT], {
        "rewritten": _string(),
        "context_turn": _string(),
        "confidence": {"type": "number"},
        "changed": {"type": "boolean"},
        "preserved_terms": _array(_string()),
        # The prompt lists four allowed warnings; the CODE validates against no vocabulary, so
        # there is no enumeration to derive and none is invented here.
        "rewrite_warnings": _array(_string()),
    })


def _ner() -> "dict[str, Any]":
    properties: "dict[str, dict[str, Any]]" = {
        "confidence": {"type": "number"},
        # Sub-keys may be left out (the stage reads an absent one as an empty list) and the
        # prompt asks for two more the stage ignores, so this object is open.
        "entities": {"type": "object",
                     "properties": {key: _array(_string()) for key in _ENTITY_KEYS}},
        "location": _nullable(_string()),
        "goal": _nullable(_string()),
        # "Include ONLY relevant categories": no key is required, and none outside the closed
        # set is allowed — the one object here a strict schema would not have to change.
        "aristotelian": {"type": "object",
                         "properties": {key: _string() for key in sorted(VALID_ARISTOTELIAN)},
                         "additionalProperties": False},
    }
    for name, (vocabulary, nullable) in _ENUM_FIELDS.items():
        properties[name] = _nullable(_enum(vocabulary)) if nullable else _enum(vocabulary)
    for name, vocabulary in NER_LIST_VOCABULARIES.items():
        properties[name] = _array(_enum(vocabulary))
    for name in _PHRASE_LISTS:
        properties[name] = _array(_string())
    for name in _FLAGS:
        properties[name] = {"type": "boolean"}
    return _required(NER_REPAIR_FIELDS[DEFAULT], properties)


def _verdict(key: str, fields: "tuple[str, ...]") -> "dict[str, Any]":
    """What the verdict stage's PROMPT asks for: the boolean and the text beside it, always."""
    return _required((key, *fields), {key: {"type": "boolean"},
                                      **{name: _string() for name in fields}})


def build_stage_schemas() -> "dict[str, dict[str, Any]]":
    """Build every stage's schema from the vocabularies AS THEY ARE NOW.

    :data:`STAGE_SCHEMAS` is this, called once at import. It is a function so the derivation can
    be exercised: a test adds a word to a vocabulary, calls it again and reads the word back.
    """
    return {
        "noumeno": _noumeno(),
        "ner": _ner(),
        "superego_scope": _verdict(SCOPE_VERDICT_KEY, SCOPE_REPAIR_FIELDS[DEFAULT]),
        "superego_judge": _verdict(JUDGE_VERDICT_KEY, JUDGE_REPAIR_FIELDS[DEFAULT]),
    }


#: The schema of each stage's reply, keyed by the ``StageMetrics.stage`` of the call — the keys
#: of :data:`cogno_anima.repairs.REPAIRS_BY_STAGE`, minus the selector. Read it through
#: :func:`schema_for`: these objects are shared, and a caller that edits one edits it for all.
STAGE_SCHEMAS: "dict[str, dict[str, Any]]" = build_stage_schemas()


def schema_for(stage: str) -> "Optional[dict[str, Any]]":
    """A COPY of the schema of ``stage``'s reply, or ``None`` for a stage that has none (the
    selector, the voice, the executor, an unknown label). Never raises."""
    schema = STAGE_SCHEMAS.get(stage) if isinstance(stage, str) else None
    return copy.deepcopy(schema) if schema is not None else None
