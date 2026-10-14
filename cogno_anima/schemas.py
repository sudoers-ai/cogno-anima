"""
cogno_anima.schemas — the JSON Schema of each stage's reply, DERIVED from the closed vocabularies.

Four stages ask a model for a JSON object whose fields this package then validates by hand
against closed vocabularies: the NOUMENO, the NER, the scope guard and the judge. A provider that
accepts a schema with the request can refuse most of what that validation repairs — a word
outside the vocabulary, a key left out, a ``null`` where a value belongs, a boolean spelled as a
string. This module is that schema, one per stage.

**OFF BY DEFAULT. One reader, behind a switch.** :func:`cogno_anima.structured.schema_ask` is the
only thing in this package that reads this module, and only on a turn whose host stamped
``metakeys.STAGE_SCHEMA`` for the stage. Without that key no stage passes a schema to a backend,
no prompt mentions one and no result changes: every call a stage makes is the same
``backend.generate(system, prompt)``. With it, the schema travels BESIDE the prompt — the prompt
is the same bytes either way. The schema a transport carries is therefore never a second copy of
the vocabularies.

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

**Two forms, and the difference between them is what THIS package knows.** :func:`schema_for`
takes the form the host asked for (``cogno_synapse.SCHEMA_FORMS``):

* ``"open"`` — the schema above, as open as the stages are: the NER ignores keys it does not
  read, its ``entities`` and ``aristotelian`` maps may leave a key out, and a field the prompt
  lets the model answer with ``null`` is a union with ``null``. The one object that IS closed is
  ``aristotelian``: its keys are the closed ``VALID_ARISTOTELIAN``, and a key outside it is a
  repair today.
* ``"strict"`` — the same schema with the keys the stage's PROMPT asks the model for and the
  reader ignores added as PERMITTED (optional). A provider's strict mode closes every object, so
  a key the prompt asks for and the schema does not hold is a key the model can no longer write
  — and the prompt does not change. Only this package knows what a stage's prompt asks for:
  :data:`NER_PROMPT_ONLY_KEYS` and :data:`NER_PROMPT_ONLY_ENTITY_KEYS`, each declared ONCE and
  tied to the prompt's own examples by a test (a key that leaves the prompt, or a new one the
  prompt starts asking for, fails it). The reader goes on ignoring them, with no repair mark.

**The provider's mechanics are NOT here.** Closing every object, requiring every key and
widening a key that was optional to ``type | null`` is ``cogno_synapse.schema.to_dialect``'s, per
dialect, at the leaf backend. This module does not know a provider exists.

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

from cogno_synapse import SCHEMA_FORMS

from cogno_anima.repairs import (DEFAULT, JUDGE_REPAIR_FIELDS, NER_REPAIR_FIELDS,
                                 NOUMENO_REPAIR_FIELDS, SCOPE_REPAIR_FIELDS)
from cogno_anima.security.pii import VALID_PII_TYPES
from cogno_anima.stages.ner import _ENUM_FIELDS, _FLAGS, _PHRASE_LISTS
from cogno_anima.vocab import NER_KNOWLEDGE_DOMAINS, VALID_ARISTOTELIAN, VALID_MANDATORY

__all__ = [
    "JUDGE_VERDICT_KEY",
    "SCOPE_VERDICT_KEY",
    "NER_LIST_VOCABULARIES",
    "NER_PROMPT_ONLY_KEYS",
    "NER_PROMPT_ONLY_ENTITY_KEYS",
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


#: The top-level keys the NER's PROMPT asks the model for and ``IntentAnalyzer._build`` never
#: reads — every example object in the prompt carries them. ``"array"`` is an array of strings.
#: In the ``"strict"`` form they are PERMITTED (optional properties); in the ``"open"`` form they
#: need no mention, the object being open. Declared once, here; ``langue`` is not among them (the
#: prompt tells the model to omit it, and no example carries it).
NER_PROMPT_ONLY_KEYS: "dict[str, str]" = {
    "abstract_tags": "array",
    "comparatives": "array",
    "pii_risk": "string",
    "raw_intent_class": "string",
    "raw_domains": "array",
    "raw_goal": "string",
}

#: The same, for the sub-keys of ``entities``.
NER_PROMPT_ONLY_ENTITY_KEYS: "dict[str, str]" = {
    "pronouns": "array",
    "possessives": "array",
}


def _string() -> "dict[str, Any]":
    return {"type": "string"}


def _permitted(kind: str) -> "dict[str, Any]":
    """The type of a key the prompt asks for and nobody reads: the plainest one that fits."""
    return _array(_string()) if kind == "array" else _string()


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


def _strict_form(stage: str, schema: "dict[str, Any]") -> "dict[str, Any]":
    """``schema`` (a copy) with the keys ``stage``'s prompt asks for and the reader ignores added
    as PERMITTED: properties that are not required. Nothing is closed, required or made nullable
    here — that is the provider's rule and the transport's job."""
    if stage == "ner":
        for name, kind in NER_PROMPT_ONLY_KEYS.items():
            schema["properties"][name] = _permitted(kind)
        for name, kind in NER_PROMPT_ONLY_ENTITY_KEYS.items():
            schema["properties"]["entities"]["properties"][name] = _permitted(kind)
    return schema


def schema_for(stage: str, form: str) -> "Optional[dict[str, Any]]":
    """A COPY of the schema of ``stage``'s reply in the ``form`` asked for, or ``None``.

    ``form`` has no default — it is ``"open"`` or ``"strict"`` (``cogno_synapse.SCHEMA_FORMS``),
    and a caller that forgets it fails here rather than get one chosen for it:

    * ``"open"`` — :data:`STAGE_SCHEMAS` as derived, byte for byte;
    * ``"strict"`` — the same, plus the keys the stage's prompt asks for and the reader ignores
      (see the module docstring). Today only the NER has any.

    ``None`` for a stage that has no schema (the selector, the voice, the executor, an unknown
    label) and for a form outside the two words. Never raises.
    """
    schema = STAGE_SCHEMAS.get(stage) if isinstance(stage, str) else None
    if schema is None or not isinstance(form, str) or form not in SCHEMA_FORMS:
        return None
    mine = copy.deepcopy(schema)
    return _strict_form(stage, mine) if form == "strict" else mine
