"""
cogno_anima.repairs — the CLOSED alphabet of what a stage had to REPAIR in a model's JSON reply.

Five stages ask a model for a JSON object and then make it usable by hand: the NOUMENO, the NER,
the scope guard, the judge and the scope-option selector. Every one of them tolerates a reply
that is not quite the object it asked for — a code fence around it, a key left out, a value
outside the vocabulary, a boolean spelled as a string — and until this module almost none of
that tolerance left a record. One log line (``intent_class_fallback``), one DEBUG line
(``domains_fallback``) and one WARNING (``truncated_response``) were all there was, so "how often
does this stage repair what the model wrote?" had no answer from anything a host persists.

**This module COUNTS. It decides nothing.** A mark says that a repair happened; the repair
itself is the stage's own code, unchanged. No prompt moves, no verdict moves, no field of a
result other than the new ``repairs`` list moves.

**A mark is about the FORM of the reply, and it never carries a byte of it.** Every mark is a
constant of this module or ``<family>:<field>`` built from two closed tuples — the family and
the NAME of a field the stage reads. The model's text, a value it wrote, a key it invented: none
of them can reach a mark, so a list of marks is safe to persist as it is.

The families:

``default:<field>``      the key was absent (or, for the text a stage must have, empty) and the
                         stage used its default;
``null_field:<field>``   the value was JSON ``null`` where the contract asks for a value;
``string_bool:<field>``  a boolean spelled as a string (``"true"``, ``"false"``, ``"yes"``…);
``coerced:<field>``      the value was not what the contract names — another type, a word
                         outside the closed vocabulary (a lower-case or aliased member
                         included: a strict enumeration would have refused it), an item of a
                         list that had to be dropped;
``cut:<field>``          the value was longer than the contract allows and was cut;
``fallback:<field>``     the stage filled the field from OTHER fields by a heuristic — the two
                         fallbacks the NER has always had (``intent_class`` from the mandatory
                         tags, ``domains`` from the mandatory tags).

and three marks about the reply as a whole:

``fence_or_think_stripped``  a chain-of-thought block or a code fence came off before the read;
``salvaged_object``          what was left was still not ONE JSON object, and the object was
                             dug out of it (prose around it, two objects side by side);
``truncated_retry``          the reply arrived cut mid-stream and the call was made again.

``unparseable`` is here for ONE stage only, the selector, and it is
:data:`cogno_anima.verdict.VERDICT_UNPARSEABLE` — the same token, not a second spelling. The
guard and the judge already say how their verdict was read (``verdict_read``, from
``VALID_VERDICT_READS``), ``unparseable`` and ``string_bool`` included, so their ``repairs``
lists never repeat it: the two records are read together.

**What is NOT a repair, and is not here.** What a stage decides from values it could read: the
grounding of a named person against the original text, the union of the model's PII list with
the deterministic detector, the anaphora flip of ``context_dependent``, the
composite/sequential reconciliation, the drift reconciliation of ``changed``. Those are the
stage's judgement over a well-formed reply; a strict schema would not remove one of them.

**One form mark per field.** ``default``, ``null_field``, ``string_bool`` and ``coerced`` are
four answers to one question (what was wrong with the value), so a field carries at most one of
them, the first recorded. ``cut`` and ``fallback`` are other questions and may sit beside it.

The alphabets are per stage (:data:`REPAIRS_BY_STAGE`, keyed by ``StageMetrics.stage``), each
DERIVED from a table of ``family -> fields``: a mark the table does not name cannot be recorded
(:class:`RepairLog` drops it) and cannot be persisted (:func:`closed_repairs` drops it).
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from cogno_anima.verdict import VERDICT_UNPARSEABLE, parse_object

__all__ = [
    "SALVAGED_OBJECT",
    "FENCE_OR_THINK_STRIPPED",
    "TRUNCATED_RETRY",
    "UNPARSEABLE",
    "DEFAULT",
    "NULL_FIELD",
    "STRING_BOOL",
    "COERCED",
    "CUT",
    "FALLBACK",
    "FORM_FAMILIES",
    "REPAIR_FAMILIES",
    "NOUMENO_REPAIR_FIELDS",
    "NER_REPAIR_FIELDS",
    "SCOPE_REPAIR_FIELDS",
    "JUDGE_REPAIR_FIELDS",
    "SELECT_REPAIR_FIELDS",
    "VALID_NOUMENO_REPAIRS",
    "VALID_NER_REPAIRS",
    "VALID_SCOPE_REPAIRS",
    "VALID_JUDGE_REPAIRS",
    "VALID_SELECT_REPAIRS",
    "REPAIRS_BY_STAGE",
    "VALID_REPAIRS",
    "NULL_IS_ABSENT_RECEIPTS",
    "OPTIONAL_KEY_FIELDS",
    "RepairLog",
    "repair_mark",
    "closed_repairs",
    "nulls_as_absent",
    "reply_form_repairs",
    "note_text_field",
    "spelled_boolean",
]

# ── the reply as a whole ─────────────────────────────────────────────────────────────────
SALVAGED_OBJECT = "salvaged_object"
FENCE_OR_THINK_STRIPPED = "fence_or_think_stripped"
TRUNCATED_RETRY = "truncated_retry"
#: The verdict reader's own token, for the one stage that has no ``verdict_read`` (the selector).
UNPARSEABLE = VERDICT_UNPARSEABLE

# ── one field ────────────────────────────────────────────────────────────────────────────
DEFAULT = "default"
NULL_FIELD = "null_field"
STRING_BOOL = "string_bool"
COERCED = "coerced"
CUT = "cut"
FALLBACK = "fallback"

#: Four answers to ONE question — what was wrong with the value — so at most one per field.
FORM_FAMILIES: "tuple[str, ...]" = (DEFAULT, NULL_FIELD, STRING_BOOL, COERCED)
REPAIR_FAMILIES: "tuple[str, ...]" = FORM_FAMILIES + (CUT, FALLBACK)

#: What a model writes when it spells a boolean as a string. Wider than the verdict reader's
#: two literals on purpose: the NER has always accepted ``"1"`` and ``"yes"`` for its three
#: flags, and the mark describes what THAT reader tolerates.
_SPELLED_BOOLEANS = frozenset({"true", "false", "1", "0", "yes", "no"})


def repair_mark(family: str, field: str) -> str:
    """``<family>:<field>`` — the one place the two halves are joined."""
    return f"{family}:{field}"


def _alphabet(singles: "Iterable[str]",
              fields: "Mapping[str, tuple[str, ...]]") -> "frozenset[str]":
    return frozenset(singles) | frozenset(
        repair_mark(family, name) for family, names in fields.items() for name in names)


# ── NOUMENO ──────────────────────────────────────────────────────────────────────────────
# The six keys of the rewriter's JSON. ``confidence`` has no ``null_field`` row because a
# ``null`` there has never been repaired: ``float(None)`` raises and the turn ends.
NOUMENO_REPAIR_FIELDS: "dict[str, tuple[str, ...]]" = {
    DEFAULT: ("rewritten", "context_turn", "confidence", "changed", "preserved_terms",
              "rewrite_warnings"),
    NULL_FIELD: ("rewritten", "context_turn", "changed", "preserved_terms", "rewrite_warnings"),
    STRING_BOOL: ("changed",),
    COERCED: ("confidence", "changed", "preserved_terms", "rewrite_warnings"),
}
VALID_NOUMENO_REPAIRS: "frozenset[str]" = _alphabet(
    (SALVAGED_OBJECT, FENCE_OR_THINK_STRIPPED, TRUNCATED_RETRY), NOUMENO_REPAIR_FIELDS)

# ── NER ──────────────────────────────────────────────────────────────────────────────────
# Every key ``IntentAnalyzer._build`` reads. ``langue``, ``abstract_tags`` and ``pii_risk`` are
# not here: the stage ignores what the model writes for them. ``location``, ``goal``,
# ``parole``, ``modality`` and ``speech_act`` have no ``null_field`` row because the prompt
# ALLOWS ``null`` there; ``mandatory_tags`` and ``pii`` have none because a ``null`` there has
# never been repaired (the stage iterates the value and raises).
NER_REPAIR_FIELDS: "dict[str, tuple[str, ...]]" = {
    DEFAULT: ("intent_class", "sentiment", "confidence", "temporal_class", "triad_signal",
              "entities", "location", "mandatory_tags", "aristotelian", "domains", "goal",
              "causal_chain", "parole", "negation", "constraints", "modality", "speech_act",
              "verbs", "context_dependent", "is_composite", "is_sequential", "pii"),
    NULL_FIELD: ("intent_class", "sentiment", "confidence", "temporal_class", "triad_signal",
                 "entities", "aristotelian", "domains", "causal_chain", "negation",
                 "constraints", "verbs", "context_dependent", "is_composite", "is_sequential"),
    STRING_BOOL: ("context_dependent", "is_composite", "is_sequential"),
    COERCED: ("intent_class", "sentiment", "confidence", "temporal_class", "triad_signal",
              "entities", "location", "mandatory_tags", "aristotelian", "domains", "goal",
              "causal_chain", "parole", "negation", "constraints", "modality", "speech_act",
              "verbs", "context_dependent", "is_composite", "is_sequential", "pii"),
    CUT: ("mandatory_tags", "aristotelian", "goal", "causal_chain", "negation", "constraints",
          "verbs"),
    FALLBACK: ("intent_class", "domains"),
}
VALID_NER_REPAIRS: "frozenset[str]" = _alphabet(
    (SALVAGED_OBJECT, FENCE_OR_THINK_STRIPPED, TRUNCATED_RETRY), NER_REPAIR_FIELDS)

# ── scope guard / judge ──────────────────────────────────────────────────────────────────
# The VERDICT key (``blocked`` / ``approved``) is deliberately absent from both tables: how it
# was read is ``verdict_read``, and repeating it here would be a second alphabet for one fact.
# What is left is the text that travels BESIDE the verdict, read only when the verdict asks for
# it (a refusal message on a block, a critique on a rejection).
SCOPE_REPAIR_FIELDS: "dict[str, tuple[str, ...]]" = {
    DEFAULT: ("refusal_message",),
    NULL_FIELD: ("refusal_message",),
    COERCED: ("refusal_message",),
}
VALID_SCOPE_REPAIRS: "frozenset[str]" = _alphabet(
    (SALVAGED_OBJECT, FENCE_OR_THINK_STRIPPED), SCOPE_REPAIR_FIELDS)

JUDGE_REPAIR_FIELDS: "dict[str, tuple[str, ...]]" = {
    DEFAULT: ("critique",),
    NULL_FIELD: ("critique",),
    COERCED: ("critique",),
}
VALID_JUDGE_REPAIRS: "frozenset[str]" = _alphabet(
    (SALVAGED_OBJECT, FENCE_OR_THINK_STRIPPED), JUDGE_REPAIR_FIELDS)

# ── scope-option selector ────────────────────────────────────────────────────────────────
# The selector has no ``verdict_read``, and its ``error`` outcome is three different things:
# the call raised (no mark at all), no JSON object came back (``unparseable``), or an object
# came back without the lists (``default:answers`` + ``default:maybe``, or ``coerced:…`` when a
# list is not a list). An option that is not on the closed list is NOT a mark: it is already
# counted, as ``discarded``.
SELECT_REPAIR_FIELDS: "dict[str, tuple[str, ...]]" = {
    DEFAULT: ("answers", "maybe", "asked"),
    NULL_FIELD: ("answers", "maybe", "asked"),
    COERCED: ("answers", "maybe", "asked"),
}
VALID_SELECT_REPAIRS: "frozenset[str]" = _alphabet(
    (SALVAGED_OBJECT, FENCE_OR_THINK_STRIPPED, UNPARSEABLE), SELECT_REPAIR_FIELDS)

#: The alphabet of each stage, keyed by the ``StageMetrics.stage`` of the call that produced
#: the reply. A reader that persists marks closes them with the row of ITS stage, never with
#: the union — ``cut:goal`` on a judge row is not a fact, it is a carrier gone wrong.
REPAIRS_BY_STAGE: "dict[str, frozenset[str]]" = {
    "noumeno": VALID_NOUMENO_REPAIRS,
    "ner": VALID_NER_REPAIRS,
    "superego_scope": VALID_SCOPE_REPAIRS,
    "superego_judge": VALID_JUDGE_REPAIRS,
    "superego_select": VALID_SELECT_REPAIRS,
}

#: Every mark of every stage — for a reader that only needs "is this a repair mark at all".
VALID_REPAIRS: "frozenset[str]" = frozenset().union(*REPAIRS_BY_STAGE.values())

# ── a reply written under a schema that makes every key mandatory ────────────────────────
#: The schema RECEIPTS (``cogno_synapse.schema_applied_of``) under which a provider makes the
#: model emit EVERY key of the schema, so a key the stage treats as optional arrives as ``null``
#: instead of being left out. ``tests/unit/test_strict_null_is_absent.py`` ties this set to the
#: transport: it is exactly the dialects whose conversion turns an optional key into a required,
#: nullable one.
NULL_IS_ABSENT_RECEIPTS: "frozenset[str]" = frozenset({"openai_strict"})

#: Per stage, the fields that are OBJECTS whose keys the stage reads as optional — the only
#: place a mandatory-keys schema changes what a well-formed reply looks like. Under a receipt of
#: :data:`NULL_IS_ABSENT_RECEIPTS`, a ``null`` under one of those keys is the model saying "this
#: key does not apply", which is what leaving the key out says without the schema; read as it
#: stands it would be a ``coerced`` mark on every such reply, and a comparison of a schema'd
#: reply with an unschema'd one would count the schema's own spelling as its defect. The same
#: test ties this table to the schemas: a field is here exactly when its canonical schema is an
#: object with a key that is not required. The NOUMENO, the guard and the judge have none.
OPTIONAL_KEY_FIELDS: "dict[str, tuple[str, ...]]" = {
    "ner": ("entities", "aristotelian"),
}


class RepairLog:
    """The marks ONE stage call recorded, in the order it recorded them.

    Closed by construction: a mark outside the stage's alphabet is dropped, never written, so
    nothing a caller passes — and nothing a future edit forgets to declare — can put a string
    of its own into a list a host persists. Never raises: a count must not cost a turn.
    """

    __slots__ = ("_alphabet", "_marks", "_formed")

    def __init__(self, stage: str) -> None:
        self._alphabet = REPAIRS_BY_STAGE.get(stage, frozenset())
        self._marks: "list[str]" = []
        self._formed: "set[str]" = set()

    def add(self, mark: str) -> None:
        """Record a whole-reply mark (once)."""
        if isinstance(mark, str) and mark in self._alphabet and mark not in self._marks:
            self._marks.append(mark)

    def field(self, family: str, name: str) -> None:
        """Record ``<family>:<name>``. A second FORM mark for the same field is ignored."""
        mark = repair_mark(family, name)
        if mark not in self._alphabet:
            return
        if family in FORM_FAMILIES:
            if name in self._formed:
                return
            self._formed.add(name)
        self.add(mark)

    def extend(self, marks: "Iterable[str]") -> None:
        for mark in marks:
            self.add(mark)

    def marks(self) -> "list[str]":
        """A copy — the list a result carries is never the log's own."""
        return list(self._marks)

    def __len__(self) -> int:
        return len(self._marks)


def closed_repairs(raw: Any, stage: str) -> "list[str]":
    """``raw`` reduced to the marks of ``stage``'s alphabet, in order, each once. Pure; never
    raises.

    For the reader on the far side of a carrier (a metadata key, a ledger row, a stand-in
    stage's result): whatever arrived, only members of the closed alphabet leave. A value that
    is not a list or a tuple, an unknown stage, a carrier that cannot be iterated: ``[]``.
    """
    alphabet = REPAIRS_BY_STAGE.get(stage) if isinstance(stage, str) else None
    if not alphabet or not isinstance(raw, (list, tuple)):
        return []
    out: "list[str]" = []
    for item in raw:
        if isinstance(item, str) and item in alphabet and item not in out:
            out.append(item)
    return out


def nulls_as_absent(data: Any, stage: str, applied: Any) -> Any:
    """``data`` with the ``null``-valued keys of ``stage``'s optional-key fields LEFT OUT — when,
    and only when, the reply was written under a receipt of :data:`NULL_IS_ABSENT_RECEIPTS`.

    The rule for reading a reply a mandatory-keys schema shaped: there, ``null`` under an
    optional key IS the absent key. Dropping it before the stage reads means the stage, and the
    marks, see exactly the object the model would have written without the schema — so the
    stage's decisions cannot differ, and no mark is recorded for the schema's own spelling.

    With any other receipt (``None`` — no schema travelled —, an open dialect, ``"refused"``)
    ``data`` is returned AS IS, the same object: a ``null`` there is still what it always was.
    A ``null`` at the TOP level is never touched (every top-level key is required in every
    form, so a top-level ``null`` means what it says). Pure — the input is never mutated —
    and it never raises.
    """
    if applied not in NULL_IS_ABSENT_RECEIPTS or not isinstance(data, dict):
        return data
    fields = OPTIONAL_KEY_FIELDS.get(stage, ()) if isinstance(stage, str) else ()
    out = None
    for name in fields:
        value = data.get(name)
        if isinstance(value, dict) and any(item is None for item in value.values()):
            if out is None:
                out = dict(data)
            out[name] = {key: item for key, item in value.items() if item is not None}
    return data if out is None else out


def spelled_boolean(value: Any) -> bool:
    """Is ``value`` a STRING spelling a boolean (``"true"``, ``"False"``, ``" yes "``, ``"0"``)?"""
    return isinstance(value, str) and value.strip().casefold() in _SPELLED_BOOLEANS


def reply_form_repairs(text: str, *, cot_stripped: bool) -> "list[str]":
    """What had to come off a reply before its ONE JSON object could be read — for the stages
    that read through :func:`cogno_anima.verdict.parse_object` (guard, judge, selector).

    ``text`` is the reply AFTER the chain-of-thought pass and ``cot_stripped`` is what that pass
    reported. Pure; never raises. It answers with marks only:

    * :data:`FENCE_OR_THINK_STRIPPED` when the pass removed something, or the reply is a fenced
      block;
    * :data:`SALVAGED_OBJECT` when an object WAS read and the reply (fence off) is not, by
      itself, that one object — prose around it, a list holding it.

    It never says the reply could not be read: that is the verdict reader's
    (``verdict_read``), or the selector's own :data:`UNPARSEABLE`.
    """
    marks: "list[str]" = []
    body = (text or "").strip()
    fenced = body.startswith("```")
    if cot_stripped or fenced:
        marks.append(FENCE_OR_THINK_STRIPPED)
    if fenced:
        lines = body.splitlines()[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        body = "\n".join(lines).strip()
    found, _ = parse_object(body)
    if found is None:
        return marks
    try:
        direct = json.loads(body)
    except (ValueError, RecursionError):
        direct = None
    if not isinstance(direct, dict):
        marks.append(SALVAGED_OBJECT)
    return marks


def note_text_field(log: RepairLog, data: Any, key: str, *,
                    blank_is_default: bool = True) -> None:
    """Record the form of a TEXT field a stage reads beside a verdict (a critique, a refusal
    message, the selector's ``asked``): absent → ``default``, ``null`` → ``null_field``,
    anything that is not a string → ``coerced``. A BLANK string is ``default`` too when the
    stage then falls back to a text of its own (``blank_is_default``, the critique and the
    refusal message); where an empty string is a legitimate answer (``asked``) it is no mark.
    Pure; never raises."""
    if not isinstance(data, dict):
        return
    if key not in data:
        log.field(DEFAULT, key)
        return
    value = data[key]
    if value is None:
        log.field(NULL_FIELD, key)
    elif not isinstance(value, str):
        log.field(COERCED, key)
    elif blank_is_default and not value.strip():
        log.field(DEFAULT, key)
