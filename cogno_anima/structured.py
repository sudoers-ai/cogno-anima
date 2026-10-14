"""
cogno_anima.structured — the SWITCH that lets a stage ask its backend to carry a JSON Schema.

Four stages read a JSON object out of a model's reply: the NOUMENO, the NER, the scope guard and
the judge. :mod:`cogno_anima.schemas` derives the schema of each reply; ``cogno-synapse`` can send
one with the request (``StructuredBackend``). This module is what sits between them: it reads the
per-turn switch, builds what a stage hands to the funnel (:func:`cogno_anima.utils.call_json_stage`)
and shapes the record a stage leaves on its ``StageMetrics``.

**OFF BY DEFAULT, and off is byte for byte.** The switch is ``ctx.metadata[mk.STAGE_SCHEMA]``, a
mapping ``{stage: "open" | "strict"}`` the HOST stamps per turn — this library never reads the
environment. With the key absent, or holding anything else for a stage, :func:`schema_ask` answers
an ask that is OFF: no schema is derived, the backend is not probed, and the call the stage makes
is the ``backend.generate(system, prompt)`` it always made.

**ON never changes a byte of the prompt.** The schema travels outside ``system`` and ``prompt``;
both are the same bytes with the switch off, ``open`` or ``strict``.

**The record says both sides**: ``{"requested": <form>, "applied": <receipt>}``. ``requested`` is
the form the host asked for; ``applied`` is what the backend that SERVED the call says travelled —
a dialect, ``"refused"``, or ``None`` (the serving backend has no dialect for that form, so the
call went as it always did). A stage that was not asked for a schema records nothing (``None``),
which is a different fact from "asked, and it did not travel".
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from cogno_synapse import SCHEMA_FORMS, SCHEMA_RECEIPTS

from cogno_anima import metakeys as mk

__all__ = [
    "REQUESTED",
    "APPLIED",
    "SchemaAsk",
    "closed_schema_record",
    "requested_form",
    "schema_ask",
]

#: The two keys of the record a stage leaves on ``StageMetrics.schema_use``.
REQUESTED = "requested"
APPLIED = "applied"


def requested_form(metadata: Any, stage: str) -> Optional[str]:
    """The form the host asked for ``stage`` this turn, or ``None`` — OFF.

    ``None`` for everything that is not exactly a member of ``cogno_synapse.SCHEMA_FORMS`` under
    this stage's own label: no key, a key that is not a mapping, another stage's row, ``True``,
    ``"STRICT"``, ``"on"``. A switch that a typo could turn on is not a switch. Never raises.
    """
    if not isinstance(metadata, Mapping):
        return None
    table = metadata.get(mk.STAGE_SCHEMA)
    if not isinstance(table, Mapping):
        return None
    form = table.get(stage)
    return form if isinstance(form, str) and form in SCHEMA_FORMS else None


class SchemaAsk:
    """What ONE stage call hands the funnel, and what came back.

    ``schema``/``name``/``form`` are the three keywords of
    :func:`cogno_anima.utils.call_json_stage`; ``schema`` and ``form`` are both ``None`` when the
    ask is OFF. ``applied`` is the receipt of the LAST call made with this ask (a stage that
    calls twice — a truncation retry, the NOUMENO's second reading — keeps the last, like
    ``StageMetrics.system_fingerprint``), and ``called`` says a call completed at all.
    """

    __slots__ = ("schema", "name", "form", "applied", "called")

    def __init__(self, name: str, schema: "Optional[Mapping[str, Any]]" = None,
                 form: Optional[str] = None) -> None:
        self.name = name
        self.schema = schema if form is not None else None
        self.form = form if schema is not None else None
        self.applied: Optional[str] = None
        self.called = False

    @property
    def on(self) -> bool:
        return self.form is not None

    def kwargs(self) -> "dict[str, Any]":
        """The funnel's three keywords, as this ask holds them."""
        return {"schema": self.schema, "name": self.name, "form": self.form}

    def note(self, applied: Optional[str]) -> None:
        """Record the receipt of a call that COMPLETED. Last call wins, ``None`` included."""
        self.applied = applied
        self.called = True

    def record(self) -> "Optional[dict[str, Optional[str]]]":
        """``{"requested", "applied"}`` for ``StageMetrics.schema_use`` — or ``None`` when the
        stage was not asked for a schema, or no call completed (nothing can be said of a call
        that raised)."""
        if not self.on or not self.called:
            return None
        return {REQUESTED: self.form, APPLIED: self.applied}


def schema_ask(metadata: Any, stage: str) -> SchemaAsk:
    """The ask of ``stage`` for THIS turn: ON with the stage's schema in the requested form, or
    OFF. Never raises.

    OFF derives nothing — :func:`cogno_anima.schemas.schema_for` is not called, and its module is
    not even imported here — and probes nothing. A stage with no schema (the selector, the
    pre-judge) is OFF whatever the switch says.
    """
    form = requested_form(metadata, stage)
    if form is None:
        return SchemaAsk(stage)
    # Imported HERE: ``schemas`` imports the stages (it derives from their tables), so a stage
    # cannot import it at module level — and an OFF turn has no business importing it at all.
    from cogno_anima.schemas import schema_for
    schema = schema_for(stage, form)
    if schema is None:
        return SchemaAsk(stage)
    return SchemaAsk(stage, schema, form)


def closed_schema_record(raw: Any) -> "Optional[dict[str, Optional[str]]]":
    """``raw`` reduced to a record of the two closed alphabets, or ``None``. Pure; never raises.

    For the reader on the far side of a carrier (a trace writer, a ledger row): ``requested``
    must be a member of ``SCHEMA_FORMS`` or there is no record; ``applied`` is a member of
    ``SCHEMA_RECEIPTS`` or ``None``. Nothing else leaves — not a third key, not a string of the
    carrier's own.
    """
    if not isinstance(raw, Mapping):
        return None
    requested = raw.get(REQUESTED)
    if not isinstance(requested, str) or requested not in SCHEMA_FORMS:
        return None
    applied = raw.get(APPLIED)
    return {REQUESTED: requested,
            APPLIED: applied if isinstance(applied, str) and applied in SCHEMA_RECEIPTS else None}
