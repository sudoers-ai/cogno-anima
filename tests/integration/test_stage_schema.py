"""The schema switch against a REAL provider — the half no double can stand in for.

``tests/unit/test_stage_schema_switch.py`` and ``test_strict_null_is_absent.py`` pin the
MECHANISM: the schema reaches the backend's request, the prompt does not move, the record says
what was applied, and a strict-shaped reply is read with no extra mark. None of that says what a
provider DOES with the schema — whether it accepts it, and what the model then writes.

Two questions, each on the NER (the stage with optional keys, and the only one whose strict form
differs from its open one):

* the schema TRAVELS — ``schema_use.applied`` is the dialect the backend announces, on every
  call (a comparison of forms where the receipt is not 100 % is not a comparison);
* a STRICT reply needs NO FORM REPAIR — the model writes every key, ``null`` under the ones
  that do not apply, and the stage reads that with none of ``default`` / ``null_field`` /
  ``string_bool`` / ``coerced``. This is the model half of ``repairs.nulls_as_absent``.

**Cloud-only, and not run by its author.** It was written without a model. On an Ollama spec it
skips, saying so: nothing is known about a local model under ``format=<schema>`` with the NER's
schema (a constrained decode over a schema this size on a CPU runner is unmeasured, and this
suite's default model serves live traffic on the box it would run on). Point it at a provider::

    COGNO_TEST_MODEL=openai:gpt-4o-mini python3 -m pytest tests/integration/test_stage_schema.py

``n=3`` per case, because a model's reply is a sample; the input is invented.
"""

from __future__ import annotations

import pytest
from cogno_synapse import schema_dialect_of

from cogno_anima import metakeys as mk
from cogno_anima.repairs import FORM_FAMILIES
from cogno_anima.stages.ner import VALID_INTENTS, IntentAnalyzer
from cogno_anima.types import NoumenoResult, PipelineContext, StageMetrics
from tests.integration import backends

N = 3
FORM = tuple(family + ":" for family in FORM_FAMILIES)


def _ctx(switch: "dict | None") -> PipelineContext:
    text = "Marca a reunião de equipa para amanhã às 10h e manda o resumo à Marlene."
    noumeno = NoumenoResult(
        original=text,
        rewritten="Schedule the team meeting for tomorrow at 10am and send Marlene the summary.",
        context_turn="", language="pt", canonical_language="en", drift_score=0.0,
        drift_tag="REWRITTEN", changed=True, confidence=0.9, change_subject=False,
        subject_similarity=None, context_used=False, preserved_terms=[], rewrite_warnings=[],
        metrics=StageMetrics(stage="noumeno", elapsed_ms=0.0, tokens_in=0, tokens_out=0,
                             model="fixture"))
    ctx = PipelineContext(user_input=text, noumeno=noumeno)
    if switch is not None:
        ctx.metadata[mk.STAGE_SCHEMA] = switch
    return ctx


async def _backend_for(form: str):
    spec = backends.model_spec("COGNO_NER_MODEL", backends.model_spec())
    if backends.is_ollama(spec):
        pytest.skip(f"{spec}: unmeasured on a local model — nothing is known about format=<schema> "
                    "with the NER's schema there; point COGNO_TEST_MODEL at a provider")
    reason = await backends.backend_unavailable_reason(spec)
    if reason:
        pytest.skip(reason)
    backend = backends.json_backend(spec)
    dialect = schema_dialect_of(backend, form)
    if dialect is None:
        pytest.skip(f"{spec}: this backend announces no dialect for the {form!r} form")
    return backend, dialect


@pytest.mark.asyncio
@pytest.mark.parametrize("form", ["open", "strict"])
async def test_the_schema_travels_and_the_receipt_says_so(form):
    backend, dialect = await _backend_for(form)
    for _ in range(N):
        ctx = _ctx({"ner": form})
        await IntentAnalyzer().process(ctx, backend)
        assert ctx.intent.metrics.schema_use == {"requested": form, "applied": dialect}
        assert ctx.intent.intent_class in VALID_INTENTS
    # the control: the same turn with the switch off records nothing
    off = _ctx(None)
    await IntentAnalyzer().process(off, backend)
    assert off.intent.metrics.schema_use is None


@pytest.mark.asyncio
async def test_a_strict_reply_needs_no_form_repair():
    backend, dialect = await _backend_for("strict")
    for _ in range(N):
        ctx = _ctx({"ner": "strict"})
        await IntentAnalyzer().process(ctx, backend)
        assert ctx.intent.metrics.schema_use == {"requested": "strict", "applied": dialect}
        form_marks = [mark for mark in ctx.intent.repairs if mark.startswith(FORM)]
        assert form_marks == [], form_marks
