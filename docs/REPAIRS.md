# Repairs of a model's JSON reply

Moved verbatim from `CLAUDE.md` on 2026-11-08: the contract did not change. `CLAUDE.md` keeps a short summary and points here. Update this file in the same pull request as any change to `cogno_anima/repairs.py`, as `CLAUDE.md` used to be updated.

### Repairs of a model's JSON reply (`cogno_anima/repairs.py`)

Five stages ask a model for a JSON object and make the reply usable by hand: the NOUMENO, the
NER, the scope guard, the judge and the scope-option selector. **`cogno_anima.repairs` counts
what each of them had to repair (0.1.9). It decides nothing.**

- **Where.** A `repairs` list on `NoumenoResult`, `IntentResult`, `ScopeCheckResult` (also
  `ctx.metadata[mk.SCOPE_REPAIRS]`, because the orchestrator drops that result),
  `SuperegoResult` (filled by `evaluate`, `[]` on `voice`) and `OptionSelection` (also in its
  `record()`).
- **What a mark is.** A constant, or `<family>:<field>` from a closed family and the NAME of a
  field the stage reads. Never a byte of the reply, so the list is safe to persist.
- **The alphabet is per stage and derived.** One table of `family -> fields` per stage, in the
  module; `REPAIRS_BY_STAGE` is keyed by `StageMetrics.stage`. The module is the enumeration —
  this section does not copy it. `RepairLog` drops a mark the stage's table does not name, and
  `closed_repairs(raw, stage)` does the same for a reader on the far side of a carrier.
- **`coerced` is the strict question**: would a closed enumeration have refused this value? A
  lower-case or aliased member is a mark even though the stage recovers it.
- **The guard and the judge do not repeat `verdict_read`.** How the verdict key was read is
  that field's (`unparseable`, `string_bool`…). The list says what came off the reply around it
  and what the text beside the verdict needed. `unparseable` is in ONE alphabet, the
  selector's, and it is the verdict reader's own token.
- **Not a repair**: what a stage decides from values it could read (grounding, the PII union,
  the anaphora flip, the composite/sequential and drift reconciliations).
- **The marking is a separate function per stage** (`_reply_repairs` in `stages/noumeno.py`,
  `stages/ner.py`, `stages/scope_options.py`), fenced: if it raises, the turn continues and a
  WARNING says so. The two NER fallbacks are marked at the branch that takes them.

**The one read that changed.** A JSON `null` in the NOUMENO's `rewritten` or `context_turn`
raised `AttributeError` and ended the turn. It now reads as the absent key always did, marked
`null_field:<field>`.

**What a stage REFUSES is typed, and named (0.1.12).** The NOUMENO's contract is "a usable
object OR `StageParseError`". For an object with a field in a form the stage has no repair for,
it was neither.
- Measured over 237 replies (each of the six fields under 38 values): 102 ended the turn, with
  `AttributeError` (50), `TypeError` (31), a pydantic `ValidationError` (12) or a bare
  `ValueError` (9). None with `StageParseError`.
- `_unreadable_field` (`stages/noumeno.py`) now refuses those replies where they are parsed.
  `StageParseError.mark` says which field in what form, from the closed table
  `repairs.NOUMENO_UNREADABLE_FIELDS`. It is the two halves of a repair mark and never a value.
- A mark on the error is a refusal, not a repair. It is never on a result's `repairs` list.
- The primary call and the echo retry are held to it alike. A retry the stage cannot use keeps
  the first answer, flagged, where it used to be adopted and then end the turn.
- What the stage has always read is still read: the other 135 replies give the same result.
- **The NER's two lists (0.1.13).** A `mandatory_tags` or `pii` that is `null`, a number or a
  boolean — the JSON values that cannot be iterated — ended the turn with a raw `TypeError` (14
  of 603 in the 0.1.9 sweep, 16 of the 0.1.10 schema sweep). `ner._iterable_or_absent` reads
  them as the absent key (`NER.UNKNOWN`; the PII detector alone), marked `null_field:<field>`
  (a row both fields gained in `NER_REPAIR_FIELDS`) or `coerced:<field>`. A string or an object
  is read as it was. `tests/unit/test_ner_reads_a_null_list_as_absent.py`.

**A `null` beside a verdict is absent (0.1.12).** The judge's `critique` and the guard's
`refusal_message` were read with `str(data.get(key, ""))`, so a JSON `null` became the text
`"None"`: the reason handed to the executor's retry, or what a contact would be told.
`_text_beside_verdict` (`stages/superego.py`) reads a `null` as the absent key always was. The
mark `null_field:<field>` stays on `repairs`. Any other non-string value is rendered as before
and counted as `coerced:<field>`.

**Measured, no model.** `tests/unit/test_repairs_red_twins.py` is the defect per stage (7 red on
`d15a6ea`); `tests/unit/test_repairs_are_counted.py` drives every mark of every alphabet and
proves that switching the marking off changes nothing but the list. The RATE of repair per
stage is a number only traffic gives.
