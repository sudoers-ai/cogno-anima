# The schema of a JSON stage's reply

Moved verbatim from `CLAUDE.md` on 2026-11-08: the contract did not change. `CLAUDE.md` keeps a short summary and points here. Update this file in the same pull request as any change to `cogno_anima/schemas.py`, as `CLAUDE.md` used to be updated.

### The schema of a JSON stage's reply (`cogno_anima/schemas.py`) — inert

One JSON Schema per JSON stage (NOUMENO, NER, scope guard, judge), **derived** from the closed
vocabularies the stages validate against and from the field tables of `cogno_anima.repairs`
(0.1.10). **Nothing reads it**: no module imports it, no stage passes it to a backend, no prompt
mentions it.

- **Derived in both directions.** A vocabulary is the `enum` of its field, and every `enum` is a
  vocabulary the package declares. `build_stage_schemas()` rebuilds from the vocabularies as
  they are; a test adds a word and reads it back.
- **The bridge to the repair marks.** A reply the schema accepts needs no form mark (`default`,
  `null_field`, `string_bool`, `coerced`) and never ends the turn. Declared exceptions: the
  right type with the wrong content (a range, a blank string, a duplicate).
- **Not the request a strict provider takes.** The objects are as open as the stages are.
  Closing them and requiring every key is a provider's rule, and it changes what the model is
  asked to emit.
- **No switch and no transport.** Sending a schema needs a protocol a backend can announce, a
  rule for the wrappers a host stacks around a backend, and a provider's confirmed shape. None
  of the three is decided. See `CHANGELOG.md` (0.1.10, *Not in this release*).

`tests/unit/test_schemas_are_derived.py` holds all of it, the validator included.
