# NER — semantic analysis

Moved verbatim from `CLAUDE.md` on 2026-11-08: the contract did not change. `CLAUDE.md` keeps a short summary and points here. Update this file in the same pull request as any change to the NER stage, as `CLAUDE.md` used to be updated.

2. **NER** (`cogno_anima/stages/ner.py`, class `IntentAnalyzer`) — semantic analysis:
   - Requires `ctx.noumeno` to be populated first
   - Builds prompt context from `ctx.noumeno` plus carry-over metadata (`last_goal`, `last_pii_risk`, `active_domains`, `turn_number`); if `noumeno.change_subject` is true, prior context is dropped
   - Calls the LLM and parses/sanitizes the JSON response into an `IntentResult` — every field is validated/coerced against a fixed vocabulary (`VALID_INTENTS`, `VALID_SENTIMENTS`, `VALID_TEMPORAL`, `VALID_TRIAD`, `VALID_MODALITY`, `VALID_SPEECH_ACTS`, `VALID_PAROLE`, `NER_KNOWLEDGE_DOMAINS`, etc.), with fallback heuristics when the LLM omits/garbles fields (e.g., `intent_class=UNKNOWN` falls back based on `mandatory_tags`; empty `domains` falls back from `mandatory_tags` via `_TAG_TO_DOMAIN`)
   - PII detection: raw `pii` strings are normalized/aliased (`security/pii.py: normalize_pii_types`) and risk is computed deterministically (`compute_pii_risk`) — never trust the LLM's own risk judgment; a `DATE_OF_BIRTH` survives only with birth vocabulary in the text (`has_birth_context`: PT, EN and ES — `nasc*`/`nacido|nacida|nacimiento`, `idade`/`edad`, `aniversário`/`cumpleaños`, …)
   - Populates `ctx.intent: IntentResult`
