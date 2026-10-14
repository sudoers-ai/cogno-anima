# Logging — this library's convention

This library **emits** logs; the **host configures** them (handlers, format, level, tenant
context). The rules:

1. Use `logging.getLogger(__name__)` at the top of the module. No handlers, no formatters, no
   `basicConfig`, no `get_logger` of our own.
2. A message is the domain fact only, as `key=value`, always lazy:
   `logger.info("stage=ner route=%s pii_risk=%s", route, risk)`.
   Do NOT put a tenant id, a timestamp or a channel in the message — the host injects them
   through contextvars and a `Filter` on the root logger (stamped on every `LogRecord`).
3. Levels:
   - **ERROR** → never here; a fatal error becomes an exception and propagates (the host logs
     the ERROR).
   - **WARNING** → a condition that was recovered or handled (a fallback, a parse coercion, a
     verification that failed).
   - **INFO** → a milestone of the turn.
   - **DEBUG** → the full-fidelity trace (prompt, raw reply, scores). DEV-ONLY, never switched on
     in multi-tenant production. Redact secrets (API keys).
4. Level control is per package: `logging.getLogger("cogno_anima").setLevel(...)`.

The host attaches the handler (its tenant filter and its formatter) to the real root logger; the
reference host documents its side in its own `LOGGING.md`.

## What is specific to cogno-anima

One logger per module (`cogno_anima.stages.ner`, `.id`, `.ego`, …). Signal the **cognitive
decisions and the degradations**:

- **WARNING** — a fallback coercion (the model returned JSON a stage had to repair or could not
  read: `scope_verdict_unreadable`, `judge_verdict_unreadable`); in the EGO, a duplicate call
  blocked (`duplicate_call`, `duplicate_in_step`), a tool the model invented (`unknown_tool`) and
  a loop that ended `interrupted=true`;
  in the SUPEREGO, the judge rejecting (`event=judge approved=false`, with the first 80 characters
  of its critique — the one line here that carries model-written text) or failing closed, the
  scope guard failing open, and the output backstops: `preserved_mutated_in_output`,
  `voice_json_unwrapped` (the voice answered inside a JSON envelope and the backstop opened it —
  count these: a net nobody counts becomes the mechanism), and the outgoing-PII rule.
  The outgoing-PII line records **how many** values the rule withheld and of which **types** —
  never the values, not even truncated: a log line is one of the transports that has already
  leaked personal data, and masking a reply only to reprint it in the log protects nobody. In
  observation mode the rule decided and masked nothing; it is that count, with the class
  (`pii:withheld_<type>`), that a person reads before moving a tenant to `enforce`.
  **`pii:flagged_in_output` is not a WARNING.** It fires on every reply that repeats the
  contact's own e-mail — the RIGHT case — and a warning that shouts on the happy path is a
  warning nobody reads. It stays in `adjustments`, with the same meaning; the WARNING is for what
  the rule WITHHELD.
- **INFO** — one line per stage per turn: the NOUMENO (`lang`, `drift`, `tag`), the NER
  (`intent`, `sentiment`, `pii_risk`, `domains`), the ID (`route`, `goal_status`, `blocked`), the
  EGO's start, each of its steps (`tool`, `ok`, `side_effect` — never the arguments) and its
  end, the cumulative drift, the scope verdict, the judge's approval and the voice.
- **DEBUG** — two lines today: a retry the EGO loop blocked (`blocked_retry`) and the NER's
  domains fallback (`domains_fallback`).

The rewritten text and the user's input are user content (personal data): DEBUG only, dev-only.
