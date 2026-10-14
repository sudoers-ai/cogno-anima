# Logging — this library's convention

This library **emits** logs; the **host configures** them (handlers, format, level, tenant
context). The rules:

1. One module-level logger per module, from `logging.getLogger(...)`. No handlers, no
   formatters, no `basicConfig`, no `get_logger` of our own, no call on the root logger.
2. A message is the domain fact only, in one grammar, always lazy:

   ```
   [stage=<word> ]event=<name>( <key>=<value>)*[ — <prose>]
   ```

   `logger.info("stage=ego event=step step=%d tool=%s ok=%s side_effect=%s", …)`.
   Every `%` placeholder sits directly behind its `key=`, so every value a line carries has a
   name. The prose after ` — ` is fixed text and carries no value.
   Do NOT put a tenant id, a timestamp or a channel in the message — the host injects them
   through contextvars and a `Filter` on the root logger (stamped on every `LogRecord`).
3. **No message carries text somebody else wrote** — at any level, DEBUG included. Not the
   model's reply or a piece of it, not the contact's message, not a tool's result or an
   argument the model chose, not a value the tenant typed, not an exception's message. A line
   says HOW MUCH and WHICH KIND: a length, a count, a member of a closed vocabulary, a class
   name. The text itself is on the result the stage returns, and the host decides where it may
   go. A level is one configuration line away from production, so "DEBUG only" protects
   nothing. The two things the library does not close are written down under
   [What a line carries](#what-a-line-carries): the names the host declared, and five
   tracebacks.
4. Level control is per package: `logging.getLogger("cogno_anima").setLevel(...)`.

The host attaches the handler (its tenant filter and its formatter) to the real root logger; the
reference host documents its side in its own `LOGGING.md`.

## Level policy

- **ERROR** — never emitted. A fatal error becomes an exception and propagates; the host logs
  the ERROR.
- **WARNING** — 52 calls. A condition that was recovered or handled: a reply a stage had to
  repair or could not read, a fail-open or fail-closed path taken, a backstop that fired, a
  host carrier that could not be used, a callback that raised.
- **INFO** — 23 calls. The milestones of a turn: one `done` line per stage (`noumeno`, `ner`,
  `id`, `ego`), the EGO's `start` and each `step`, the cumulative drift, the scope verdict, the
  judge's approval, the voice, and what a tool wrapper decided.
- **DEBUG** — 2 calls: a retry the EGO loop blocked (`blocked_retry`) and the NER's domains
  fallback (`domains_fallback`). They carry what every other line carries, no more.

The judge REJECTING is a WARNING (`event=judge approved=false`), one line per rejection. It
carries the branch of the criteria, how the verdict was read, the attempt, the LENGTH of the
critique and the repair marks of the reply. It does not carry the critique: a critique is text
the model wrote about the turn, and it quotes the turn.

**`pii:flagged_in_output` is not a WARNING.** It fires on every reply that repeats the
contact's own e-mail — the RIGHT case — and a warning that shouts on the happy path is a
warning nobody reads. It stays in `adjustments`, with the same meaning; the WARNING is for what
the outgoing-PII rule WITHHELD (`pii_redacted_in_output` in `enforce` mode,
`pii_would_redact_in_output` in `observe` mode), with how many values and of which types —
never the values, not even truncated.

## Events

This table is the code's list. `tests/unit/test_logging_md_lists_what_the_code_emits.py` reads
every log call of the package by AST and compares it with the rows below, in both directions:
the logger, the event, the level, the `key=` fields and whether the line carries a traceback.

| Logger | Event | Level | When |
|---|---|---|---|
| `cogno_anima.noumeno` | `echo_backstop` (`examples`, `rule_illustrations`) | INFO | the stage was built: how many example rewrites and rule illustrations the echo backstop harvested from the prompt |
| `cogno_anima.noumeno` | `language_undetected` (`error`) | WARNING | no language was given and `langdetect` raised; the language is `und` |
| `cogno_anima.noumeno` | `few_shot_echo` (`rewritten_len`) | WARNING | the rewrite of a short reply matches one of the prompt's own examples; the call is made again without the examples |
| `cogno_anima.noumeno` | `few_shot_echo_retry_empty` | WARNING | that second call returned no rewrite; the first answer is kept, flagged |
| `cogno_anima.noumeno` | `few_shot_echo_retry_degenerate` (`retried_len`) | WARNING | the second call returned a bare or pass-through rewrite; it is adopted, flagged |
| `cogno_anima.noumeno` | `few_shot_echo_genuine` | INFO | the second call reproduced the same sentence without the examples in its context |
| `cogno_anima.noumeno` | `few_shot_echo_retry_failed` (`error`) | WARNING | the second call raised; the first answer is kept, flagged |
| `cogno_anima.noumeno` | `few_shot_echo_rule_illustration` (`rewritten_len`) | WARNING | the rewrite matches an illustration in the prompt's rules and nothing in the input or the conversation supports it; flagged |
| `cogno_anima.noumeno` | `changed_unreadable` (`read`) | WARNING | `changed` was present and was not a JSON boolean; it is recorded as False |
| `cogno_anima.noumeno` | `preserved_not_in_input` (`dropped`, `kept`) | WARNING | the rewriter listed an e-mail or a URL the contact never typed; it is dropped |
| `cogno_anima.noumeno` | `embed_unavailable` (`error`) | WARNING | the embedder raised; the similarity it was asked for is unknown and the turn goes on |
| `cogno_anima.noumeno` | `repair_marks_failed` | WARNING | the repair marks of the reply could not be computed; the turn goes on without them |
| `cogno_anima.noumeno` | `done` (`lang`, `drift`, `tag`, `changed`, `change_subject`, `degraded`) | INFO | the stage finished |
| `cogno_anima.ner` | `intent_class_fallback` (`llm=UNKNOWN\|out_of_vocabulary`, `coerced`) | WARNING | the model's `intent_class` was `UNKNOWN` or outside the vocabulary; `coerced` is the class the stage used |
| `cogno_anima.ner` | `domains_fallback` (`derived`) | DEBUG | `domains` came back empty and was derived from the mandatory tags |
| `cogno_anima.ner` | `repair_marks_failed` | WARNING | a repair mark of the reply could not be recorded; the turn goes on without it |
| `cogno_anima.ner` | `done` (`intent`, `sentiment`, `pii_risk`, `domains`, `composite`) | INFO | the stage finished |
| `cogno_anima.id` | `done` (`turn`, `route`, `goal_status`, `blocked`, `complexity`, `sim`) | INFO | the stage finished |
| `cogno_anima.drift` | `cumulative` (`cumulative`, `action`, `stages`, `scores`) | INFO | the cumulative drift was recomputed; `scores` is `<stage>:<score>` for each stage that was measured |
| `cogno_anima.ego` | `circling_streak_unusable` (`type`) | WARNING | the host's `circling_streak` is not a count; the hint is skipped |
| `cogno_anima.ego` | `circling_warned` (`streak`) | INFO | the circling hint was rendered into the task context |
| `cogno_anima.ego` | `prompt_sha_unavailable` (`reason=tools_not_json`) | WARNING | the tool schemas could not be serialised, so the attempt has no prompt digest |
| `cogno_anima.ego` | `start` (`path`, `tools`, `max_steps`, `attempt`) | INFO | the loop is about to start; `tools` is how many were offered |
| `cogno_anima.ego` | `confirmed_call_dropped` (`tool`, `reason=not_in_dispatcher`) | WARNING | a call the host marked as confirmed names a tool the dispatcher does not have this turn |
| `cogno_anima.ego` | `confirmed_exec` (`tool`, `ok`) | INFO | a confirmed call was replayed |
| `cogno_anima.ego` | `confirmed_call_still_asks` (`tool`) | WARNING | a confirmed call was run and the skill asked for confirmation again; it is failed, nothing was committed |
| `cogno_anima.ego` | `unknown_tool` (`step`, `name_len`) | WARNING | the model called a tool that is not in the catalogue; the name it wrote is on the trace, not here |
| `cogno_anima.ego` | `blocked_retry` (`step`, `tool`) | DEBUG | the model re-issued a call that had already failed with the same arguments |
| `cogno_anima.ego` | `duplicate_in_step` (`step`, `tool`) | WARNING | the same read was issued twice in one step |
| `cogno_anima.ego` | `duplicate_call` (`step`, `tool`) | WARNING | the same call was issued more often than the loop allows |
| `cogno_anima.ego` | `pending_confirmation` (`step`, `tool`) | INFO | gate B held a call the host declared as needing confirmation |
| `cogno_anima.ego` | `pending_confirmation` (`source=skill`, `step`, `tool`) | INFO | gate C: the skill ran and asked for confirmation of this call |
| `cogno_anima.ego` | `side_effect_without_success` (`tool`) | WARNING | the dispatcher stamped `side_effect` on a call that failed |
| `cogno_anima.ego` | `policy_is_mutating_failed` (`tool`) | WARNING | the host's `is_mutating` raised; the call is recorded without that fact |
| `cogno_anima.ego` | `step` (`step`, `tool`, `ok`, `side_effect`) | INFO | a tool call was executed — never its arguments, never its result |
| `cogno_anima.ego` | `done` (`steps`, `tools`, `interrupted=false`) | INFO | the loop ended by itself; `tools` is how many calls ran |
| `cogno_anima.ego` | `done` (`steps`, `tools`, `interrupted=true`, `reason`) | WARNING | the loop was cut: `reason=max_steps\|duplicate_calls` |
| `cogno_anima.superego` | `voice_traits_dropped` (`persona`, `dropped`, `kept`) | WARNING | the persona's declared traits held values the vocabulary does not have (how many were dropped, how many kept); once per configuration per process |
| `cogno_anima.superego` | `contact_state_unusable` (`persona`, `shape`, `type`) | WARNING | the host's contact-state carrier is malformed; once per shape per process |
| `cogno_anima.superego` | `hint_within_own_normal` (`dropped=tone:empathetic`, `persona`, `sentiment`, `neutral`, `n`) | INFO | the turn's negative sentiment is this contact's normal, so the empathy hint is not rendered |
| `cogno_anima.superego` | `traits_modulated` (`declared`, `effective`, `sentiment`) | INFO | the turn changed the persona's declared traits |
| `cogno_anima.superego` | `scope_verdict_unreadable` (`read`) | WARNING | the scope guard's verdict was not a JSON boolean; the turn is allowed (fail-open) |
| `cogno_anima.superego` | `scope` (`blocked`) | INFO | the scope guard answered |
| `cogno_anima.superego` | `scope_failed` (`error`) | WARNING | the scope guard's call raised; the turn is allowed (fail-open) |
| `cogno_anima.superego` | `judge_verdict_unreadable` (`read`, `branch`) | WARNING | the judge's verdict was not a JSON boolean; not approved (fail-closed) |
| `cogno_anima.superego` | `judge` (`approved=true`, `branch`) | INFO | the judge approved |
| `cogno_anima.superego` | `judge` (`approved=false`, `branch`, `read`, `attempt`, `critique_len`, `repairs`) | WARNING | the judge rejected; one line per rejection |
| `cogno_anima.superego` | `judge_failed` (`error`) | WARNING | the judge's call raised; not approved (fail-closed) |
| `cogno_anima.superego` | `execution_trace_unreadable` | WARNING | the execution trace could not be read to choose the judge's criteria; the stricter branch is used |
| `cogno_anima.superego` | `consult_trace_unreadable` | WARNING | the consulted specialist's trace could not be rendered for the judge |
| `cogno_anima.superego` | `voice_json_unwrapped` | WARNING | the voice answered inside a JSON envelope and the backstop opened it |
| `cogno_anima.superego` | `voice_diverged_from_approved_draft` (`lost`, `invented`) | WARNING | the voiced reply lost or invented values against the draft the judge approved; the voice is asked once more |
| `cogno_anima.superego` | `preserved_mutated_in_output` | WARNING | a preserved figure, e-mail or URL appears altered in the reply |
| `cogno_anima.superego` | `voice_kept_value_dropped` | WARNING | a statement the prompt ordered kept did not reach the reply with all its values |
| `cogno_anima.superego` | `voice_figure_from_critique` (`count`) | WARNING | figures in the reply come from the judge's critique and from nothing that was read |
| `cogno_anima.superego` | `pii_redacted_in_output` (`count`, `types`) | WARNING | `enforce` mode: the outgoing-PII rule masked values in the reply |
| `cogno_anima.superego` | `pii_would_redact_in_output` (`count`, `types`) | WARNING | `observe` mode: the rule decided and masked nothing |
| `cogno_anima.superego` | `voice` (`len`, `cot_stripped`, `adjustments`) | INFO | the reply was written |
| `cogno_anima.stages.scope_options` | `scope_options_failed` (`error`) | WARNING | the option selector's call raised; the refusal stands |
| `cogno_anima.stages.scope_options` | `scope_options` (`outcome`, `offered`, `discarded`, `covered_unsupported`) | INFO | the option selector answered |
| `cogno_anima.tools.composite` | `tool_name_collision` (`name`) | WARNING | two sources offer a tool of the same name; the first one wins |
| `cogno_anima.tools.confirm_args` | `confirm_args_too_many` (`tool`, `n`) | WARNING | a proposal named more confirmation arguments than the recorder keeps; they are dropped |
| `cogno_anima.tools.id_provenance` | `id_provenance_refused` (`tool`, `arg`, `id_len`) | WARNING | a guarded write named an id no read of this turn returned; `arg` is the argument's NAME |
| `cogno_anima.tools.idempotency` | `idempotency_withheld` (`tool`, `state`, `at`) | INFO | the same declared call already succeeded, or is still in flight, inside its window; it was not executed again |
| `cogno_anima.tools.idempotency` | `idempotency_store_unavailable` (`tool`, `step`, `error`) | WARNING | the store did not answer at `step=claim\|complete\|release`; the call goes on without the repeat check |
| `cogno_anima.tools.idempotency` | `idempotency_unavailable_hook_failed` (`tool`) | WARNING | the caller's `on_unavailable` hook raised |
| `cogno_anima.tools.pre_judge` | `judge_pre` (`tool`, `verdict`, `ms`) | INFO | a pre-judgement of a write finished |
| `cogno_anima.tools.pre_judge` | `judge_pre_failed` (`tool`, `error`) | WARNING | the pre-judge callback raised; the verdict is `error` |
| `cogno_anima.tools.pre_judge` | `judge_pre_enforced` (`tool`, `verdict`, `outcome`) | INFO | an enforced write waited for its judgement and was run or held |
| `cogno_anima.tools.pre_judge` | `judge_pre_confirm_not_a_proposal` (`tool`) | WARNING | the host's `confirm` returned something that is not a proposal; a neutral one is used |
| `cogno_anima.tools.pre_judge` | `judge_pre_settle_failed` | WARNING, with a traceback | the end-of-turn settling of the pending judgements raised |
| `cogno_anima.tools.pre_judge` | `judge_pre_enforce_policy_failed` (`tool`) | WARNING, with a traceback | the host's `enforce` raised; the call is not enforced |
| `cogno_anima.tools.pre_judge` | `judge_pre_confirmed_failed` (`tool`) | WARNING, with a traceback | the host's `confirmed` raised; the call is treated as not confirmed |
| `cogno_anima.tools.pre_judge` | `judge_pre_confirm_failed` (`tool`) | WARNING, with a traceback | the host's `confirm` raised; a neutral proposal is used |
| `cogno_anima.tools.pre_judge` | `judge_pre_launch_failed` (`tool`) | WARNING, with a traceback | a pre-judgement could not be started; the call runs unjudged |
| `cogno_anima.utils` | `truncated_response` (`stage`, `attempt`, `attempts`, `chars`) | WARNING | a JSON stage's reply arrived cut mid-stream and the call is made again; `stage` is the stage that asked |

`judge_pre_enforced` is emitted from two places with the same fields. It has one row.

A line of a stage opens with `stage=<name>`: `noumeno`, `ner`, `id`, `ego`, `drift`,
`superego` (the option selector logs as `stage=superego` on its own logger). A line of a tool
wrapper has no `stage=`. `cogno_anima.utils` logs on behalf of the stage that called it, so
there the stage is a value.

## Modules that log nothing

- `cogno_anima`
- `cogno_anima._version`
- `cogno_anima.errors`
- `cogno_anima.metakeys`
- `cogno_anima.preserved`
- `cogno_anima.prompts`
- `cogno_anima.repairs`
- `cogno_anima.routing`
- `cogno_anima.routing.attention`
- `cogno_anima.routing.goal`
- `cogno_anima.routing.intention`
- `cogno_anima.schemas`
- `cogno_anima.security.contact_memo`
- `cogno_anima.security.detector`
- `cogno_anima.security.pii`
- `cogno_anima.security.prompt_guard`
- `cogno_anima.security.redaction`
- `cogno_anima.stages`
- `cogno_anima.stages.base`
- `cogno_anima.stages.proposal_judge`
- `cogno_anima.tools`
- `cogno_anima.tools.base`
- `cogno_anima.tools.binding`
- `cogno_anima.tools.commit_sink`
- `cogno_anima.types`
- `cogno_anima.verdict`
- `cogno_anima.vocab`

## What a line carries

Every value a line carries is named by its `key=`, and every key has ONE class.
`tests/unit/test_logs_carry_no_third_party_text.py` reads the table below, holds every log
call's arguments to it, and pins the expression each key is given — a new key, or a new
expression under an old key, fails until it is classified here.

The classes:

- **count** — a number, a boolean or a clock time this library computed: a length, a step, a
  score, how many. Never something parsed out of text.
- **closed** — a member, or a list of members, of a closed vocabulary this library defines: a
  route, a verdict read, a repair mark, a PII type.
- **type** — a Python class name, `type(x).__name__`: of an exception, or of a carrier that
  could not be used. Never the exception's message.
- **name** — an identifier the HOST declared: a tool of its catalogue, the name of an argument,
  a persona label, a language tag. This is the one class the library does not close, and it is
  here on purpose: a name the host declared is how an operator finds the tool or the persona a
  line is about. A host that puts a person's name in a tool name or a persona label is logging
  it.
- **constant** — the key only ever appears with a literal value in the message.

There is no class for text, and no digest is logged today.

| Field | Class | What the value is |
|---|---|---|
| `action` | closed | the drift action: `none`, `warn`, `ask_user`, `self_correct` |
| `adjustments` | closed | the voice's adjustment tokens, comma-joined (`-` when none) |
| `approved` | constant | `true` or `false`, written in the message |
| `arg` | name | the NAME of the id argument the caller declared for a guarded tool |
| `at` | count | `HH:MM`, when the earlier identical call was claimed |
| `attempt` | count | the correction attempt, or the try of a truncated reply |
| `attempts` | count | how many tries a truncated reply gets |
| `blocked` | count | a boolean |
| `branch` | closed | the judge's criteria branch |
| `change_subject` | count | a boolean |
| `changed` | count | a boolean |
| `chars` | count | the length of the reply that arrived cut |
| `coerced` | closed | the intent class the stage used |
| `complexity` | closed | `LOW`, `MEDIUM`, `HIGH`, `EXPERT` |
| `composite` | count | a boolean |
| `cot_stripped` | count | a boolean |
| `count` | count | how many values |
| `covered_unsupported` | count | how many `covered` picks had no evidence |
| `critique_len` | count | the length of the judge's critique, in characters |
| `cumulative` | count | the cumulative drift |
| `declared` | closed | the persona's traits, after the sanitizer |
| `degraded` | closed | the stage's degradation tokens, comma-joined (`-` when none) |
| `derived` | closed | the knowledge domains derived from the tags |
| `discarded` | count | how many options the selector named that were not on the list |
| `domains` | closed | the knowledge domains |
| `drift` | count | the drift score, or `unknown` when it could not be measured |
| `dropped` | count | how many were dropped (written as `tone:empathetic` in one message: the hint that was not rendered) |
| `effective` | closed | the traits the turn ended with |
| `error` | type | the class of the exception |
| `examples` | count | how many example rewrites were harvested |
| `goal_status` | closed | `NEW`, `ONGOING`, `COMPLETED`, `ABANDONED` |
| `id_len` | count | the length of the id the call named |
| `intent` | closed | the intent class |
| `interrupted` | constant | `true` or `false`, written in the message |
| `invented` | count | how many values the reply invented |
| `kept` | count | how many were kept |
| `lang` | name | the language tag the host set, or the code `langdetect` returned |
| `len` | count | the length of the reply |
| `llm` | closed | `UNKNOWN` or `out_of_vocabulary` — never what the model wrote |
| `lost` | count | how many values the reply lost |
| `max_steps` | count | the loop's budget |
| `ms` | count | wall time in milliseconds |
| `n` | count | how many |
| `name` | name | a tool name, from a source's own schema |
| `name_len` | count | the length of the tool name the model wrote |
| `neutral` | count | the contact's valence average, or `?` |
| `offered` | count | how many options were offered |
| `ok` | count | a boolean |
| `outcome` | closed | the selector's outcome, or the pre-judge's (`executed`, `held`, `executed_fail_open`) |
| `path` | closed | `native` or `text` |
| `persona` | name | the host's persona label, cut at 40 characters, control characters removed |
| `pii_risk` | closed | the PII risk level |
| `read` | closed | how a verdict or a boolean was read (`VALID_VERDICT_READS`) |
| `reason` | closed | why the loop was cut (`max_steps`, `duplicate_calls`); otherwise a literal |
| `repairs` | closed | the repair marks of the reply, comma-joined (`-` when none) |
| `retried_len` | count | the length of the second rewrite |
| `rewritten_len` | count | the length of the rewrite |
| `route` | closed | the triad route |
| `rule_illustrations` | count | how many rule illustrations were harvested |
| `scores` | count | `<stage>:<score>` for each measured stage, comma-joined (`-` when none) |
| `sentiment` | closed | the sentiment, or `?` |
| `shape` | closed | `not-a-mapping` or `no-valence` |
| `side_effect` | count | a boolean, or `None` |
| `sim` | count | the goal similarity |
| `source` | constant | `skill`, written in the message |
| `stage` | closed | the stage that asked, when a helper logs on its behalf |
| `stages` | count | how many stages were measured |
| `state` | closed | the earlier claim's state: `done` or `pending` |
| `step` | count | the index of the loop's step |
| `step` in `cogno_anima.tools.idempotency` | closed | `claim`, `complete` or `release` |
| `steps` | count | how many steps the loop took |
| `streak` | count | the circling streak |
| `tag` | closed | the drift tag |
| `tool` | name | a tool of the host's catalogue |
| `tools` | count | how many tools |
| `turn` | count | the turn number |
| `type` | type | the class of the carrier that could not be used |
| `types` | closed | the PII types, comma-joined |
| `verdict` | closed | the pre-judge's verdict: `approved`, `critique`, `error`, `timeout` |

### Tracebacks

Five lines carry a traceback, all in `cogno_anima.tools.pre_judge`: `judge_pre_settle_failed`,
`judge_pre_enforce_policy_failed`, `judge_pre_confirmed_failed`, `judge_pre_confirm_failed`
and `judge_pre_launch_failed`. A traceback says what the exception says. These five are the
wrapper's own bookkeeping or a callback the HOST injected (`enforce`, `confirmed`, `confirm`),
never a model call or a tool: the stack is what tells the host which of its callbacks is
broken. A host whose callback puts an argument in its exception message is logging it.
`judge_pre_failed` — the judge callback itself, which is a model call — logs the class only.

### What is not in the log

- The judge's critique. It is on `SuperegoResult.critique`.
- The rewrite, the reply and the tail of a reply that arrived cut.
- A tool's arguments and results. A tool name the model invented is on the trace
  (`ToolExecution.tool`), with its length here.
- The trait values a persona declared and the vocabulary does not have.
  `cogno_anima.vocab.sanitize_voice_traits` returns them.
- An exception's message. `StageParseError` quotes the reply it could not read (120 characters
  of it, in its message, and the whole reply on `.raw`): the library never logs it, and a host
  that logs the exception is logging that.
