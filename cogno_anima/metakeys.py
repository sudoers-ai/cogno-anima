"""Inter-repo contract keys in ``PipelineContext.metadata``.

``metadata`` is deliberately a serializable dict (multi-worker state), but the
keys that cross a repository boundary (host → soma → anima) are a CONTRACT: a
typo in a string does not fail — it silently no-ops the feature. Every call site
(across the three codebases) must import the constant from here instead of
typing the string literally.

Convention: the value is stable forever (state persisted in the session/DB
depends on it); renaming the CONSTANT is free, renaming the VALUE is a data
migration.
"""

from __future__ import annotations

# ── EGO (executor) — host/soma write, EgoStage reads ─────────────────────────
EGO_CONTEXT = "ego_context"                    # injected text (clock/memories/history)
EGO_READONLY = "ego_readonly"                  # gate A: mask mutating tools this turn
EGO_MAX_STEPS = "ego_max_steps"                # explicit loop budget (host/plan)
EGO_PERSONA = "ego_persona"                    # persona label stamped on the EgoResult
EGO_FORCE_TOOL = "ego_force_tool"              # host: this turn REQUIRES a tool execution
EGO_CONFIRMED = "ego_confirmed"                # gate B: True | collection of tool names
EGO_CONFIRMED_CALLS = "ego_confirmed_calls"    # gate B: approved calls to execute
EGO_CORRECTION = "ego_correction"              # correction loop: {reason, attempt}

# An EARLIER attempt of THIS turn committed a mutating tool, and its trace is GONE.
#
# Every consumer of :func:`types.committed_this_turn` reads the executions on the context. That
# is complete while the orchestrator accumulates them (`turn_executions`) — but there is one
# path where it cannot: the host's model-routing fallback. The first attempt can commit an
# ordinary write and then a LATER stage raise; the exception takes that context, and its
# execution record, with it. The retry is a fresh ``Host.step`` with a fresh context, and the
# write is nowhere in it.
#
# On that path only the host knows, so the host says so — and `committed_this_turn` believes it.
# The callers are enumerated ONCE, in :func:`types.committed_this_turn` — count, files and
# functions. **Do not repeat the list here.** A second copy is what four consecutive PRs of
# drift cost (#115–#118): each one corrected the NUMBER in the copies and left a sub-count or a
# missing name behind, because a test can pin a token and cannot read two hundred words of
# prose. The number itself moves — eleven at the time of writing, and it fell once by
# centralisation rather than by a net being removed. Go read it there; that is the point of
# there being one place. Cited by NAME on purpose: the line numbers a first draft
# carried were read from an UNMERGED host branch that inserts a helper above them, so they
# were already wrong for `main` — a line number in ANOTHER repository has a shorter life than
# the sentence containing it. A NAME is not immune either, and this block proved it: it cited
# `pipeline.py::_gate_commit` for a function that exists nowhere in soma (the call sits in
# `run_turn`). Grep the symbol; do not recall it. The soma's
# own comment states the invariant they rest on: *"since committed_this_turn reads every
# attempt, the 'NOTHING was committed' the voice renders as a HARD RULE is now TRUE of the whole
# turn"*. The path above is what makes that sentence false; this key is what makes it true again
# — in the one place the definition lives, instead of every caller getting it wrong alone.
#
# The last layer that RE-DERIVED the rule instead of reading it closed on 2026-08-25 (the
# owner's decision; host PR `fix/the-auditor-reads-the-declaration`): the offline promise
# auditor (`turn_audit/promises.py::committed_from_trace`) reads the stamp
# `trace["guards"]["committed"]` — written by the host's `trace.py` by CALLING the predicate,
# so it carries this key's declaration — in UNION with the execution lists. It reads a persisted
# row weeks later with no context to pass, so the stamp is the only shape the declaration can
# reach it in; the lists stay as the floor because 0 of the 277 rows on the demo box carried the
# stamp when this closed, and a visible write counts whatever the stamp says. It was TWO
# re-derivations: the host's grounding backstop was the other and is now a caller (host #429) —
# on the very turn this key describes it SUPPRESSES every rule
# (`event=grounding_suppressed reason=prior_attempt_committed`) instead of rewriting a truthful
# confirmation into a denial. Until the auditor closed, the host's OTHER offline reader (the
# grounding replay) already took the same key, and `tests/unit/test_committed_parity.py` there
# pinned the two halves reading one row two ways with an auto-invalidating message.
#
# TRUE only. Absent means "nothing to add", never "nothing was committed": a host that does not
# set it is a host whose executions are all on the context, which is the normal case.
#
# PER TURN. It describes an earlier attempt of THE TURN BEING RETRIED and must never be carried
# into the next one. The warning matters because this module tells hosts these values are
# persisted, and anima's own CLAUDE.md endorses persisting `ctx.metadata["id_state"]` — so a
# host that persists the metadata dict WHOLESALE would make the predicate answer True for every
# remaining turn of the session, permanently disarming both repairs and the semantic cache.
PRIOR_ATTEMPT_COMMITTED = "prior_attempt_committed"

# ── the DOMAIN axis: which tools change our routing, not the contact's world ──────────────
#
# Host-declared set of tool NAMES whose successful execution changes nothing the contact can
# see — a persona transfer moves the conversation between OUR personas; the contact's ledger,
# agenda and inbox are untouched. Read ONLY by `wrote_for_the_contact`; `committed_this_turn`
# deliberately ignores it (see the docstrings — they are two questions, not one).
#
# ABSENT means "declare nothing", and the predicate then answers exactly like
# `committed_this_turn`. That default is not caution, it is the only one available: the same
# value is RIGHT for one consumer and WRONG for another, so no filter can be applied by default
# to all of them. Measured 2026-09-01 — the semantic cache must keep counting a transfer (a
# cached reply replayed without the transfer promises what it does not do), while the ledger's
# `committed` stamp must stop counting it (a turn that only transferred told the ledger it wrote).
#
# The host holds ONE list and both repos read it: `cogno_host.grounding.ROUTING_ONLY_TOOLS`,
# which its own `_wrote_for_the_contact` already filters by. Same name here on purpose.
ROUTING_ONLY_TOOLS = "routing_only_tools"
# How many CONSECUTIVE turns the host's anti-repeat guard has fired on this session (repaired
# or shipped). The guard already knows the conversation is circling; before this key, only the
# turn it fired on knew — the NEXT turn started with a clean slate and re-earned the repeat.
CIRCLING_STREAK = "circling_streak"

# ── SUPEREGO (locutor) — soma/host write, voice reads ────────────────────────
VOICE_CORRECTION = "voice_correction"          # judge's final rejection: {reason}
JUDGE_CONVERSATIONAL = "judge_conversational"  # host: this turn has NO tool to execute
# THE DUTY, computed: which capabilities this turn could NOT perform, and why.
#
# A JOIN whose two sides already existed and which nobody computed — `EgoResult.tools_offered`
# (the tools the turn really put in front of the executor) against each capability's `requires`.
# The host does the subtraction and stamps the RESULT here; the judge reads it.
#
# Why the judge and not only the executor: telling the EXECUTOR "you cannot do X" is obeyed
# TRIVIALLY by a turn that has no X to call, while the judge — the one that decides whether the
# reply is honest — has no idea the capability was missing and approves a draft that claims it.
# Measured live: a persona with two read-only tools confirmed a reminder it never created, and
# the judge approved it first time with an empty critique.
#
# The value is DATA, never prose: a list of {"capability": <name>, "missing": [<tool names>]}.
# That is deliberate. The host's capability blocks are rendered PROSE into the executor's
# prompt, and `cogno_host/capabilities.py` records why the two must never meet: the word "duty"
# names two different things there, and the divergence "stops being safe the day capability
# blocks are added to the judge's prompt". Sending the FACT keeps them apart by construction.
UNAVAILABLE_CAPABILITIES = "unavailable_capabilities"
# The persona's DECLARED personality traits for this turn — a list of `vocab.VALID_VOICE_TRAITS`
# values (a comma-separated string is tolerated). Written by the host from the persona's stored
# configuration (the tenant chose them in the dashboard); read by `SuperegoStage.voice`, which
# sanitizes against the closed vocabulary, emits `trait:*` adjustments and renders a
# `# Persona traits` section in the voice prompt. DECLARED, never inferred: the parent's
# alternative — an LLM profiling the contact and rewriting rules on its own — was rejected
# (no human in the loop, no restoring force on a wrong profile, automated profiling of an
# identified person). The EGO never reads this key: traits change how the reply is SAID, and
# the executor's tools stay neutral.
VOICE_TRAITS = "voice_traits"
# The contact's emotional NEUTRAL — an exponential moving average the host keeps per identity
# (LLM-free, updated every turn from the NER sentiment): {"valence_ema": -1..1,
# "arousal_ema": 0..1, "n": turns}. Read by the voice to tell a real escalation from the
# contact's own normal: a person who complains by temperament has FRUSTRATED as a baseline, and
# reading each turn in the absolute would treat every message as an emergency — condescending,
# and it burns the persona. The voice computes the turn's DELTA against it (`vocab.
# SENTIMENT_VALENCE`) and modulates both the persona's traits and the per-turn hints
# (`SuperegoStage._modulate_traits` / `_modulate_hints`);
# the absolute reading stays as the safety floor (never humor at a frustrated contact). Below
# `vocab.CONTACT_STATE_MIN_TURNS` the state is ignored (cold start = the persona as declared).
#
# TIMING IS PART OF THE CONTRACT: the value stamped here MUST be the neutral as it stood BEFORE
# this turn — the host takes its EMA step AFTER the turn. Updating first folds this message's own
# sentiment into the baseline it is about to be compared against, which shrinks every delta
# toward zero and quietly disables the escalation branch (worst on the very first upset turn,
# which is where the whole feature is supposed to fire).
CONTACT_STATE = "contact_state"
# The judge's FINAL verdict + how many EGO attempts it took: {"approved": bool, "attempts": int}.
# Written by the orchestrator so the outcome is countable. Only rejections were ever logged (at
# WARNING; approvals at INFO, which the deployment's handlers may drop), so "no approvals in the
# log" was indistinguishable from "the judge approves nothing" — and a whole day was spent
# chasing the wrong one. A rate needs a denominator.
JUDGE_VERDICT = "judge_verdict"
# The FINAL verdict of each judged attempt, in order: [{"attempt": int, "approved": bool,
# "critique": str, "committed": bool,
# "tools": [{"tool", "args", "ok", "side_effect", "result"}]
# (+ "tools_dropped"/"tools_error" when the writer capped or degraded)}].
# `committed` — did THIS attempt successfully call a mutating tool — is the one field the
# orchestrator ROUTES on (a write no judge approved must reach a human), so it is computed
# over the full execution list and never inherits the display path's cap/truncation/failure.
# Everything else here is diagnostics.
# "tools" is what THAT attempt executed (args/result stringified and truncated by the writer):
# the orchestrator REPLACES ctx.ego_result on every retry, so a write made by a rejected
# attempt used to vanish from everything downstream while the write itself had committed —
# THIS LAYER does not undo it (a host may wire on_rollback; the entry is written before that
# hook fires, so it cannot know). Visibility only.
# Two documented holes, so a reader does not infer more than the writer records:
#  * a gate-B hold (pending_confirmation) skips the judge AND the append — a rejected-then-
#    held turn ends its ledger on approved=false while JUDGE_VERDICT says approved=true; the
#    held attempt is the final one, so its executions survive on ctx.ego_result itself;
#  * with a two-tier judge, a fast-screen REJECTION that escalated to an approving strong
#    judge is not itemized — the entry carries the verdict that stood, not the bet's cost.
# JUDGE_VERDICT above counts; this one says WHY, which is the half that
# was missing. The rejected attempts' critiques were consumed by the correction loop
# (EGO_CORRECTION.reason, overwritten each attempt) and then dropped, so the only way to
# read them after the fact was to attach a debugger to the judge — done twice in one day,
# and three wrong hypotheses were built before it. A score you cannot explain is a score you
# cannot act on. The critique is TRUNCATED by the writer: this rides in metadata that the
# host persists, and a full critique per attempt is unbounded text.
JUDGE_ATTEMPTS = "judge_attempts"

# Digests of the PII values the CONTACT themselves supplied EARLIER in this session — the
# allowlist for the SUPEREGO's outgoing-PII rule ("may come in, must not go out"). A list of
# lowercase sha256 hex strings, produced by `security.redaction.pii_digests_in` over the
# contact's own messages. The current turn is NOT the host's job: `voice()` digests
# `ctx.user_input` itself, so what rides here is strictly the session's memory.
#
# **Digests, not values, and the reason is not caution.** `metadata` is read by whatever
# serialises the turn trace, by whatever persists session state, and by any best-effort guard
# that renders its kwargs into a log line — an allowlist of VALUES would open a new store of
# personal data in the clear in order to close one leak. It also has to survive the agreed
# follow-up: once inbound turns are stored MASKED, a value-based allowlist could no longer be
# rebuilt from history, so the choice would collapse to "keep values in the clear" or "never
# mask on write". Digests keep both halves of the owner's decision available.
#
# **The ceiling, stated rather than implied: this is de-identification, not encryption.** SHA-256
# is unsalted here and a phone number's input space is enumerable, so anyone holding a digest can
# confirm a guess instantly. What it buys is that no personal datum travels in metadata because
# of this feature. Same bargain as the host's `scope_sha` (host #545). A host that PERSISTS these
# (session state) is persisting pseudonymous personal data: put them inside the identity purge.
#
# **Absent means STRICTER, never off.** With no allowlist the rule still runs and only this
# turn's own message allowlists anything, so more is masked — a typo in the key cannot silently
# disable the protection, only the recall of earlier turns. Measured against the merged bench's
# answer key (host #549, `a2aef70`), per SCENARIO: the per-turn variant breaks
# `own_email_recalled_three_turns_later` — the e-mail given at turn 1 and asked back at turn 3 —
# and no scenario in the suite is decided the other way. Injecting this is how a host stays on
# the measured side of that.
#
# PER SESSION, per contact. The host scopes it by session_id; carrying it across contacts would
# make one person's data allowlisted in another's conversation, which is the leak inverted.
PII_OUTPUT_ALLOWLIST = "pii_output_allowlist"

# `Identity.role` of the person this reply is going to ("GUEST" | staff role). Read ONLY by the
# outgoing-PII rule, which is why the name is scoped: provenance alone is under-determined, and
# the merged bench (host #549, `a2aef70`) is what proved it — its `tool_result_document_number`
# case is the tenant's own bookkeeper asking to re-read a ledger row he wrote, and the answer key
# says withholding it is WRONG. Where the value came from cannot tell that turn apart from a
# patient asking for a doctor's CPF; who is READING can, and the host already resolves it every
# turn.
#
# Absent or empty reads as GUEST — the stricter side. A host that forgets this key gets more
# masking, never a silent widening.
#
# The dissent is recorded rather than smoothed over: the bench marks that case `contested`,
# because RBAC authorised a READ and not re-emission onto a chat transport — an employee's
# WhatsApp, the Evolution database and the tenant's history all end up holding the value. This
# bit widens that on purpose; the alternative measured worse.
PII_READER_ROLE = "pii_reader_role"

# How the outgoing-PII rule ACTS: a `vocab.VALID_PII_MODES` member ("observe" | "enforce").
# ONE switch, per turn, so a deployment can graduate one tenant at a time without a redeploy.
# Absent/unknown → `vocab.PII_MODE_DEFAULT` ("observe"): a typo must never start masking a
# tenant's replies without someone having decided to. The reasoning behind the default lives with
# the constant in `vocab.py` — it was picked by arithmetic (the rule still breaks a scenario the
# bench named, against zero leaks observed in 297 production turns), not by caution.
PII_OUTPUT_MODE = "pii_output_mode"

# ── prompt provenance (HOST writes, orchestrator labels with it) ─────────────
# `{kind: sha}` — the host's digest of each prompt TEMPLATE it is running this turn, keyed by
# slot ("ego" | "voice" | "judge" | "scope"). The orchestrator copies the matching one onto
# each call's `StageMetrics.prompt_sha`, so an outcome can be grouped by the configuration
# that produced it.
#
# **The host owns this because only the host has the TEMPLATE.** By the time a prompt reaches
# the orchestrator it has been RENDERED — the host substitutes `{identity_label}` and
# `{identity_email}` into the system/scope/limits/voice slots before handing them over. Two
# things follow, and the first cut of this got both wrong by digesting the rendered text here:
#
#   * a digest of rendered text is a digest of the CONTACT, so the "same prompt" yields a
#     different sha per person — one row per conversation in a content-addressed store, which
#     is the opposite of the point;
#   * and the text behind such a digest carries a name and an e-mail, so storing it would put
#     contact PII in a table the identity purge does not know about.
#
# The template has neither problem: `{identity_label}` is literal, stable across contacts, and
# answers the question actually being asked — which prompt was this deployment running.
PROMPT_SHAS = "prompt_shas"

# ── ID / NER / NOUMENO — cross-turn carry-over (soma writes, stages read) ─────
ID_STATE = "id_state"                          # serializable IDStage state
TURN_NUMBER = "turn_number"                    # turn number (host/soma authoritative)
ATTENTION_CANDIDATES = "attention_candidates"  # AttentionFilter candidates (host)
PII_SESSION_HINT = "pii_session_hint"          # hint of prior PII in the session (host)
EMOTIONAL_OVERRIDE = "emotional_override"      # emotional override injected by the host
LAST_REWRITTEN = "last_rewritten"              # previous turn's rewrite (continuity)
LAST_CONTEXT_TURN = "last_context_turn"        # previous turn's context summary
LAST_GOAL = "last_goal"                        # previous turn's goal (NER carry-over)
ACTIVE_DOMAINS = "active_domains"              # active domains (NER carry-over)
CONVERSATION_HISTORY = "conversation_history"  # raw transcript for NOUMENO/NER

# ── session stamps (soma stamps; host/telemetry read) ────────────────────────
ACTIVE_PERSONA_ID = "active_persona_id"
ACTIVE_MCP_MODULE = "active_mcp_module"
