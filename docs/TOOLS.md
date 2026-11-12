# Tool dispatch adapters

Moved verbatim from `CLAUDE.md` on 2026-11-08: the contract did not change. `CLAUDE.md` keeps a short summary and points here. Update this file in the same pull request as any change to `cogno_anima/tools/`, as `CLAUDE.md` used to be updated.

### Tool dispatch adapters (`cogno_anima/tools/`)

`ToolDispatcher`/`ToolPolicyDispatcher` (`tools/base.py`) is the contract; `CompositeDispatcher`
(`tools/composite.py`) is the ROUTER that unions many sources behind it. The rest of the package is
the **wrapper** side of the same contract — the generic, business-free adapters a host stacks around
the composed dispatcher. They were written in the host, one per live incident, and moved here
because a wrapper that implements this protocol belongs beside the protocol: each one is a rule
that every wrapper author was otherwise re-deriving alone, and the hand-written copies are the ones
that failed. **What did NOT move is every policy that names a tool** — which writes ask before they
commit, which are exempt, which roles see what, which ids are guarded. That is the host's catalog
and its business; the core ships the mechanism and takes the declaration as a parameter.

- **`bind_delegated(wrapper, inner, *members)`** (`tools/binding.py`) — bind DELEGATED protocol
  members onto the INSTANCE, conditionally, so `isinstance(x, ToolPolicyDispatcher)` keeps telling
  the truth about the source underneath. Three shapes fail and this is the one that does not:
  `__getattr__` alone is invisible to Python 3.12's `getattr_static` (the wrapper leaves the probe
  and gate B silently stops firing); declaring on the class LIES for a source with no policy (a
  safety wrapper arming a gate over guesses); `__slots__` is the same lie with a descriptor, and it
  reads True on 3.10 **and** 3.12 — the older, quieter trap, only ever hidden by `__getattr__`
  failing the other way first. A member the wrapper's own class defines is never overwritten, so a
  wrapper that TIGHTENS `requires_confirmation` cannot be disarmed by forwarding the inner's.
- **`CommitRecordingDispatcher` + `new_sink()`/`committed()`** (`tools/commit_sink.py`) — records a
  turn's commit (`ok` AND `side_effect`, the same pair `committed_this_turn` requires) into a
  container the CALLER pre-placed. Two facts make the shape: a turn's metadata is copied shallowly on
  the way in, so a pre-placed object is the same object all the way down while a key stamped from
  inside never comes back **and fails silently**; and the record has to land in `execute`, not in a
  post-turn hook, because the branch it exists for is "attempt 1 wrote, a later stage raised" — the
  branch where a hook after the loop never fires at all.
- **`ConfirmArgumentRecordingDispatcher` + `call_key`/`held_calls`** (`tools/confirm_args.py`) — the
  RETURN half of the EGO's gate C. A tool that asks about THIS call names what it needs back; the
  recorder captures it keyed by `(tool, arguments)` (so a step with two proposals keeps them apart)
  and `held_calls` merges it into the replay, **only ever ADDING** — the proposal named a row and the
  commit must be that row. Read duck-typed off a `confirm_arguments` attribute that `ToolResult`
  deliberately does not declare: what a skill needs in order to commit is the skill's business, so the
  carrier belongs to the bridge (`cogno_mcp.MCPToolResult`). `held_calls`' keyword has no default on
  purpose — forgetting the return trip is otherwise invisible until production.
- **`IdProvenanceDispatcher`** (`tools/id_provenance.py`) — refuses (recoverably) a guarded write
  whose id did not appear in a successful read EARLIER IN THE SAME TURN, so a stale id recalled from
  an old session's memories cannot reach a write. `guarded` (`tool → (id argument, the READ that
  legitimises it)`) is INJECTED with no default: which tools carry ids, and which read names them, is
  the caller's catalog — and the read tool is part of the REFUSAL, so naming one the persona does not
  have turns a self-correction into a loop. Build a fresh instance per turn.
- **`IdempotentDispatcher` + `IdempotencyStore`/`InMemoryIdempotencyStore`/`idempotency_key`**
  (`tools/idempotency.py`) — the same effect never goes out twice without saying so. Before a
  DECLARED call runs, the digest of `(scope, tool, normalised arguments)` is CLAIMED atomically in a
  store; when the same key already succeeded inside the tool's window the call is NOT executed and
  the executor gets `ok=True, side_effect=False` with a note saying when (so `committed_this_turn`
  and its family read it as what it is — this turn wrote nothing — and the contact is always told,
  never blocked in silence). The key is EXACT: normalisation drops only what carries no meaning
  (key order, whitespace runs, NFC, `None`, `150.0`), case and every word stay, so one word changed
  is another message. Claim before, **complete** on `ok ∧ side_effect`, **release** when nothing was
  committed (`ok=False`, no write, a gate-C proposal), and **keep PENDING when the call raised** —
  the outcome is unknown, so an identical call inside the window is told it *may* have gone out and
  is not run (at most once; the pending claim expires with the window). A dead store fails OPEN by
  default (`event=idempotency_store_unavailable` + the caller's `on_unavailable` hook), because a
  side table must not switch off every write — **unless the tool's rule sets `fail_closed=True`**
  (2026-11-12): then a store it cannot reach makes the call REFUSE instead
  (`ok=False, side_effect=False`, `event=idempotency_refused_unavailable`, the contact told to
  retry via `closed_notice`/`default_closed_notice`), because a silently repeated financial entry
  or appointment is worse than a retryable refusal. `rules` (which tools, which window, optionally
  which arguments, optionally `fail_closed`), `scope` (an empty one DISABLES the guard rather than
  merging contacts) and `notice`/`closed_notice` (the sentences in the tenant's language) are all
  injected — the caller's catalog. **What the guard does not offer is a way to repeat inside the
  window**: a "shall I send it again?" whose "yes" lands on the same key would be answered "already
  done" again, so the note names the exit that works (change what is sent, or let the window pass);
  an explicit repeat needs the gate-C return trip and is a separate change. A caller CAN,
  deliberately, un-remember one key with **`IdempotencyStore.forget(key)`** — it erases a key
  whatever its state, the `DONE` row included (which `release` never does), for a confirmed "yes,
  this one is a DIFFERENT one" on a write where two identical calls can both be legitimate (a
  ledger entry); the next identical call then runs once. The guard never calls it itself.
  Implementers of `IdempotencyStore` add `forget(key)` (a new protocol member; the bundled
  in-memory store has it). Build a fresh instance per turn; the store is shared.
- **`PreJudgeDispatcher` + `PreJudgeSink`/`Proposal`/`PreJudgment`** (`tools/pre_judge.py`, F2.3a) —
  the judge reads a WRITE before it goes out, **in SHADOW**: it changes nothing. For every call the
  source underneath classifies as a write (`is_mutating`; a source with no policy is never judged —
  the core does not guess, and judging every call would spend a model call on every read), an
  INJECTED `judge` callback is started over the PROPOSAL (tool + a deep COPY of the arguments the
  executor chose — a layer below may edit them in place) **beside the call, never in front of it**:
  `execute` does not await it, does not read it and returns exactly the inner's result, exception
  included. The verdict lands in a caller-placed sink from a closed alphabet
  (`PRE_VERDICTS = approved|critique|error|timeout`) with **no text** — a critique quotes the
  arguments, and the arguments are contact data — plus the tool, the wall ms and `committed`
  (`ok ∧ side_effect` of the call's own result, `None` when it raised): the facts the activation
  decision needs (agreement with the post-execution judge; writes the pre-judge would have stopped
  that went out anyway). The cost is a `StageMetrics` whose stage the WRAPPER forces to
  `JUDGE_PRE_STAGE = "judge_pre"` whatever the callback labelled it, so a ledger grouping by stage
  books it on its own line and can never fold it into the judge's. The ceiling is a timer beside the
  task (not `wait_for`, which before 3.12 runs the judge as a second task and makes an answered
  verdict need several loop turns); `settle(grace_s=0.0)` at the end of the turn yields one loop
  turn, waits at most `grace_s`, then CANCELS the stragglers and files them as `timeout` — a task
  that outlives its turn is a model call nobody accounts for, and grace 0 means the shadow never
  delays a reply. A judgement cut AFTER its callback handed the prompt over is charged the
  callback's estimate (`Proposal.note_prompt(n)`, called BEFORE the await — since 0.1.4
  `note_prompt(n, prompt_sha)`, so the estimated row carries the digest of the template that was
  SENT, and `ProposalJudge` stamps `prompt_digest(_SYSTEM, <its # Decide block>)` on every row it
  builds: these rows are filed by the host and pass through no orchestrator's stamp, so only the
  layer that authors the text can label them; `ProposalJudge` uses
  `estimate_prompt_tokens`, len//4, declared) on its own ledger line, `JUDGE_PRE_ESTIMATED_STAGE =
  "judge_pre:estimated"` — the provider bills a request it received, and a 0 there makes the
  activation's cost per turn come out low (F2.3a-bis). A callback that never calls the hook, or a
  judgement cut before its first step, records 0 as before (unknown; the `timeout` beside it is
  what makes that readable); an errored one records what the callback measured. The callback this
  package ships is `cogno_anima.stages.ProposalJudge` (`stages/proposal_judge.py`): criterion #1 of
  the judge asked of the CALL — the contact's message, the reply a "yes" answers, the tool's own
  description and the fenced arguments; no rules, no tool results (named in its prompt: an id the
  executor READ is not wrong by itself) — and STRICT about the verdict (only a JSON boolean counts;
  a string `"false"` is `error`, never coerced — since 0.1.3 through `cogno_anima.verdict.read_verdict`,
  the one reader the judge and the scope guard share, which also refuses a verdict key written twice). **F2.3a-v2 gave it the context the first replay
  showed it lacked** — of 43 writes labelled right it rejected 20, every one for want of it — as
  four OPTIONAL inputs, each fenced as the tenant's data and each rendering its own rule inside
  `# Decide` only when given: `now` (the HOST's clock in the tenant's zone, never this library's —
  relative dates are resolved against it), `persona` (the running persona's purpose: an action that
  is its job, with values given in passing, is what was asked), `personas` (`PersonaCard` id → name
  → purpose, the only transfer targets: moving the conversation to the persona the user NAMED, or
  the owner of the task, IS the request; to any other is a different action) and
  `facts_not_wording` (a free-text argument is judged by the facts it states, never its phrasing).
  Without them the prompt is byte for byte the first cut's, pinned by digest with a control
  (`test_pre_judge_context.py`). It lives apart from `superego.py` and shares only
  its parser, so no rendering of the post-execution judge moves by a byte. **Enforcement is
  opt-in and per tool (F2.3a-on)**: with `enforce=<(tool) -> bool>` (host-injected — the core names
  no tool) a WRITE it names WAITS for its judgement (`DEFAULT_ENFORCE_TIMEOUT_S = 8.0`) and
  `approved` runs it, `critique` does NOT — the executor gets the host's `confirm` PROPOSAL (gate C,
  replaced by a neutral one if it is not a proposal, so a held call can never read as a write) —
  and `error`/`timeout` run it FAIL OPEN, counted (`PRE_OUTCOMES = executed|held|executed_fail_open`,
  two more keys on that record only). `confirmed(tool, arguments)` lets a call the contact already
  confirmed run unjudged, asked with the call's own arguments so a yes to one object never covers
  another — and it is REQUIRED whenever `enforce` names a tool: it is gate C's return trip, and
  without it the confirmed replay is judged again, so a repeated critique makes the EGO's
  `_refuse_if_still_asking` fail the call the contact confirmed. One judgement per call; without
  `enforce` the wrapper is the shadow byte for byte (`test_pre_judge_enforce.py`, `docs/ACT_CONFIRM_READONLY.md` § enforcement). Build a fresh
  instance and sink per turn.

The two recorders, the idempotency guard and the pre-judge bind their policy conditionally (they add
no verdict of their own, so claiming one would be a lie about the source); the provenance guard and the router DECLARE it, answering the EGO's
own fail-safe defaults, because both must answer on behalf of a source that may have none.
`tests/unit/test_protocol_probe_contract.py` derives the wrapper list from the package and fails a new
wrapper that arrives without the binding — the rule is pinned by mechanism, not by a hand-written list.
