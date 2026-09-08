# Design note — act_confirm → EGO read-only + confirmation gate (propose/commit)

**Status:** IMPLEMENTED. **Three** confirmation gates in the EGO — A and B host-driven and
validated end-to-end in CognoBench as HARD invariants (EGO 100%, see
`cognobench/EGO_BENCH_RESULTS.md`), C raised by the SKILL itself. This note records the
rationale + the decisions taken on the original open points. It was written when there were two
and the third arrived later; §"Fonte A vs Fonte B vs Fonte C" below is the current list, and
`CLAUDE.md` is the shorter statement of the same three.

## Problem

A tentative or question-framed action — "I think I *maybe* spent 30 on coffee?"
(`modality=UNCERTAIN`) or "*Do you think* I should record 50?"
(`speech_act=INTERROGATIVE`) — should not silently fire a side-effecting tool.

The first attempt (Block 3) added an **advisory hint** in the EGO task-context
("confirm before firing a side-effecting tool"). CognoBench showed this fails:
`mistral:latest` on the text-fallback path ignores the hint because the host's
execution prompt ("for ANY data operation you MUST call the tool") and the core's
`force_first` (forces a tool on iteration 1 for `ACTION_REQUEST`) are stronger,
system-level signals. **You cannot ask an executor to "execute, but maybe don't"
— it's a contradictory instruction.** See `cognobench/EGO_BENCH_RESULTS.md`.

## The reframe

It is not *execute vs. don't execute* — it is **execute in a read-only
capacity**. An executor with a restricted toolset is coherent: the EGO still does
its job (gather, propose), but the **mutating tools are masked off**. This turns a
hopeful soft hint into an enforceable **capability gate**: the model cannot call a
tool that is not in its toolset.

```
turn 1 (tentative):  ID detects act_confirm → routes to EGO with ego_readonly
                     → EGO sees ONLY read/query tools → consults + PROPOSES
                     → "Want 13:00 or 15:00?"
turn 2 (confirmed):  normal ACTION_REQUEST → full toolset → mutation executes
```

This is essentially a **dry-run / propose mode**; the actual mutation happens on
the next turn, after the user confirms.

### What if the next turn is about something else?

The two-turn drawing above is the shape of the handshake, **not a promise that the
answer arrives immediately**, and reading it as one cost a real conversation. The
core is stateless: a hold lives in `EgoResult.pending_confirmation` for exactly one
turn, and whether it exists on the NEXT turn is entirely the **host's** doing —
`ego_confirmed` is something a host decides to stamp. Measured in `cogno-host` on
2026-09-05 over a 13-turn replay: a proposal ("Cadastro de Empresa — Padaria Sol
Nascente. Posso seguir?"), one unrelated question from the contact ("Ah, antes disso:
vocês atendem no sábado de manhã?"), then "Sim, pode seguir." — and the host's session
state was rebuilt from scratch each turn, so by the third message nothing was held.
The "sim" ran as an ordinary turn and the model answered *"Vou cadastrar…"* with zero
tool calls. **The contact heard a promise and nothing was written.** Zero writes in
thirteen turns.

So the drawing needs a third line, and it belongs to the host:

```
turn N   (proposal):   EGO holds → host asks "may I go ahead?" and PERSISTS the hold
turn N+k (any other):  the hold is CARRIED, untouched — its clock does not restart
turn N+k (agreement):  stamp `ego_confirmed` → the EGO executes it
```

**Two properties, and they pull in opposite directions — which is why one rule
decides both.** A proposal SURVIVES a turn that is about something else, so the
contact may ask their side question and still say yes afterwards. And a hold ENDS: it
executes, or it is dropped. It never re-asks.

That second half is not a preference, it is a measurement. The first repair in
`cogno-host` (#719) carried the hold but required the agreement to be adjacent to the
proposal — any other distance RE-PROPOSED the same question instead of committing.
The property it was defending sounds right (*nothing is executed that was not
re-proposed*) and its implementation has no fixed point: a re-proposal is itself an
interruption, the contact answers it with something else, the distance grows back, and
the next agreement re-asks again. Measured over the same thirteen turns: **zero writes,
the hold still open at the end, and the conversation blocked** — the persona answered
every message with the same proposal. It was reverted (#724) and re-landed without the
re-ask. What bounds a carried hold instead are two limits, one per axis: a maximum
number of turns it may ride unanswered, and a wall clock that a passing turn does not
restart.

A hold is also **not** an answer to every affirmative-looking message. Two shapes were
measured wrong in both directions and both are the host's to close: a message naming a
time the held call does not have is a COUNTER-PROPOSAL, not a go-ahead; and a request
for another persona by name ("Perfeito. Agora me transfira de volta para a Pam") is
affirmative to any word lexicon and is not an answer at all — read as one, it hijacked
the turn and the transfer never happened. `cogno-host`'s implementation is
`assembler.decide_hold` (RELEASE / CARRY / DROP) plus `routing.explicit_persona_request`;
its `docs/ANTI_FABRICATION.md` §2-bis carries the measurements.

The same applies to **Fonte C**: the skill answered "I did not commit — ask first"
about a specific call. The rows it read may have changed while the hold was carried,
so a host that carries one across many turns is trading freshness for continuity — the
turn and clock bounds above are where that trade is declared.

## Ownership (respects the existing layer boundaries)

- **ID — detects & signals.** The `_act_confirm_caution` logic moves UP from the
  EGO to the ID (the ID already reads `modality`/`speech_act` off `IntentResult`
  and already emits routing signals: pii→SUPEREGO, `emotional_override`,
  `complexity`). The ID sets a routing flag, e.g. `ctx.metadata["ego_readonly"]`
  (or a field on `IdResult`). The EGO stops second-guessing.
- **EGO — obeys.** When the read-only flag is set, the EGO filters
  `dispatcher.tools_schema()` down to non-mutating tools. If no useful query tool
  exists, it degrades to its natural no-op → a `draft` that proposes/clarifies.
- **Host — classifies the tools.** The core must NOT hardcode which tools mutate
  (that would break infra-agnosticism). Read-vs-write is **host-declared
  metadata**. The core only reads the classification and filters.

## Why this is better for the bench

The current failing checks are **soft** ("did the model restrain itself?"). The
read-only gate makes them **hard invariants**:

- **ID:** `modality=UNCERTAIN` action → emits `ego_readonly`. Deterministic.
- **EGO:** `ego_readonly` set → the set of *mutating* tools dispatched is empty.
  A capability guarantee, asserted directly — not model goodwill.

## Fonte A vs Fonte B vs Fonte C — three sources of "needs confirmation"

These are **three different triggers** that converge on the same propose/commit
outcome. They are ordered here from broadest to most specific:

- **Fonte A — the USER is tentative** (framing): "*Should I* record 50?"
  (`speech_act=INTERROGATIVE`) / "I *maybe* spent 30?" (`modality=UNCERTAIN`).
  Detected by the **ID** (`needs_confirmation`). The host may then set
  `ego_readonly` → the EGO **masks ALL mutating tools** (broad caution).
- **Fonte B — the TOOL is dangerous** (destructiveness): the user is certain and
  commanding ("delete everything"), but the specific tool is irreversible.
  Detected in the **EGO** when the model picks a `requires_confirmation` tool →
  the EGO **holds that one call** (surgical). No `ego_readonly` involved.
- **Fonte C — the SKILL asks**, about THIS call and what it just read: the tool RAN, decided it
  must not commit, and returns `ToolResult(needs_confirmation=True)`. The EGO records it as
  pending and stops, exactly like B, and the proposal text is the **skill's own `output`** —
  grounded in the rows it read, not in the tool's name.

**Why C exists when B is already there.** B decides by tool NAME and BEFORE execution, so it
cannot know that cancelling *this* appointment is two hours away, or that *this* entry is a
hundred times the usual one. The skill knows, because it read. That also makes C the **widest**
of the three in reach even though it is the narrowest in scope: B only fires on a tool the host
declared destructive, while C can be raised by any skill on any call, with no policy declared
anywhere. And it supersedes A's trigger in practice — A guesses from how the user PHRASED the
request and has no reader; C is a fact about what the skill found.

**What C promises.** A `True` is a promise that nothing was committed, and the core enforces it
rather than trusting it: the call is recorded `ok=False`, and `committed_this_turn` requires
`ok` **and** `side_effect`, so a proposal can never be counted as a write however the skill
filled the other fields. On the return trip, a call the host CONFIRMED whose skill still asks
committed nothing and there is nobody left to ask — `_refuse_if_still_asking` fails it loudly
(`ok=False`, named error) rather than shipping "done" over a turn that wrote nothing, on both
routes a confirmed call arrives by.

**B and C hold per CALL, not per turn** — they stop the LOOP, not the STEP. A step with two
calls holds the one that asked and executes its sibling, so one turn can truthfully report both
"I am holding this" and "I committed that".

## Decisions taken (the original open points)

1. **Classification lives in a dispatcher hook** — `ToolPolicyDispatcher`
   (`is_mutating` / `requires_confirmation`), a separate optional Protocol probed
   with `isinstance` (mirrors `ToolCallingBackend`). The schema-field option was
   rejected (the native path sends `tools_schema()` to the provider API; a
   non-standard key risks rejection). **Fail-safe:** read-only mode with no policy
   → mask ALL tools (propose via draft). The confirmation gate is **opt-in** (no
   policy → the core cannot know a tool is destructive → no gate).
2. **`ToolResult.side_effect` kept separate** (observed post-exec) from the
   declared pre-exec classification. No cross-enforcement in the core (host owns
   consistency).
3. **Core renders a minimal `PROPOSE mode` marker** in `_task_context` when
   `ego_readonly` is set (so the model knows why the write tools are gone); the
   rich persona/voice prompt stays host-owned.
4. **Scoped to both sources now.** The boolean `ego_readonly` covers Fonte A; the
   `requires_confirmation` per-tool flag covers Fonte B (the originally-deferred
   "certain but destructive" cousin). DEFERRED: turning Fonte B into a richer
   "confirmation-required policy" object (per-arg thresholds etc.) — the boolean
   tool flag suffices for now. *(Fonte C landed later and is what that deferral turned into:
   rather than teaching the host's per-NAME policy about arguments and thresholds, the decision
   moved to the only party that can read the row — the skill. The deferred richer policy object
   was not built and is not needed.)*

## Plumbing summary

- `IdResult.needs_confirmation` (ID signal) · `ctx.metadata["ego_readonly"]`
  (host → EGO, Fonte A) · `ctx.metadata["ego_confirmed"]` (host → EGO, `True` or a
  set of tool names, opens Fonte B **and** Fonte C) · `ToolResult.needs_confirmation`
  (skill → EGO, Fonte C) · `EgoResult.pending_confirmation` (EGO →
  host, the held calls, from either gate — `_confirmation_stop` and anything else reading it
  deliberately cannot tell which gate produced a hold, and does not need to).
- Block 1 (judge constraints/negation) and Block 2 (parole→voice) shipped
  alongside (SUPEREGO 100%). The earlier advisory act-confirm hint in the EGO was
  removed (superseded by these capability gates).
