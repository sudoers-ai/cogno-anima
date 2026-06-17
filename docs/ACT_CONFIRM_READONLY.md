# Design note — act_confirm → EGO read-only + confirmation gate (propose/commit)

**Status:** IMPLEMENTED. Two host-driven capability gates in the EGO, both
validated end-to-end in CognoBench as HARD invariants (EGO 100%, see
`cognobench/EGO_BENCH_RESULTS.md`). This note records the rationale + the
decisions taken on the original open points.

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

## Fonte A vs Fonte B — two sources of "needs confirmation"

These are **two different triggers** that converge on the same propose/commit
outcome:

- **Fonte A — the USER is tentative** (framing): "*Should I* record 50?"
  (`speech_act=INTERROGATIVE`) / "I *maybe* spent 30?" (`modality=UNCERTAIN`).
  Detected by the **ID** (`needs_confirmation`). The host may then set
  `ego_readonly` → the EGO **masks ALL mutating tools** (broad caution).
- **Fonte B — the TOOL is dangerous** (destructiveness): the user is certain and
  commanding ("delete everything"), but the specific tool is irreversible.
  Detected in the **EGO** when the model picks a `requires_confirmation` tool →
  the EGO **holds that one call** (surgical). No `ego_readonly` involved.

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
   tool flag suffices for now.

## Plumbing summary

- `IdResult.needs_confirmation` (ID signal) · `ctx.metadata["ego_readonly"]`
  (host → EGO, Fonte A) · `ctx.metadata["ego_confirmed"]` (host → EGO, `True` or a
  set of tool names, opens Fonte B) · `EgoResult.pending_confirmation` (EGO →
  host, the held destructive calls).
- Block 1 (judge constraints/negation) and Block 2 (parole→voice) shipped
  alongside (SUPEREGO 100%). The earlier advisory act-confirm hint in the EGO was
  removed (superseded by these capability gates).
