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
# How many CONSECUTIVE turns the host's anti-repeat guard has fired on this session (repaired
# or shipped). The guard already knows the conversation is circling; before this key, only the
# turn it fired on knew — the NEXT turn started with a clean slate and re-earned the repeat.
CIRCLING_STREAK = "circling_streak"

# ── SUPEREGO (locutor) — soma/host write, voice reads ────────────────────────
VOICE_CORRECTION = "voice_correction"          # judge's final rejection: {reason}
JUDGE_CONVERSATIONAL = "judge_conversational"  # host: this turn has NO tool to execute
# The judge's FINAL verdict + how many EGO attempts it took: {"approved": bool, "attempts": int}.
# Written by the orchestrator so the outcome is countable. Only rejections were ever logged (at
# WARNING; approvals at INFO, which the deployment's handlers may drop), so "no approvals in the
# log" was indistinguishable from "the judge approves nothing" — and a whole day was spent
# chasing the wrong one. A rate needs a denominator.
JUDGE_VERDICT = "judge_verdict"
# The FINAL verdict of each judged attempt, in order: [{"attempt": int, "approved": bool,
# "critique": str, "tools": [{"tool", "args", "ok", "side_effect", "result"}]
# (+ "tools_dropped"/"tools_error" when the writer capped or degraded)}].
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
