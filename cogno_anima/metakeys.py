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

# ── SUPEREGO (locutor) — soma/host write, voice reads ────────────────────────
VOICE_CORRECTION = "voice_correction"          # judge's final rejection: {reason}

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
