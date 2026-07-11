"""Chaves do contrato inter-repo em ``PipelineContext.metadata``.

O ``metadata`` é deliberadamente um dict serializável (estado multi-worker), mas as
chaves que cruzam fronteira de repositório (host → soma → anima) são um CONTRATO:
um typo em uma string não falha — silenciosamente no-opa a feature. Todo call site
(nas três bases) deve importar a constante daqui em vez de digitar a string.

Convenção: o valor é estável para sempre (estado persistido em sessão/DB depende
dele); renomear a CONSTANTE é livre, renomear o VALOR é uma migração de dados.
"""

from __future__ import annotations

# ── EGO (executor) — host/soma escrevem, EgoStage lê ─────────────────────────
EGO_CONTEXT = "ego_context"                    # texto injetado (clock/memórias/história)
EGO_READONLY = "ego_readonly"                  # gate A: mascarar tools mutantes neste turno
EGO_MAX_STEPS = "ego_max_steps"                # orçamento explícito do loop (host/plano)
EGO_PERSONA = "ego_persona"                    # rótulo de persona estampado no EgoResult
EGO_FORCE_TOOL = "ego_force_tool"              # host: este turno EXIGE execução de tool
EGO_CONFIRMED = "ego_confirmed"                # gate B: True | coleção de nomes de tool
EGO_CONFIRMED_CALLS = "ego_confirmed_calls"    # gate B: calls aprovadas a executar
EGO_CORRECTION = "ego_correction"              # loop de correção: {reason, attempt}

# ── SUPEREGO (locutor) — soma/host escrevem, voice lê ────────────────────────
VOICE_CORRECTION = "voice_correction"          # rejeição final do juiz: {reason}

# ── ID / NER / NOUMENO — carry-over entre turnos (soma escreve, estágios leem) ─
ID_STATE = "id_state"                          # estado serializável do IDStage
TURN_NUMBER = "turn_number"                    # nº do turno (host/soma autoritativo)
ATTENTION_CANDIDATES = "attention_candidates"  # candidatos do AttentionFilter (host)
PII_SESSION_HINT = "pii_session_hint"          # dica de PII prévia na sessão (host)
EMOTIONAL_OVERRIDE = "emotional_override"      # override emocional injetado pelo host
LAST_REWRITTEN = "last_rewritten"              # rewrite do turno anterior (continuidade)
LAST_CONTEXT_TURN = "last_context_turn"        # resumo de contexto do turno anterior
LAST_GOAL = "last_goal"                        # goal do turno anterior (carry NER)
ACTIVE_DOMAINS = "active_domains"              # domínios ativos (carry NER)
CONVERSATION_HISTORY = "conversation_history"  # transcript cru p/ NOUMENO/NER

# ── carimbos de sessão (soma estampa; host/telemetria leem) ──────────────────
ACTIVE_PERSONA_ID = "active_persona_id"
ACTIVE_MCP_MODULE = "active_mcp_module"
