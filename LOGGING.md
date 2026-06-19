# Logging — convenção desta lib

Esta biblioteca **emite** logs; o **host configura** (handlers, formato, nível,
contexto de tenant). Regras:

1. Use `logging.getLogger(__name__)` no topo do módulo. Nada de handlers,
   formatters, `basicConfig` ou um `get_logger` próprio.
2. Mensagem = só o fato de domínio, em `key=value`, sempre lazy:
   `logger.info("stage=ner route=%s pii_risk=%s", route, risk)`.
   NÃO coloque tenant_id / timestamp / channel na mensagem — o host injeta
   via contextvars + Filter no root logger (carimbado em todo LogRecord).
3. Níveis:
   - **ERROR**  → nunca aqui; erro fatal vira exceção e propaga (host loga ERROR).
   - **WARNING**→ condição recuperada/tratada (fallback, parse coercion, verify falho).
   - **INFO**   → marco caro e raro; NÃO happy-path por request.
   - **DEBUG**  → trace de fidelidade total (prompt/raw/scores). DEV-ONLY,
                  jamais ligado em produção multi-tenant. Redija secrets (apikey).
4. Controle de nível é por pacote: `logging.getLogger("cogno_anima").setLevel(...)`.

O host anexa o handler (TenantFilter + JsonFormatter) ao root logger real;
veja `cogno/core/logging.py` no host como referência.

## Nota específica do cogno-anima

Logger por stage (`cogno_anima.stages.ner`, `.id`, `.ego`, …). Sinalize as
**decisões cognitivas e degradações**, não o happy-path:

- **WARNING** — `StageParseError`/coerção de fallback (LLM devolveu JSON
  inválido → heurístico); EGO: duplicate-call bloqueada, tool alucinada,
  `interrupted=true`; SUPEREGO: judge fail-closed devolveu `critique`, backstops
  `pii:flagged_in_output` / `preserved:mutated_in_output`.
- **INFO** — rota resolvida pelo ID (`route`/`goal_status`/`blocked`).
- **DEBUG** — `drift_score`/tags do NOUMENO, `pii_raw`→`pii_risk` no NER, cada
  passo do loop do EGO (tool/args/ok).

O texto reescrito e o input do usuário são conteúdo de usuário (PII) → DEBUG
apenas (dev-only).
