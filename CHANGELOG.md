# Changelog

## Unreleased

### Fixed

- **Uma persona SEM tools era ensinada a chamar tools, e emitia a tag.** No caminho de
  fallback textual o bloco de mecânica do `<TOOL_CALL>` era anexado incondicionalmente — a
  LISTA de tools já era condicional, só a lição não era. Um catálogo vazio recebia na mesma o
  formato, e o modelo usa-o: medido ao vivo, uma persona sem tools emitiu a tag e ela chegou ao
  contato, porque nada a jusante remove um bloco que nomeia uma tool que ninguém oferece.

  O prompt lia como coerente para quem o inspecionasse — nenhuma tool listada, e um formato para
  as chamar. Agora a lição só sai com o catálogo.

## 0.1.0 — 2026-07-25

First public release on PyPI.

- The five-stage cognitive pipeline: NOUMENO (perception/rewrite), NER
  (semantic analysis), ID (heuristic router & goal continuity), EGO (executor
  & tool dispatch), SUPEREGO (judge & voicer) + the pure Drift calculator.
- Deterministic PII detection and risk scoring (`compute_pii_risk`) — the
  LLM's own risk judgment is never trusted.
- Dual-path tool calling: native function calling for capable backends, a
  `<TOOL_CALL>` text-fallback for plain ones; confirmation gates (read-only
  mask + destructive-tool hold) behind a host-declared tool policy.
- Infrastructure-agnostic: model transport lives in `cogno-synapse`; the host
  owns persistence, execution, and escalation.
