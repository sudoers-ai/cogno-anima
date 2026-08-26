# Changelog

## Unreleased — a taxa do envelope JSON estava 17x abaixo do real (2026-08-26)

### Fixed

- **A prosa dizia "1 turno em 283"; a contagem sobre a tabela INTEIRA de traços da caixa dá 5 em
  297.** O primeiro número saiu de uma fatia, não do conjunto — e foi ele que sustentou a leitura
  "isto é raro, a rede determinística chega". Uma taxa errada numa doc pública é pior do que
  nenhuma: quem lê decide o tamanho da resposta por ela. Re-contado a 2026-08-26 com
  `SELECT count(*) FROM turn_traces WHERE trace::text LIKE '%json_unwrapped%'` na base viva:
  **5**, nos dias 2026-08-25 às 11:03, 11:19, 11:20, 22:10 e 22:11 BRT — **um isolado e dois
  PARES**. Dois pares em cinco eventos não é acaso, o que reforça a leitura já escrita de que o
  gatilho está na ENTRADA e não no acaso do modelo. Nada muda no comportamento: a rede
  (`unwrap_envelope`) já estava lá, o `voice:json_unwrapped` já era carimbado, e a captura do
  bloco `# Context` no host (cogno-host #510/#514/#533) está viva para apanhar o próximo.

## Unreleased

### Fixed

- **A guarda de duplicados não via duas chamadas idênticas dentro do MESMO passo.** O contador
  `MAX_DUPLICATE_CALLS` bloqueia a terceira repetição de uma assinatura, e isso está certo
  ENTRE passos — uma leitura depois de uma escrita pode legitimamente devolver outra coisa.
  Dentro de um passo é demonstravelmente errado: as duas chamadas saíram do mesmo turno do
  modelo, sem nada a correr no meio, logo a segunda só pode devolver o que a primeira devolveu.

  Medido no bench do doctor (2026-08-25), instrumentando o despachante: um único passo emitiu
  `resolve_date({'expression': 'July 7, 2026'})` **duas vezes** e as duas executaram. Inócuo
  para uma data — a mesma porta está aberta para uma escrita.

  **Restrito a tool que o host declarou NÃO-mutante, e a restrição é o ponto.** Dois
  `record_expense(5, "café")` idênticos num passo podem ser DOIS CAFÉS: bloquear o segundo
  apagaria em silêncio um lançamento real — o defeito oposto, e mais calado. Escrita repetida é
  o que os portões de confirmação (B e C) tratam, e eles retêm por CHAMADA, portanto já veem a
  segunda. Sem política declarada não há afirmação sobre a tool e não há bloqueio — mesma
  direção à prova de falha da máscara só-leitura, que mascara em vez de assumir.

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
