# Changelog

## Unreleased — o juiz aprende o que o turno NÃO PODIA fazer (2026-08-27)

### Added

- **`mk.UNAVAILABLE_CAPABILITIES` — o Duty computado chega ao JUIZ.** O host subtrai o
  `requires` de cada capacidade das ferramentas que o turno realmente ofereceu e carimba a
  diferença; o `_build_judge_prompt` renderiza-a como `# NOT AVAILABLE this turn`.

  **É o JOIN que o `cogno_host/capabilities.py` tinha RESERVADO e que ninguém computava** — os
  dois lados já existiam (`EgoResult.tools_offered` e `Capability.variants[*].requires`).

  **Porque o juiz e não só o executor:** dizer ao executor *"não podes fazer X"* é obedecido
  TRIVIALMENTE por um turno que não tem X para chamar. O juiz é quem decide se a resposta é
  HONESTA, e sem esta linha **não distingue "não havia ferramenta" de "havia e não foi usada"** —
  os dois aparecem como `(no tools executed)`. Medido ao vivo: uma persona com duas ferramentas
  de leitura confirmou um lembrete que nunca criou, e o juiz aprovou à primeira, com crítica vazia.

  **DADO, nunca prosa.** O host renderiza texto de capacidade no prompt do EXECUTOR, e esse
  módulo regista que a palavra "duty" nomeia duas coisas diferentes nas duas camadas e que a
  divergência *"deixa de ser segura no dia em que blocos de capacidade forem acrescentados ao
  prompt do juiz"*. Atravessa o **facto** (nomes nossos, vocabulário fechado), nunca o texto — e
  `test_no_capability_PROSE_reaches_the_judge` transforma essa condição documentada num TESTE.

  A instrução diz também que **admitir o limite é uma resposta CORRECTA e COMPLETA** — sem isso
  o juiz rejeita a recusa honesta, e já medimos o que isso custa: o laço esgota e entrega um
  encaminhamento em vez da resposta. Sem o sinal, o prompt é byte-idêntico ao de antes.

## Unreleased — a PII pode entrar, mas não sai: a voz decide por proveniência (2026-08-26)

### Added

- **O backstop de PII na saída passou a DECIDIR por proveniência, e a redigir quando actua.** A
  regra do dono: *"a validação e proteção já deveria estar automaticamente no SUPEREGO, podendo
  entrar, mas nunca sair"* — com a única excepção de que o dado do PRÓPRIO contato pode
  voltar-lhe: confirmar o e-mail que a pessoa acabou de escrever **é a resposta**, não uma fuga.

  **Mascara, nunca recusa.** Uma recusa custa ao contato a resposta dele e entrega o turno ao
  laço de re-vozeamento — já medido a despachar um handoff em vez de uma marcação. A frase sai;
  o valor não.

- **Três entradas na decisão, e a terceira é o LEITOR.** O turno CORRENTE é derivado aqui, de
  `ctx.user_input`. Tudo o que é ANTERIOR na sessão chega em
  `ctx.metadata[mk.PII_OUTPUT_ALLOWLIST]`, porque `PipelineContext` vê um turno — por desenho. E
  `mk.PII_READER_ROLE` (o `Identity.role`) diz a quem a resposta vai.

  **Por SESSÃO, não por turno**, medido contra o gabarito do bench mergeado (host #549,
  `a2aef70`), por CENÁRIO: a variante por turno quebra `own_email_recalled_three_turns_later` —
  o e-mail dado no turno 1 e pedido de volta no 3 — e nenhum cenário da suíte é decidido ao
  contrário. (A manchete por CORRIDA que aquele bench publicou primeiro foi retirada pelo próprio
  autor por ser aritmética; a unidade é o cenário, ou a corrida DECIDÍVEL.)

  **O papel do leitor** porque a proveniência sozinha é sub-determinada, e é o mesmo bench que o
  prova: em `tool_result_document_number` o contabilista do próprio tenant relê uma linha do
  livro que ele escreveu — MESMA origem que o CPF de uma médica entregue a uma paciente, e
  gabarito OPOSTO. Medido de forma determinística (zero chamadas ao modelo) sobre o gabarito
  mergeado, o bit do papel move a concordância **5/7 → 6/7** e as respostas boas quebradas
  **2/5 → 1/5**; vazamentos barrados ficam em 2/2 nas duas variantes. Ausente, em branco ou só
  espaços lê-se GUEST — quem esquece a chave recebe MAIS máscara, nunca um alargamento calado.

- **A lista carrega DIGESTS, nunca valores** (`security/redaction.py`). `metadata` é lido por
  quem serializa o traço, por quem persiste o estado da sessão e por qualquer guarda que
  renderize os seus kwargs numa linha de log — uma lista de VALORES abriria um armazém novo de
  dado pessoal em claro para fechar um vazamento, exactamente a classe que se está a fechar. E
  colide com o passo seguinte já acordado: quando os turnos de entrada forem guardados
  MASCARADOS, uma lista de valores deixa de poder ser reconstruída a partir do histórico.

  **O tecto, dito e não insinuado: isto é de-identificação do fluxo, não cifra.** SHA-256 sem
  sal sobre um telefone tem espaço de entrada enumerável — quem tem o digest confirma um palpite
  em microssegundos. O que se compra é que nenhum dado pessoal viaja em `metadata`, chega a uma
  linha de traço ou a um log **por causa desta funcionalidade**. Mesmo negócio, e mesmo tecto,
  que o `scope_sha` do host (#545): um digest persistido é dado pessoal pseudonimizado e
  pertence DENTRO da purga de identidade.

- **Entra em modo de OBSERVAÇÃO, e o padrão foi escolhido por aritmética** — não por prudência
  (`mk.PII_OUTPUT_MODE`, `vocab.VALID_PII_MODES`, omissão `observe`; um erro de escrita cai em
  `observe`, nunca em `enforce`). Mesmo com o bit do papel, a regra ainda mascara um dos sete
  cenários que não devia: o telefone da recepção do próprio tenant, `red_by_design`. E não é uma
  forma de laboratório — varrido o detector sobre respostas com a forma de produção, o **CNPJ** e
  o **CEP** do próprio tenant mascaram igualmente ("Nosso CNPJ e 11.222.333/0001-81",
  "Rua das Flores, 123 - CEP 01310-100"), que é o que uma persona de recepção diz o dia inteiro.
  A classe de falsos positivos é **estreita e frequente**. Do outro lado: **zero vazamentos em
  297 turnos** de produção. Impor sobre estes números troca um dano nunca observado por um que
  cai num contato real amanhã.

  Em observação a regra corre inteira — detecção, proveniência, papel, registo — e o texto sai
  **byte-idêntico**, carimbando `pii:would_redact_in_output` **e a CLASSE**
  (`pii:withheld_<tipo>`). A classe vai junto porque as classes têm vereditos OPOSTOS: um
  `ADDRESS`/`TAX_ID` retido é quase sempre o CEP/CNPJ do próprio tenant — o falso positivo
  frequente; um `NATIONAL_ID` retido é quase sempre uma pessoa que não é quem está a ler — o
  vazamento. Contados juntos não decidem nada, e "observar" vira "esquecer". `PHONE` e `EMAIL`
  são o par genuinamente ambíguo, e são esses que um humano tem mesmo de olhar.

  **O critério de saída é um NÚMERO, não uma frase num PR que ninguém relê**
  (`vocab.PII_OBSERVATION_MIN_TURNS` = 200): gradua-se um tenant quando, ao longo de pelo menos
  200 turnos com bloco SUPEREGO, nenhum carimbo `pii:withheld_*` é dado do PRÓPRIO tenant. Regra
  de três — zero eventos em n ensaios limita a taxa real abaixo de 3/n a ~95% de confiança, logo
  zero em 200 põe o falso positivo abaixo de 1,5% — e 200 é cerca de uma ordem de grandeza acima
  da amostra que existe hoje (dos 297 traços da caixa, só 9 podiam carregar o bloco), que é o que
  distingue "não vimos nenhum" de "não olhámos". **Por tenant, e é uma DECISÃO**: nada neste
  pacote promove ninguém — um humano lê as contagens e põe `mk.PII_OUTPUT_MODE`. Uma regra que se
  graduasse sozinha estaria a impor com base numa semana calma.

  Um teste fixa que os dois modos DECIDEM igual e só diferem no texto — é ele que faz a contagem
  da observação valer o que a imposição faria; outro fixa as DUAS direcções do alfabeto de
  classes (todo tipo que o detector produz é alcançável; todo símbolo emitido está no
  vocabulário), no molde do `test_voice_blocks_sync`.

- **Cada decisão deixa registo com a razão que a causou.** `SuperegoResult.pii_findings`
  (tipo + `vocab.VALID_PII_PROVENANCE` + veredito — **nunca o valor**) e o alfabeto fechado de
  `adjustments`: `pii:flagged_in_output` (mantido, com o significado de sempre),
  `pii:redacted_in_output` **ou** `pii:would_redact_in_output` (o MODO está no símbolo, não só
  numa configuração que ninguém relê), e `pii:provenance_<valor>` para cada decisão — as
  permitidas incluídas, porque a pergunta que decide se isto sobrevive a conversas reais
  (*está a atrapalhar?*) precisa do denominador, e uma contagem de redações não o tem.

- **A decisão é UMA função pura** — `decide_provenance(digest, ProvenanceContext)`. A forma é que
  é a entrega: o segundo bit em falta está NOMEADO e NÃO CONSTRUÍDO (contatos que o tenant
  DECLARA citáveis, que resolvem o cenário da recepção), e tem de entrar como um CAMPO e um ramo
  — nunca como uma condição re-derivada em cada sítio que pergunta.

- **O digest inclui o TIPO, e a normalização é consciente do tipo** — dois falsos POSITIVOS de
  permissão, medidos, não imaginados. Sem o prefixo do tipo, o telefone do próprio contato
  `(52) 99822-4725` e o CPF de um desconhecido `529.982.247-25` reduzem-se aos MESMOS onze
  dígitos (cerca de 1 em 100 telemóveis BR calha num CPF de checksum válido), portanto quem
  escreveu o telefone permitia o documento alheio durante o resto da sessão. E apagar pontuação
  de um valor não-numérico fazia `ana@example.com` e `an@aexample.com` — domínios DIFERENTES —
  colidirem. Agora: valor todo-dígitos → só dígitos (é para isso que serve); o resto → apenas
  `strip` + `casefold`. Um e-mail nunca é re-pontuado entre a mensagem e a resposta, logo não
  havia nada a ganhar com o contrário.

- **Padrões sobrepostos são AGRUPADOS, não descartados.** O `\S{4,}` do `credential_kv` pára no
  espaço e o do cartão não, portanto `"senha: 4539 1488 0343 6467"` produz uma sobreposição
  PARCIAL. Descartar o segundo achado — correcto para uma sobreposição ANINHADA — deixava
  `"[CREDENTIAL REDACTED] 1488 0343 6467"`: catorze dos dezanove dígitos do cartão em claro, uma
  máscara enfiada no meio de um valor, e o CREDIT_CARD ausente de `findings` — logo, em modo de
  observação, a contagem que decide a graduação teria sub-reportado em silêncio. Um grupo é
  substituído de uma vez e todos os membros ficam registados; o mesmo TIPO aninhado num intervalo
  maior conta UMA vez (os packs de telefone sobrepõem-se de propósito, e inflar essa contagem
  manteria toda a gente em observação para sempre).

- `PiiDetector.find()` — valores localizados (tipo, texto, extensão), e `detect()` passou a ser
  expresso sobre ele. **Uma definição só**: o caminho que sinaliza e o caminho que mascara não
  podem discordar sobre o que conta como PII. A mascaragem na ESCRITA, o passo seguinte, tem de
  reutilizar este mesmo detector — nunca uma segunda definição.

### Changed

- O drift de síntese e o backstop de termos preservados passam a ler o texto **VOZEADO**, não o
  mascarado. A máscara é obra desta guarda; realimentá-la faria a protecção parecer uma
  fabricação e podia disparar o laço de correcção contra si própria.

## Unreleased — a taxa do envelope JSON estava 17x abaixo do real (2026-08-26)

### Fixed

- **A prosa dizia "1 turno em 283"; o defeito estava no DENOMINADOR.** O primeiro número dividiu
  por todos os traços guardados, incluindo turnos ANTERIORES ao campo existir — e foi ele que sustentou a leitura
  "isto é raro, a rede determinística chega". Uma taxa errada numa doc pública é pior do que
  nenhuma: quem lê decide o tamanho da resposta por ela. Re-contado a 2026-08-26 com
  a caixa tem 297 traços mas só **9** carregam bloco `superego` (o campo é persistido desde
  25/08 11:03) — os outros 288 nunca poderiam ter mostrado o carimbo. Dos 9 que podiam, **5
  mostraram**: 25/08 às 11:03, 11:19, 11:20, 22:10 e 22:11 BRT. A FORMA diz mais que a contagem:
  três são `turn 1` de três sessões DIFERENTES cuja primeira mensagem é quase a mesma frase, e
  dois são os turnos 10 e 11 de UMA sessão, também quase idênticos entre si — logo não é
  re-vozeamento do mesmo turno após rejeição do juiz. Reforça a leitura já escrita de que o
  gatilho está na ENTRADA. Nada muda no comportamento: a rede
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
