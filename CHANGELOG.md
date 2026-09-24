# Changelog

## Unreleased — o critério de graduação da PII conta os turnos em que HAVIA PII (2026-09-23)

### Changed (documentação; comportamento inalterado)

- **`vocab.PII_OBSERVATION_MIN_TURNS` (200) passa a contar turnos em que o detector ACHOU PII na
  resposta (`pii:flagged_in_output`), não «turnos com bloco SUPEREGO».** O critério antigo
  cumpria-se no VAZIO: uma resposta sem dado pessoal não retém nada e nunca erra, portanto 200
  turnos «limpos» provam que não houve nada para redigir, não que a rede redige bem. Medido no
  box a 23/09: 0 respostas com PII detectada em 352 turnos. **Hoje nenhum tenant está pronto.**
  Só muda a redacção (docstring em `vocab.py`, `CLAUDE.md`, docstring do teste); o valor, o
  modo por omissão e a rede ficam byte a byte. Nada NESTA biblioteca conta para o critério. No
  host HÁ quem o conte, e um grep pelo nome da constante não o encontra: o relatório
  `scripts/nets_by_class.py` imprime o critério com o denominador ANTIGO (todos os turnos com
  bloco) sem usar o nome `PII_OBSERVATION_MIN_TURNS` na contagem. Vai ser alinhado num PR do
  host, junto com a redacção antiga do docstring de `PATCH /tenant/{id}/pii-output-mode` (a
  rota escreve o que uma pessoa decidiu e não lê contagem nenhuma).

## Unreleased — no esgotamento, a resposta é o rascunho e os dados: nada além deles (2026-09-22)

### Fixed

- **A voz, no esgotamento, deixa de reconstruir o que nenhuma leitura devolveu.** Medido num
  turno REAL do dono (`turn_traces` 1970, turno 107, 22/09 20:38; host `2f0d2cd6`, anima
  `d34822a5`): «e de novembro?», um turno depois da lista de outubro. A ÚNICA ferramenta que
  correu foi a estimativa de remuneração de novembro — um bloco POR DISCIPLINA (8 aulas, 32 h,
  R$ 3.840,00), sem lista de dias — e devolveu `ok=True`; o juiz rejeitou 1/1 (ramo `readonly`);
  o host declarou `judge_rejected_all` → `last_draft_voiced`; e a voz entregou a estimativa E,
  por cima dela, «Aulas de novembro de 2026» com SETE DATAS (01/11…07/11), cada uma com turma e
  disciplina, que nenhuma ferramenta leu neste turno — contraditórias com a própria estimativa
  (7 datas contra 8 aulas; 2 contra 3 numa disciplina), e 01/11/2026 é domingo. A lista de
  OUTUBRO do turno 106 era real (`get_professor_schedule`) e estava no `# Context` da voz
  (4 530 chars): a voz completou novembro por analogia com outubro.

  **O princípio do dono, textual: «quem escreve não pode escrever coisas que não sabe».** A voz
  não é fonte; proveniência ou nada; no esgotamento nunca se reconstrói.

  **Desenho.** Uma cláusula INCONDICIONAL, `_NOTHING_BEYOND_WHAT_WAS_READ`, escrita uma vez e
  spliced por referência como ÚLTIMA palavra dos três renders dos dois cabeçalhos de
  esgotamento — `# Execution verdict (HARD RULE)` e os dois ramos de `# Review verdict (HARD
  RULE)`: a resposta é o que o prompt mostra — os dados do executor e, quando é mostrada, a
  resposta do executor — na voz da persona (regras e limites configurados incluídos), e NADA
  além deles: nenhuma lista, data, item, nome ou valor que não esteja lá escrito; o que um turno
  ANTERIOR leu não é o que este turno leu, e uma lista completada por analogia com uma no
  Context, por padrão ou de memória é INVENTADA; quando o pedido implica algo que NÃO foi lido,
  diz-se que não foi lido — numa frase, AO LADO do que foi lido, nunca em vez dele — e não se
  põe nada no lugar. **A metade positiva é uma ORDEM, e isso foi medido, não preferido:** a
  primeira redacção dizia «a resposta É os dados … diz que não foi lido» — a positiva descritiva,
  a negativa imperativa — e o canário com modelo (qwen3:8b, `temperature=0`, este mesmo turno)
  obedeceu à ordem e deixou cair os dados: nenhuma data de novembro, e nenhuma estimativa
  («Ainda não há um calendário de aulas para novembro de 2026 disponível», sobre uma leitura que
  tinha devolvido a remuneração do mês). Uma mordaça é o espelho do defeito, não a sua correcção;
  a reprodução passou a ordem («state it, reproducing every figure, date, name and identifier…
  exactly as written»), vem PRIMEIRO, e o limite fica ao lado. **E a segunda redacção falhou no
  outro sentido, no mesmo canário:** com a ordem, a estimativa saiu inteira — e por cima dela,
  outra vez, «Aulas de novembro de 2026», quatro linhas com `11/11` em todas: uma data DERIVADA
  do cabeçalho `11/2026` do próprio bloco, numa secção copiada da FORMA da resposta anterior que
  está no Context (lista de aulas por data, depois a estimativa). «Nenhuma lista que não esteja
  escrita» não a apanhou, porque cada ITEM da lista estava nos dados e só a coluna da data foi
  inventada. A terceira redacção nomeia os dois mecanismos, e nomeia-os por ÚLTIMO: uma data
  nunca é derivada (de um mês, período, contagem ou padrão — se não está escrita carácter a
  carácter no que foi lido, não entra), e a resposta anterior não é template (uma secção que
  ela tinha e este turno não leu não existe aqui); e o que o pedido pedia e nenhuma ferramenta
  leu «não é teu para escrever» — porque `nothing_tried`, uma cláusula acima, diz «a crítica
  diz o que FALTOU… escreve ISSO» (escrito para uma CONFIRMAÇÃO em falta, legível num turno
  só de leitura como «escreve a lista em falta»). Incondicional pela mesma razão da cláusula
  «a crítica é para o executor»: proíbe um movimento que a voz nunca pode fazer legitimamente,
  logo não há facto que a condicione. Sem cabeçalho novo — `_VOICE_BLOCKS` e o inventário
  persistido não se movem.

  **Porque a regra que já lá estava não chegou.** `_FIGURES_HAVE_A_SOURCE` (HARD RULE no
  `# Task`) já proibia datas sem fonte e já excluía o Context — e não segurou: no esgotamento o
  rascunho está retido, a crítica diz que a resposta ficou aquém, e o bloco mais longo do prompt
  é a resposta do turno anterior com exactamente a forma da que se pede. A cláusula repete a
  regra DENTRO do veredicto, como última palavra, e sobre LISTAS e ITENS — o que foi inventado
  foi uma lista. As fontes são nomeadas exactamente como `_FIGURES_HAVE_A_SOURCE` as nomeia
  (dados; resposta do executor QUANDO É MOSTRADA — neste caminho `_draft_section` retém-na, e a
  frase não pode afirmar o contrário; regras da persona), para que as duas regras não apontem em
  sentidos opostos: estreitar «só aos dados» recompraria o turno 3 de `_FIGURES_HAVE_A_SOURCE`
  (a taxa R$ 120/h nas regras da persona, dita desconhecida).

  **Pinos.** `tests/unit/test_voice_on_exhaustion_does_not_reconstruct.py` — presença primeiro
  (o prompt carrega as 14 datas de outubro no Context e a estimativa nos dados, e NENHUMA data
  de novembro: uma na resposta é invenção por construção), a cláusula nos três renders e como
  última palavra, incondicional sobre 4 shapes × 2 kinds, definição única, o rascunho continua
  retido (com o seu controlo), o caminho real via `voice()`, e os CONTROLOS: sem esgotamento
  (sem rejeição, `repeated_reply`, rascunho aprovado, conversacional) o prompt é byte-idêntico à
  main, por digest medido em `d34822a5` com esta fixture. Sobre a main o ficheiro falha por
  asserção, não por ImportError. `tests/unit/test_voice_review_verdict_after_a_read.py` foi
  re-pinado (o literal `MAIN_REVIEW_VERDICT` e a tabela `_MAIN_SECTIONS`, re-medida no ramo: as
  células de veredicto moveram-se JUNTAS e só `already_said` manteve o digest). Canário com
  modelo em `tests/integration/test_superego.py` (`…does_not_reconstruct_a_list_nobody_read`;
  só com spec de nuvem, `temperature=0`): a resposta não contém nenhuma data `dd/11` e contém a
  estimativa, em VALORES — as duas asserções, porque cada redacção que falhou falhou uma delas.
  A fixture é o t107 anonimizado (sem pessoa, tenant nem id); o resultado
  persistido está cortado a 240 chars e o bloco inteiro é assumido pela FORMA (por disciplina,
  sem lista de dias) — dito no docstring.

  **(a) `nothing_tried` só renderiza quando uma ACÇÃO foi pedida (decisão do Director, 22/09).**
  A cláusula «NOTHING WAS EVEN TRIED … the critique says what was MISSING … write THAT
  instead» foi escrita para uma CONFIRMAÇÃO em falta num turno de ESCRITA (espécime 1198:
  `resolve_date` só, despesa nunca registada), mas o seu gate, `write_attempted_this_turn`, é
  falso em QUALQUER turno só de leitura — logo todo o esgotamento de leitura entregava à voz
  «escreve o que a crítica diz que falta», e no t107 a crítica nomeava as aulas do mês. O gate
  passa a exigir também `intent.intent_class == "ACTION_REQUEST"` (vocabulário fechado
  `vocab.VALID_INTENTS`; o mesmo sinal que o EGO já lê para forçar uma chamada no 1.º passo): o
  sujeito da própria cláusula é «the requested action», e um turno que não pediu acção nenhuma
  não lhe dá referente — no mundo só-de-leitura manda a cláusula irmã (`read_worked`). Porque
  não «uma ferramenta mutante na mesa»: o carrier não o tem — `EgoResult.tools_offered` são só
  nomes, e qual nome ESCREVE é política do dispatcher, que `voice()` nunca recebe; o sinal
  mais fino seria uma metakey nova. Intent ausente lê-se como «nenhuma acção pedida» (a regra é
  «só quando uma escrita era possível»; desconhecido não o estabelece). O PREDICADO não muda
  (`test_having_writing_tools_on_the_table_is_not_an_attempt` continua verdadeiro): é uma
  segunda condição na CLÁUSULA. Gémeos no mesmo ficheiro de teste: t107 (só leitura,
  `INFORMATION_REQUEST`) → não renderiza; espécime 1198 (`ACTION_REQUEST`) → continua a
  renderizar, byte-idêntico; intent trocado/ausente → não renderiza; controlos por digest
  medidos em `29eafab6` (cláusula sem gate): escrita tentada e falhada sob qualquer intent, e
  turno aprovado, byte-idênticos. `test_voice_does_not_deny_a_read_that_worked.py` passou a
  mostrar a coexistência das duas cláusulas num turno `ACTION_REQUEST` e ganhou o gémeo
  só-de-leitura; `test_voice_never_invents_a_failure.py` intacto em comportamento.

  **As DUAS medições, e a leitura (22/09).** Canário `tests/integration/test_superego.py::
  test_voice_on_exhaustion_does_not_reconstruct_a_list_nobody_read` (t107 reconstruído: rascunho
  + estimativa + lista de outubro no Context + esgotamento declarado), `temperature=0`:

  | modelo | árvore | n | resultado | leitura |
  |---|---|---|---|---|
  | qwen3:8b (local) | cláusula, 3 redacções, e cláusula + (a) | 4 corridas (11,6 s · 30,5 s · 31,7 s · 31,3 s de GPU) | **0/4** | «Aulas de novembro» com `11/11` em cada linha por cima da estimativa inteira; a 1.ª redacção mordaçou a estimativa em vez de inventar datas |
  | gpt-4o-mini (a voz de produção do t107) | `main` d34822a5, sem cláusula | 3 (corrida do Director) | **0/3** | «não consegui encontrar as aulas de novembro» e larga a estimativa — a MORDAÇA, a outra face da mesma classe; sem datas inventadas nesta fixture |
  | gpt-4o-mini | cláusula só (29eafab6) | 1 | 1/1 | — |
  | gpt-4o-mini | cláusula + (a) (832d0ab1) | 3 (corrida do Director) | **3/3** | sem `dd/11`, estimativa inteira |

  **Instrução não é garantia.** No modelo de produção a cláusula move (0/3 → 3/3); no qwen3
  não move (0/4). As duas coisas são verdadeiras e este registo diz as duas. Por isso o canário
  corre **só com spec de nuvem** — `pytest.skip` quando o backend resolvido por
  `tests/integration/backends.py` é Ollama, com a razão medida no texto do skip — e os três
  shards ollama do CI ficam verdes E honestos (um skip que diz porquê, nunca um verde vazio). A
  garantia desta classe não é um parágrafo: é a rede determinística do host
  (`unread_date_claim`), que corre em runtime, onde o resultado da ferramenta está inteiro.

  **Fora de alcance, com nome.** A PREVALÊNCIA desta classe não é mensurável no traço enquanto
  38 % das leituras chegarem cortadas ao `turn_traces` (o próprio t107 é mecanicamente
  indecidível; positivo pela LEITURA do conteúdo). Uma rede `unread_date_claim` em runtime, e o
  dia-da-semana verificável sem modelo nenhum (01/11/2026 é domingo), ficam parqueados com nome.
  Custo conhecido, assumido: no esgotamento um NOME que só o Context traga (a saudação pelo nome
  do contacto) cai debaixo da mesma proibição — é a fronteira do princípio, e a lista inventada
  era de nomes.

## Unreleased — um e-mail preservado que o contacto nunca escreveu é do modelo, não do contacto (2026-09-22)

### Fixed

- **A NOUMENO descarta um termo preservado que seja E-MAIL ou URL e não esteja, inteiro, nas
  palavras do contacto — antes de chegar ao juiz e à voz.** Medido 3/3, determinístico
  (`qwen3:8b`, `temperature=0`), ao regravar a cassete do onboarding no host (#954): o rewriter
  listou o e-mail do contacto em `preserved_terms` com UM carácter alterado (mesmo domínio,
  parte local a distância de edição 1) e o valor foi parar ao prompt do juiz sob `# Preserved
  terms — VALUES`. Como o juiz exige a reprodução EXACTA de cada valor preservado (critério #4,
  #172), a resposta com o e-mail CERTO era rejeitada por «o ter alterado», o laço esgotava e o
  contacto recebia a frase de falha — ou a voz repetia o endereço errado.

  **Só e-mails e URLs, de propósito.** São tokens exactos por natureza: não há reescrita
  legítima de um endereço, logo um que não está no que o contacto escreveu é do modelo. Uma
  FIGURA não é: a NOUMENO normaliza o formato a caminho do inglês («R$ 1.000,00» pode sair
  `1000`) e o #172 decidiu que um termo preservado é um VALOR, não uma grafia — um descarte por
  «não está verbatim» apanharia figuras legítimas. Figuras, nomes e frases nunca passam por
  este caminho (controlo pinado).

  **A comparação é uma correspondência de endereço INTEIRO contra `ctx.user_input`** — o texto
  cru, nunca a reescrita, que carrega a mesma mutação — **com maiúsculas e tudo.** Exacta porque
  os dois erros não são simétricos: um descarte a mais custa só a marcação (as palavras do
  contacto já estão no prompt do juiz, verbatim, em `# User request`); um descarte a menos é o
  defeito medido. Dobrar maiúsculas não recupera um carácter trocado — só admitiria uma
  normalização que o MODELO escolheu, e o juiz passaria a exigir a grafia do modelo sobre a do
  contacto. Endereço inteiro e não substring nua porque, a distância de edição 1, um carácter
  caído em qualquer das PONTAS da parte local («na.silva@…» em «ana.silva@…») ou do domínio
  («…@example.com» em «…@example.com.br») deixa uma substring do que foi escrito que continua a
  não ser o endereço.

  **O descarte deixa marca — uma rede que ninguém conta vira o mecanismo.**
  `vocab.PRESERVED_NOT_IN_INPUT` em `NoumenoResult.degradations` (alfabeto fechado, ao lado de
  `EMBED_UNAVAILABLE`), a CONTAGEM em `NoumenoResult.preserved_dropped`,
  `NOUMENO.PRESERVED_NOT_IN_INPUT` em `DriftMetrics.to_tags()` (novo campo
  `noumeno_degradations`, semeado por `DriftCalculator.compute`) e um WARNING com a contagem.
  Nunca o valor: um endereço é PII e o registo sobrevive ao turno. O traço do host lê hoje
  `preserved_terms` e `warnings` do resultado da NOUMENO e não `degradations` — persistir a
  marca é uma linha do lado do host (`trace.py`, bloco `noumeno`).

  **Uma definição de «crítico», agora três leitores.** `_CRITICAL_TERM_RE` (figura, e-mail, URL)
  vivia só no SUPEREGO; passa a `cogno_anima.preserved.CRITICAL_TERM_RE`, e a metade de tokens
  exactos é `EXACT_TOKEN_RE`, construída dos MESMOS fragmentos. O SUPEREGO mantém o nome local
  e importa-o; `test_judge_preserved_is_a_value` continua a pinar os dois leitores dele, e o
  teste novo pina a identidade `superego._CRITICAL_TERM_RE is preserved.CRITICAL_TERM_RE`.

  `tests/unit/test_preserved_terms_are_the_contacts.py`: gémeo, três controlos, contagem sem
  valor, arestas da comparação, alfabeto e tag. Quatro mutações medidas: sem descarte → gémeo
  morre; descartar figuras → controlo 2 morre; descartar sem registar → os testes da marca
  morrem; comparar contra `rewritten` → gémeo morre. Um caso de integração no shard
  `perception` afirma o invariante contra um modelo real.

## Unreleased — o critério de GROUNDING contradizia o bloco que estava no mesmo prompt (2026-09-08)

### Fixed

- **`_GROUNDING_SOURCES`: os ramos `execution` e `readonly` do juiz deixam de declarar os
  resultados de ferramenta a ÚNICA verdade do turno.** O prompt do juiz sempre carregou mais
  fundamento do que as chamadas: `# Persona limits` é onde o host renderiza as regras de
  negócio configuradas pelo INQUILINO (e as enquadra, lá, como fundamento legítimo) e
  `# Context` traz o relógio, as memórias e o histórico. O ramo `conversational` enumera os
  três desde que foi escrito («NOT in the Context above, in the persona's limits, or in what
  the user said»); os outros dois nunca o fizeram — e diziam o contrário, na forma mais forte
  disponível: «the reads are the ONLY ground truth this reply has» / «backed by the tool
  results».

  **MEDIDO NO PROMPT RENDERIZADO** (`turn_traces` id=1440, host `b1901a6`, juiz
  `gpt-5.6-luna`). As `custom_rules` do papel EMPLOYEE de um inquilino configuram o valor da
  hora-aula e a tabela de bónus; o contacto perguntou exactamente isso; o executor respondeu
  certo; as três leituras vieram vazias («No faculty records found.», «Nothing recorded
  about…»). O juiz rejeitou: *«The draft fabricates the hourly rate, bonus amounts,
  eligibility rules, invoice deadline, and payment date. The successful searches found no
  faculty records or knowledge about teacher rates and bonus rules»* — que é o critério #1
  APLICADO CORRECTAMENTE, palavra por palavra, a um prompt que também carregava, 120 linhas
  acima, o bloco a dizer «An execution/answer grounded in them is CORRECTLY grounded».

  **Não era uma cláusula em falta, e a distinção era o trabalho todo.** Uma sonda determinista
  sobre o prompt renderizado encontrou a linha do próprio inquilino — `- Aula - R$ 120,00 por
  hora` — presente, verbatim, sob `# Tenant rules (legitimate grounding)`, no mesmo prompt que
  produziu aquela crítica. Logo o conserto não é fazer o bloco chegar (ele chega): é parar os
  critérios de afirmarem o oposto. Acrescentar um quinto parágrafo a um prompt cujo
  enquadramento já é ignorado é o movimento fraco — é o mesmo achado que decidiu o ramo
  `JUDGE_READONLY`, e chega aqui pela outra porta.

  **Não é uma flexibilização, e a última frase é o que a mantém honesta:** o conjunto de fontes
  continua FECHADO e todos os seus membros estão DENTRO deste prompt. Um número que não aparece
  em nenhuma das três continua a ser fabricação e continua rejeitado com a mesma dureza; a
  leitura vazia continua a fundamentar apenas uma resposta NEGATIVA — agora sobre O QUE AQUELA
  FERRAMENTA COBRE, que é a única correcção que a medição obriga. Escrito UMA vez, pela razão
  por que `_ADMITTING_A_LIMIT` é escrito uma vez. O ramo `conversational` fica byte a byte
  igual: era ele o precedente.

### Added

- **`SuperegoResult.judge_branch` + `SuperegoStage.judge_prompt_inventory`** — que critérios o
  juiz recebeu, e que secções o prompt dele carregava, POR TENTATIVA. O ramo era uma linha de
  LOG e nada mais, e ler UM turno rejeitado custou reconstruí-lo off-line contra a linha viva de
  `tenant_personas`: nada do que ficou persistido dizia se as regras do inquilino tinham sequer
  chegado ao prompt. «Em falta» e «ignorado» têm consertos OPOSTOS — um é fiação, o outro
  substitui os critérios — e um traço que não os distingue manda o leitor seguinte pelo caminho
  errado.

  Gémeo exacto de `voice_prompt_inventory`, e com a mesma propriedade de segurança: o
  **alfabeto de saída é FECHADO** (`_JUDGE_BLOCKS`), portanto o slug nunca sai do texto que
  casou. Aqui isso pesa MAIS do que na voz, porque um dos blocos — `# Persona limits` — é onde
  o host renderiza markdown escrito pelo inquilino: um cabeçalho forjado lá dentro compra uma
  LINHA duplicada (visível, e reportada como duplicada em vez de fundida) e nada mais; nunca um
  byte seu. O prompt renderizado continua a NÃO ser persistido pelo core.

  Gravado em TODOS os caminhos que construíram um prompt, o *fail-CLOSED* incluído: «a chamada
  rebentou» e «o juiz leu estes critérios e disse não» são falhas diferentes, e a segunda é a
  comum.

## Unreleased — um assunto sem token na lista fechada não tem dono (2026-09-06)

### Added

- **`MARKETING` entra em `NER_KNOWLEDGE_DOMAINS`** (e, na mesma alteração, na lista `domains`
  do prompt do NER — as duas cópias do facto que `test_code_domains_match_prompt_domains_exactly`
  e `test_all_vocab_values_are_taught_by_the_prompt` obrigam a andar juntas).

  **A lista é lida pelas duas pontas.** O NER responde o assunto do turno a partir dela; uma
  persona declara o assunto que POSSUI no mesmo vocabulário (`cogno_persona.Persona.domains`,
  o campo irmão desta alteração), e é assim que um host entrega um turno a quem é dono do
  domínio. Um assunto que falta aqui não é só inclassificável: **não tem dono**, e a única
  maneira de chegar à persona cujo trabalho ele é passa a ser dizer o nome dela. Medido numa
  corrida viva — um pedido de planeamento mensal de campanhas não abriu delegação nenhuma
  (`tokens=[]`, `target=''`).

  **Cresceu um token, não uma família.** `ADVERTISING`, `BRANDING`, `SOCIAL_MEDIA`, `SALES` e
  `MKT` continuam a cair no filtro da lista fechada (`_canonical_domains`), sem entrada nova em
  `_DOMAIN_ALIASES` — um alias tem vítima e nenhum foi medido a fazer falta. Sonda de
  sobre-aperto contra o corpus inteiro do bench e uma corrida `--only ner` de controlo contra
  a base: números no PR.
## Unreleased — o cache do provider deixa de ser invisível ao medidor (2026-09-06)

### Added

- **`StageMetrics.cached_tokens` — a parte de `tokens_in` que o provider serviu do PRÓPRIO
  cache.** Medido ao vivo a 2026-09-03: a 2.ª chamada com o mesmo prefixo devolveu **2432 em
  cache de 2625** tokens de prompt (**92,6%**). O campo chegava em toda a resposta da OpenAI e
  **não era lido em lado nenhum** — nem no `cogno-synapse`, nem no `cogno_meter/pricing.py` — e o
  padrão de produção é exactamente o que activa o cache: as tentativas de correcção do EGO do
  mesmo turno reenviam o mesmo prompt de sistema com segundos de intervalo (as repetições são
  **40,2%** dos tokens do mês). O provider cobra-as a uma taxa muito menor; um medidor que não as
  vê preça toda a retentativa como prompt novo e reporta um custo que não foi pago.

  **É SUBCONJUNTO, não parcela nova, e por isso NÃO entra no `tokens_total`.** Já está dentro do
  `tokens_in`: somá-lo contaria os mesmos tokens duas vezes na franquia, no rollup e em cada
  painel que os lê.

  Os cinco estágios que correm modelo gravam-no, e cada um soma no **mesmo eixo** em que já soma
  os tokens — NOUMENO/NER por TENTATIVA (o `generate_json_resilient` passa a devolver quatro
  valores; uma retentativa de truncamento reenvia o mesmo prefixo, logo é precisamente a chamada
  com mais probabilidade de estar em cache), o EGO por PASSO do laço e nos **dois** caminhos
  (FC nativo e fallback de texto), o SUPEREGO nas três chamadas. Os atalhos que **não** chamam
  modelo passam 0 de propósito: o backend é partilhado, e ler o atributo ali cobraria a este
  turno o cache de outro.

  A leitura é sempre `cogno_synapse.cached_tokens_of(backend)` **imediatamente a seguir ao
  `await`, sem outro `await` pelo meio** — é esse o contrato que torna um valor por instância
  seguro num backend partilhado entre turnos concorrentes. Um backend que não reporta nada
  (Ollama, um duplo, o aluno destilado) dá 0, e 0 a jusante é preço **cheio**: o comportamento de
  hoje, byte a byte.

  Requer `cogno-synapse` com `cached_tokens_of`. O preço vive no `cogno-meter`
  (`cached_input` por modelo; sem taxa → preço cheio, declarado) e a coluna no `cogno-host`.

## Unreleased — o router encaminhava duas perguntas de política e não a terceira (2026-09-01)

### Documentation

- **`docs/ACT_CONFIRM_READONLY.md` desenhava dois turnos e nada sobre o terceiro.** A referência
  dos três portões mostra «turno 1 propõe → turno 2 confirma → executa» e não dizia o que
  acontece a uma retenção que atravessa uma pergunta pelo meio — e ler o desenho como promessa
  custou uma conversa real. Medido em `cogno-host` a 2026-09-05: proposta («marco seu agendamento
  para 22/06 às 14h. Posso seguir?»), uma pergunta lateral do contacto, depois «sim» — o estado
  da sessão era reconstruído do zero a cada turno, à terceira mensagem não havia nada retido, e o
  modelo respondeu «Vou agendar…» com ZERO chamadas de ferramenta. O contacto ouviu uma promessa
  e nada foi escrito. O documento ganha a terceira linha do desenho (a que pertence ao host) e a
  propriedade que nenhum host pode violar: **nunca executar algo que não foi re-proposto ao
  contacto** — reter sem essa regra troca «a proposta é esquecida» por «um sim ambíguo comita»,
  que é o pior dos dois. Sem alteração de código: o núcleo é sem estado por desenho, e o tempo de
  vida de uma retenção é do host (`cogno-host`: `assembler.decide_hold` + `_reask_gate`,
  `docs/ANTI_FABRICATION.md` §2-bis).

### Fixed

- **A pergunta que faltava era a que uma protecção do host precisava de atravessar.** O
  `CompositeDispatcher` roteia `is_mutating` e `requires_confirmation` à fonte dona, mas não
  `source_requires_confirmation` — e não tem `__getattr__`. Era uma parede.

  **Medido ponta a ponta, não lido.** Um contacto que escreveu *«prefiro não falar com robô, me
  passa pra uma pessoa por favor»* recebeu *«Só pra confirmar: executo human handoff. Posso
  seguir?»* — jargão de sistema, uma pergunta, e **nenhuma escalada**. O chão anti-retenção do
  host, que existe exactamente para impedir isso, precisa daquele predicado para separar *"a
  fonte declarou destrutivo"* de *"é escrita e ninguém isentou"*; ao alcançá-lo pelos wrappers
  bateu neste router e recuou para o seu conservador, que **retém**.

  **A forma do erro vale mais que o conserto: premissa verdadeira, conclusão falsa.** O docstring
  do chão dizia que o recuo era *"inalcançável em produção: toda fonte é embrulhada"*. Toda fonte
  **é** embrulhada — o embrulho fica **dentro** do router. **"Embrulhada" não é "o método
  atravessa":** a alcançabilidade depende da TRAVESSIA, e a verificação parou na premissa. Não é
  medir a coisa errada; é medir a coisa certa e parar um passo antes do que a conclusão precisa.
  Quatro medições falharam porque **todas deixaram o router de fora** — e sem ele o defeito não
  aparece.

  **`__getattr__` genérico foi recusado de propósito:** é a resposta certa para um WRAPPER (uma
  camada sobre UMA fonte) e a errada para um ROUTER sobre muitas, que não sabe a que fonte
  encaminhar e teria de escolher uma.

  **Os recuos seguem a convenção já declarada da classe**, e o do meio é o que mantém isto
  seguro: uma fonte de política **sem** o predicado fino tem no seu `requires_confirmation` o
  próprio veredicto, portanto responder `False` ali deixaria o chão engolir um `destructiveHint`
  — a única coisa que ele nunca pode fazer.

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
