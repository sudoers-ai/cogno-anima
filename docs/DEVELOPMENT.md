# Development notes: the integration suites, CognoBench, the version

Moved verbatim from `CLAUDE.md` on 2026-11-08: the contract did not change. `CLAUDE.md` keeps a short summary and points here. Update this file in the same pull request as any change to the integration suites, the bench defaults or the version, as `CLAUDE.md` used to be updated.

`--embed-model` aceita `provider:model`, a mesma gramática do `--model` (2026-08-19). O default
segue Ollama local — grátis, determinístico, e por ser fixo é a **variável de controle** quando
um A/B mede outra coisa. Antes disso o embedder era local por construção e uma rodada de LLM em
nuvem **recusava-se a começar** sem Ollama de pé: o único componente que nunca foi o modelo sob
teste prendia a suíte à máquina. Medido de passagem, contra a expectativa: 100% nuvem pontuou
melhor no `--only id` (98,1% vs 97,1%), não pior.

Integration tests use real models and are written to be deterministic (`temperature=0.0`). **CognoBench's default `--model` is `qwen3:8b`** (every local run uses it; `mistral:latest` remains the top ID-bench scorer but is not judge-capable — it approved a wrong execution 3/3).

**The integration suites take a `provider:model` spec, on two axes** (2026-09-06): `COGNO_TEST_MODEL` (NER reads `COGNO_NER_MODEL` first) and `COGNO_TEST_EMBED_MODEL`, the same grammar CognoBench's `--model`/`--embed-model` already use, resolved through `cogno_synapse.create_backend`/`create_embedder` in the ONE place every suite now asks — `tests/integration/backends.py`. Defaults are unchanged: Ollama `qwen3:8b` and `nomic-embed-text:latest`, `temperature=0.0`, `format="json"` on the JSON-consuming ops, and Ollama's own `COGNO_OLLAMA_TIMEOUT` rather than the factory's constant — a unit test compares the built object attribute by attribute against a directly-constructed `OllamaBackend`. Until then every file built its own `OllamaBackend(model=…)`, so the run was **Ollama-only by construction** and this line — which said the suites read `COGNO_TEST_MODEL`, and was true — read as a choice it did not offer: the variable picked WHICH LOCAL MODEL. That is not cosmetic. The served box runs the same `qwen3:8b`, so a run puts every live turn in a queue behind the suite (measured: a host integration run sat 17 minutes on 5 seconds of CPU waiting for Ollama), and there was nowhere else to point it. **The skip travels with the spec** (`backend_unavailable_reason`): the local server for an Ollama spec, the provider's key for a cloud one — a suite pointed at OpenAI that skips itself saying "Ollama is not running" is an instrument lying about why it measured nothing.

**The version is written once**, in `cogno_anima/_version.py` (0.1.14). `cogno_anima.__version__` is that constant and `pyproject.toml` reads the same line (`dynamic = ["version"]`); it declares no number. It used to come from `importlib.metadata`, which answers for the INSTALLED distribution: from a source tree the code said 0.1.0 while the tree declared 0.1.5. A release bumps that file and adds the `CHANGELOG.md` entry; `tests/unit/test_version_has_one_source.py` compares the package, the build configuration and the changelog's first heading, and the wheel test checks the wheel it builds.

The **CognoBench** cognitive benchmark lives in `cognobench/` — see `cognobench/CLAUDE.md` for its design (loaded when working there).

**`docs/NETWORK_PERSONA_CHANNEL.md`** is the contract for the three things this library does NOT
own — the **persona** (four prompt strings and an opaque label), the **channel** (nothing at all:
`cogno_anima` has no channel field, metakey or vocabulary, and that emptiness is the contract),
and the **network** between personas (one carrier, `ctx.consult_result`; one host-declared name
set, `mk.ROUTING_ONLY_TOOLS`; and no graph — who may consult or transfer to whom is never asked
here). Each section ends with *who decides*, which is the host every time. Its counterpart is
`cogno-host/docs/NETWORK_PERSONA_CHANNEL.md`; the two are written to be read together and neither
repeats the other.
