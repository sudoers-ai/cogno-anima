# Examples

## `host_min.py` — a minimal runnable host

A self-contained host that wires `cogno-anima` end-to-end, importing only from `cogno_anima`
and its transport lib `cogno_synapse` (never `cognobench`). It's the runnable companion to
[`docs/HOST_INTEGRATION.md`](../docs/HOST_INTEGRATION.md).

```bash
python3 examples/host_min.py
```

Needs a local [Ollama](https://ollama.com) at `http://localhost:11434` with
`mistral:latest` and `nomic-embed-text` pulled. It prints a note and exits if
Ollama is unreachable.

### What it shows

- The full orchestration a host owns: NOUMENO → NER → ID → PII gate → scope guard
  → EGO ⇄ SUPEREGO(judge) correction loop → SUPEREGO(voice).
- A real `ToolDispatcher` (`LedgerDispatcher`) with host **transaction semantics**:
  side effects are buffered, `rollback()` discards a judge-rejected attempt,
  `commit()` persists only after approval.
- An **idempotency guard** (a host concern): a weaker model may emit the same
  write several times in one turn — the host must not double-charge. In the demo
  the model calls `record_expense` multiple times but the balance drops once.
- **Cross-turn state** persisted by the host (`id_state`, `last_rewritten`,
  `turn_number`) — here in an in-memory dict, in production your DB/Redis.

### Note on prompts

The NOUMENO and NER prompt templates ship INSIDE the package
(`cogno_anima/prompt_templates/<stage>/`, declared as package data), so the example runs the same
from the source tree and from an installed wheel. The persona's own prompts — the executor's, the
judge's limits, the voice — are the host's, and this example writes its own inline.

## `quickstart_60s.py` — perception + routing in a minute

The first three stages only (NOUMENO → NER → ID): no tools, no dispatcher, no database. It needs
an `LLMBackend` and an `Embedder` and prints, for each sentence, the rewrite, the intent, the PII
risk, the route and the tokens.

```bash
ollama pull mistral && ollama pull nomic-embed-text
python3 examples/quickstart_60s.py

# or any other backend spec that cogno-synapse's create_backend() understands:
COGNO_BACKEND="groq:llama-3.3-70b-versatile" GROQ_API_KEY=... python3 examples/quickstart_60s.py
```

`COGNO_BACKEND` defaults to `mistral:latest`.
