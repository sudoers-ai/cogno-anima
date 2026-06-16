# Examples

## `host_min.py` — a minimal runnable host

A self-contained host that wires `cogno-anima` end-to-end, importing **only** from
`cogno_anima` (never `cognobench`). It's the runnable companion to
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

The example runs from the source tree, where the NOUMENO/NER prompt templates
live in `prompts/` at the repo root. (Packaging those into the wheel is a known
follow-up so a plain `pip install` host can load them without the source tree.)
