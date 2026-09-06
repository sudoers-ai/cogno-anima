"""The integration suites' backends, proved BY CONSTRUCTION — no model is called.

Two properties are in tension and both matter.

**Nothing changes by default.** Point nothing anywhere and the suites must build
exactly what they built when every file said ``OllamaBackend(model=MODEL, …)``
itself: local Ollama, ``qwen3:8b``, ``temperature=0.0``, ``format="json"`` where it
was set — and Ollama's OWN timeout, which is the one the nightly CI job steers with
``COGNO_OLLAMA_TIMEOUT`` because a CPU runner needs 600 s. The factory would
otherwise impose its own 600/120 and that knob would go quiet.

**The variable finally means something.** ``COGNO_TEST_MODEL`` used to choose WHICH
OLLAMA MODEL and nothing else, while the documentation said the suites "read
COGNO_TEST_MODEL" — true, and read as a choice it did not offer. Given a cloud spec
the fixture must NOT hand back an Ollama backend, and the skip must stop talking
about a server nobody asked for.

The cloud half is asserted WITHOUT the provider SDK and WITHOUT a key, so it runs in
CI (which installs neither): a cloud spec with no key raises ``MissingAPIKeyError``
from the factory, and an ``OllamaBackend`` never raises that whatever you name it.
The construction that needs the SDK is here too, guarded, for a dev box that has it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from cogno_synapse import CachingEmbedder, MissingAPIKeyError, OllamaBackend, OllamaEmbedder

from tests.integration import backends

ROOT = Path(__file__).resolve().parent.parent.parent

# Every attribute an OllamaBackend carries. Named one by one, not compared with
# `__dict__`, so a new field in cogno-synapse shows up as a failure to look at
# rather than passing silently on both sides.
_BACKEND_ATTRS = ("model", "base_url", "temperature", "num_ctx", "max_tokens",
                  "format", "think", "timeout")


def _clear(monkeypatch) -> None:
    for var in (backends.MODEL_ENV, backends.EMBED_MODEL_ENV, "COGNO_NER_MODEL",
                "COGNO_OLLAMA_TIMEOUT"):
        monkeypatch.delenv(var, raising=False)


# ── half one: with no variable set, nothing moved ────────────────────────────────

def test_default_json_backend_is_the_ollama_backend_it_always_was(monkeypatch):
    _clear(monkeypatch)
    twin = OllamaBackend(model="qwen3:8b", temperature=0.0, format="json")
    got = backends.json_backend()

    assert isinstance(got, OllamaBackend)
    for attr in _BACKEND_ATTRS:
        assert getattr(got, attr) == getattr(twin, attr), attr


def test_default_text_backend_is_the_ollama_backend_it_always_was(monkeypatch):
    _clear(monkeypatch)
    twin = OllamaBackend(model="qwen3:8b", temperature=0.0)
    got = backends.text_backend()

    assert isinstance(got, OllamaBackend)
    assert got.format is None, "the EGO's <TOOL_CALL> tags are illegal under JSON decoding"
    for attr in _BACKEND_ATTRS:
        assert getattr(got, attr) == getattr(twin, attr), attr


def test_both_default_backends_are_deterministic(monkeypatch):
    """temperature=0.0 is why these suites are worth running at all."""
    _clear(monkeypatch)
    assert backends.json_backend().temperature == 0.0
    assert backends.text_backend().temperature == 0.0


def test_default_embedder_is_the_cached_local_one(monkeypatch):
    _clear(monkeypatch)
    twin = OllamaEmbedder(model="nomic-embed-text:latest")
    got = backends.embedder()

    assert isinstance(got, CachingEmbedder)
    inner = got._inner
    assert isinstance(inner, OllamaEmbedder)
    for attr in ("model", "base_url", "timeout"):
        assert getattr(inner, attr) == getattr(twin, attr), attr


def test_the_ollama_timeout_knob_still_reaches_the_backend(monkeypatch):
    """``COGNO_OLLAMA_TIMEOUT`` is what keeps the nightly CPU-runner canary green.

    The factory takes its own ``timeout`` (600 s for a backend, 120 s for an embedder)
    and would pin the value regardless of the variable — quietly, since both numbers
    look plausible. Going through the factory must not cost the knob."""
    monkeypatch.setenv("COGNO_OLLAMA_TIMEOUT", "537")
    monkeypatch.delenv(backends.MODEL_ENV, raising=False)
    monkeypatch.delenv(backends.EMBED_MODEL_ENV, raising=False)

    assert backends.json_backend().timeout == 537
    assert backends.text_backend().timeout == 537
    assert backends.embedder()._inner.timeout == 537


# ── half two: the variable actually chooses a PROVIDER ───────────────────────────

def test_a_cloud_spec_does_not_build_an_ollama_backend(monkeypatch):
    """THE twin. No SDK and no key needed: the factory refuses a keyless cloud spec,
    and an ``OllamaBackend`` is built from any string without ever raising."""
    _clear(monkeypatch)
    monkeypatch.setenv(backends.MODEL_ENV, "openai:gpt-4o-mini")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    for build in (backends.json_backend, backends.text_backend):
        with pytest.raises(MissingAPIKeyError) as exc:
            build()
        assert "OPENAI_API_KEY" in str(exc.value)


def test_a_cloud_spec_really_constructs_the_cloud_backend(monkeypatch):
    """The same twin with the outcome named, where the provider SDK is installed."""
    pytest.importorskip("openai", reason="cogno-synapse[openai] not installed here")
    from cogno_synapse import OpenAIBackend

    _clear(monkeypatch)
    monkeypatch.setenv(backends.MODEL_ENV, "openai:gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key-nothing-is-called")

    backend = backends.text_backend()
    assert not isinstance(backend, OllamaBackend)
    assert isinstance(backend, OpenAIBackend)
    assert backend.model == "gpt-4o-mini", "the provider prefix must not survive into the model"
    assert backend.temperature == 0.0


def test_the_embedder_is_steered_by_its_own_variable(monkeypatch):
    """Two axes, like CognoBench's ``--model`` / ``--embed-model``: a cloud LLM run can
    keep the free local embedder, or go cloud end to end when the GPU is busy."""
    _clear(monkeypatch)
    monkeypatch.setenv(backends.EMBED_MODEL_ENV, "openai:text-embedding-3-small")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(MissingAPIKeyError) as exc:
        backends.embedder()
    assert "OPENAI_API_KEY" in str(exc.value)


# ── the skip has to be about what is actually configured ─────────────────────────

async def test_the_skip_names_the_local_server_when_that_is_what_is_missing(monkeypatch):
    _clear(monkeypatch)

    async def _down() -> bool:
        return False

    monkeypatch.setattr(backends, "_ollama_up", _down)
    reason = await backends.backend_unavailable_reason()
    assert reason and "Ollama" in reason


async def test_the_skip_names_the_key_and_never_mentions_ollama_on_a_cloud_run(monkeypatch):
    """An instrument that reports "Ollama is not running" while pointed at OpenAI is
    lying about why it measured nothing. It must not even ASK the local server."""
    _clear(monkeypatch)
    monkeypatch.setenv(backends.MODEL_ENV, "openai:gpt-4o-mini")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def _explode() -> bool:
        raise AssertionError("a cloud run must not probe the local Ollama server")

    monkeypatch.setattr(backends, "_ollama_up", _explode)
    reason = await backends.backend_unavailable_reason()
    assert reason and "OPENAI_API_KEY" in reason
    assert "Ollama" not in reason


async def test_a_reachable_default_reports_no_reason_to_skip(monkeypatch):
    _clear(monkeypatch)

    async def _up() -> bool:
        return True

    monkeypatch.setattr(backends, "_ollama_up", _up)
    assert await backends.backend_unavailable_reason() is None
    assert await backends.embedder_unavailable_reason() is None


# ── the fix has to stay fixed ────────────────────────────────────────────────────

_REAL_CONSTRUCTION = re.compile(r"\b(OllamaBackend|OllamaEmbedder)\s*\(")


def test_no_integration_suite_builds_its_own_real_backend():
    """One place decides which models the integration run talks to.

    This is the defect itself, as a guard: eight files each constructing their own
    ``OllamaBackend`` is what made the suite Ollama-only BY CONSTRUCTION, so that
    ``COGNO_TEST_MODEL`` chose a local model and nothing more. A ninth call site added
    later would restore it one file at a time.

    Monkeypatched sites are exempt and named on the line: they never reach a server, so
    the model string there is a label, not a pull."""
    offenders = []
    for f in sorted((ROOT / "tests" / "integration").glob("*.py")):
        if f.name == "backends.py":
            continue
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if _REAL_CONSTRUCTION.search(line) and "mocked" not in line:
                offenders.append(f"{f.name}:{i}: {line.strip()}")
    assert not offenders, (
        "these build a transport client directly instead of asking "
        "tests/integration/backends.py, which pins them to Ollama: " + "; ".join(offenders)
    )
