"""Where the integration suites get their models.

Every suite used to construct its own ``OllamaBackend(model=…)``, so the whole
integration run was **Ollama-only by construction**: ``COGNO_TEST_MODEL`` picked
WHICH LOCAL MODEL and nothing else. A variable shaped like a provider switch that
is not one is worse than no variable at all — and the documentation said the
suites "read ``COGNO_TEST_MODEL``", which was true and read as a choice it never
offered.

It is not a cosmetic gap. The box that serves real traffic runs the same
``qwen3:8b`` these tests ask for, so a run puts every live turn in a queue behind
the suite, and there was no way to point the suite anywhere else.

Both specs now go through the ``cogno-synapse`` factory and take the same
``provider:model`` grammar CognoBench already uses on both of its axes
(``--model`` / ``--embed-model``)::

    pytest tests/integration                                   # local Ollama, as before
    COGNO_TEST_MODEL=openai:gpt-4o-mini pytest tests/integration
    COGNO_TEST_MODEL=openai:gpt-4o-mini \
        COGNO_TEST_EMBED_MODEL=openai:text-embedding-3-small pytest tests/integration

With neither variable set the suites build exactly what they built before: Ollama,
``qwen3:8b``, ``temperature=0.0``, ``format="json"`` on the JSON backends, and the
local ``nomic-embed-text:latest`` embedder — down to Ollama's own timeout rather
than the factory's (see ``_timeout_kwarg``).

The SKIP travels with the spec. A suite pointed at OpenAI that skipped itself
because "Ollama is not running" would be an instrument lying about why it measured
nothing, so ``backend_unavailable_reason`` asks about whatever provider is actually
configured: the local server for Ollama, the provider's API key for a cloud model.
"""

from __future__ import annotations

import os

import httpx
import pytest
from cogno_synapse import (
    CachingEmbedder,
    Embedder,
    LLMBackend,
    MissingAPIKeyError,
    OllamaBackend,
    create_backend,
    create_embedder,
    parse_model_string,
)

#: Steer the generation model. ``provider:model``; a bare name is Ollama.
MODEL_ENV = "COGNO_TEST_MODEL"
#: Steer the embedding model, independently — a cloud LLM run can keep the free,
#: deterministic local embedder, or go cloud end to end when the GPU is busy.
EMBED_MODEL_ENV = "COGNO_TEST_EMBED_MODEL"

DEFAULT_MODEL = "qwen3:8b"
DEFAULT_EMBED_MODEL = "nomic-embed-text:latest"
BASE_URL = "http://localhost:11434"


def model_spec(env: str = MODEL_ENV, default: str = DEFAULT_MODEL) -> str:
    """The generation spec this run is pointed at. ``env`` lets the NER suite
    read ``COGNO_NER_MODEL`` first, as it always has."""
    return os.environ.get(env) or default


def embed_spec() -> str:
    """The embedding spec this run is pointed at."""
    return os.environ.get(EMBED_MODEL_ENV) or DEFAULT_EMBED_MODEL


def is_ollama(spec: str) -> bool:
    """Does this spec resolve to Ollama? ASKS the factory instead of re-deciding.

    ``partition(":")`` reads ``nomic-embed-text:latest`` as provider
    ``nomic-embed-text``; the factory maps an unknown prefix back to Ollama. The
    bench learned that one the expensive way (``harness.embedder_is_local``)."""
    provider, _model = parse_model_string(spec)
    return provider == "ollama"


def _timeout_kwarg(spec: str) -> dict:
    """Hand the timeout decision back to Ollama instead of taking the factory's.

    ``create_backend``/``create_embedder`` impose their own defaults (600 s / 120 s).
    A direct ``OllamaBackend(model=…)`` — what every call site here used to build —
    asks ``default_timeout()``, i.e. ``$COGNO_OLLAMA_TIMEOUT`` or 120. The nightly
    CI job sets that variable to 600 *because* a CPU runner needs it, so silently
    swapping in a constant here would either re-break that job or hide the knob.
    ``None`` asks the backend for its own default rather than restating it."""
    return {"timeout": None} if is_ollama(spec) else {}


def json_backend(spec: str | None = None) -> LLMBackend:
    """Backend for the JSON-consuming ops (NOUMENO, NER, scope guard, judge).

    On Ollama that means ``format="json"`` — structured decoding, which is an
    Ollama concept and therefore not a factory parameter. Cloud backends are not
    response-format-locked (the JSON stages regex-extract the object out of plain
    text), so there the JSON and text backends are the same object, exactly as in
    CognoBench's ``build_cloud``."""
    spec = spec or model_spec()
    backend = create_backend(spec, temperature=0.0, **_timeout_kwarg(spec))
    if isinstance(backend, OllamaBackend):
        backend.format = "json"
    return backend


def text_backend(spec: str | None = None) -> LLMBackend:
    """Backend for the free-text ops (EGO, voice). No JSON constraint: the EGO's
    text-fallback path emits ``<TOOL_CALL>`` tags, which JSON decoding forbids."""
    spec = spec or model_spec()
    return create_backend(spec, temperature=0.0, **_timeout_kwarg(spec))


def embedder(spec: str | None = None) -> Embedder:
    """The suites' embedder — cached, like every production call site."""
    spec = spec or embed_spec()
    return CachingEmbedder(create_embedder(spec, base_url=BASE_URL, **_timeout_kwarg(spec)))


async def _ollama_up() -> bool:
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            return (await client.get(f"{BASE_URL}/")).status_code == 200
    except Exception:
        return False


async def _unavailable(spec: str, build) -> str | None:
    if is_ollama(spec):
        if await _ollama_up():
            return None
        return f"Local Ollama server ({BASE_URL}) is not running."
    # A cloud spec never touches Ollama, so neither does its skip. The factory
    # itself is the one that knows which key a provider needs — asking it beats
    # keeping a second copy of that table here.
    try:
        build(spec)
    except MissingAPIKeyError as exc:
        return str(exc)
    except ImportError as exc:  # the provider's optional SDK is not installed here
        return f"{spec}: provider SDK is not installed ({exc})."
    return None


async def backend_unavailable_reason(spec: str | None = None) -> str | None:
    """Why the configured generation model cannot run here — ``None`` when it can."""
    return await _unavailable(spec or model_spec(), create_backend)


async def embedder_unavailable_reason(spec: str | None = None) -> str | None:
    """Why the configured embedding model cannot run here — ``None`` when it can."""
    return await _unavailable(spec or embed_spec(), create_embedder)


async def skip_unless_available(*, backend: bool = True, embed: bool = False) -> None:
    """``pytest.skip`` naming what is actually missing, for the models this test needs."""
    if backend:
        reason = await backend_unavailable_reason()
        if reason:
            pytest.skip(reason)
    if embed:
        reason = await embedder_unavailable_reason()
        if reason:
            pytest.skip(reason)
