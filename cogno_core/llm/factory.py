"""
cogno_core.llm.factory — resolve a "provider:model" string to a backend.

A slim, infra-agnostic factory: parse the provider prefix, validate the API key
for cloud providers, instantiate the right backend. A bare string (or "ollama:")
→ Ollama.

Deliberately NOT ported from the parent: the business `_FALLBACK_MATRIX` (model
ladders) and the tenant-key contextvar — those are host concerns. A host that
wants a failover chain composes ``FallbackBackend`` itself.
"""

from __future__ import annotations

import os

from cogno_core.errors import MissingAPIKeyError
from cogno_core.llm.base import LLMBackend

_EXTERNAL = {"openai", "anthropic", "groq", "gemini", "bedrock"}
_KEY_ENV = {
    "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY", "bedrock": "AWS_ACCESS_KEY_ID",
}


def parse_model_string(model_string: str) -> tuple[str, str]:
    """"provider:model" → (provider, model). Bare/unknown prefix → ("ollama", ...)."""
    if not model_string:
        return "ollama", "llama3.2"
    if ":" in model_string:
        prefix, rest = model_string.split(":", 1)
        prefix = prefix.lower()
        if prefix in _EXTERNAL or prefix == "ollama":
            return prefix, rest
    return "ollama", model_string


def _key_present(provider: str) -> bool:
    val = os.environ.get(_KEY_ENV[provider], "")
    return bool(val) and val.lower() != "dummy" and "sk-proj-*" not in val


def create_backend(
    model_string: str,
    *,
    base_url: str = "http://localhost:11434",
    temperature: float | None = None,
    num_ctx: int | None = 8192,
    max_tokens: int = 4096,
    timeout: int = 600,
) -> LLMBackend:
    """Instantiate a single backend for ``model_string`` (e.g. "openai:gpt-4o-mini").

    Raises ``MissingAPIKeyError`` if a cloud provider is requested without a
    usable key (fail loudly, never silently degrade).
    """
    provider, model = parse_model_string(model_string)

    if provider in _EXTERNAL and not _key_present(provider):
        raise MissingAPIKeyError(
            f"Model '{model_string}' needs {_KEY_ENV[provider]}, which is unset or a placeholder."
        )

    if provider == "openai":
        from cogno_core.llm.openai_backend import OpenAIBackend
        return OpenAIBackend(model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    if provider == "anthropic":
        from cogno_core.llm.anthropic_backend import AnthropicBackend
        return AnthropicBackend(model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    if provider == "groq":
        from cogno_core.llm.groq_backend import GroqBackend
        return GroqBackend(model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    if provider == "gemini":
        from cogno_core.llm.gemini_backend import GeminiBackend
        return GeminiBackend(model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    if provider == "bedrock":
        from cogno_core.llm.bedrock_backend import BedrockBackend
        return BedrockBackend(model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout)

    from cogno_core.llm.ollama import OllamaBackend
    return OllamaBackend(model=model, base_url=base_url, temperature=temperature,
                         num_ctx=num_ctx, max_tokens=max_tokens, timeout=timeout)
