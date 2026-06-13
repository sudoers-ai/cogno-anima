from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import httpx

from cogno_core.llm.base import LLMBackend, Embedder
from cogno_core.utils import cosine_similarity

logger = logging.getLogger("cogno_core.llm.ollama")


class OllamaBackend(LLMBackend):
    """
    Concrete LLM backend that calls a local Ollama instance.
    """
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = 8192,
        max_tokens: Optional[int] = 4096,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self._endpoint = f"{self.base_url}/api/generate"

    async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
        payload: dict = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
        }
        options: dict = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        if self.max_tokens is not None:
            options["num_predict"] = self.max_tokens
        if options:
            payload["options"] = options

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self._endpoint, json=payload)
        elapsed = (time.perf_counter() - t0) * 1000

        response.raise_for_status()
        data = response.json()

        text = data.get("response", "")
        tokens_in = data.get("prompt_eval_count", 0)
        tokens_out = data.get("eval_count", 0)

        logger.debug(
            "Ollama generate done: elapsed_ms=%.1f tokens_in=%d tokens_out=%d",
            elapsed, tokens_in, tokens_out,
        )
        return text, tokens_in, tokens_out

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                await client.get(f"{self.base_url}/api/tags")
            return True
        except Exception:
            return False


class OllamaEmbedder(Embedder):
    """
    Local embedding provider using Ollama's /api/embeddings endpoint.
    """
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._cache: dict[str, list[float]] = {}

    async def embed(self, text: str) -> list[float]:
        if not text:
            return []

        key = text.strip().lower()
        if key in self._cache:
            return self._cache[key]

        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        embedding = data.get("embedding", [])
        if embedding:
            self._cache[key] = embedding
        return embedding

    async def similarity(self, a: str, b: str) -> float:
        vec_a, vec_b = await asyncio.gather(self.embed(a), self.embed(b))
        return cosine_similarity(vec_a, vec_b)
