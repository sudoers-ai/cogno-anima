import pytest
from typing import Optional

class StubBackend:
    """Zero-network LLM client test double."""
    def __init__(
        self,
        response: str = '{"rewritten": "Hello.", "context_turn": "", "confidence": 0.95, "changed": false, "preserved_terms": [], "rewrite_warnings": []}',
        tokens_in: int = 10,
        tokens_out: int = 5,
        model: str = "stub"
    ):
        self.response = response
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.model = model

    async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
        return self.response, self.tokens_in, self.tokens_out


class StubEmbedder:
    """Zero-network Embedder test double."""
    def __init__(self, vectors: Optional[dict[str, list[float]]] = None, default_similarity: float = 0.8):
        self.vectors = vectors or {}
        self.default_similarity = default_similarity

    async def embed(self, text: str) -> list[float]:
        # Return a deterministic vector based on text length or direct lookup
        if text in self.vectors:
            return self.vectors[text]
        # Return a simple 3D vector for testing
        length = len(text)
        return [float(length), float(length * 2), 1.0]

    async def similarity(self, a: str, b: str) -> float:
        # If they are exactly the same, return 1.0
        if a == b:
            return 1.0
        # If we have vectors, calculate cosine similarity
        v1 = await self.embed(a)
        v2 = await self.embed(b)
        
        # Calculate manually
        import math
        dot = sum(x * y for x, y in zip(v1, v2))
        mag1 = math.sqrt(sum(x * x for x in v1))
        mag2 = math.sqrt(sum(y * y for y in v2))
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot / (mag1 * mag2)


@pytest.fixture
def stub_backend():
    return StubBackend()


@pytest.fixture
def stub_embedder():
    return StubEmbedder()
