from typing import Protocol, runtime_checkable

@runtime_checkable
class LLMBackend(Protocol):
    """Protocol that any LLM client (OpenAI, Ollama, Bedrock, etc.) must implement."""
    model: str

    async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
        """
        Executes a generation call to the LLM.
        Returns a tuple: (response_text, tokens_in, tokens_out)
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    """Protocol for calculating embeddings and semantic similarity."""
    async def embed(self, text: str) -> list[float]:
        """Generates embedding vector for the given text."""
        ...

    async def similarity(self, a: str, b: str) -> float:
        """Calculates cosine similarity between two texts [0.0, 1.0]."""
        ...
