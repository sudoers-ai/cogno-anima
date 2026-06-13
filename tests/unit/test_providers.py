import pytest
import json
import httpx
from cogno_core.llm import OllamaBackend, OllamaEmbedder

@pytest.mark.asyncio
async def test_ollama_backend_generate_success(monkeypatch):
    """Successful generation returns text and token counts."""
    class MockResponse:
        status_code = 200
        def json(self):
            return {
                "response": "mocked response",
                "prompt_eval_count": 10,
                "eval_count": 5
            }
        def raise_for_status(self):
            pass

    async def mock_post(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    backend = OllamaBackend(model="llama3")
    text, tokens_in, tokens_out = await backend.generate("system prompt", "user prompt")
    
    assert text == "mocked response"
    assert tokens_in == 10
    assert tokens_out == 5

@pytest.mark.asyncio
async def test_ollama_backend_connection_error(monkeypatch):
    """ConnectError must propagate to the caller."""
    async def mock_post(*args, **kwargs):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    backend = OllamaBackend(model="llama3")
    with pytest.raises(httpx.ConnectError, match="Connection refused"):
        await backend.generate("system prompt", "user prompt")

@pytest.mark.asyncio
async def test_ollama_backend_timeout_error(monkeypatch):
    """TimeoutException must propagate to the caller."""
    async def mock_post(*args, **kwargs):
        raise httpx.ReadTimeout("Request timed out")

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    backend = OllamaBackend(model="llama3")
    with pytest.raises(httpx.TimeoutException):
        await backend.generate("system prompt", "user prompt")

@pytest.mark.asyncio
async def test_ollama_backend_http_error(monkeypatch):
    """HTTP error status (e.g. 500) must propagate via raise_for_status."""
    class MockResponse:
        status_code = 500
        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "Internal Server Error",
                request=httpx.Request("POST", "http://test"),
                response=self,
            )
        def json(self):
            return {}

    async def mock_post(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    backend = OllamaBackend(model="llama3")
    with pytest.raises(httpx.HTTPStatusError):
        await backend.generate("system prompt", "user prompt")

@pytest.mark.asyncio
async def test_ollama_backend_options_payload(monkeypatch):
    """Verify temperature, num_ctx, max_tokens are forwarded correctly in payload."""
    captured_payloads = []

    class MockResponse:
        status_code = 200
        def json(self):
            return {"response": "ok", "prompt_eval_count": 1, "eval_count": 1}
        def raise_for_status(self):
            pass

    async def mock_post(client, url, *args, **kwargs):
        captured_payloads.append(kwargs.get("json", {}))
        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    backend = OllamaBackend(model="test", temperature=0.7, num_ctx=4096, max_tokens=2048)
    await backend.generate("sys", "usr")

    assert len(captured_payloads) == 1
    payload = captured_payloads[0]
    assert payload["model"] == "test"
    assert payload["system"] == "sys"
    assert payload["prompt"] == "usr"
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0.7
    assert payload["options"]["num_ctx"] == 4096
    assert payload["options"]["num_predict"] == 2048

@pytest.mark.asyncio
async def test_ollama_backend_is_available(monkeypatch):
    """is_available returns True when Ollama responds, False when it doesn't."""
    class MockResponse:
        status_code = 200

    async def mock_get_ok(*args, **kwargs):
        return MockResponse()

    async def mock_get_fail(*args, **kwargs):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get_ok)
    backend = OllamaBackend(model="test")
    assert await backend.is_available() is True

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get_fail)
    assert await backend.is_available() is False

@pytest.mark.asyncio
async def test_ollama_embedder_success(monkeypatch):
    """Successful embed returns the embedding vector."""
    class MockResponse:
        status_code = 200
        def json(self):
            return {"embedding": [0.1, 0.2, 0.3]}
        def raise_for_status(self):
            pass

    async def mock_post(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    embedder = OllamaEmbedder(model="nomic-embed-text")
    vec = await embedder.embed("hello")
    assert vec == [0.1, 0.2, 0.3]

    # Test cache hit (no new network call needed)
    vec2 = await embedder.embed("hello")
    assert vec2 == [0.1, 0.2, 0.3]

@pytest.mark.asyncio
async def test_ollama_embedder_empty_text():
    """Empty text returns empty vector without making any network call."""
    embedder = OllamaEmbedder()
    vec = await embedder.embed("")
    assert vec == []

@pytest.mark.asyncio
async def test_ollama_embedder_network_error_propagates(monkeypatch):
    """Network errors during embedding must propagate to the caller."""
    async def mock_post(*args, **kwargs):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    embedder = OllamaEmbedder(model="nomic-embed-text")
    with pytest.raises(httpx.ConnectError):
        await embedder.embed("hello")

@pytest.mark.asyncio
async def test_ollama_embedder_similarity(monkeypatch):
    """Similarity computes cosine distance between two embeddings."""
    vectors = {
        "hello": [1.0, 0.0],
        "world": [0.8, 0.6]
    }
    embedder = OllamaEmbedder()
    async def mock_embed(text):
        return vectors.get(text, [0.0, 0.0])
    
    monkeypatch.setattr(embedder, "embed", mock_embed)
    sim = await embedder.similarity("hello", "world")
    assert sim == pytest.approx(0.8)

@pytest.mark.asyncio
async def test_ollama_embedder_cache_is_case_insensitive(monkeypatch):
    """Cache key normalization: 'Hello' and 'hello' hit the same cache entry."""
    call_count = 0

    class MockResponse:
        status_code = 200
        def json(self):
            return {"embedding": [1.0, 2.0, 3.0]}
        def raise_for_status(self):
            pass

    async def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    embedder = OllamaEmbedder()
    v1 = await embedder.embed("Hello")
    v2 = await embedder.embed("hello")
    assert v1 == v2
    assert call_count == 1  # Only one network call because of cache
