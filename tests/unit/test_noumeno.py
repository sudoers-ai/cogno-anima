import pytest
from pathlib import Path
from cogno_core.types import PipelineContext
from cogno_core.stages.noumeno import Noumeno, classify_drift
from cogno_core.utils import cosine_similarity
from tests.conftest import StubBackend, StubEmbedder

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"
SLANGS = {
    "vc": "você",
    "pq": "porque",
    "pfv": "por favor",
}


def make_noumeno(
    response_json: str = '{"rewritten": "Hello.", "context_turn": "", "confidence": 0.95, "changed": false, "preserved_terms": [], "rewrite_warnings": []}',
    embedder=None,
) -> Noumeno:
    emb = embedder or StubEmbedder()
    return Noumeno(
        embedder=emb,
        prompts_dir=PROMPTS_DIR,
        slangs=SLANGS,
    )


class FixedSimilarityEmbedder(StubEmbedder):
    """Embedder that always returns the same similarity value regardless of inputs."""
    def __init__(self, sim_value: float):
        super().__init__()
        self.sim_value = sim_value

    async def similarity(self, a: str, b: str) -> float:
        return self.sim_value


# ────────────────────────────────────────────────────────────────────
#  classify_drift — Pure function tests
# ────────────────────────────────────────────────────────────────────

class TestClassifyDrift:

    def test_pass_through(self):
        assert classify_drift(0.0) == "PASS_THROUGH"

    def test_rewritten_range(self):
        assert classify_drift(0.01) == "REWRITTEN"
        assert classify_drift(0.15) == "REWRITTEN"
        assert classify_drift(0.20) == "REWRITTEN"   # upper boundary inclusive

    def test_compressed_range(self):
        assert classify_drift(0.21) == "COMPRESSED"
        assert classify_drift(0.30) == "COMPRESSED"
        assert classify_drift(0.40) == "COMPRESSED"   # upper boundary inclusive

    def test_expanded_range(self):
        assert classify_drift(0.41) == "EXPANDED"
        assert classify_drift(0.55) == "EXPANDED"
        assert classify_drift(0.60) == "EXPANDED"     # upper boundary inclusive

    def test_drift_range(self):
        assert classify_drift(0.61) == "DRIFT"
        assert classify_drift(0.70) == "DRIFT"
        assert classify_drift(1.0) == "DRIFT"


# ────────────────────────────────────────────────────────────────────
#  cosine_similarity — Pure function tests
# ────────────────────────────────────────────────────────────────────

class TestCosineSimilarity:

    def test_identical_vectors(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_opposite_vectors(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_known_angle(self):
        # [1, 0] vs [0.8, 0.6] = 0.8
        assert cosine_similarity([1.0, 0.0], [0.8, 0.6]) == pytest.approx(0.8)

    def test_empty_vectors(self):
        assert cosine_similarity([], [1.0]) == 0.0
        assert cosine_similarity([1.0], []) == 0.0
        assert cosine_similarity([], []) == 0.0

    def test_mismatched_lengths(self):
        assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0

    def test_zero_vectors(self):
        assert cosine_similarity([0.0], [0.0]) == 0.0
        assert cosine_similarity([1.0], [0.0]) == 0.0
        assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0


# ────────────────────────────────────────────────────────────────────
#  Noumeno Stage — Full process() tests
# ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestNoumenoStage:

    # ── Result Shape ──────────────────────────────────────────────

    async def test_result_shape(self):
        """Must return a PipelineContext with populated NoumenoResult and all expected fields."""
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="olá, como vai você?")
        ctx = await noumeno.process(ctx, StubBackend(
            response='{"rewritten": "Hello, how are you?", "context_turn": "greeting", "confidence": 0.99, "changed": true, "preserved_terms": ["Bitcoin"], "rewrite_warnings": ["ambiguity"]}'
        ))

        assert ctx.noumeno is not None
        assert ctx.noumeno.original == "olá, como vai você?"
        assert ctx.noumeno.rewritten == "Hello, how are you?"
        assert ctx.noumeno.context_turn == "greeting"
        assert ctx.noumeno.language == "pt"
        assert ctx.noumeno.canonical_language == "en"
        assert ctx.noumeno.changed is True
        assert ctx.noumeno.confidence == 0.99
        assert ctx.noumeno.preserved_terms == ["Bitcoin"]
        assert ctx.noumeno.rewrite_warnings == ["ambiguity"]
        assert ctx.noumeno.metrics.stage == "noumeno"

    async def test_context_is_returned_same_object(self):
        """process() returns the same PipelineContext object it received."""
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="test")
        result = await noumeno.process(ctx, StubBackend())
        assert result is ctx

    # ── Slang Expansion ──────────────────────────────────────────

    async def test_slang_expansion(self):
        """Slang terms must be expanded prior to further processing."""
        captured_prompts = []

        class CaptureBackend(StubBackend):
            async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
                captured_prompts.append(prompt)
                return self.response, self.tokens_in, self.tokens_out

        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="pfv, vc pode me ajudar?")
        await noumeno.process(ctx, CaptureBackend(
            response='{"rewritten": "please, can you help me?", "context_turn": "", "confidence": 0.9, "changed": true, "preserved_terms": [], "rewrite_warnings": []}'
        ))

        assert len(captured_prompts) == 1
        assert "por favor, você pode me ajudar?" in captured_prompts[0]

    async def test_no_slangs_dict(self):
        """When slangs dict is empty, text passes through unchanged."""
        captured_prompts = []

        class CaptureBackend(StubBackend):
            async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
                captured_prompts.append(prompt)
                return self.response, self.tokens_in, self.tokens_out

        noumeno = Noumeno(embedder=StubEmbedder(), prompts_dir=PROMPTS_DIR)
        ctx = PipelineContext(user_input="vc tá aí?")
        await noumeno.process(ctx, CaptureBackend())
        assert "vc tá aí?" in captured_prompts[0]

    # ── Language Detection ───────────────────────────────────────

    async def test_language_detection_pt(self):
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="olá, como vai você?")
        ctx = await noumeno.process(ctx, StubBackend())
        assert ctx.noumeno.language == "pt"

    async def test_language_detection_en(self):
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="what is the price of bitcoin today?")
        ctx = await noumeno.process(ctx, StubBackend())
        assert ctx.noumeno.language == "en"

    async def test_language_detection_es(self):
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="¿dónde está mi dinero?")
        ctx = await noumeno.process(ctx, StubBackend())
        assert ctx.noumeno.language == "es"

    async def test_force_language_overrides_detection(self):
        """If force_language is set in context, bypass language detection entirely."""
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="ola como vai", force_language="fr")
        ctx = await noumeno.process(ctx, StubBackend())
        assert ctx.noumeno.language == "fr"

    async def test_language_detection_failure_defaults_to_und(self, monkeypatch):
        """If langdetect raises, language defaults to 'und'."""
        import langdetect
        def broken_detect(text):
            raise langdetect.lang_detect_exception.LangDetectException(0, "boom")

        monkeypatch.setattr("langdetect.detect", broken_detect)

        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="hello world")
        ctx = await noumeno.process(ctx, StubBackend())
        assert ctx.noumeno.language == "und"

    # ── Subject Continuity ───────────────────────────────────────

    async def test_subject_continuity_same_subject(self):
        """High similarity → same subject, context_turn preserved."""
        embedder = FixedSimilarityEmbedder(0.85)
        noumeno = make_noumeno(embedder=embedder)

        ctx = PipelineContext(user_input="bitcoin price")
        ctx.metadata["last_rewritten"] = "what is the price of bitcoin?"
        ctx.metadata["last_context_turn"] = "crypto trading"

        ctx = await noumeno.process(ctx, StubBackend(
            response='{"rewritten": "bitcoin price", "context_turn": "crypto trading", "confidence": 0.95, "changed": false, "preserved_terms": [], "rewrite_warnings": []}'
        ))
        assert ctx.noumeno.change_subject is False
        assert ctx.noumeno.subject_similarity == 0.85
        assert ctx.noumeno.context_turn == "crypto trading"
        assert ctx.noumeno.context_used is True

    async def test_subject_continuity_new_subject(self):
        """Low similarity → new subject, context_turn cleared."""
        embedder = FixedSimilarityEmbedder(0.3)
        noumeno = make_noumeno(embedder=embedder)

        ctx = PipelineContext(user_input="weather today")
        ctx.metadata["last_rewritten"] = "what is the price of bitcoin?"
        ctx.metadata["last_context_turn"] = "crypto trading"

        ctx = await noumeno.process(ctx, StubBackend(
            response='{"rewritten": "weather today", "context_turn": "weather", "confidence": 0.95, "changed": true, "preserved_terms": [], "rewrite_warnings": []}'
        ))
        assert ctx.noumeno.change_subject is True
        assert ctx.noumeno.subject_similarity == 0.3
        assert ctx.noumeno.context_turn == ""        # Cleared because change_subject
        assert ctx.noumeno.context_used is False

    async def test_no_history_skips_subject_check(self):
        """Without last_rewritten in metadata, similarity defaults to 1.0."""
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="hello")
        ctx = await noumeno.process(ctx, StubBackend())
        assert ctx.noumeno.subject_similarity == 1.0
        assert ctx.noumeno.change_subject is False

    async def test_custom_subject_threshold(self):
        """subject_threshold parameter controls the cutoff for subject change detection."""
        embedder = FixedSimilarityEmbedder(0.50)

        noumeno_low_threshold = Noumeno(
            embedder=embedder, prompts_dir=PROMPTS_DIR, subject_threshold=0.40
        )
        noumeno_high_threshold = Noumeno(
            embedder=embedder, prompts_dir=PROMPTS_DIR, subject_threshold=0.60
        )

        ctx_low = PipelineContext(user_input="test")
        ctx_low.metadata["last_rewritten"] = "previous"
        ctx_low = await noumeno_low_threshold.process(ctx_low, StubBackend())
        assert ctx_low.noumeno.change_subject is False  # 0.50 >= 0.40

        ctx_high = PipelineContext(user_input="test")
        ctx_high.metadata["last_rewritten"] = "previous"
        ctx_high = await noumeno_high_threshold.process(ctx_high, StubBackend())
        assert ctx_high.noumeno.change_subject is True  # 0.50 < 0.60

    # ── Drift Score & Reconciliation ────────────────────────────

    async def test_drift_low_score(self):
        """Low drift → REWRITTEN tag, changed preserved from LLM."""
        embedder = FixedSimilarityEmbedder(0.9)
        noumeno = make_noumeno(embedder=embedder)

        ctx = PipelineContext(user_input="original input")
        ctx = await noumeno.process(ctx, StubBackend(
            response='{"rewritten": "rewritten input", "context_turn": "", "confidence": 0.9, "changed": false, "preserved_terms": [], "rewrite_warnings": []}'
        ))
        assert ctx.noumeno.drift_score == pytest.approx(0.1)
        assert ctx.noumeno.drift_tag == "REWRITTEN"
        assert ctx.noumeno.changed is False

    async def test_drift_high_score_reconciliation(self):
        """High drift (>0.50) → forces changed=True and drift_tag='DRIFT'."""
        embedder = FixedSimilarityEmbedder(0.4)
        noumeno = make_noumeno(embedder=embedder)

        ctx = PipelineContext(user_input="original input")
        ctx = await noumeno.process(ctx, StubBackend(
            response='{"rewritten": "completely different output", "context_turn": "", "confidence": 0.9, "changed": false, "preserved_terms": [], "rewrite_warnings": []}'
        ))
        assert ctx.noumeno.drift_score > 0.50
        assert ctx.noumeno.drift_tag == "DRIFT"
        assert ctx.noumeno.changed is True  # Reconciled!

    async def test_drift_zero_when_identical(self):
        """If input == rewritten, drift should be 0.0 and tag PASS_THROUGH."""
        embedder = FixedSimilarityEmbedder(1.0)
        noumeno = make_noumeno(embedder=embedder)

        ctx = PipelineContext(user_input="hello world")
        ctx = await noumeno.process(ctx, StubBackend(
            response='{"rewritten": "hello world", "context_turn": "", "confidence": 1.0, "changed": false, "preserved_terms": [], "rewrite_warnings": []}'
        ))
        assert ctx.noumeno.drift_score == 0.0
        assert ctx.noumeno.drift_tag == "PASS_THROUGH"

    # ── Exception Propagation ───────────────────────────────────

    async def test_llm_failure_propagates(self):
        """If LLM raises, the exception must propagate to the caller."""
        class FailingLLM(StubBackend):
            async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
                raise RuntimeError("Fatal API Error")

        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="test input")
        with pytest.raises(RuntimeError, match="Fatal API Error"):
            await noumeno.process(ctx, FailingLLM())

    async def test_subject_similarity_failure_propagates(self):
        """If embedder.similarity() raises during subject check, exception propagates."""
        class FailingEmbedder(StubEmbedder):
            async def similarity(self, a: str, b: str) -> float:
                raise RuntimeError("Embedder down")

        noumeno = make_noumeno(embedder=FailingEmbedder())
        ctx = PipelineContext(user_input="ethereum")
        ctx.metadata["last_rewritten"] = "bitcoin"
        with pytest.raises(RuntimeError, match="Embedder down"):
            await noumeno.process(ctx, StubBackend())

    async def test_drift_similarity_failure_propagates(self):
        """If embedder.similarity() raises during drift computation, exception propagates."""
        class FailOnDrift(StubEmbedder):
            async def similarity(self, a: str, b: str) -> float:
                if b == "rewritten text":
                    raise RuntimeError("Drift computation failed")
                return 0.9

        noumeno = make_noumeno(embedder=FailOnDrift())
        ctx = PipelineContext(user_input="original")
        with pytest.raises(RuntimeError, match="Drift computation failed"):
            await noumeno.process(ctx, StubBackend(
                response='{"rewritten": "rewritten text", "context_turn": "", "confidence": 0.9, "changed": false, "preserved_terms": [], "rewrite_warnings": []}'
            ))



    # ── LLM Response Parsing ───────────────────────────────────

    async def test_json_wrapped_in_markdown_fences(self):
        """If response is wrapped in ```json ... ```, parse successfully."""
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="test")
        ctx = await noumeno.process(ctx, StubBackend(
            response='```json\n{"rewritten": "test text", "context_turn": "", "confidence": 0.9, "changed": false, "preserved_terms": [], "rewrite_warnings": []}\n```'
        ))
        assert ctx.noumeno.rewritten == "test text"

    async def test_json_parse_fails_on_invalid_json(self):
        """If LLM returns invalid JSON, raise json.JSONDecodeError."""
        import json
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="original")
        with pytest.raises(json.JSONDecodeError):
            await noumeno.process(ctx, StubBackend(
                response='This is not JSON at all, just a plain text rewrite.'
            ))

    async def test_empty_llm_response_fails_parse(self):
        """If LLM returns empty string, raise json.JSONDecodeError."""
        import json
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="original")
        with pytest.raises(json.JSONDecodeError):
            await noumeno.process(ctx, StubBackend(response=""))

    async def test_whitespace_only_response_fails_parse(self):
        """If LLM returns whitespace only, raise json.JSONDecodeError."""
        import json
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="original")
        with pytest.raises(json.JSONDecodeError):
            await noumeno.process(ctx, StubBackend(response="   "))

    async def test_json_missing_rewritten_field(self):
        """If JSON is valid but missing 'rewritten', use original input."""
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="my question")
        ctx = await noumeno.process(ctx, StubBackend(
            response='{"context_turn": "topic", "confidence": 0.8, "changed": false, "preserved_terms": [], "rewrite_warnings": []}'
        ))
        assert ctx.noumeno.rewritten == "my question"

    async def test_json_empty_rewritten_field(self):
        """If JSON has empty 'rewritten', fall back to original input."""
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="my question")
        ctx = await noumeno.process(ctx, StubBackend(
            response='{"rewritten": "", "context_turn": "", "confidence": 0.8, "changed": false, "preserved_terms": [], "rewrite_warnings": []}'
        ))
        assert ctx.noumeno.rewritten == "my question"

    # ── Metrics ─────────────────────────────────────────────────

    async def test_metrics_populated(self):
        """Metrics must correctly populate token counts, model and elapsed time."""
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="valid test input")
        ctx = await noumeno.process(ctx, StubBackend(
            response='{"rewritten": "result", "context_turn": "", "confidence": 0.9, "changed": false, "preserved_terms": [], "rewrite_warnings": []}',
            tokens_in=42,
            tokens_out=24,
            model="custom-stub"
        ))
        m = ctx.noumeno.metrics
        assert m.stage == "noumeno"
        assert m.tokens_in == 42
        assert m.tokens_out == 24
        assert m.tokens_total == 66
        assert m.model == "custom-stub"
        assert m.elapsed_ms > 0.0

    async def test_model_name_from_backend(self):
        """Model name in metrics matches the active LLM backend's model property."""
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="test")
        ctx = await noumeno.process(ctx, StubBackend(model="llama3-model"))
        assert ctx.noumeno.metrics.model == "llama3-model"

    async def test_pipeline_context_aggregate_metrics(self):
        """PipelineContext properties (noumeno_metrics, total_tokens, etc.) work correctly."""
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="test")
        ctx = await noumeno.process(ctx, StubBackend(tokens_in=10, tokens_out=5))

        assert ctx.noumeno_metrics is not None
        assert ctx.noumeno_metrics.tokens_total == 15
        assert ctx.total_tokens == 15
        assert ctx.total_elapsed_ms > 0.0
        assert len(ctx.stage_metrics) == 1

    # ── History Injection ───────────────────────────────────────

    async def test_history_injected_when_same_subject(self):
        """When subject stays, the prompt includes history from metadata."""
        captured_prompts = []

        class CaptureBackend(StubBackend):
            async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
                captured_prompts.append(prompt)
                return self.response, self.tokens_in, self.tokens_out

        embedder = FixedSimilarityEmbedder(0.9)  # Same subject
        noumeno = make_noumeno(embedder=embedder)

        ctx = PipelineContext(user_input="tell me more")
        ctx.metadata["last_rewritten"] = "What is quantum physics?"
        ctx.metadata["last_context_turn"] = "science discussion"

        await noumeno.process(ctx, CaptureBackend())

        assert "What is quantum physics?" in captured_prompts[0]
        assert "science discussion" in captured_prompts[0]

    async def test_history_not_injected_when_subject_changes(self):
        """When subject changes, the prompt does NOT include previous history."""
        captured_prompts = []

        class CaptureBackend(StubBackend):
            async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
                captured_prompts.append(prompt)
                return self.response, self.tokens_in, self.tokens_out

        embedder = FixedSimilarityEmbedder(0.2)  # New subject
        noumeno = make_noumeno(embedder=embedder)

        ctx = PipelineContext(user_input="weather forecast")
        ctx.metadata["last_rewritten"] = "What is quantum physics?"
        ctx.metadata["last_context_turn"] = "science discussion"

        await noumeno.process(ctx, CaptureBackend())

        assert "What is quantum physics?" not in captured_prompts[0]
        assert "science discussion" not in captured_prompts[0]
