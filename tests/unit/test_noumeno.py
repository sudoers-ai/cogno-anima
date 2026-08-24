from cogno_anima.errors import StageParseError
import pytest
from pathlib import Path
from cogno_anima.types import PipelineContext
from cogno_anima.stages.noumeno import Noumeno, classify_drift
from cogno_anima.utils import cosine_similarity
from tests.conftest import StubBackend, StubEmbedder

PROMPTS_DIR = Path(__file__).parent.parent.parent / "cogno_anima" / "prompt_templates"
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
        # History, so `context_turn` survives and this stays a test of the RESULT SHAPE. Without
        # it the field is now cleared on purpose (a model cannot know prior context that was
        # never in its prompt — see the fabricated-context test below), and this case would be
        # asserting the no-history rule instead of the shape it is named for.
        ctx.metadata["last_rewritten"] = "Hello there."
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

    async def test_the_langdetect_fallback_is_DETERMINISTIC(self):
        """Unseeded, langdetect draws random samples and answers differently for the same
        string — measured 2026-08-24: "Consulta 14h" resolved to three different languages
        across three consecutive calls. That is not a cosmetic flake. This branch decides the
        language the turn is REWRITTEN from, and every stage downstream reads that rewrite, so
        an unseeded fallback means one contact's retry can be understood as a different
        language than their first attempt.

        Two assertions, because either alone is weak: the seed is pinned exactly (a mutation
        removing the line dies immediately), and the behaviour is checked over repeats (so a
        future change that seeds some other way still has to keep the property).
        """
        from langdetect import DetectorFactory

        DetectorFactory.seed = 7                      # any other value, to prove we set it
        noumeno = make_noumeno()                      # no force_language, no default → fallback
        ctx = await noumeno.process(
            PipelineContext(user_input="¿dónde está mi dinero?"), StubBackend())
        assert DetectorFactory.seed == 0, "the fallback ran without pinning langdetect's seed"

        first = ctx.noumeno.language
        for _ in range(8):
            again = await noumeno.process(
                PipelineContext(user_input="¿dónde está mi dinero?"), StubBackend())
            assert again.noumeno.language == first, "the same input resolved differently"

    async def test_force_language_overrides_detection(self):
        """If force_language is set in context, bypass language detection entirely."""
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="ola como vai", force_language="fr")
        ctx = await noumeno.process(ctx, StubBackend())
        assert ctx.noumeno.language == "fr"

    async def test_default_language_used_when_no_force(self, monkeypatch):
        """default_language is used when the request has no force_language (langdetect NOT called)."""
        def boom(text):
            raise AssertionError("langdetect must not run when default_language is set")
        monkeypatch.setattr("langdetect.detect", boom)

        noumeno = Noumeno(embedder=StubEmbedder(), prompts_dir=PROMPTS_DIR, default_language="pt-BR")
        ctx = PipelineContext(user_input="qualquer entrada")
        ctx = await noumeno.process(ctx, StubBackend())
        assert ctx.noumeno.language == "pt-BR"

    async def test_force_language_overrides_default_language(self):
        """Per-request force_language wins over the stage default_language (tenant precedence)."""
        noumeno = Noumeno(embedder=StubEmbedder(), prompts_dir=PROMPTS_DIR, default_language="pt-BR")
        ctx = PipelineContext(user_input="hello there", force_language="es")
        ctx = await noumeno.process(ctx, StubBackend())
        assert ctx.noumeno.language == "es"

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

    async def test_a_fabricated_context_turn_is_dropped_on_a_first_turn(self):
        """No history → no context, whatever the model claims.

        The prompt only carries a "Last Query" block when `last_rewritten` exists, but a model
        asked to produce `context_turn` fills it anyway. Measured against a real model on turn
        ONE with no history: it returned "The user is asking about cryptocurrency prices." — a
        summary of the CURRENT turn dressed as prior context — and the pipeline recorded
        `context_used=True` on a first contact.

        Same class as trusting the model's own PII verdict: the host knows whether history
        exists, so the host decides. Only an Ollama-backed integration test covered this, and
        CI skips those, so it reached main invisibly.
        """
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="Qual o preço do bitcoin?")   # no last_rewritten
        ctx = await noumeno.process(ctx, StubBackend(
            response='{"rewritten": "What is the price of Bitcoin?", "context_turn": "The user '
                     'is asking about cryptocurrency prices.", "confidence": 0.95, '
                     '"changed": true, "preserved_terms": [], "rewrite_warnings": []}'
        ))
        assert ctx.noumeno.context_used is False
        # `context_turn` itself is KEPT. It is a restatement of the current turn rather than
        # prior context, and the NER — its only consumer — measurably uses it: clearing it
        # flipped `temporal` from MIXED to RECENT on a cognobench case, 3/3 runs either way.
        # The defect was the flag claiming provenance, not the text being there.
        assert ctx.noumeno.context_turn != ""

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
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="original")
        with pytest.raises(StageParseError):
            await noumeno.process(ctx, StubBackend(
                response='This is not JSON at all, just a plain text rewrite.'
            ))

    async def test_json_with_trailing_extra_data_parses_first_object(self):
        """A cloud backend without format=json emits the object + extra text ("Extra
        data") — the first valid object is the response (raw_decode fallback)."""
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="test")
        ctx = await noumeno.process(ctx, StubBackend(
            response='{"rewritten": "clean text", "context_turn": "", "confidence": 0.9, '
                     '"changed": false, "preserved_terms": [], "rewrite_warnings": []} '
                     'Some trailing commentary the model added.'
        ))
        assert ctx.noumeno.rewritten == "clean text"

    async def test_empty_llm_response_fails_parse(self):
        """If LLM returns empty string, raise json.JSONDecodeError."""
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="original")
        with pytest.raises(StageParseError):
            await noumeno.process(ctx, StubBackend(response=""))

    async def test_whitespace_only_response_fails_parse(self):
        """If LLM returns whitespace only, raise json.JSONDecodeError."""
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="original")
        with pytest.raises(StageParseError):
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

    async def test_embedding_tokens_captured_in_metrics(self):
        """Embedding token cost (from similarity calls) is recorded in StageMetrics."""

        class UsageEmbedder(StubEmbedder):
            async def similarity_with_usage(self, a: str, b: str) -> tuple[float, int]:
                return 0.9, 6   # 6 tokens per similarity call

        noumeno = make_noumeno(embedder=UsageEmbedder())
        ctx = PipelineContext(user_input="first turn")
        # No history → exactly one similarity call (drift) → 6 tokens, 2 embeds.
        ctx = await noumeno.process(ctx, StubBackend(tokens_in=10, tokens_out=5))

        m = ctx.noumeno.metrics
        assert m.embedding_tokens == 6
        assert m.embedding_calls == 2
        # tokens_total folds embeddings in: 10 + 5 + 6
        assert m.tokens_total == 21
        assert ctx.total_tokens == 21
        assert ctx.total_llm_tokens == 15
        assert ctx.total_embedding_tokens == 6

    async def test_embedding_tokens_accumulate_across_similarity_calls(self):
        """With history, both the subject-check and drift similarities are billed."""

        class UsageEmbedder(StubEmbedder):
            async def similarity_with_usage(self, a: str, b: str) -> tuple[float, int]:
                return 0.95, 4

        noumeno = make_noumeno(embedder=UsageEmbedder())
        ctx = PipelineContext(user_input="follow up")
        ctx.metadata["last_rewritten"] = "previous english query"
        ctx = await noumeno.process(ctx, StubBackend(tokens_in=10, tokens_out=5))

        m = ctx.noumeno.metrics
        assert m.embedding_tokens == 8   # two similarity calls × 4
        assert m.embedding_calls == 4

    async def test_embedding_tokens_zero_for_plain_embedder(self):
        """A plain Embedder (no usage method) reports 0 embedding tokens, no crash."""
        noumeno = make_noumeno()   # StubEmbedder has no similarity_with_usage
        ctx = PipelineContext(user_input="hello there")
        ctx = await noumeno.process(ctx, StubBackend(tokens_in=10, tokens_out=5))
        m = ctx.noumeno.metrics
        assert m.embedding_tokens == 0
        assert m.embedding_calls == 2   # still counts the operations
        assert m.tokens_total == 15

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


def _resp(rewritten: str, warnings: "list[str] | None" = None) -> str:
    import json as _json
    return _json.dumps({
        "rewritten": rewritten, "context_turn": "", "confidence": 0.95,
        "changed": False, "preserved_terms": [], "rewrite_warnings": warnings or []})


class SeqBackend(StubBackend):
    """Scripted backend: pops one response per call and records each system prompt,
    so tests can assert the echo retry really dropped the Examples block."""

    def __init__(self, responses: list[str]):
        super().__init__()
        self.responses = list(responses)
        self.systems: list[str] = []

    async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
        self.systems.append(system)
        return self.responses.pop(0), self.tokens_in, self.tokens_out


@pytest.mark.asyncio
class TestFewShotEchoBackstop:
    """The stage must not ship a rewrite copied from its own few-shot examples.

    Live defect (CLOSER, 2026-08-18): a bare "Sim" was rewritten as the configure-it
    example output byte-for-byte, fabricating the task every downstream stage then
    executed — the conversation looped on the same question for 6 turns."""

    # A real example output from prompt_templates/noumeno/system.txt — the tests break
    # if the prompt and this constant drift apart, which is the point: the detector's
    # material IS the prompt.
    ECHO = "Yes, I would like to know how to repot the fern."

    async def test_echo_is_retried_without_examples_and_retry_wins(self):
        backend = SeqBackend([_resp(self.ECHO),
                              _resp("Yes, I have lost customers due to slow replies.")])
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="Sim")

        await noumeno.process(ctx, backend)

        assert ctx.noumeno.rewritten == "Yes, I have lost customers due to slow replies."
        assert len(backend.systems) == 2
        assert "Examples:" in backend.systems[0]
        assert "Examples:" not in backend.systems[1]
        assert "repot the fern" not in backend.systems[1]
        assert ctx.noumeno.rewrite_warnings == []
        # Both calls are billed: the retry must show up in metering.
        assert ctx.noumeno.metrics.tokens_in == 2 * backend.tokens_in
        assert ctx.noumeno.metrics.tokens_out == 2 * backend.tokens_out

    async def test_retry_reproducing_the_sentence_is_genuine(self):
        """The retry cannot parrot what is not in its context — same answer twice
        means real resolution, and it ships unflagged."""
        backend = SeqBackend([_resp(self.ECHO), _resp(self.ECHO)])
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="Sim")

        await noumeno.process(ctx, backend)

        assert ctx.noumeno.rewritten == self.ECHO
        assert len(backend.systems) == 2
        assert ctx.noumeno.rewrite_warnings == []

    async def test_head_swapped_copy_is_detected_by_the_tail(self):
        """"Yes, …" pasted over the example's "Maybe, …" keeps the distinctive tail —
        observed live as "Yes, I would like you to schedule the appointment."."""
        swapped = "Yes, I want you to water the succulents today."
        backend = SeqBackend([_resp(swapped), _resp("Yes, water them today, please.")])
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="talvez")

        await noumeno.process(ctx, backend)

        assert ctx.noumeno.rewritten == "Yes, water them today, please."
        assert len(backend.systems) == 2

    async def test_unparseable_retry_keeps_first_answer_flagged(self):
        """The backstop never kills a turn the primary call served: garbage on the
        retry ships the first answer WITH the doubt flag."""
        backend = SeqBackend([_resp(self.ECHO), "not json at all"])
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="Sim")

        await noumeno.process(ctx, backend)

        assert ctx.noumeno.rewritten == self.ECHO
        assert "FEW_SHOT_ECHO" in ctx.noumeno.rewrite_warnings

    async def test_long_input_matching_an_example_is_coincidence(self):
        """Expansion (and parroting) risk is short-reply territory; a long input that
        lands on an example is legitimate and must cost one call only."""
        backend = SeqBackend([_resp(self.ECHO)])
        noumeno = make_noumeno()
        ctx = PipelineContext(
            user_input="quero saber como replantar a minha samambaia grande")

        await noumeno.process(ctx, backend)

        assert ctx.noumeno.rewritten == self.ECHO
        assert len(backend.systems) == 1

    async def test_passthrough_of_the_input_is_never_a_parrot(self):
        """"bloop zorg fnarg" IS an example output — but echoing the user's own words
        is pass-through behaviour, not fabrication."""
        backend = SeqBackend([_resp("bloop zorg fnarg")])
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="bloop zorg fnarg")

        await noumeno.process(ctx, backend)

        assert ctx.noumeno.rewritten == "bloop zorg fnarg"
        assert len(backend.systems) == 1

    async def test_rule_illustration_parrot_without_anchor_is_flagged_not_retried(self):
        """"Book with Dr. Vinicius Vale" lives in the RULES — the retry prompt keeps it,
        so no retry can clear it (review finding): an unsupported match ships flagged,
        in ONE call."""
        backend = SeqBackend([_resp("Book with Dr. Vinicius Vale.")])
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="com a Ana")

        await noumeno.process(ctx, backend)

        assert len(backend.systems) == 1
        assert "FEW_SHOT_ECHO" in ctx.noumeno.rewrite_warnings

    async def test_rule_illustration_with_anchor_in_input_is_clean(self):
        backend = SeqBackend([_resp("Book with Dr. Vinicius Vale.")])
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="com o Vinicius Vale")

        await noumeno.process(ctx, backend)

        assert len(backend.systems) == 1
        assert ctx.noumeno.rewrite_warnings == []

    async def test_rule_illustration_with_anchor_in_conversation_is_clean(self):
        """"Sim" right after the assistant offered Vinicius Vale IS real resolution —
        the anchor lives in the conversation, not the input."""
        from cogno_anima import metakeys as mk
        backend = SeqBackend([_resp("Book with Dr. Vinicius Vale.")])
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="Sim")
        ctx.metadata[mk.CONVERSATION_HISTORY] = (
            "Assistant: Quer que eu agende com o Vinicius Vale?")

        await noumeno.process(ctx, backend)

        assert len(backend.systems) == 1
        assert ctx.noumeno.rewrite_warnings == []

    async def test_harvest_stays_aligned_with_the_shipped_prompt(self):
        """The detector's material IS the prompt: harvested example count must equal an
        independent parse (Output: blocks), the rules region must carry exactly the one
        known illustration, and the Examples block must stay terminal (a rule bullet
        after it would silently weaken the retry prompt)."""
        noumeno = make_noumeno()
        system = noumeno._system
        tail = system[system.find("\nExamples:"):]
        assert len(noumeno._example_rewrites) == tail.count("Output:")
        assert len(noumeno._example_rewrites) >= 6
        assert len(noumeno._rule_rewrites) == 1
        assert not any(line.startswith("* ") for line in tail.splitlines())

    async def test_slang_expansion_does_not_disarm_the_gate(self):
        """The short-reply gate is measured on the RAW reply: 'sim pfv pode ser'
        (4 words) expands past 4 words and must still be guarded."""
        backend = SeqBackend([_resp(self.ECHO),
                              _resp("Yes, that works for me, please.")])
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="sim pfv pode ser")

        await noumeno.process(ctx, backend)

        assert len(backend.systems) == 2

    async def test_transport_error_on_retry_keeps_first_answer_flagged(self):
        """The backstop must never kill a turn the primary call served — ANY retry
        failure (not just parse) degrades to first-answer-plus-flag."""
        class BoomBackend(SeqBackend):
            async def generate(self, system, prompt):
                if self.responses[0] == "BOOM":
                    self.systems.append(system)
                    raise RuntimeError("connect timeout")
                return await super().generate(system, prompt)

        backend = BoomBackend([_resp(self.ECHO), "BOOM"])
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="Sim")

        await noumeno.process(ctx, backend)

        assert ctx.noumeno.rewritten == self.ECHO
        assert "FEW_SHOT_ECHO" in ctx.noumeno.rewrite_warnings

    async def test_degenerate_retry_is_adopted_but_flagged(self):
        """The retry is the model's uncontaminated reading — even bare ('Yes.') it
        beats a possibly-fabricated first answer (wrong is worse than unresolved,
        measured live: the primary call parroted the Sedex example WHOLE for
        'WhatsApp'); the bare answer ships carrying the doubt flag."""
        backend = SeqBackend([_resp(self.ECHO), _resp("Yes.")])
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="Sim")

        await noumeno.process(ctx, backend)

        assert ctx.noumeno.rewritten == "Yes."
        assert "FEW_SHOT_ECHO" in ctx.noumeno.rewrite_warnings

    async def test_empty_retry_keeps_first_answer_flagged(self):
        backend = SeqBackend([_resp(self.ECHO), _resp("")])
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="Sim")

        await noumeno.process(ctx, backend)

        assert ctx.noumeno.rewritten == self.ECHO
        assert "FEW_SHOT_ECHO" in ctx.noumeno.rewrite_warnings

    async def test_rule_illustration_anchored_in_last_query_is_clean(self):
        """The model legitimately resolves from the Last Query hint too — an anchor
        there must count (hosts may wire last_rewritten without a transcript)."""
        from cogno_anima import metakeys as mk
        backend = SeqBackend([_resp("Book with Dr. Vinicius Vale.")])
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="Sim")
        ctx.metadata[mk.LAST_REWRITTEN] = (
            "Do you want to book with Dr. Vinicius Vale?")

        await noumeno.process(ctx, backend)

        assert len(backend.systems) == 1
        assert ctx.noumeno.rewrite_warnings == []

    async def test_a_single_stray_anchor_does_not_clear_the_flag(self):
        """'vale' inside "vale a pena" is not an anchor for the NAME Vinicius Vale —
        one stray common word must not launder the wrong-name fabrication."""
        from cogno_anima import metakeys as mk
        backend = SeqBackend([_resp("Book with Dr. Vinicius Vale.")])
        noumeno = make_noumeno()
        ctx = PipelineContext(user_input="com a Ana")
        ctx.metadata[mk.CONVERSATION_HISTORY] = (
            "Assistant: vale a pena marcar amanhã? Com qual profissional?")

        await noumeno.process(ctx, backend)

        assert len(backend.systems) == 1
        assert "FEW_SHOT_ECHO" in ctx.noumeno.rewrite_warnings
