import pytest
import httpx
from pathlib import Path

from cogno_core.stages.noumeno import Noumeno
from cogno_core.stages.ner import IntentAnalyzer
from cogno_core.stages.drift import DriftCalculator
from cogno_core.llm import OllamaBackend, OllamaEmbedder
from cogno_core.types import PipelineContext

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"
SLANGS = {"vc": "você", "pq": "porque", "blz": "beleza", "pfv": "por favor"}


async def is_ollama_available() -> bool:
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.get("http://localhost:11434/")
            return resp.status_code == 200
    except Exception:
        return False


def _make_real_pipeline() -> tuple[Noumeno, IntentAnalyzer, DriftCalculator, OllamaBackend]:
    llm = OllamaBackend(model="llama3.1:8b", temperature=0.0)
    embedder = OllamaEmbedder(model="nomic-embed-text:latest")
    noumeno = Noumeno(embedder=embedder, prompts_dir=PROMPTS_DIR, slangs=SLANGS)
    analyzer = IntentAnalyzer(backend=llm, prompts_dir=PROMPTS_DIR)
    drift_calc = DriftCalculator()
    return noumeno, analyzer, drift_calc, llm


pytestmark = pytest.mark.asyncio


async def test_ner_integration_flow():
    """Real E2E integration validation of NOUMENO -> NER -> Drift pipeline."""
    if not await is_ollama_available():
        pytest.skip("Ollama is not running")

    noumeno, analyzer, drift_calc, llm = _make_real_pipeline()

    # User input
    ctx = PipelineContext(user_input="Quero agendar uma consulta com o veterinário amanhã, por favor.")
    
    # 1. Run Noumeno
    ctx = await noumeno.process(ctx, llm)
    assert ctx.noumeno is not None
    
    # 2. Run NER
    ctx = await analyzer.process(ctx, llm)
    assert ctx.intent is not None
    assert ctx.intent.intent_class in ("ACTION_REQUEST", "INFORMATION_REQUEST")
    
    # 3. Compute Drift
    drift = drift_calc.compute(ctx.noumeno, ctx.intent)
    drift_calc.compute_ontological(drift, ctx.noumeno, ctx.intent)
    drift_calc.compute_cumulative(drift)
    
    ctx.drift = drift
    assert ctx.drift.drift_score >= 0.0
