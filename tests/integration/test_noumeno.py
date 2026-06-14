"""
tests/integration/test_noumeno.py — NOUMENO stage integration tests.

Uses REAL LLM calls via OllamaBackend + OllamaEmbedder.
Auto-skipped if Ollama is not available.

Validates:
  - Dialogue flows extracted directly from Cogno turn database (PostgreSQL)
  - Subject continuity behavior (history injection vs history clearing)
  - Accurate query rewriting preserving key entity names, context, and intent
  - Refusal prevention for real-world PT-BR inputs
  - Full structural validation of NoumenoResult outputs across all tests
"""

import pytest
import json
import httpx
from pathlib import Path

from cogno_core.stages.noumeno import Noumeno, NoumenoResult
from cogno_core.llm import OllamaBackend, OllamaEmbedder, CachingEmbedder
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


def _make_real_noumeno() -> tuple[Noumeno, OllamaBackend]:
    # Using temperature=0.0 to prevent hallucinations and make integration tests deterministic
    llm = OllamaBackend(model="llama3.1:8b", temperature=0.0)
    embedder = CachingEmbedder(OllamaEmbedder(model="nomic-embed-text:latest"))
    noumeno = Noumeno(embedder=embedder, prompts_dir=PROMPTS_DIR, slangs=SLANGS)
    return noumeno, llm


class SpyLLM:
    """Wrapper backend to capture the exact prompts sent to the LLM during execution."""
    def __init__(self, target):
        self.target = target
        self.captured_prompts = []

    @property
    def model(self):
        return self.target.model

    async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
        self.captured_prompts.append(prompt)
        return await self.target.generate(system, prompt)


def _assert_valid_noumeno_result(res: NoumenoResult, original_input: str, model_name: str):
    """Rigorous structural and range validation of all fields returned in NoumenoResult."""
    assert res is not None
    assert res.original == original_input
    assert isinstance(res.rewritten, str)
    assert res.rewritten.strip()
    assert isinstance(res.context_turn, str)
    assert isinstance(res.language, str)
    assert len(res.language) >= 2
    assert res.canonical_language == "en"
    
    assert isinstance(res.drift_score, float)
    assert res.drift_tag in ("PASS_THROUGH", "REWRITTEN", "COMPRESSED", "EXPANDED", "DRIFT")
    
    assert isinstance(res.changed, bool)
    assert isinstance(res.confidence, float)
    assert 0.0 <= res.confidence <= 1.0
    assert isinstance(res.change_subject, bool)
    
    assert isinstance(res.subject_similarity, float)
    assert isinstance(res.context_used, bool)
    
    assert isinstance(res.preserved_terms, list)
    assert isinstance(res.rewrite_warnings, list)
    
    # Verify metrics
    assert res.metrics is not None
    assert res.metrics.stage == "noumeno"
    assert res.metrics.tokens_in > 0
    assert res.metrics.tokens_out > 0
    assert res.metrics.model == model_name
    assert res.metrics.elapsed_ms > 0



pytestmark = pytest.mark.asyncio


# ────────────────────────────────────────────────────────────────────
#  1. Real Examples from Postgres turns table
# ────────────────────────────────────────────────────────────────────

# Each tuple contains: (desc, raw_input, expected_keywords_alternatives)
# For each expected keyword slot, at least one of the alternatives must match (case-insensitive).
REAL_TURNS_INPUTS = [
    # Veterinary Clinic
    ("pet registration", "Quero cadastrar meu cachorro...Thor, Golden Retriever...", [["thor"], ["golden", "retriever", "dog"]]),
    ("pet sheet view", "Boa tarde, quero ver a ficha do Thor do Vinicius Vale", [["thor"], ["vinicius", "vini", "vale", "valley"]]),
    ("consultation opening", "Abrir consulta para o Thor, motivo: consulta com vacinação", [["thor"], ["consult", "vaccin", "open"]]),
    ("consultation items", "Adicionar ao atendimento atual do Thor os itens do catálogo: Vacina V8/V10 e Consulta Clínica Geral", [["thor"], ["v8", "v10"]]),
    ("status report", "Fechar o atendimento, diagnóstico: animal saudável, vacinação em dia", [["close", "clos", "fechar", "finish"], ["health", "saud", "vaccin"]]),
    
    # Restaurant / Cardápio
    ("menu query", "Olá, o que temos de cardápio para hoje?", [["menu", "cardapio", "today", "hoje"]]),
    ("add item with detail", "Coloque 2 Hambúrgueres Artesanais no carrinho. Um deles sem cebola.", [["burger", "hamburguer"], ["onion", "cebola"]]),
    ("add drink", "Adicionar também uma Coca-Cola.", [["coca", "coke", "cola"]]),
    ("order checkout", "Pode confirmar o pedido. Entregar na Rua das Flores, 123 e o pagamento vai ser via Pix.", [["confirm"], ["flores", "flowers", "123"], ["pix"]]),
    
    # MBA / Reminders / Calendar
    ("reminder setup", "Crie um lembrete para daqui a 10 minutos, preciso acesse a planilha de aulas e que traga todas as próximas aulas do MBA de Data Engineering para o mês de Junho", [["reminder", "lembrete"], ["mba", "class", "aula", "engineering"]]),
    ("reminder general", "Crie uma lembrete para daqui a 5 minutos sobre minhas aulas do MBA", [["reminder", "lembrete"], ["mba"]]),
    ("schedule blockage", "Bloqueie minha agenda no dia 2026-06-09 das 09:00 às 10:00.", [["block", "bloque"], ["agenda", "schedule"]]),
    ("schedule unblockage", "Desbloquear minha agenda do dia 11", [["unblock", "unlock", "desbloque", "agenda", "schedule"]]),
    
    # Finance / Market
    ("market news", "Gostaria de saber as notícias do mercado", [["news", "noticia"], ["market", "mercado"]]),
    ("crypto alert", "Lembre me todo dias as 17 me passando o valor do bitcoin.", [["bitcoin", "btc"]]),
    ("commodity query", "E as commodities?", [["commodit"]]),
    ("euro query", "E o euro?", [["euro"]])
]


@pytest.mark.parametrize("desc,raw_input,expected_keywords", REAL_TURNS_INPUTS, ids=[d for d, _, _ in REAL_TURNS_INPUTS])
async def test_noumeno_produces_valid_result(desc, raw_input, expected_keywords):
    """Each input type extracted from database should produce a valid rewrite containing the expected keywords."""
    if not await is_ollama_available():
        pytest.skip("Local Ollama server (http://localhost:11434) is not running.")

    noumeno, llm = _make_real_noumeno()
    ctx = PipelineContext(user_input=raw_input)
    ctx = await noumeno.process(ctx, llm)
    result = ctx.noumeno

    # Full structural validation
    _assert_valid_noumeno_result(result, raw_input, llm.model)
    
    # Semantic verification
    rewritten_lower = result.rewritten.lower()
    for alternatives in expected_keywords:
        assert any(alt.lower() in rewritten_lower for alt in alternatives), (
            f"None of the alternative keywords {alternatives!r} found in rewritten query!\n"
            f"  Original:  {raw_input!r}\n"
            f"  Rewritten: {result.rewritten!r}"
        )


# ────────────────────────────────────────────────────────────────────
#  2. PT-BR Refusal Prevention
# ────────────────────────────────────────────────────────────────────

_REFUSAL_PHRASES = [
    "i can only", "i cannot", "i'm sorry", "i am sorry",
    "cannot process", "unable to", "not able to",
    "please provide", "could you please",
]

PTBR_DB_INPUTS = [
    ("top obrigado", "top, obrigado"),
    ("valeu era isso", "valeu, era isso"),
    ("beleza entendi", "beleza, entendi"),
    ("manda ver", "manda ver"),
    ("show", "show"),
    ("joia ta bom", "joia, tá bom"),
    ("ta por ai Pam", "Ta por ai Pam?"),
    ("oi tudo bem", "OI, tudo bem?"),
    ("pode confirmar", "Pode confirmar o pedido."),
    ("confirmado", "Confirmado")
]


@pytest.mark.parametrize("desc,raw_input", PTBR_DB_INPUTS, ids=[d for d, _ in PTBR_DB_INPUTS])
async def test_noumeno_no_refusal_for_ptbr(desc, raw_input):
    """NOUMENO must rewrite PT-BR chat signals and confirmations without triggering LLM refusal templates."""
    if not await is_ollama_available():
        pytest.skip("Local Ollama server (http://localhost:11434) is not running.")

    noumeno, llm = _make_real_noumeno()
    ctx = PipelineContext(user_input=raw_input)
    ctx = await noumeno.process(ctx, llm)
    result = ctx.noumeno

    # Full structural validation
    _assert_valid_noumeno_result(result, raw_input, llm.model)

    rewritten_lower = result.rewritten.lower()
    for phrase in _REFUSAL_PHRASES:
        assert phrase not in rewritten_lower, (
            f"NOUMENO refused to rewrite PT-BR input!\n"
            f"  Input:    {raw_input!r}\n"
            f"  Rewritten: {result.rewritten!r}\n"
            f"  Refusal:  '{phrase}' found in output"
        )


# ────────────────────────────────────────────────────────────────────
#  3. Conversational Flows & Domain Shifts (Subject Continuity)
# ────────────────────────────────────────────────────────────────────

CONVERSATIONAL_FLOWS = {
    "vet_to_finance_shift": [
        {"input": "Quero cadastrar meu cachorro Thor que é um Golden Retriever", "contains": [["thor"], ["golden", "retriever", "dog"]]},
        {"input": "Abrir consulta para o Thor, motivo: consulta com vacinação", "contains": [["thor"], ["consult", "vaccin"]]},
        {"input": "Adicione também Vacina V8/V10 e Consulta Clínica Geral", "contains": [["v8", "v10"]]},
        {"input": "Qual a cotação do dólar hoje?", "contains": [["dollar", "dolar"]]},
        {"input": "E o euro?", "contains": [["euro"]]},
        {"input": "quais lembretes eu tenho?", "contains": [["reminder", "lembrete", "list"]]}
    ],
    
    "restaurant_to_weather_shift": [
        {"input": "Olá, o que temos de cardápio para hoje?", "contains": [["menu", "cardapio"]]},
        {"input": "Coloque 2 Hambúrgueres Artesanais no carrinho, um deles sem cebola.", "contains": [["burger", "hamburguer"], ["onion", "cebola"]]},
        {"input": "Adicione também uma Coca-Cola.", "contains": [["coca", "coke", "cola"]]},
        {"input": "Quero ver o meu carrinho.", "contains": [["cart", "carrinho"]]},
        {"input": "Pode confirmar o pedido. Entregar na Rua das Flores, 123", "contains": [["confirm", "order"], ["flores", "flowers", "123"]]},
        {"input": "Como está a previsão do tempo pra Nova York?", "contains": [["weather", "tempo"], ["york"]]}
    ],

    "mba_to_finance_shift": [
        {"input": "Quando é minha próxima aula do MBA de fundamentos de Data Engineering?", "contains": [["class", "aula", "mba"], ["engineering"]]},
        {"input": "Crie um lembrete para amanha as 08:00 trazendo a agenda dos meus subordinados.", "contains": [["subordinate", "employee", "subordinado", "agenda", "reminder"]]},
        {"input": "E as commodities?", "contains": [["commodit"]]}
    ]
}


@pytest.mark.parametrize("flow_name", CONVERSATIONAL_FLOWS.keys())
async def test_noumeno_conversational_flows(flow_name):
    """Verifies output quality and history injection behavior based on continuity decisions during conversation flows."""
    if not await is_ollama_available():
        pytest.skip("Local Ollama server (http://localhost:11434) is not running.")

    noumeno, base_llm = _make_real_noumeno()
    spy_llm = SpyLLM(base_llm)
    flow = CONVERSATIONAL_FLOWS[flow_name]
    
    last_rewritten = None
    last_context_turn = None

    for i, turn in enumerate(flow):
        user_input = turn["input"]
        expected_keywords = turn["contains"]

        ctx = PipelineContext(user_input=user_input)
        if last_rewritten:
            ctx.metadata["last_rewritten"] = last_rewritten
            ctx.metadata["last_context_turn"] = last_context_turn

        ctx = await noumeno.process(ctx, spy_llm)
        res = ctx.noumeno

        # Full structural validation
        _assert_valid_noumeno_result(res, user_input, base_llm.model)

        # 1. Output Validation (semantic matching with alternatives)
        rewritten_lower = res.rewritten.lower()
        for alternatives in expected_keywords:
            assert any(alt.lower() in rewritten_lower for alt in alternatives), (
                f"Turn {i} rewrite does not contain any of the expected keywords: {alternatives!r}\n"
                f"  Original:  {user_input!r}\n"
                f"  Rewritten: {res.rewritten!r}"
            )

        # 2. Prompt History Injection Validation
        latest_prompt = spy_llm.captured_prompts[-1]
        if i == 0:
            assert "Recent conversation:" not in latest_prompt
        else:
            if not res.change_subject:
                assert "Recent conversation:" in latest_prompt, (
                    f"Subject stayed, but history was NOT injected!\n"
                    f"  Turn {i}: {user_input!r}\n"
                    f"  Similarity: {res.subject_similarity:.4f}\n"
                    f"  Prompt:\n{latest_prompt}"
                )
            else:
                assert "Recent conversation:" not in latest_prompt, (
                    f"Subject changed, but history WAS injected!\n"
                    f"  Turn {i}: {user_input!r}\n"
                    f"  Similarity: {res.subject_similarity:.4f}\n"
                    f"  Prompt:\n{latest_prompt}"
                )

        last_rewritten = res.rewritten
        last_context_turn = res.context_turn


# ────────────────────────────────────────────────────────────────────
#  4. Hermetic Mocked Integration (no network)
# ────────────────────────────────────────────────────────────────────

async def test_noumeno_mocked_ollama_integration(monkeypatch):
    """Full pipeline with monkeypatched HTTP — validates wiring without network."""

    async def mock_post(client, url, *args, **kwargs):
        class MockResponse:
            def __init__(self, data):
                self._data = data
                self.status_code = 200
            def json(self):
                return self._data
            def raise_for_status(self):
                pass

        url_str = str(url)
        if "/api/generate" in url_str:
            resp_body = {
                "rewritten": "hello you, how are you?",
                "context_turn": "greeting",
                "confidence": 0.95,
                "changed": True,
                "preserved_terms": [],
                "rewrite_warnings": []
            }
            return MockResponse({
                "response": json.dumps(resp_body),
                "prompt_eval_count": 25,
                "eval_count": 15
            })
        elif "/api/embeddings" in url_str:
            return MockResponse({"embedding": [1.0, 0.0, 0.0]})

        return MockResponse({})

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    llm = OllamaBackend(model="llama3")
    embedder = CachingEmbedder(OllamaEmbedder(model="nomic-embed-text"))
    noumeno = Noumeno(embedder=embedder, prompts_dir=PROMPTS_DIR, slangs=SLANGS)

    ctx = PipelineContext(user_input="olá vc, como vai?")
    ctx.metadata["last_rewritten"] = "Hello, how are you?"
    ctx.metadata["last_context_turn"] = "greeting"

    ctx = await noumeno.process(ctx, llm)

    res = ctx.noumeno
    assert res is not None
    assert res.original == "olá vc, como vai?"
    assert res.rewritten == "hello you, how are you?"
    assert res.context_turn == "greeting"
    assert res.language == "pt"
    assert res.canonical_language == "en"
    assert res.drift_score == 0.0
    assert res.drift_tag == "PASS_THROUGH"
    assert res.changed is True
    assert res.confidence == 0.95
    assert res.change_subject is False
    assert res.subject_similarity == 1.0
    assert res.context_used is True
    assert res.preserved_terms == []
    assert res.rewrite_warnings == []

    # Verify metrics
    assert res.metrics.stage == "noumeno"
    assert res.metrics.tokens_in == 25
    assert res.metrics.tokens_out == 15
    assert res.metrics.model == "llama3"
    assert res.metrics.elapsed_ms > 0


# ────────────────────────────────────────────────────────────────────
#  5. Determinism — Same input must yield identical output
# ────────────────────────────────────────────────────────────────────

async def test_noumeno_determinism():
    """With temperature=0.0, running the same input twice must produce byte-identical rewrites."""
    if not await is_ollama_available():
        pytest.skip("Local Ollama server (http://localhost:11434) is not running.")

    noumeno, llm = _make_real_noumeno()
    raw = "Quero cadastrar meu cachorro Thor que é um Golden Retriever"

    ctx1 = PipelineContext(user_input=raw)
    ctx1 = await noumeno.process(ctx1, llm)

    ctx2 = PipelineContext(user_input=raw)
    ctx2 = await noumeno.process(ctx2, llm)

    assert ctx1.noumeno.rewritten == ctx2.noumeno.rewritten, (
        f"Non-deterministic output!\n"
        f"  Run 1: {ctx1.noumeno.rewritten!r}\n"
        f"  Run 2: {ctx2.noumeno.rewritten!r}"
    )
    assert ctx1.noumeno.confidence == ctx2.noumeno.confidence
    assert ctx1.noumeno.changed == ctx2.noumeno.changed
    assert ctx1.noumeno.preserved_terms == ctx2.noumeno.preserved_terms


# ────────────────────────────────────────────────────────────────────
#  6. Anaphoric Reference Resolution — "E isso?" with history
# ────────────────────────────────────────────────────────────────────

ANAPHORIC_CASES = [
    {
        "desc": "bitcoin_price_followup",
        "history_rewritten": "What is the price of Bitcoin?",
        "history_context": "The user is asking about cryptocurrency prices.",
        "input": "E o preço do Bitcoin tá subindo?",
        "must_contain": [["bitcoin", "btc", "price"]],
    },
    {
        "desc": "thor_consultation_followup",
        "history_rewritten": "Open a consultation for Thor, reason: vaccination appointment.",
        "history_context": "The user is opening a veterinary consultation for their dog Thor.",
        "input": "Qual o peso do Thor?",
        "must_contain": [["weight", "thor"]],
    },
    {
        "desc": "euro_rate_followup",
        "history_rewritten": "What is the Euro exchange rate today?",
        "history_context": "The user is asking about the Euro currency rate.",
        "input": "O Euro caiu hoje?",
        "must_contain": [["euro", "drop", "fell", "down", "declin", "today"]],
    },
]


@pytest.mark.parametrize("case", ANAPHORIC_CASES, ids=[c["desc"] for c in ANAPHORIC_CASES])
async def test_noumeno_anaphoric_reference_resolution(case):
    """Short references like 'isso', 'dele', 'ela' must be resolved using injected history."""
    if not await is_ollama_available():
        pytest.skip("Local Ollama server (http://localhost:11434) is not running.")

    # Using subject_threshold=0.40 to handle cross-lingual (PT input vs EN history) similarity
    llm = OllamaBackend(model="llama3.1:8b", temperature=0.0)
    embedder = CachingEmbedder(OllamaEmbedder(model="nomic-embed-text:latest"))
    noumeno = Noumeno(embedder=embedder, prompts_dir=PROMPTS_DIR, slangs=SLANGS, subject_threshold=0.40)
    spy_llm = SpyLLM(llm)

    ctx = PipelineContext(user_input=case["input"])
    ctx.metadata["last_rewritten"] = case["history_rewritten"]
    ctx.metadata["last_context_turn"] = case["history_context"]

    ctx = await noumeno.process(ctx, spy_llm)
    res = ctx.noumeno

    _assert_valid_noumeno_result(res, case["input"], llm.model)

    # History MUST have been injected (same subject)
    latest_prompt = spy_llm.captured_prompts[-1]
    assert "Recent conversation:" in latest_prompt, (
        f"History was not injected for anaphoric input!\n"
        f"  Input: {case['input']!r}\n"
        f"  Similarity: {res.subject_similarity:.4f}"
    )

    # The rewrite must resolve the reference
    rewritten_lower = res.rewritten.lower()
    for alternatives in case["must_contain"]:
        assert any(alt.lower() in rewritten_lower for alt in alternatives), (
            f"Anaphoric reference not resolved!\n"
            f"  Input:    {case['input']!r}\n"
            f"  History:  {case['history_rewritten']!r}\n"
            f"  Rewritten: {res.rewritten!r}\n"
            f"  Expected one of: {alternatives!r}"
        )


# ────────────────────────────────────────────────────────────────────
#  7. Slang Expansion — Verify slangs are expanded BEFORE hitting LLM
# ────────────────────────────────────────────────────────────────────

async def test_noumeno_slang_expansion_in_prompt():
    """Slang dictionary entries must be expanded in the prompt sent to the LLM."""
    if not await is_ollama_available():
        pytest.skip("Local Ollama server (http://localhost:11434) is not running.")

    noumeno, base_llm = _make_real_noumeno()
    spy_llm = SpyLLM(base_llm)

    # "vc" should be expanded to "você" and "pq" to "porque" before reaching the LLM
    ctx = PipelineContext(user_input="vc sabe pq o bitcoin caiu?")
    ctx = await noumeno.process(ctx, spy_llm)

    _assert_valid_noumeno_result(ctx.noumeno, "vc sabe pq o bitcoin caiu?", base_llm.model)

    prompt_sent = spy_llm.captured_prompts[-1]
    assert "você" in prompt_sent, (
        f"Slang 'vc' was not expanded to 'você' in the prompt!\n"
        f"  Prompt: {prompt_sent!r}"
    )
    assert "porque" in prompt_sent, (
        f"Slang 'pq' was not expanded to 'porque' in the prompt!\n"
        f"  Prompt: {prompt_sent!r}"
    )
    # The original slang should NOT appear as a standalone word in the prompt
    # (it might appear inside "você" but not as the raw abbreviation)
    prompt_words = prompt_sent.lower().split()
    assert "vc" not in prompt_words, "Raw slang 'vc' leaked into the LLM prompt without expansion"


# ────────────────────────────────────────────────────────────────────
#  8. Drift Reconciliation — High drift forces changed=True
# ────────────────────────────────────────────────────────────────────

async def test_noumeno_drift_reconciliation(monkeypatch):
    """When drift_score > 0.50, the reconciliation logic must force changed=True and drift_tag='DRIFT'."""

    call_count = 0

    async def mock_post(client, url, *args, **kwargs):
        nonlocal call_count

        class MockResponse:
            def __init__(self, data):
                self._data = data
                self.status_code = 200
            def json(self):
                return self._data
            def raise_for_status(self):
                pass

        url_str = str(url)
        if "/api/generate" in url_str:
            # Return a rewrite that is semantically very different from the input
            resp_body = {
                "rewritten": "The quantum flux capacitor needs recalibration.",
                "context_turn": "",
                "confidence": 0.9,
                "changed": False,  # LLM says no change, but drift will disagree
                "preserved_terms": [],
                "rewrite_warnings": []
            }
            return MockResponse({
                "response": json.dumps(resp_body),
                "prompt_eval_count": 10,
                "eval_count": 8
            })
        elif "/api/embeddings" in url_str:
            call_count += 1
            if call_count == 1:
                # Embedding for the original input
                return MockResponse({"embedding": [1.0, 0.0, 0.0]})
            else:
                # Embedding for the rewritten — deliberately orthogonal to simulate high drift
                return MockResponse({"embedding": [0.0, 1.0, 0.0]})
        return MockResponse({})

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    llm = OllamaBackend(model="llama3")
    embedder = CachingEmbedder(OllamaEmbedder(model="nomic-embed-text"))
    noumeno = Noumeno(embedder=embedder, prompts_dir=PROMPTS_DIR, slangs=SLANGS)

    ctx = PipelineContext(user_input="Qual o preço do bitcoin?")
    ctx = await noumeno.process(ctx, llm)

    res = ctx.noumeno
    # Drift should be 1.0 (orthogonal vectors → cosine_sim = 0.0 → drift = 1.0)
    assert res.drift_score == 1.0
    assert res.drift_tag == "DRIFT"
    # Reconciliation must override the LLM's "changed: false"
    assert res.changed is True, (
        f"Drift reconciliation failed! drift_score={res.drift_score} "
        f"but changed={res.changed}"
    )


# ────────────────────────────────────────────────────────────────────
#  9. context_used Consistency — Logical invariant validation
# ────────────────────────────────────────────────────────────────────

async def test_noumeno_context_used_consistency():
    """
    context_used must equal: bool(context_turn) AND NOT change_subject.

    Scenario A: Same subject + LLM returns context_turn → context_used = True
    Scenario B: Subject changes → context_used = False (even if LLM returns context_turn)
    Scenario C: No history at all → context_used = False
    """
    if not await is_ollama_available():
        pytest.skip("Local Ollama server (http://localhost:11434) is not running.")

    noumeno, llm = _make_real_noumeno()

    # Scenario C: No history → context_used must be False
    ctx_c = PipelineContext(user_input="Qual o preço do bitcoin?")
    ctx_c = await noumeno.process(ctx_c, llm)
    res_c = ctx_c.noumeno
    _assert_valid_noumeno_result(res_c, "Qual o preço do bitcoin?", llm.model)
    # With no history, change_subject=False but there's no last_rewritten,
    # so context_turn from LLM should be "" (no history to reference).
    # context_used = bool("") and not False = False
    assert res_c.context_used is False, (
        f"Scenario C: context_used should be False without history, "
        f"got context_turn={res_c.context_turn!r}"
    )

    # Scenario A: Same subject continuation → context_used should be True
    ctx_a = PipelineContext(user_input="E a Ethereum?")
    ctx_a.metadata["last_rewritten"] = "What is the price of Bitcoin?"
    ctx_a.metadata["last_context_turn"] = "User asking about cryptocurrency prices."
    ctx_a = await noumeno.process(ctx_a, llm)
    res_a = ctx_a.noumeno
    _assert_valid_noumeno_result(res_a, "E a Ethereum?", llm.model)
    if not res_a.change_subject and res_a.context_turn:
        assert res_a.context_used is True, (
            f"Scenario A: Same subject with context_turn={res_a.context_turn!r} "
            f"should have context_used=True"
        )

    # Scenario B: Abrupt domain shift → context_used should be False
    ctx_b = PipelineContext(user_input="Bloqueie minha agenda amanhã das 09:00 às 10:00.")
    ctx_b.metadata["last_rewritten"] = "What is the price of Bitcoin?"
    ctx_b.metadata["last_context_turn"] = "User asking about cryptocurrency prices."
    ctx_b = await noumeno.process(ctx_b, llm)
    res_b = ctx_b.noumeno
    _assert_valid_noumeno_result(res_b, "Bloqueie minha agenda amanhã das 09:00 às 10:00.", llm.model)
    if res_b.change_subject:
        assert res_b.context_used is False, (
            f"Scenario B: Subject changed but context_used={res_b.context_used}"
        )
        assert res_b.context_turn == "", (
            f"Scenario B: Subject changed but context_turn={res_b.context_turn!r} (should be empty)"
        )

