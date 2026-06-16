"""
End-to-end pipeline tests through the cognobench ReferencePipeline.

Deterministic (a sequence-scripted backend + stub embedder/dispatcher), so they
exercise the SEAMS that per-stage tests cannot: full-chain wiring, routing
branches, the EGO↔SUPEREGO correction loop, multiple tool calls in one turn
(is_composite), retry_metrics accumulation, and multi-turn state threading.
"""

import json
from pathlib import Path

import pytest

from tests.conftest import StubEmbedder
from cognobench.pipeline import ReferencePipeline
from cogno_core.types import PipelineContext, ToolResult

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


# ── a single backend scripted by call order (all stages use .generate) ──

class SequenceBackend:
    """Pops canned responses in call order; native FC off → EGO uses the text path."""

    def __init__(self, responses, model="seq", ti=6, to=4):
        self.responses = list(responses)
        self.model = model
        self.ti = ti
        self.to = to

    async def generate(self, system, prompt):
        resp = self.responses.pop(0) if self.responses else ""
        return resp, self.ti, self.to

    async def chat_with_tools(self, messages, tools, tool_choice=None):
        return {"content": ""}, self.ti, self.to

    def supports_native_tools(self):
        return False


class Dispatcher:
    def __init__(self, schema_names, side_effects=None):
        self._schema = [{"type": "function",
                         "function": {"name": n, "description": n, "parameters": {}}}
                        for n in schema_names]
        self._side = side_effects or {}
        self.executed = []

    def tools_schema(self):
        return self._schema

    async def execute(self, name, arguments):
        self.executed.append((name, dict(arguments)))
        return ToolResult(output=f"{name} ok", side_effect=self._side.get(name, False))


def _noumeno_json(rewritten):
    return json.dumps({"rewritten": rewritten, "context_turn": "", "confidence": 0.95,
                       "changed": True, "preserved_terms": [], "rewrite_warnings": []})


def _ner_json(intent_class="ACTION_REQUEST", sentiment="NEUTRAL", goal="record expense",
              tags=None, pii=None, domains=None):
    return json.dumps({
        "intent_class": intent_class, "sentiment": sentiment, "confidence": 0.9,
        "temporal_class": "TIMELESS", "triad_signal": "EGO" if intent_class == "ACTION_REQUEST" else "BALANCED",
        "goal": goal, "domains": domains or ["FINANCE"],
        "mandatory_tags": tags if tags is not None else ["SYSTEM"],
        "pii": pii or [],
    })


def _tool_tag(name, args):
    return f'<TOOL_CALL>{{"tool": "{name}", "args": {json.dumps(args)}}}</TOOL_CALL>'


def _pipe(embedder=None):
    return ReferencePipeline(prompts_dir=PROMPTS_DIR, embedder=embedder or StubEmbedder())


KW = dict(ego_prompt="You execute finance tasks.", limits_prompt="confirm before write",
          voice_prompt="You are a warm finance assistant.")


# ── 1. happy ACTION turn: full chain, EGO executes, judge approves, voice writes ──

@pytest.mark.asyncio
async def test_e2e_action_turn_happy_path():
    backend = SequenceBackend([
        _noumeno_json("record an expense of 50 for lunch"),
        _ner_json(),
        _tool_tag("record_expense", {"amount": 50}),     # EGO iter 0
        "Recorded.",                                      # EGO iter 1 (no tool → stop)
        '{"approved": true}',                             # judge
        "Pronto! Registrei R$50 do almoço. ✅",           # voice
    ])
    disp = Dispatcher(["record_expense"], side_effects={"record_expense": True})
    ctx = await _pipe().run_turn(
        PipelineContext(user_input="registra 50 do almoço"),
        gen_backend=backend, ego_backend=backend, dispatcher=disp, **KW)

    assert ctx.id_result.triad_route == "EGO"
    assert [t.tool for t in ctx.ego_result.tools_executed] == ["record_expense"]
    assert disp.executed == [("record_expense", {"amount": 50})]
    assert ctx.superego_result.response == "Pronto! Registrei R$50 do almoço. ✅"
    assert ctx.superego_result.blocked is False
    assert ctx.stop_reason == "completed"
    assert ctx.total_tokens > 0


# ── 2. composite: TWO tool calls in one EGO turn both execute (is_composite) ──

@pytest.mark.asyncio
async def test_e2e_composite_executes_multiple_tools():
    backend = SequenceBackend([
        _noumeno_json("record an expense of 50 and show this month summary"),
        _ner_json(goal="record expense and show summary"),
        _tool_tag("record_expense", {"amount": 50}) + "\n" + _tool_tag("get_summary", {"period": "month"}),
        "Done both.",
        '{"approved": true}',
        "Registrei os R$50 e aqui está seu resumo do mês. 📊",
    ])
    disp = Dispatcher(["record_expense", "get_summary"])
    ctx = await _pipe().run_turn(
        PipelineContext(user_input="registra 50 e mostra o resumo do mês"),
        gen_backend=backend, ego_backend=backend, dispatcher=disp, **KW)

    names = [t.tool for t in ctx.ego_result.tools_executed]
    assert names == ["record_expense", "get_summary"]          # BOTH executed in one turn
    assert len(disp.executed) == 2
    assert len(ctx.ego_result.steps[0].tool_calls) == 2        # same step


# ── 3. correction loop: judge rejects → EGO retries → approves → voice ──

@pytest.mark.asyncio
async def test_e2e_correction_loop():
    backend = SequenceBackend([
        _noumeno_json("record an expense of 50 for lunch"),
        _ner_json(),
        _tool_tag("record_income", {"amount": 50}),      # EGO attempt 1 — WRONG tool
        "Recorded income.",
        '{"approved": false, "critique": "recorded income instead of expense"}',   # judge rejects
        _tool_tag("record_expense", {"amount": 50}),     # EGO attempt 2 — corrected
        "Recorded expense.",
        '{"approved": true}',                            # judge approves
        "Prontinho, corrigi para despesa de R$50. ✅",    # voice
    ])
    disp = Dispatcher(["record_income", "record_expense"])
    ctx = await _pipe().run_turn(
        PipelineContext(user_input="registra despesa de 50"),
        gen_backend=backend, ego_backend=backend,
        dispatcher=disp, max_corrections=3, **KW)

    assert ctx.metadata["ego_correction"]["attempt"] == 2          # loop fired once
    assert [n for n, _ in disp.executed] == ["record_income", "record_expense"]
    assert ctx.superego_result.response.startswith("Prontinho")
    assert ctx.needs_handoff is False
    # retry_metrics carries the 2 judge calls + the failed EGO attempt
    judge_calls = [m for m in ctx.retry_metrics if m.stage == "superego_judge"]
    assert len(judge_calls) == 2
    assert any(m.stage == "ego" for m in ctx.retry_metrics)


# ── 3b. correction loop EXHAUSTED: judge keeps rejecting → human handoff ──

@pytest.mark.asyncio
async def test_e2e_correction_loop_exhausted_handoff():
    # Judge rejects on every attempt; after max_corrections the host policy in
    # ReferencePipeline flags needs_handoff/stop_reason="human_handoff" and skips
    # voice (the core only SIGNALS — the host owns the handoff decision).
    backend = SequenceBackend([
        _noumeno_json("record an expense of 50 for lunch"),
        _ner_json(),
        _tool_tag("record_income", {"amount": 50}),       # EGO attempt 1 — wrong tool
        "Recorded income.",
        '{"approved": false, "critique": "recorded income instead of expense"}',  # reject 1
        _tool_tag("record_income", {"amount": 50}),       # EGO attempt 2 — still wrong
        "Recorded income again.",
        '{"approved": false, "critique": "still income, not expense"}',           # reject 2 → exhausted
        # no voice response — handoff returns before voice
    ])
    disp = Dispatcher(["record_income", "record_expense"])
    ctx = await _pipe().run_turn(
        PipelineContext(user_input="registra despesa de 50"),
        gen_backend=backend, ego_backend=backend,
        dispatcher=disp, max_corrections=2, **KW)

    assert ctx.needs_handoff is True
    assert ctx.stop_reason == "human_handoff"
    assert ctx.superego_result is None                    # voice never ran
    assert [n for n, _ in disp.executed] == ["record_income", "record_income"]
    assert ctx.metadata["ego_correction"]["attempt"] == 2
    judge_calls = [m for m in ctx.retry_metrics if m.stage == "superego_judge"]
    assert len(judge_calls) == 2                          # judged on every attempt


# ── 4. SOCIAL routes to SUPEREGO voice directly — EGO never runs ──

@pytest.mark.asyncio
async def test_e2e_social_skips_ego():
    backend = SequenceBackend([
        _noumeno_json("thank you, that's all"),
        _ner_json(intent_class="SOCIAL", goal="", tags=[]),
        "De nada! Precisando, é só chamar. 😊",          # voice (no EGO, no judge)
    ])
    disp = Dispatcher(["record_expense"])
    ctx = await _pipe().run_turn(
        PipelineContext(user_input="obrigado, era só isso"),
        gen_backend=backend, ego_backend=backend, dispatcher=disp, **KW)

    assert ctx.id_result.triad_route == "SUPEREGO"
    assert ctx.ego_result is None
    assert disp.executed == []
    assert ctx.superego_result.response.startswith("De nada")


# ── 5. PII-CRITICAL → blocked, EGO/voice skipped ──

@pytest.mark.asyncio
async def test_e2e_pii_critical_blocked():
    backend = SequenceBackend([
        _noumeno_json("my bank password is X"),
        _ner_json(intent_class="ACTION_REQUEST", pii=["CREDENTIAL"]),
    ])
    disp = Dispatcher(["record_expense"])
    ctx = await _pipe().run_turn(
        PipelineContext(user_input="minha senha do banco é X"),
        gen_backend=backend, ego_backend=backend, dispatcher=disp, **KW)

    assert ctx.id_result.blocked is True
    assert ctx.superego_result.blocked is True
    assert ctx.stop_reason == "pii_blocked"
    assert ctx.ego_result is None and disp.executed == []


# ── 6. multi-turn threads id_state ──

@pytest.mark.asyncio
async def test_e2e_multi_turn_threads_state():
    emb = StubEmbedder()
    pipe = _pipe(emb)
    disp = Dispatcher(["get_summary"])

    b1 = SequenceBackend([
        _noumeno_json("what is the dollar today"),
        _ner_json(intent_class="INFORMATION_REQUEST", goal="dollar price", tags=[]),
        "O dólar está em X hoje.",
    ])
    ctx1 = await pipe.run_turn(PipelineContext(user_input="quanto tá o dólar?"),
                               gen_backend=b1, ego_backend=b1, dispatcher=disp, **KW)
    id_state = ctx1.metadata.get("id_state")
    assert id_state, "ID should persist cross-turn state"

    # turn 2 carries id_state forward
    b2 = SequenceBackend([
        _noumeno_json("and the euro"),
        _ner_json(intent_class="INFORMATION_REQUEST", goal="euro price", tags=[]),
        "O euro está em Y.",
    ])
    ctx2 = PipelineContext(user_input="e o euro?")
    ctx2.metadata["id_state"] = id_state
    ctx2.metadata["turn_number"] = 2
    ctx2 = await pipe.run_turn(ctx2, gen_backend=b2, ego_backend=b2, dispatcher=disp, **KW)
    assert ctx2.id_result.turn_number == 2
