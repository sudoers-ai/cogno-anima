"""
examples/host_min.py — a minimal, runnable host wiring cogno-anima end-to-end.

The runnable companion to ``docs/HOST_INTEGRATION.md``: a tiny host that owns the
things the library deliberately does NOT ship — orchestration (the control flow +
the EGO⇄SUPEREGO correction loop), a ``ToolDispatcher`` (the "hands"), atomicity
(a buffered ledger with rollback/commit), and cross-turn state persistence — while
``cogno_anima`` provides the cognition.

It imports ONLY from ``cogno_anima`` (never from ``cognobench``), so it mirrors the
real dependency surface of a host.

Run:  python3 examples/host_min.py
Needs a local Ollama (http://localhost:11434). Prints a note and exits if absent.
"""

from __future__ import annotations

import asyncio

import httpx

from cogno_anima.llm import OllamaBackend, OllamaEmbedder, CachingEmbedder
from cogno_anima.stages.noumeno import Noumeno
from cogno_anima.stages.ner import IntentAnalyzer
from cogno_anima.stages.id import IDStage
from cogno_anima.stages.ego import EgoStage
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.types import PipelineContext, ToolResult, SuperegoResult, StageMetrics

MODEL = "mistral:latest"
BASE_URL = "http://localhost:11434"


# ── 1. The host's "hands": a ToolDispatcher over an in-memory ledger ──────────
#     Side effects are BUFFERED (pending) so the host can roll back an EGO attempt
#     the judge rejects, and only commit() after approval. This is the atomicity
#     the core leaves to the host ("EGO = brain, dispatcher = hands").

TOOLS = [
    {"type": "function", "function": {
        "name": "get_balance", "description": "Get the current account balance in BRL.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "record_expense", "description": "Record an expense (money spent).",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number"}, "description": {"type": "string"}},
            "required": ["amount", "description"]}}},
]


class LedgerDispatcher:
    """In-memory finance dispatcher with host-side transaction semantics."""

    def __init__(self, opening_balance: float = 1000.0) -> None:
        self._balance = opening_balance
        self._pending: list[float] = []      # buffered, uncommitted expenses
        self._seen: set[float] = set()       # idempotency keys (amounts) for THIS turn
        self.executed: list[tuple[str, dict]] = []

    # --- ToolDispatcher protocol (what the EGO calls) ---
    def tools_schema(self) -> list[dict]:
        return TOOLS

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        self.executed.append((name, dict(arguments)))   # honest trace (may show repeats)
        if name == "get_balance":
            return ToolResult(output=f"Current balance: {self._balance:.2f} BRL.")
        if name == "record_expense":
            amount = float(arguments.get("amount", 0))
            # Idempotency guard — a HOST concern: a weaker model may emit the same
            # write several times in one turn (sometimes with slightly varied args);
            # the host must not double-charge. Demo simplification: one expense per
            # distinct amount per turn — production uses a real business idempotency key.
            if amount in self._seen:
                return ToolResult(output="Expense already staged (idempotent no-op).")
            self._seen.add(amount)
            self._pending.append(amount)      # buffer — not committed yet
            return ToolResult(output=f"Expense of {amount:.2f} BRL staged.", side_effect=True)
        return ToolResult(output="", ok=False, error=f"unknown tool {name!r}")

    # --- host transaction hooks (NOT part of the core protocol) ---
    def commit(self) -> None:
        for amount in self._pending:
            self._balance -= amount
        self._pending.clear()
        self._seen.clear()

    def rollback(self) -> None:
        self._pending.clear()                 # discard the rejected attempt's effects
        self._seen.clear()


# ── 2. Persona prompts (host-owned; you'd store/version these per tenant) ─────

SCOPE_PROMPT = ("A personal finance assistant. In scope: balance, expenses, income, "
                "summaries, greetings. Out of scope: recipes, trivia, politics.")
EGO_PROMPT = ("You are the execution engine of a finance assistant. For ANY data "
              "operation you MUST call the appropriate tool — never invent data. "
              "If the user is only chatting, do not call a tool.")
LIMITS_PROMPT = ("Do exactly what was asked (right operation, right amounts). Never "
                 "expose data the user did not ask for.")
VOICE_PROMPT = ("You are a warm, concise finance assistant. Reply in the user's "
                "language; keep figures exactly as in the tool data.")


# ── 3. The host orchestrator (mirrors ReferencePipeline, but as host code) ────

class MiniHost:
    def __init__(self, gen_backend, ego_backend, voice_backend, embedder) -> None:
        self._gen = gen_backend
        self._ego_backend = ego_backend
        self._voice_backend = voice_backend
        self._embedder = embedder
        self._noumeno = Noumeno(embedder=embedder)
        self._ner = IntentAnalyzer()
        self._id = IDStage()
        self._ego = EgoStage()
        self._superego = SuperegoStage()
        # The host persists cross-turn state. Here: an in-memory dict per session
        # (a real host puts this in Postgres/Redis next to its sessions/turns rows).
        self._sessions: dict[str, dict] = {}

    async def turn(self, session_id: str, text: str, dispatcher: LedgerDispatcher,
                   *, max_corrections: int = 2) -> PipelineContext:
        saved = self._sessions.get(session_id, {})
        ctx = PipelineContext(user_input=text)
        ctx.metadata.update(saved.get("metadata", {}))
        ctx.metadata["turn_number"] = saved.get("turn_number", 0) + 1

        # perception + routing (no tools, no DB)
        ctx = await self._noumeno.process(ctx, self._gen)
        ctx = await self._ner.process(ctx, self._gen)
        ctx = await self._id.process(ctx, self._embedder)

        # 1) PII-CRITICAL safety gate
        if ctx.id_result and ctx.id_result.blocked:
            ctx.superego_result = self._superego._blocked_response(ctx)
            ctx.stop_reason = "pii_blocked"
            return self._persist(session_id, ctx)

        # 2) cheap scope guard (fail-open)
        scope = await self._superego.check_input_scope(ctx, self._gen, scope_prompt=SCOPE_PROMPT)
        if scope.blocked:
            ctx.superego_result = SuperegoResult(
                response=scope.refusal_message, blocked=True,
                metrics=StageMetrics(stage="superego_voice", elapsed_ms=0.0,
                                     tokens_in=0, tokens_out=0, model="none"))
            ctx.stop_reason = "scope_blocked"
            return self._persist(session_id, ctx)

        # 3) EGO route: execute + the correction loop (host-owned)
        if ctx.id_result and ctx.id_result.triad_route == "EGO":
            attempt = 1
            judge = None
            while True:
                ctx = await self._ego.process(ctx, self._ego_backend, dispatcher,
                                              system_prompt=EGO_PROMPT)
                judge = await self._superego.evaluate(ctx, self._gen, limits_prompt=LIMITS_PROMPT)
                if judge.approved or attempt >= max_corrections:
                    break
                dispatcher.rollback()                       # undo the rejected attempt
                ctx.metadata["ego_correction"] = {"reason": judge.critique, "attempt": attempt + 1}
                attempt += 1
            if judge is not None and not judge.approved:    # retries exhausted → escalate
                ctx.needs_handoff = True
                ctx.stop_reason = "human_handoff"
                return self._persist(session_id, ctx)
            dispatcher.commit()                             # approved → persist side effects

        # 4) SUPEREGO writes the final reply (for EGO and chat paths)
        ctx.superego_result = await self._superego.voice(ctx, self._voice_backend,
                                                         voice_prompt=VOICE_PROMPT)
        return self._persist(session_id, ctx)

    def _persist(self, session_id: str, ctx: PipelineContext) -> PipelineContext:
        """Save the cross-turn state the next turn needs (host's job)."""
        meta = {"id_state": ctx.metadata.get("id_state", {})}
        if ctx.noumeno:
            meta["last_rewritten"] = ctx.noumeno.rewritten
        self._sessions[session_id] = {
            "metadata": meta,
            "turn_number": ctx.id_result.turn_number if ctx.id_result else 0,
        }
        return ctx


# ── 4. Demo ───────────────────────────────────────────────────────────────────

async def _ollama_up() -> bool:
    try:
        async with httpx.AsyncClient(timeout=1.0) as c:
            return (await c.get(f"{BASE_URL}/")).status_code == 200
    except Exception:
        return False


async def main() -> None:
    if not await _ollama_up():
        print(f"Ollama not reachable at {BASE_URL} — start it (and `ollama pull {MODEL}`) "
              "to run this demo.")
        return

    gen = OllamaBackend(model=MODEL, base_url=BASE_URL, temperature=0.0, format="json")
    text = OllamaBackend(model=MODEL, base_url=BASE_URL, temperature=0.0)   # EGO/voice (TOOL_CALL)
    embedder = CachingEmbedder(OllamaEmbedder(model="nomic-embed-text", base_url=BASE_URL))

    host = MiniHost(gen_backend=gen, ego_backend=text, voice_backend=text, embedder=embedder)
    dispatcher = LedgerDispatcher(opening_balance=1000.0)

    for user_text in ["bom dia!", "registra uma despesa de 50 do almoço", "qual é o meu saldo?"]:
        ctx = await host.turn("session-1", user_text, dispatcher)
        route = ctx.id_result.triad_route if ctx.id_result else "?"
        tools = [t.tool for t in ctx.ego_result.tools_executed] if ctx.ego_result else []
        reply = ctx.superego_result.response if ctx.superego_result else "(no reply)"
        print(f"\n👤 {user_text}")
        print(f"   route={route}  tools={tools}  stop={ctx.stop_reason}  tokens={ctx.total_tokens}")
        print(f"🤖 {reply}")


if __name__ == "__main__":
    asyncio.run(main())
