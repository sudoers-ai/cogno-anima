"""CognoBench — BOOKKEEPER persona tool-selection cases (ports the parent ``persona_cases``).

The parent SaaS shipped a bench for its ANALYST/bookkeeper persona: given a user utterance,
does the EGO pick the right financial tool with the right shape? This is the clean-room port,
run through CognoBench's EGO harness (text-fallback path) against the **real 8 bookkeeper tools**
(matching ``cogno_praxis/bookkeeper/server.py``). Self-contained: the tool schemas are hand-built
here (like the parent), so it needs neither the vertical nor the host — only the EGO stage + a
text backend.

Run:  ``python -m cognobench.bookkeeper_cases [model]``   (default qwen3:8b; needs Ollama)
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field

from cogno_anima.stages.ego import EgoStage
from cogno_anima.types import ToolResult

from cognobench.dimensions import _ego_ctx
from cognobench.ego_cases import EgoCase
from cognobench.harness import build_ollama_text, ollama_available

# ── The real bookkeeper toolset (mirrors cogno_praxis/bookkeeper/server.py) ──
BOOKKEEPER_TOOLS: list[dict] = [
    {"type": "function", "function": {
        "name": "add_income",
        "description": "Record an income (entrada) — money the business received, optionally from a client.",
        "parameters": {"type": "object", "properties": {
            "description": {"type": "string"}, "amount": {"type": "string"},
            "client": {"type": "string"}, "date": {"type": "string"}},
            "required": ["description", "amount"]}}},
    {"type": "function", "function": {
        "name": "add_outcome",
        "description": "Record an expense (saída) — money the business spent (a bill, rent, supplies).",
        "parameters": {"type": "object", "properties": {
            "description": {"type": "string"}, "amount": {"type": "string"}, "date": {"type": "string"}},
            "required": ["description", "amount"]}}},
    {"type": "function", "function": {
        "name": "get_summary",
        "description": "Financial summary — totals of income and expenses over a period.",
        "parameters": {"type": "object", "properties": {
            "date_from": {"type": "string"}, "date_to": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "list_clients",
        "description": "List the known clients of the business.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "search",
        "description": "Search transactions by keyword (and optional date range).",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "remove_by_search",
        "description": "Find and permanently remove your most recent transaction matching a keyword.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_usage",
        "description": "Report AI token/usage consumption (NOT a financial expense).",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "help",
        "description": "Explain what the bookkeeper does / redirect an out-of-scope request to reception.",
        "parameters": {"type": "object", "properties": {}}}},
]

VALID_TOOLS = {t["function"]["name"] for t in BOOKKEEPER_TOOLS}
SIDE_EFFECT_TOOLS = {"add_income", "add_outcome", "remove_by_search"}
DESTRUCTIVE_TOOLS = {"remove_by_search"}

BOOKKEEPER_SYSTEM = (
    "You are the execution engine of a financial bookkeeper for a small service business. "
    "For ANY data operation you MUST call the right tool — never invent or compute figures. "
    "Record income with add_income and expenses with add_outcome; report totals with get_summary; "
    "list_clients, search and remove_by_search for the obvious operations; get_usage is for AI "
    "token usage (NOT money spent); help explains scope or redirects an out-of-scope request. "
    "If the user is only chatting, reply briefly without a tool."
)


class BookkeeperDispatcher:
    """Deterministic in-memory dispatcher for the bench (satisfies ToolPolicyDispatcher)."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, dict]] = []

    def tools_schema(self) -> list[dict]:
        return BOOKKEEPER_TOOLS

    def is_mutating(self, name: str) -> bool:
        return name in SIDE_EFFECT_TOOLS

    def requires_confirmation(self, name: str) -> bool:
        return name in DESTRUCTIVE_TOOLS

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        self.executed.append((name, dict(arguments)))
        side = name in SIDE_EFFECT_TOOLS
        canned = {
            "add_income": "Income recorded.", "add_outcome": "Expense recorded.",
            "get_summary": "Income 1200, expenses 800, net +400.",
            "list_clients": "Clients: João, Maria.", "search": "1 match found.",
            "remove_by_search": "Removed the matching entry.",
            "get_usage": "You have used 12k AI tokens.", "help": "I handle finances only.",
        }
        if name in canned:
            return ToolResult(output=canned[name], side_effect=side)
        return ToolResult(output="", ok=False, error=f"unknown tool {name!r}")


# ── Cases ported from the parent persona_cases (ANALYST) ─────────────────────
BOOKKEEPER_CASES: list[EgoCase] = [
    # add_income (5 variants)
    EgoCase("bk_income_haircut", "income after a haircut",
            "A client just paid 40 reais for a haircut.", expect_tool="add_income"),
    EgoCase("bk_income_cash", "income no client",
            "I received 100 reais in cash today.", expect_tool="add_income"),
    EgoCase("bk_income_beard", "income with client name",
            "Carlos paid 25 reais for a beard trim.", expect_tool="add_income"),
    EgoCase("bk_income_pet", "income pet service",
            "A client paid 60 reais for their dog Pandora's bath.", expect_tool="add_income"),
    EgoCase("bk_income_fullname", "income full client name",
            "Maria Souza paid 150 reais for hair coloring.", expect_tool="add_income"),
    # add_outcome (3 variants)
    EgoCase("bk_outcome_bill", "expense electricity bill",
            "I paid a 200 reais electricity bill.", expect_tool="add_outcome"),
    EgoCase("bk_outcome_rent", "expense salon rent",
            "I paid 2500 reais for the salon rent.", expect_tool="add_outcome"),
    EgoCase("bk_outcome_supply", "expense supplies",
            "I spent 80 reais on shampoo supplies.", expect_tool="add_outcome"),
    # get_summary (3 variants)
    EgoCase("bk_summary_today", "summary today", "How much did I earn today?",
            intent_class="INFORMATION_REQUEST", expect_tool="get_summary"),
    EgoCase("bk_summary_week", "summary week", "Give me this week's financial summary.",
            intent_class="INFORMATION_REQUEST", expect_tool="get_summary"),
    EgoCase("bk_summary_month", "summary month expenses", "What did I spend this month?",
            intent_class="INFORMATION_REQUEST", expect_tool="get_summary"),
    # list_clients / search
    EgoCase("bk_list_clients", "list clients", "List my clients.",
            intent_class="INFORMATION_REQUEST", expect_tool="list_clients"),
    EgoCase("bk_search_rent", "search rent", "Find my rent payments.",
            intent_class="INFORMATION_REQUEST", expect_tool="search"),
    # remove_by_search — destructive: must be HELD for confirmation, not executed
    EgoCase("bk_remove_electricity", "remove last electricity expense",
            "Delete the electricity expense I just recorded.",
            expect_pending="remove_by_search", expect_no_mutation=True),
    # get_usage vs expenses (the parent's tokens-vs-despesas distinction)
    EgoCase("bk_usage", "AI token usage", "How many AI tokens have I used?",
            intent_class="INFORMATION_REQUEST", expect_tool="get_usage"),
    # scope guardrail
    EgoCase("bk_help", "scope / capabilities", "What can you do?",
            intent_class="INFORMATION_REQUEST", expect_tool="help"),
]


@dataclass
class _Score:
    total: int = 0
    tool_ok: int = 0
    invariant_ok: int = 0
    invariant_total: int = 0
    fails: list = field(default_factory=list)


async def run(backend, cases: list[EgoCase]) -> _Score:
    """Score BOOKKEEPER tool-selection over the EGO stage (mirrors run_ego, bookkeeper toolset)."""
    stage = EgoStage()
    sc = _Score()
    for case in cases:
        sc.total += 1
        ctx = _ego_ctx(case)
        disp = BookkeeperDispatcher()
        ctx = await stage.process(ctx, backend, disp, system_prompt=BOOKKEEPER_SYSTEM)
        res = ctx.ego_result
        names = [t.tool for t in (res.tools_executed if res else [])]
        dispatched = [n for n, _ in disp.executed]

        # hard invariant: only valid tools dispatched
        sc.invariant_total += 1
        valid = all(n in VALID_TOOLS for n in dispatched)
        sc.invariant_ok += int(valid)

        ok = True
        if case.expect_tool:
            ok = case.expect_tool in names
        elif case.expect_no_tool:
            ok = not names
        if case.expect_pending:                       # destructive held, not executed
            held = [t.tool for t in (res.pending_confirmation if res else [])]
            ok = ok and case.expect_pending in held and case.expect_pending not in dispatched
        if case.expect_no_mutation:
            ok = ok and not [n for n in dispatched if n in SIDE_EFFECT_TOOLS]

        sc.tool_ok += int(ok)
        if not (ok and valid):
            want = case.expect_tool or (f"HOLD {case.expect_pending}" if case.expect_pending
                                        else "no-tool")
            sc.fails.append((case.id, want, names or dispatched or "—"))
    return sc


async def _main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3:8b"
    if not await ollama_available():
        print("Ollama not available at localhost:11434 — skipping.")
        return 0
    print(f"BOOKKEEPER tool-selection bench — model={model}, {len(BOOKKEEPER_CASES)} cases\n")
    sc = await run(build_ollama_text(model), BOOKKEEPER_CASES)
    for cid, want, got in sc.fails:
        print(f"  FAIL {cid}: wanted {want!r}, got {got}")
    pct = 100.0 * sc.tool_ok / sc.total if sc.total else 0.0
    print(f"\ntool-selection: {sc.tool_ok}/{sc.total} ({pct:.0f}%)   "
          f"valid-dispatch invariant: {sc.invariant_ok}/{sc.invariant_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
