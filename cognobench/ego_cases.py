"""
EGO (Stage 4) benchmark cases — tool selection + agent-loop behaviour.

The EGO is an EXECUTOR: it decides which tool to call and runs it (via a host
dispatcher), gathering data; it does NOT write the user reply (the SUPEREGO
does). So this dimension scores **tool selection** and **loop hygiene**, NOT
prose. It is deliberately decoupled from NER quality: each case hand-builds the
NOUMENO+NER context (the NER dimension already covers extraction), so an EGO
miss reflects the model's tool-use ability, not upstream NER noise.

The default ``OllamaBackend`` has no native FC, so the EGO runs the **text
fallback path** (``<TOOL_CALL>`` tags) — the same path the distilled student
will use. Tool execution is delegated to ``BenchDispatcher`` (in-memory,
deterministic; no DB/MCP).

Hard invariants (always enforced): a valid EgoResult with ≥1 step, and every
*dispatched* tool is a real tool (the loop blocks hallucinated names). Soft
(model-dependent, ``--calibrate``able): the expected tool was selected, or a
chat turn called no tool.
"""

from __future__ import annotations

from dataclasses import dataclass

from cogno_anima.types import ToolResult

# ── The bench toolset (clear, finance + utility, to exercise selection) ──
BENCH_TOOLS: list[dict] = [
    {"type": "function", "function": {
        "name": "record_expense",
        "description": "Record an expense — money the user spent.",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number", "description": "value in BRL"},
            "description": {"type": "string", "description": "what it was spent on"},
        }, "required": ["amount", "description"]}}},
    {"type": "function", "function": {
        "name": "record_income",
        "description": "Record an income — money the user received.",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number", "description": "value in BRL"},
            "description": {"type": "string", "description": "source of the income"},
        }, "required": ["amount", "description"]}}},
    {"type": "function", "function": {
        "name": "get_balance",
        "description": "Get the user's current account balance.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_summary",
        "description": "Summarise income and expenses over a period (financial overview).",
        "parameters": {"type": "object", "properties": {
            "period": {"type": "string", "description": "e.g. 'this month', 'week'"}}}}},
    {"type": "function", "function": {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another.",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number"},
            "from_currency": {"type": "string"},
            "to_currency": {"type": "string"},
        }, "required": ["amount", "from_currency", "to_currency"]}}},
    {"type": "function", "function": {
        "name": "delete_all_records",
        "description": "Permanently delete ALL of the user's financial records.",
        "parameters": {"type": "object", "properties": {}}}},
]

VALID_TOOLS = {t["function"]["name"] for t in BENCH_TOOLS}

# Mutating = writes / side effects (drives the read-only mask). Destructive =
# irreversible, must be confirmed (drives the confirmation gate). The host owns
# this classification; the bench dispatcher declares it via ToolPolicyDispatcher.
SIDE_EFFECT_TOOLS = {"record_expense", "record_income", "delete_all_records"}
DESTRUCTIVE_TOOLS = {"delete_all_records"}

EGO_SYSTEM = (
    "You are the execution engine of a personal finance assistant. For ANY data "
    "operation you MUST call the appropriate tool — never invent, compute, or "
    "guess the data yourself. If the user is only chatting (a greeting or a "
    "thank-you), reply briefly WITHOUT calling any tool. When the task is done, "
    "give a short confirmation."
)


class BenchDispatcher:
    """Deterministic in-memory dispatcher for the bench (no DB/MCP).

    Satisfies ToolPolicyDispatcher (``is_mutating``/``requires_confirmation``) so
    the read-only mask and the confirmation gate can be exercised end-to-end.
    """

    def __init__(self) -> None:
        self.executed: list[tuple[str, dict]] = []

    def tools_schema(self) -> list[dict]:
        return BENCH_TOOLS

    def is_mutating(self, name: str) -> bool:
        return name in SIDE_EFFECT_TOOLS

    def requires_confirmation(self, name: str) -> bool:
        return name in DESTRUCTIVE_TOOLS

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        self.executed.append((name, dict(arguments)))
        side = name in SIDE_EFFECT_TOOLS
        if name == "record_expense":
            return ToolResult(output=f"Recorded expense of {arguments.get('amount')} BRL.", side_effect=side)
        if name == "record_income":
            return ToolResult(output=f"Recorded income of {arguments.get('amount')} BRL.", side_effect=side)
        if name == "get_balance":
            return ToolResult(output="Current balance: 1000 BRL.")
        if name == "get_summary":
            return ToolResult(output="This period: income 1200, expenses 800, net +400.")
        if name == "convert_currency":
            return ToolResult(output=f"{arguments.get('amount')} {arguments.get('from_currency')} "
                                     f"= {arguments.get('amount')} {arguments.get('to_currency')} (demo rate).")
        if name == "delete_all_records":
            return ToolResult(output="All records permanently deleted.", side_effect=side)
        return ToolResult(output="", ok=False, error=f"unknown tool {name!r}")


@dataclass
class EgoCase:
    id: str
    description: str
    task: str                      # canonical-English task (the NOUMENO rewrite)
    intent_class: str = "ACTION_REQUEST"
    expect_tool: str = ""          # soft: the tool the model should pick ("" = skip)
    expect_no_tool: bool = False   # soft: a chat turn should call no tool
    # Capability gates (HARD invariants, deterministic — not model goodwill):
    readonly: bool = False         # Fonte A: host masked mutating tools this turn
    expect_no_mutation: bool = False     # assert no mutating tool was dispatched
    expect_pending: str = ""       # Fonte B: this destructive tool must be HELD (not run)


EGO_CASES: list[EgoCase] = [
    EgoCase("expense_explicit", "Explicit expense", "Record an expense of 50 reais for lunch.",
            expect_tool="record_expense"),
    EgoCase("expense_colloquial", "Colloquial expense", "I just spent 30 reais on coffee.",
            expect_tool="record_expense"),
    EgoCase("income_with_client", "Income with client",
            "Add an income of 200 reais from a haircut for client Maria.",
            expect_tool="record_income"),
    EgoCase("balance", "Balance query", "What is my current balance?",
            intent_class="INFORMATION_REQUEST", expect_tool="get_balance"),
    EgoCase("summary_period", "Summary for a period", "Give me this month's financial summary.",
            intent_class="INFORMATION_REQUEST", expect_tool="get_summary"),
    EgoCase("summary_fuzzy", "Fuzzy overview → summary", "How am I doing financially?",
            intent_class="INFORMATION_REQUEST", expect_tool="get_summary"),
    EgoCase("currency", "Currency conversion", "Convert 100 dollars to reais.",
            intent_class="INFORMATION_REQUEST", expect_tool="convert_currency"),
    EgoCase("farewell", "Pure chat — no tool", "Thank you so much, that's all for now!",
            intent_class="SOCIAL", expect_no_tool=True),
    EgoCase("greeting", "Greeting — no tool", "Hi there, good morning!",
            intent_class="SOCIAL", expect_no_tool=True),

    # ── Read-only mask (Fonte A) — host flagged the user as tentative ──
    # The mutating tools are masked, so the model CANNOT commit (hard invariant:
    # no mutating tool dispatched) — it consults/proposes instead.
    EgoCase("readonly_propose", "Tentative action → propose, never commit",
            "I think I maybe spent around 30 reais on coffee, but I'm not sure.",
            readonly=True, expect_no_mutation=True),

    # ── Confirmation gate (Fonte B) — destructive tool, unconfirmed ──
    # The model may pick delete_all_records, but the core HOLDS it (hard
    # invariant: it is never executed; it surfaces as pending_confirmation).
    EgoCase("destructive_needs_confirmation", "Destructive action held for confirmation",
            "Delete all of my financial records.",
            expect_pending="delete_all_records", expect_no_mutation=True),
]
