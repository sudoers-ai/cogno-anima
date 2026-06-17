"""
SUPEREGO (Stage 5) benchmark cases — scope guard + judge + voicer.

Three case kinds, decoupled from NER quality (contexts are hand-built):
  * scope  — Early Input Scope Guard: in/out-of-scope → ALLOW/BLOCK.
  * judge  — quality gate over the EGO execution; criterion #1 is goal↔execution
             ("asked X, did X not Y").
  * voice  — writes the final response grounded in the tool data.

Hard invariants (always): scope.blocked is bool, judge.approved is bool,
voice.response is a non-empty str. Soft (model-dependent, --calibrate-able):
expect_blocked / expect_approved / a grounding substring in the voiced response.
"""

from __future__ import annotations

from dataclasses import dataclass, field

FINANCE_SCOPE = (
    "You are a personal finance assistant. You help ONLY with money, expenses, "
    "income, budgets, balances and financial summaries."
)


@dataclass
class SuperegoCase:
    id: str
    kind: str                       # "scope" | "judge" | "voice"
    user: str
    intent_class: str = "ACTION_REQUEST"
    # scope
    scope_prompt: str = ""
    expect_blocked: bool | None = None
    # judge
    goal: str = ""
    tool: str = ""
    args: dict = field(default_factory=dict)
    result: str = ""
    expect_approved: bool | None = None
    # User pragmatic restrictions (Block 1) — the judge must verify they were
    # honored; a violated `negation` should drive a rejection.
    constraints: list[str] = field(default_factory=list)
    negation: list[str] = field(default_factory=list)
    # voice
    expect_contains: str = ""       # grounding substring that must appear
    parole: str = ""                # NER register (Block 2) → voice accommodation


SUPEREGO_CASES: list[SuperegoCase] = [
    # ── scope guard ──
    SuperegoCase("scope_block_recipe", "scope", "Como faço um bolo de chocolate?",
                 intent_class="INFORMATION_REQUEST", scope_prompt=FINANCE_SCOPE, expect_blocked=True),
    SuperegoCase("scope_block_trivia", "scope", "Quem descobriu o Brasil?",
                 intent_class="INFORMATION_REQUEST", scope_prompt=FINANCE_SCOPE, expect_blocked=True),
    SuperegoCase("scope_allow_finance", "scope", "Quanto gastei esse mês?",
                 intent_class="INFORMATION_REQUEST", scope_prompt=FINANCE_SCOPE, expect_blocked=False),
    SuperegoCase("scope_allow_greeting", "scope", "Oi, bom dia!",
                 intent_class="SOCIAL", scope_prompt=FINANCE_SCOPE, expect_blocked=False),

    # ── judge: goal↔execution ──
    SuperegoCase("judge_correct_expense", "judge", "registra uma despesa de 50 do almoço",
                 goal="record an expense of 50 for lunch", tool="record_expense",
                 args={"amount": 50, "description": "lunch"}, result="Recorded expense of 50 BRL",
                 expect_approved=True),
    SuperegoCase("judge_wrong_kind", "judge", "registra uma despesa de 50 do almoço",
                 goal="record an expense of 50 for lunch", tool="record_income",
                 args={"amount": 50, "description": "lunch"}, result="Recorded income of 50 BRL",
                 expect_approved=False),   # income instead of expense → reject
    SuperegoCase("judge_correct_balance", "judge", "qual meu saldo?",
                 intent_class="INFORMATION_REQUEST", goal="get the account balance",
                 tool="get_balance", args={}, result="Current balance: 1000 BRL",
                 expect_approved=True),

    # ── judge: user constraints / negation (Block 1) ──
    SuperegoCase("judge_violates_negation", "judge",
                 "registra uma despesa de 50 do almoço, mas NÃO categorize",
                 goal="record an expense of 50 for lunch",
                 negation=["do not categorize the expense"],
                 tool="record_expense", args={"amount": 50, "category": "food"},
                 result="Recorded expense of 50 BRL in category 'food'",
                 expect_approved=False),   # categorized despite being told not to → reject
    SuperegoCase("judge_honors_constraint", "judge",
                 "registra uma despesa de 50 do almoço, só desse mês",
                 goal="record an expense of 50 for lunch",
                 constraints=["only for the current month"],
                 tool="record_expense", args={"amount": 50, "description": "lunch"},
                 result="Recorded expense of 50 BRL for the current month",
                 expect_approved=True),    # constraint honored → approve

    # ── voice: grounded ──
    SuperegoCase("voice_balance", "voice", "qual meu saldo?", intent_class="INFORMATION_REQUEST",
                 goal="get balance", tool="get_balance", args={}, result="Current balance: 1000 BRL",
                 expect_contains="1000"),
    SuperegoCase("voice_expense_confirm", "voice", "registra 50 de almoço",
                 goal="record expense", tool="record_expense", args={"amount": 50},
                 result="Recorded expense of 50 BRL", expect_contains="50"),
    # Register accommodation (Block 2): parole feeds the voice prompt. Register
    # adherence is qualitative (inspect via --calibrate); the scored check stays
    # grounding (no regression from carrying the signal).
    SuperegoCase("voice_academic_register", "voice",
                 "Solicito a apresentação do saldo atual da conta",
                 intent_class="INFORMATION_REQUEST", goal="get balance",
                 tool="get_balance", args={}, result="Current balance: 1000 BRL",
                 parole="ACADEMICO", expect_contains="1000"),
]
