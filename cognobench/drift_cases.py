"""
Drift quality cases.

NOTE — recalibration vs the parent:
The parent Cogno drift cases were "calibrated against the heuristic (non-LLM)
drift calculator". cogno-core's drift is different by design: epistemological
drift is embedding-based and owned by NOUMENO, and DriftCalculator is a pure
consumer. So the parent's exact cumulative bands do NOT transfer.

Therefore the numeric `min_cumulative`/`max_cumulative` here are treated as a
SOFT band: the drift dimension always validates the hard invariants
(action ∈ valid set, cumulative ∈ [0,1], no crash) and records the actual
cumulative so the bands can be recalibrated against real embedding output.
Run with `--calibrate` to print actuals without failing on the soft band.
"""

from __future__ import annotations

from dataclasses import dataclass, field

VALID_ACTIONS = {"none", "warn", "ask_user", "self_correct"}


@dataclass
class DriftCase:
    """A single drift quality benchmark case."""
    id: str
    input: str
    expected_action: str = ""          # soft expectation; "" = don't check
    min_cumulative: float = 0.0
    max_cumulative: float = 1.0
    history: list[str] = field(default_factory=list)
    description: str = ""


DRIFT_CASES: list[DriftCase] = [
    # ── Clear inputs → low drift ─────────────────────────────────────────
    DriftCase(id="drift_clear_math", input="quanto é 2+2?",
              expected_action="none", max_cumulative=0.65,
              description="Simple math — should stay low drift"),
    DriftCase(id="drift_clear_news", input="notícias sobre tecnologia",
              expected_action="none", min_cumulative=0.0, max_cumulative=0.55,
              description="News request"),
    DriftCase(id="drift_clear_greeting", input="hello, how are you?",
              expected_action="none", max_cumulative=0.30,
              description="Simple greeting"),

    # ── Moderate ambiguity → detectable drift ────────────────────────────
    DriftCase(id="drift_moderate_ambiguous", input="me ajuda com aquilo de ontem",
              min_cumulative=0.0, max_cumulative=0.65,
              description="Vague reference + temporal shift"),
    DriftCase(id="drift_moderate_vague", input="faz aquela coisa",
              min_cumulative=0.0, max_cumulative=0.65,
              description="Vague request"),

    # ── High ambiguity → higher drift ────────────────────────────────────
    DriftCase(id="drift_high_nonsense", input="purpura elefante teclado lua abacaxi",
              min_cumulative=0.0, max_cumulative=0.90,
              description="Nonsensical — high ontological/synthesis drift expected"),
    DriftCase(id="drift_high_contradictory",
              input="delete everything but keep it all safe and undo it",
              min_cumulative=0.0, max_cumulative=0.90,
              description="Contradictory instruction"),

    # ── Faithfulness ─────────────────────────────────────────────────────
    DriftCase(id="drift_faithful_response", input="busque notícias de inteligência artificial",
              expected_action="none", min_cumulative=0.0, max_cumulative=0.55,
              description="Standard request"),

    # ── Garbage input ────────────────────────────────────────────────────
    DriftCase(id="drift_garbage", input="asdf ghjk qwer zxcv bnm 12345 !@#$%",
              min_cumulative=0.0, max_cumulative=1.0,
              description="Garbage input — embedding drift should be high"),

    # ── Multi-turn drift (cumulative context shifts) ─────────────────────
    DriftCase(id="drift_multi_turn_abrupt_shift", input="e qual a cotação do dólar hoje?",
              min_cumulative=0.0, max_cumulative=0.80,
              history=["olá, me faça um resumo do livro Dom Casmurro", "quem é Capitu?"],
              description="Abrupt topic shift literature → finance → situational drift"),
]
