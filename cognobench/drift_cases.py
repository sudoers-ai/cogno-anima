"""
Drift quality cases.

CALIBRATION (2026-06, llama3.1:8b + nomic-embed-text, language forced pt-BR),
AFTER the cumulative renormalization + ontological degenerate-case fix:

`compute_cumulative` now renormalizes over the stages actually computed, so with
only NOUMENO + NER the cumulative is the weighted average of epistemological +
ontological on a full [0,1] scale (no longer capped at 0.30). Ontological drift
is left UNCOMPUTED (None) for content-poor rewrites (greetings) so it does not
false-trigger an action. As a result the drift now discriminates and the action
thresholds (warn .50 / ask_user .70 / self_correct .85) fire meaningfully:

    clear inputs        cumulative ~0.11–0.25  → none
    vague request       ~0.62                  → warn
    garbage             ~0.50                  → warn
    nonsense            ~0.76                  → ask_user

Bands are a regression guard with margin for LLM/embedding nondeterminism, not a
tight contract. `expected_action` is asserted only where it is comfortably away
from a threshold; borderline cases leave it blank (band-only). Re-run
`--calibrate` and widen if they flake. Hard invariants (valid action,
cumulative ∈ [0,1]) are always enforced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

VALID_ACTIONS = {"none", "warn", "ask_user", "self_correct"}


@dataclass
class DriftCase:
    """A single drift quality benchmark case.

    expected_action : soft expectation ("" = skip, used for near-threshold cases).
    min/max_cumulative : soft band, calibrated from real output (+margin).
    epist_observed : reference epistemological drift from calibration (informational).
    """
    id: str
    input: str
    expected_action: str = ""
    min_cumulative: float = 0.0
    max_cumulative: float = 1.0
    history: list[str] = field(default_factory=list)
    description: str = ""
    epist_observed: float = 0.0


DRIFT_CASES: list[DriftCase] = [
    # ── Clear inputs → low drift, action none ────────────────────────────
    DriftCase(id="drift_clear_math", input="quanto é 2+2?",
              expected_action="none", min_cumulative=0.05, max_cumulative=0.40,
              epist_observed=0.194, description="Simple math"),
    DriftCase(id="drift_clear_news", input="notícias sobre tecnologia",
              expected_action="none", min_cumulative=0.05, max_cumulative=0.40,
              epist_observed=0.363, description="News request"),
    DriftCase(id="drift_clear_greeting", input="hello, how are you?",
              expected_action="none", min_cumulative=0.0, max_cumulative=0.30,
              epist_observed=0.119,
              description="Greeting — onto uncomputed (content-poor), no false warn"),
    DriftCase(id="drift_clear_contradictory",
              input="delete everything but keep it all safe and undo it",
              expected_action="none", min_cumulative=0.0, max_cumulative=0.35,
              epist_observed=0.227, description="LLM normalizes → low drift"),
    DriftCase(id="drift_faithful_response", input="busque notícias de inteligência artificial",
              expected_action="none", min_cumulative=0.05, max_cumulative=0.42,
              epist_observed=0.433, description="Standard request"),
    DriftCase(id="drift_multi_turn_abrupt_shift", input="e qual a cotação do dólar hoje?",
              expected_action="none", min_cumulative=0.05, max_cumulative=0.45,
              epist_observed=0.491,
              history=["olá, me faça um resumo do livro Dom Casmurro", "quem é Capitu?"],
              description="Topic shift literature → finance"),

    # ── Moderate ambiguity → still none but higher ───────────────────────
    DriftCase(id="drift_moderate_ambiguous", input="me ajuda com aquilo de ontem",
              expected_action="none", min_cumulative=0.10, max_cumulative=0.48,
              epist_observed=0.567, description="Vague reference"),

    # ── Near/over threshold → band-only (action left blank: borderline) ──
    DriftCase(id="drift_moderate_vague", input="faz aquela coisa",
              min_cumulative=0.40, max_cumulative=0.85,
              epist_observed=0.624, description="Vague request → ~warn"),
    DriftCase(id="drift_garbage", input="asdf ghjk qwer zxcv bnm 12345 !@#$%",
              min_cumulative=0.25, max_cumulative=0.80,
              epist_observed=0.000, description="Garbage → onto 1.0 → ~warn"),
    DriftCase(id="drift_high_nonsense", input="purpura elefante teclado lua abacaxi",
              min_cumulative=0.45, max_cumulative=0.95,
              epist_observed=0.514, description="Nonsensical → epist + onto 1.0 → ~ask_user"),
]
