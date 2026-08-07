"""
NOUMENO quality cases — perception/normalization layer.

Validates the NOUMENO contract:
  - language detection (BCP-47 short code)
  - rewrite is non-empty and (for non-English input) actually rewritten to English
  - drift_tag is a valid classification
  - short-reply resolution: a bare answer ("Sim", "WhatsApp", "20") to the
    assistant's pending question must be resolved into a full statement carrying
    the question's content (regression: CLOSER session where bare "Sim" wiped the
    goal and restarted the sales arc)
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Suite version (plan 0.6): bump on ANY case addition/removal/edit, then
# re-record with `python -m cognobench.suites --update`. Published numbers
# cite this id; different versions never share a table.
SUITE_ID = "noumeno-v1"

VALID_DRIFT_TAGS = {"PASS_THROUGH", "REWRITTEN", "COMPRESSED", "EXPANDED", "DRIFT"}


@dataclass
class NoumenoCase:
    """A single NOUMENO quality benchmark case."""
    id: str
    input: str
    expect_language: str = ""          # pt | en | es (matched against language prefix)
    expect_changed: bool | None = None  # True if a structural rewrite is expected
    conversation: str = ""              # verbatim transcript fed as conversation_history
    expect_in_rewrite: tuple[str, ...] = field(default_factory=tuple)
    # ^ case-insensitive substrings the resolved rewrite must contain (short-reply
    #   resolution: the question's content must be merged into the answer). Each
    #   entry may hold "|"-separated alternatives ("20|twenty"): any one satisfies.


NOUMENO_CASES: list[NoumenoCase] = [
    NoumenoCase(id="noumeno_pt_rewrite", input="me explica como funciona a fotossíntese",
                expect_language="pt", expect_changed=True),
    NoumenoCase(id="noumeno_en_passthrough", input="explain how neural networks work",
                expect_language="en"),
    NoumenoCase(id="noumeno_es_rewrite", input="explica qué es la inteligencia artificial",
                expect_language="es", expect_changed=True),
    NoumenoCase(id="noumeno_pt_action", input="cria um lembrete para amanhã às 9h",
                expect_language="pt", expect_changed=True),
    NoumenoCase(id="noumeno_en_question", input="what is the capital of Japan?",
                expect_language="en"),
    # ── Short-reply resolution (all four from the real CLOSER failure log) ──
    NoumenoCase(id="noumeno_short_sim",
                input="Sim",
                conversation=("User: Por hoje eu me cadastro no Cogno.\n"
                              "Assistant: Você gostaria de conhecer mais sobre o Cogno?"),
                expect_in_rewrite=("cogno",)),
    NoumenoCase(id="noumeno_short_channel",
                input="WhatsApp",
                conversation=("User: Quero melhorar meu atendimento.\n"
                              "Assistant: Por onde os seus clientes chegam até você?"),
                expect_in_rewrite=("whatsapp", "client")),
    NoumenoCase(id="noumeno_short_number",
                input="20",
                conversation=("User: Meus clientes chegam pelo WhatsApp.\n"
                              "Assistant: Quantos clientes chegam até você por dia?"),
                expect_in_rewrite=("20|twenty", "client")),
    NoumenoCase(id="noumeno_short_hedge",
                input="Talvez",
                conversation=("User: Entendi como funciona.\n"
                              "Assistant: Quer marcar uma conversa curta com o nosso time?"),
                expect_in_rewrite=("maybe",)),
]
