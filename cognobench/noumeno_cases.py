"""
NOUMENO quality cases — perception/normalization layer.

Validates the NOUMENO contract:
  - language detection (BCP-47 short code)
  - rewrite is non-empty and (for non-English input) actually rewritten to English
  - drift_tag is a valid classification
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_DRIFT_TAGS = {"PASS_THROUGH", "REWRITTEN", "COMPRESSED", "EXPANDED", "DRIFT"}


@dataclass
class NoumenoCase:
    """A single NOUMENO quality benchmark case."""
    id: str
    input: str
    expect_language: str = ""          # pt | en | es (matched against language prefix)
    expect_changed: bool | None = None  # True if a structural rewrite is expected


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
]
