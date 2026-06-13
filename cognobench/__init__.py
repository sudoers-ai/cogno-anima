"""
cognobench — Cognitive benchmark for cogno-core.

A self-contained, dependency-light benchmark for the cognitive layer
(NOUMENO → NER → Drift) of cogno-core. Unlike the parent Cogno SaaS
benchmark, this one has NO business/infra coupling: it drives the stages
directly through dependency injection (any LLMBackend + Embedder).

Dimensions available today (the implemented stages):
  - noumeno : rewrite non-empty, language detection, drift_tag validity
  - ner     : intent/sentiment/temporal/entities/langue/pii/speech_act/...
  - drift   : cumulative drift + action band (embedding-based, pure)

ID / EGO / SUPEREGO dimensions will be added as those stages land.
"""

from cognobench.types import CheckResult, DimensionResult, BenchReport

__all__ = ["CheckResult", "DimensionResult", "BenchReport"]
