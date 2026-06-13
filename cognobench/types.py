"""Result types for cognobench — self-contained, no external deps."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CheckResult:
    """A single field-level check within a case."""
    case_id: str
    field: str
    expected: str
    actual: str
    correct: bool


@dataclass
class DimensionResult:
    """Aggregated results for one benchmark dimension (ner, drift, noumeno...)."""
    name: str
    checks: list[CheckResult] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)  # (case_id, message)

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def correct_count(self) -> int:
        return sum(1 for c in self.checks if c.correct)

    @property
    def accuracy(self) -> float:
        return (self.correct_count / self.total * 100.0) if self.total else 0.0

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.correct]

    def to_dict(self) -> dict:
        return {
            "dimension": self.name,
            "total": self.total,
            "correct": self.correct_count,
            "accuracy": round(self.accuracy, 1),
            "errors": len(self.errors),
        }


@dataclass
class BenchReport:
    """Top-level report across all dimensions."""
    dimensions: list[DimensionResult] = field(default_factory=list)
    model: str = ""

    @property
    def total(self) -> int:
        return sum(d.total for d in self.dimensions)

    @property
    def correct_count(self) -> int:
        return sum(d.correct_count for d in self.dimensions)

    @property
    def accuracy(self) -> float:
        return (self.correct_count / self.total * 100.0) if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "overall_accuracy": round(self.accuracy, 1),
            "total": self.total,
            "correct": self.correct_count,
            "dimensions": [d.to_dict() for d in self.dimensions],
        }
