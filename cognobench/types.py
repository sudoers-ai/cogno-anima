"""Result types for cognobench — self-contained, no external deps.

Two invariants live here (plan Fase 0):

* **Nothing is aggregate-only.** ``to_dict`` can serialize every ``CheckResult``
  (``include_checks=True``, the ``--out`` path), because the noise floor, the
  stable-score rule and the discrimination report all join runs per check —
  an aggregate-only artifact cannot answer any of those questions.
* **An invalid dimension never scores.** A transport failure marks the whole
  dimension invalid (``invalid_reason``); its checks are excluded from every
  total rather than silently shrinking the denominator (the 497/515-vs-622/642
  splice is the incident this rule encodes).
"""

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

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
            "correct": self.correct,
        }

    @property
    def key(self) -> tuple[str, str]:
        """Stable cross-run join key (case_id, field)."""
        return (self.case_id, self.field)


@dataclass
class DimensionResult:
    """Aggregated results for one benchmark dimension (ner, drift, noumeno...)."""
    name: str
    checks: list[CheckResult] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)  # (case_id, message)
    # Transport/instrument failure: the DIMENSION is invalid — excluded from every
    # total, rendered as such, never a silently smaller denominator.
    invalid_reason: str | None = None
    # Side metrics the score cannot see (ner fallback count, per-tool ok=False
    # rates, …) — reported next to the score, never mixed into it.
    meta: dict = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.invalid_reason is None

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

    def to_dict(self, include_checks: bool = False) -> dict:
        d = {
            "dimension": self.name,
            "valid": self.valid,
            "total": self.total,
            "correct": self.correct_count,
            "accuracy": round(self.accuracy, 1),
            "errors": len(self.errors),
        }
        if self.invalid_reason:
            d["invalid_reason"] = self.invalid_reason
        if self.meta:
            d["meta"] = self.meta
        if include_checks:
            d["checks"] = [c.to_dict() for c in self.checks]
            d["error_detail"] = [{"case_id": cid, "message": msg}
                                 for cid, msg in self.errors]
        return d


@dataclass
class BenchReport:
    """Top-level report for ONE run across all selected dimensions."""
    dimensions: list[DimensionResult] = field(default_factory=list)
    model: str = ""
    # Cost meter (filled by the runner's TokenTally): total LLM tokens/calls across
    # the run — multiply by the provider's price table for $ per sweep. 0 on --stub.
    tokens_in: int = 0
    tokens_out: int = 0
    llm_calls: int = 0
    # Run metadata (--out persists all of it; comparisons are only meaningful
    # between runs whose suite versions match — plan principle 4).
    run_id: str = ""
    timestamp: str = ""
    config: dict = field(default_factory=dict)   # models per slot, language, limit…
    suites: dict = field(default_factory=dict)   # dimension → {suite_id, suite_hash, case_ids}
    meta: dict = field(default_factory=dict)     # fallback totals, local tok/s probe…

    @property
    def valid_dimensions(self) -> list[DimensionResult]:
        return [d for d in self.dimensions if d.valid]

    @property
    def invalid_dimensions(self) -> list[DimensionResult]:
        return [d for d in self.dimensions if not d.valid]

    @property
    def total(self) -> int:
        return sum(d.total for d in self.valid_dimensions)

    @property
    def correct_count(self) -> int:
        return sum(d.correct_count for d in self.valid_dimensions)

    @property
    def accuracy(self) -> float:
        return (self.correct_count / self.total * 100.0) if self.total else 0.0

    def to_dict(self, include_checks: bool = False) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "model": self.model,
            "config": self.config,
            "suites": self.suites,
            "overall_accuracy": round(self.accuracy, 1),
            "total": self.total,
            "correct": self.correct_count,
            "invalid_dimensions": [d.name for d in self.invalid_dimensions],
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "llm_calls": self.llm_calls,
            "meta": self.meta,
            "dimensions": [d.to_dict(include_checks=include_checks)
                           for d in self.dimensions],
        }


def aggregate_runs(reports: list[BenchReport]) -> dict:
    """Aggregate N repeat runs into the stable-score / noise-floor report.

    Definitions (plan 0.2): a check is *unstable* when its result diverges across
    the runs that scored it; the *stable score* counts checks passing in the
    MAJORITY of runs; the *noise floor* per dimension is the observed unstable
    count with a conservative default of max(observed, 2). Checks join across
    runs on (dimension, case_id, field). Invalid runs of a dimension are excluded
    from its stability math (they never scored) but are reported.

    Known limit: a case that parse-fails in SOME runs contributes a
    ``case_error`` key instead of its per-field keys, so its flakiness is
    understated here — compare.py (the decision tool) expands case_error into
    FAIL votes on the sibling runs' keys; this aggregate is the quick look.
    """
    dims: dict[str, dict] = {}
    for report in reports:
        for d in report.dimensions:
            slot = dims.setdefault(d.name, {"correct": [], "invalid_runs": 0,
                                            "results": {}})
            if not d.valid:
                slot["invalid_runs"] += 1
                continue
            slot["correct"].append(d.correct_count)
            for c in d.checks:
                slot["results"].setdefault(c.key, []).append(c.correct)

    out: dict[str, dict] = {}
    for name, slot in dims.items():
        results: dict = slot["results"]
        unstable = [k for k, v in results.items() if len(set(v)) > 1]
        # STRICT majority: a tie is FAIL — conservative, order-independent
        # (Counter.most_common broke ties by insertion order, so run ORDER
        # decided a 1-of-2 split), and identical to compare.py's rule.
        stable_pass = sum(1 for v in results.values() if sum(v) * 2 > len(v))
        n_runs = len(slot["correct"])
        out[name] = {
            "runs": n_runs,
            "invalid_runs": slot["invalid_runs"],
            "checks": len(results),
            "correct_min": min(slot["correct"]) if slot["correct"] else 0,
            "correct_max": max(slot["correct"]) if slot["correct"] else 0,
            "stable_correct": stable_pass,
            "unstable_checks": len(unstable),
            "unstable_detail": [f"{cid}::{fld}" for cid, fld in sorted(unstable)],
            # Conservative floor: never trust a difference smaller than 2 checks
            # even when the repeats happened to agree (n is small).
            "noise_floor": max(len(unstable), 2),
        }
    return out
