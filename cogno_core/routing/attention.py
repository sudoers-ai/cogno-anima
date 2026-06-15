"""
cogno_core.routing.attention — Select relevant context candidates per turn.

The pipeline may have access to many memory/context keys. Rather than passing all
of them downstream (wasting tokens and diluting focus), the AttentionFilter
scores each candidate string by relevance to the current IntentResult and returns
the top-N. Candidates are plain strings supplied by the host (keys into a memory
store, skill registry, etc.); for richer matching pass enriched strings like
"RECENT:TECH:docker-daemon".

Pure heuristic — no embedding, no I/O. Scoring factors are additive.
"""

from __future__ import annotations

import re

from cogno_core.types import IntentResult


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-ZÀ-ÿ0-9]+", text.lower()))


class AttentionFilter:
    """Returns the top-N candidate strings most relevant to the current intent."""

    def __init__(self, top_n: int = 5) -> None:
        self._top_n = top_n

    def focus(self, intent: IntentResult, candidates: list[str]) -> list[str]:
        """Score candidates by relevance to `intent`; return up to top_n, desc score."""
        if not candidates:
            return []
        scored = [(self._score(intent, c), c) for c in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[: self._top_n]]

    def _score(self, intent: IntentResult, candidate: str) -> float:
        score = 0.0
        cand_upper = candidate.upper()
        cand_tokens = _tokens(candidate)

        # 1. Temporal class match.
        if intent.temporal_class and intent.temporal_class.upper() in cand_upper:
            score += 1.0

        # 2. Domain overlap.
        if any(d.upper() in cand_upper for d in intent.domains):
            score += 0.8

        # 3. Mandatory tags overlap (strip the "NER." prefix).
        if any(t.replace("NER.", "").upper() in cand_upper for t in intent.mandatory_tags):
            score += 0.7

        # 4. Named-people match (any token of any person name).
        if any(
            p.lower() in candidate.lower()
            for person in intent.entities_people
            for p in person.split()
        ):
            score += 0.9

        # 5. Goal keyword overlap.
        if intent.goal and (_tokens(intent.goal) & cand_tokens):
            score += 0.6

        return score
