"""The judge's two "the Context above" clauses, put BACK — for the digests taken before they
became conditional.

Until 0.1.6 the judge's prompt said "the Context above" twice on every turn, the `# Context`
section rendered or not. The two clauses now render only with the section
(`superego._IN_THE_CONTEXT_ABOVE`, `superego._CONTEXT_HAS_THE_CLOCK`), so the prompt of a turn
WITHOUT a context changed, on purpose, by those words.

Several files pin the digest of such a prompt as an older tree rendered it — landing-time
proofs that a later change moved nothing else. They keep their digests and hash
:func:`with_the_context_clauses` of the prompt instead: the two clauses restored BY REFERENCE,
exactly where they were. A digest that still matches says what it always said, plus one thing:
the 0.1.6 change removed these words and nothing around them.
"""

from __future__ import annotations

import re

from cogno_anima.stages.superego import _CONTEXT_HAS_THE_CLOCK, _IN_THE_CONTEXT_ABOVE

#: The words on each side of the two clauses, as the prompt reads without them.
_CLOCK_BEFORE, _CLOCK_AFTER = "your own idea of today", "). Judge only "
_FACT_BEFORE, _FACT_AFTER = "that is NOT ", "in the persona's limits, "

_SECTION = re.compile(r"(?m)^# Context \(authoritative")


def has_context_section(prompt: str) -> bool:
    return bool(_SECTION.search(prompt))


def with_the_context_clauses(prompt: str) -> str:
    """``prompt`` as it read before the two clauses travelled with the section.

    A prompt that carries the section already has them and comes back untouched. Otherwise the
    clock clause goes back into TRUST THE TOOLS (it was on every turn, so its anchor must be
    there exactly once) and the source clause into the conversational branch's criterion #1
    (the only branch that had it, so its anchor is there once or not at all)."""
    if has_context_section(prompt):
        return prompt
    clock = f"{_CLOCK_BEFORE}{_CLOCK_AFTER}"
    assert prompt.count(clock) == 1, "the TRUST THE TOOLS anchor moved — fix this helper"
    out = prompt.replace(clock, f"{_CLOCK_BEFORE}{_CONTEXT_HAS_THE_CLOCK}{_CLOCK_AFTER}")
    fact = f"{_FACT_BEFORE}{_FACT_AFTER}"
    assert out.count(fact) <= 1, "the conversational anchor is no longer unique"
    return out.replace(fact, f"{_FACT_BEFORE}{_IN_THE_CONTEXT_ABOVE}{_FACT_AFTER}")
