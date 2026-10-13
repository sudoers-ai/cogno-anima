"""The voice's two "the context" clauses, put BACK — the twin of `_judge_context_clauses`.

Until 0.1.6 the voice's prompt told the voice to "use the context for background" in its
`# Task`, and called "the context above" background in the approved-answer section, on every
turn — the `# Context (memories/history)` section rendered or not. Both now render only with
the section (`superego._USE_THE_CONTEXT`, `superego._CONTEXT_IS_BACKGROUND`).

The files that pin the digest of a context-less voice prompt as an older tree rendered it hash
:func:`with_the_context_clauses` of the prompt: the two clauses restored BY REFERENCE, exactly
where they were. A digest that still matches says the change removed these words and nothing
around them.
"""

from __future__ import annotations

import re

from cogno_anima.stages.superego import _CONTEXT_IS_BACKGROUND, _USE_THE_CONTEXT

#: The words on each side of the two clauses, as the prompt reads without them.
_TASK_BEFORE = "in the persona's voice and within its limits. "
_APPROVED_BEFORE, _APPROVED_AFTER = "PRIMARY source for what to say", ".\n"

_SECTION = re.compile(r"(?m)^# Context \(memories/history\)")


def has_context_section(prompt: str) -> bool:
    return bool(_SECTION.search(prompt))


def with_the_context_clauses(prompt: str) -> str:
    """``prompt`` as it read before the two clauses travelled with the section.

    A prompt that carries the section already has them and comes back untouched. Otherwise the
    `# Task` sentence goes back (it was on every turn, so its anchor is there exactly once) and
    so does the approved-answer clause (rendered only on a turn whose draft review approved, so
    its anchor is there once or not at all)."""
    if has_context_section(prompt):
        return prompt
    assert prompt.count(_TASK_BEFORE) == 1, "the `# Task` anchor moved — fix this helper"
    out = prompt.replace(_TASK_BEFORE, f"{_TASK_BEFORE}{_USE_THE_CONTEXT}")
    approved = f"{_APPROVED_BEFORE}{_APPROVED_AFTER}"
    assert out.count(approved) <= 1, "the approved-answer anchor is no longer unique"
    return out.replace(approved, f"{_APPROVED_BEFORE}{_CONTEXT_IS_BACKGROUND}{_APPROVED_AFTER}")
