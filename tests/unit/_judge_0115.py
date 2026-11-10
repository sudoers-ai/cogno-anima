"""The judge's three deliberate changes of 0.1.15, UNDONE — for the digests taken before them.

0.1.15 changed the judge's text on purpose in three places, each only where it applies:

* the SYSTEM message of the read-only and conversational branches (`_JUDGE_SYSTEM_ANSWER`): it
  no longer says "Default to NOT approving", which the user half of those branches overrides;
* a call held by NAME renders `HELD_STATUS`, not `ERROR`;
* the held-message rule: criterion (d) points at the GROUNDING criterion's own sources
  (`_HELD_NOT_GROUNDED`), and `_HELD_IS_THE_OUTCOME` follows the asking sentence.

Several files pin the digest of a judge rendering as an older tree produced it — landing-time
proofs that a later change moved nothing else. They keep their digests and hash
:func:`as_0114` of the rendering: the three changes undone BY REFERENCE. A digest that still
matches says what it always said, plus one thing: 0.1.15 changed those words and nothing else.
The execution branch's system message, and every turn with no held call, need no undoing — on
them this function is the identity, and `test_judge_stance_follows_the_branch.py` says so.
"""

from __future__ import annotations

from cogno_anima.stages.superego import (HELD_STATUS, _HELD_IS_THE_OUTCOME, _HELD_NOT_GROUNDED,
                                         _JUDGE_SYSTEM, _JUDGE_SYSTEM_ANSWER)

#: Criterion (d) as 0.1.14 rendered it.
HELD_NOT_SUPPORTED_0114 = ("(d) states anything the request and the successful tool results "
                           "above do not support")


def as_0114(text: str) -> str:
    """``text`` (a judge system message, prompt, or both) as 0.1.14 rendered it."""
    return (text.replace(_JUDGE_SYSTEM_ANSWER, _JUDGE_SYSTEM)
                .replace(f"→ {HELD_STATUS}:", "→ ERROR:")
                .replace(_HELD_NOT_GROUNDED, HELD_NOT_SUPPORTED_0114)
                .replace(_HELD_IS_THE_OUTCOME, ""))
