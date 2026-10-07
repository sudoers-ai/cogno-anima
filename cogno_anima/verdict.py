"""
cogno_anima.verdict — the ONE strict reader of a boolean verdict a model wrote.

Three stages ask a model a yes/no question and act on the answer: the JUDGE
(``SuperegoStage.evaluate``, key ``approved``), the scope GUARD
(``SuperegoStage.check_input_scope``, key ``blocked``) and the pre-judge
(``stages.proposal_judge.ProposalJudge``, key ``approved``). Until this module each of them read
that answer its own way, and two of the three read it with ``bool(...)``:

* the judge read ``"approved": "false"`` — a STRING — as APPROVED, because a non-empty string is
  truthy. A fail-CLOSED component failing OPEN on the one field that decides it;
* the guard read ``"blocked": "false"`` as BLOCKED for the same reason, and a reply it could not
  parse at all came back ALLOW, byte for byte the record of a classifier that read the input and
  let it through.

**The rule, said once: only a JSON boolean is a verdict.** A string (``"false"``, ``"True"``,
``"no"``), a number, ``null``, a list, a missing key, a key written TWICE, or no JSON object at
all is an ERROR — never a guess in either direction — and :func:`read_verdict` says WHICH error
from the closed alphabet :data:`VALID_VERDICT_READS`. What an error DOES is not decided here: it
is each stage's own contract (the judge fails CLOSED, the guard fails OPEN, the pre-judge answers
``error``), and every one of them records the read beside its decision so "the model said so" and
"we could not read what the model said" are never the same record again.

**A key written twice is an error, and that is why this module owns the extraction.**
``json.loads`` keeps the LAST value of a repeated key in silence, so
``{"approved": false, "critique": "...", "approved": true}`` parsed to an approval. The reader
sees the pairs before they collapse (``object_pairs_hook``) and refuses a verdict key that occurs
more than once in the top-level object, whatever the two values are — two agreeing copies are
still not ONE verdict. Only the verdict key is refused; a repeated key elsewhere reads as JSON
always read it.

**One extractor.** :func:`parse_object` is the JSON extraction ``SuperegoStage._parse_json`` has
always done (the first ``{`` to the last ``}``, decoded, an object or nothing), moved here so the
strict reader and the tolerant one are the same code; ``_parse_json`` is now expressed over it
and answers what it always answered. ONE object wrapped in anything — prose, a code fence, a
one-element list — is that object, as it always was. It does not salvage, repair or unwrap: two
objects side by side decode to nothing, and an envelope (``{"message": {"approved": false}}``)
has no top-level verdict and reads ``missing``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = [
    "VERDICT_BOOLEAN",
    "VERDICT_UNPARSEABLE",
    "VERDICT_MISSING",
    "VERDICT_DUPLICATED",
    "VERDICT_STRING_BOOL",
    "VERDICT_NOT_BOOLEAN",
    "VERDICT_CALL_FAILED",
    "VALID_VERDICT_READS",
    "VerdictRead",
    "parse_object",
    "read_verdict",
]

#: A JSON boolean was read — the only value of this alphabet under which a verdict exists.
VERDICT_BOOLEAN = "boolean"
#: No JSON object could be decoded from the reply (prose, an empty reply, a cut object, two
#: objects side by side, a bare list or scalar).
VERDICT_UNPARSEABLE = "unparseable"
#: An object was decoded and it has no such key at its top level (an envelope reads as this).
VERDICT_MISSING = "missing"
#: The key occurs more than once in the top-level object.
VERDICT_DUPLICATED = "duplicated"
#: The value is a STRING spelling the JSON literal — ``"true"``/``"false"``, any case, any
#: surrounding whitespace. Counted apart from the other non-booleans because it is the shape a
#: model that almost followed the format produces, and the one ``bool(...)`` read backwards.
VERDICT_STRING_BOOL = "string_bool"
#: Any other value: a number (``0``/``1`` included), ``null``, a list, an object, another string
#: (``"no"``, ``"não"``, ``"approved"``).
VERDICT_NOT_BOOLEAN = "not_boolean"
#: Never returned by :func:`read_verdict`: the stage records it when the model CALL itself raised
#: and there was no reply to read. Listed here so the alphabet a host persists is one tuple.
VERDICT_CALL_FAILED = "call_failed"

#: The closed alphabet of a verdict read, as a stage records it. ``""`` (not a member) means NO
#: VERDICT WAS ASKED — a bypass, nothing to judge — which is a different fact from every row here.
VALID_VERDICT_READS: "tuple[str, ...]" = (
    VERDICT_BOOLEAN,
    VERDICT_UNPARSEABLE,
    VERDICT_MISSING,
    VERDICT_DUPLICATED,
    VERDICT_STRING_BOOL,
    VERDICT_NOT_BOOLEAN,
    VERDICT_CALL_FAILED,
)

# The first ``{`` to the LAST ``}`` — greedy on purpose and unchanged: this is the extraction the
# stages have always used, and narrowing or salvaging it is a different change with its own
# measurement.
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_BOOLEAN_SPELLINGS = frozenset({"true", "false"})


@dataclass(frozen=True)
class VerdictRead:
    """What :func:`read_verdict` found.

    ``value`` is the verdict, and it is ``None`` on every read that is not
    :data:`VERDICT_BOOLEAN` — the two fields cannot disagree. ``data`` is the decoded object
    (``{}`` when there was none), for the fields that travel BESIDE the verdict (a critique, a
    refusal message); a stage that reads them on an error is guessing, and none does.
    """

    value: Optional[bool]
    read: str
    data: "dict[str, Any]" = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when a JSON boolean was read."""
        return self.read == VERDICT_BOOLEAN


def parse_object(raw: str) -> "tuple[Optional[dict[str, Any]], frozenset[str]]":
    """``(object, repeated_keys)`` decoded from ``raw`` — ``(None, frozenset())`` when no JSON
    object can be read from it.

    ``repeated_keys`` are the keys written more than once in the TOP-LEVEL object; the object
    itself holds what ``json.loads`` has always kept for them (the last value), so a caller that
    ignores the second element reads exactly what it read before this function existed. Never
    raises: a reply nested deeply enough to exhaust the decoder is a reply that cannot be read.
    """
    match = _JSON_RE.search(raw or "")
    if not match:
        return None, frozenset()
    # The hook runs once per object, innermost first, so the LAST entry is the top-level one.
    repeated: "list[frozenset[str]]" = []

    def _pairs(pairs: "list[tuple[str, Any]]") -> "dict[str, Any]":
        seen: "dict[str, Any]" = {}
        twice: "set[str]" = set()
        for key, value in pairs:
            if key in seen:
                twice.add(key)
            seen[key] = value
        repeated.append(frozenset(twice))
        return seen

    try:
        data = json.loads(match.group(), object_pairs_hook=_pairs)
    except (ValueError, RecursionError):
        return None, frozenset()
    if not isinstance(data, dict):
        return None, frozenset()
    return data, repeated[-1] if repeated else frozenset()


def read_verdict(raw: str, key: str) -> VerdictRead:
    """Read the boolean ``key`` out of a model's reply. Pure; never raises.

    ``raw`` is the reply as the stage would have parsed it (chain-of-thought already stripped).
    Only a JSON boolean at the top level of the one object in it is a verdict; everything else is
    an error named from :data:`VALID_VERDICT_READS`, with ``value=None``.
    """
    data, repeated = parse_object(raw)
    if data is None:
        return VerdictRead(None, VERDICT_UNPARSEABLE)
    if key in repeated:
        return VerdictRead(None, VERDICT_DUPLICATED, data)
    if key not in data:
        return VerdictRead(None, VERDICT_MISSING, data)
    value = data[key]
    # ``bool`` first and by TYPE: ``True == 1`` in Python, and a model's ``1`` is not a verdict.
    if isinstance(value, bool):
        return VerdictRead(value, VERDICT_BOOLEAN, data)
    if isinstance(value, str) and value.strip().casefold() in _BOOLEAN_SPELLINGS:
        return VerdictRead(None, VERDICT_STRING_BOOL, data)
    return VerdictRead(None, VERDICT_NOT_BOOLEAN, data)
