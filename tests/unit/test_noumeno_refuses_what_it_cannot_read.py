"""The NOUMENO answers a reply it cannot use with ``StageParseError`` — never with whatever the
value happened to raise.

The stage's contract is "a usable object OR ``StageParseError``". For an object whose FIELDS
were in a form the stage has no repair for it was neither. Measured on ``32803f5`` over 237
replies (each of the six fields under 38 values, plus the clean reply, a missing key and two
bare-literal forms): 102 ended the turn, and none of the 102 with ``StageParseError`` —
50 ``AttributeError``, 31 ``TypeError``, 12 pydantic ``ValidationError``, 9 bare
``ValueError``. A host that catches the typed error to retry or swap models caught none of them.

Now the reply is refused where it is parsed, and the error names the field and the form
(``StageParseError.mark``, from ``repairs.NOUMENO_UNREADABLE_FIELDS``) without a byte of the
value. What the stage has always READ — a ``null`` text, ``"0.9"`` as a confidence, an empty
list — is still read, and still counted as the repair it is.

It imports nothing the change added except through ``getattr``, so the twins run on the tree
before it and are red there for the reason above.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import cogno_anima
from cogno_anima import repairs
from cogno_anima.errors import StageParseError
from cogno_anima.stages.noumeno import Noumeno
from cogno_anima.types import PipelineContext
from tests.conftest import StubBackend, StubEmbedder

PROMPTS_DIR = Path(cogno_anima.__file__).parent / "prompt_templates"
CLEAN = {"rewritten": "Send the invoice.", "context_turn": "", "confidence": 0.9,
         "changed": False, "preserved_terms": [], "rewrite_warnings": []}
SECRET = "zqx7 odile.varnes@example.com"

#: (field, value, the mark the error carries) — every FORM that ended the turn on ``32803f5``.
REFUSED = [
    # a text that is neither a string nor null — `.strip()` raised AttributeError
    *[(key, value, f"coerced:{key}") for key in ("rewritten", "context_turn")
      for value in (0, 1, 7.5, -1, True, False, [], ["a"], [SECRET], {}, {"k": SECRET})],
    # a confidence that is null, or neither a number nor the spelling of one
    ("confidence", None, "null_field:confidence"),
    *[("confidence", value, "coerced:confidence")
      for value in ("", "  ", "true", "high", SECRET, [], [0.9], {}, {"k": 0.9})],
    # a list field that is a number or `true`, or a list holding something that is not a string
    *[(key, value, f"coerced:{key}") for key in ("preserved_terms", "rewrite_warnings")
      for value in (1, 7.5, -1, True, [None, 3, "ok", ""], [SECRET, 4], [["a"]], [{"a": 1}],
                    [1.5], [True])],
]

#: (field, value, the repair mark on the RESULT) — forms the stage has always read. They did
#: not end the turn before and must not now; ``None`` is "no mark for this field".
STILL_READ = [
    ("rewritten", None, "null_field:rewritten"),
    ("rewritten", "", "default:rewritten"),
    ("context_turn", None, "null_field:context_turn"),
    ("confidence", "0.9", "coerced:confidence"),
    ("confidence", True, "coerced:confidence"),
    ("confidence", 1, None),
    ("confidence", 7.5, None),
    ("confidence", "NaN", "coerced:confidence"),
    ("preserved_terms", None, "null_field:preserved_terms"),
    ("preserved_terms", ["R$ 40,00", "invoice"], None),
    ("preserved_terms", 0, "coerced:preserved_terms"),
    ("preserved_terms", False, "coerced:preserved_terms"),
    ("preserved_terms", "abc", "coerced:preserved_terms"),
    ("preserved_terms", {"k": "v"}, "coerced:preserved_terms"),
    ("rewrite_warnings", None, "null_field:rewrite_warnings"),
    ("rewrite_warnings", ["AMBIGUOUS"], None),
    ("rewrite_warnings", {}, "coerced:rewrite_warnings"),
]


class Replies(StubBackend):
    def __init__(self, replies):
        super().__init__()
        self.replies = list(replies)

    async def generate(self, system, prompt):
        return self.replies.pop(0), self.tokens_in, self.tokens_out


def _stage() -> Noumeno:
    return Noumeno(embedder=StubEmbedder(), prompts_dir=PROMPTS_DIR)


async def _run(reply: dict, user: str = "Manda a fatura."):
    ctx = PipelineContext(user_input=user, force_language="pt")
    await _stage().process(ctx, Replies([json.dumps(reply)]))
    return ctx.noumeno


def _id(case) -> str:
    return f"{case[0]}={json.dumps(case[1])[:28]}"


@pytest.mark.parametrize("key, value, mark", REFUSED, ids=[_id(c) for c in REFUSED])
async def test_a_field_the_stage_cannot_use_is_a_stage_parse_error(key, value, mark):
    with pytest.raises(Exception) as caught:        # noqa: PT011 — the CLASS is the assertion
        await _run(dict(CLEAN, **{key: value}))
    error = caught.value
    assert type(error) is StageParseError, f"the turn ended with {type(error).__name__}"
    assert isinstance(error, ValueError) and error.stage == "noumeno"
    assert getattr(error, "mark", None) == mark
    assert mark in getattr(repairs, "VALID_NOUMENO_UNREADABLE", ())
    # the mark is two closed halves: it names the field and never holds the value
    assert "zqx7" not in mark and "@" not in mark


@pytest.mark.parametrize("key, value, mark", STILL_READ, ids=[_id(c) for c in STILL_READ])
async def test_a_form_the_stage_has_always_read_is_still_read(key, value, mark):
    result = await _run(dict(CLEAN, **{key: value}))
    assert result.rewritten                       # a rewrite, or the contact's own text
    of_this_field = [m for m in result.repairs if m.endswith(f":{key}")]
    assert of_this_field == ([mark] if mark else [])


async def test_the_clean_reply_and_each_missing_key_are_read():
    assert (await _run(CLEAN)).repairs == []
    for key in CLEAN:
        result = await _run({k: v for k, v in CLEAN.items() if k != key})
        assert result.repairs == [f"default:{key}"]


async def test_every_mark_of_the_table_is_raised_by_some_reply_and_no_other_is():
    raised = set()
    for key, value, _ in REFUSED:
        try:
            await _run(dict(CLEAN, **{key: value}))
        except StageParseError as error:
            raised.add(error.mark)
    assert raised == repairs.VALID_NOUMENO_UNREADABLE == {
        "coerced:rewritten", "coerced:context_turn", "null_field:confidence",
        "coerced:confidence", "coerced:preserved_terms", "coerced:rewrite_warnings"}
    # a refusal is not a repair: the two alphabets share one mark, and it means two things
    assert raised & repairs.VALID_NOUMENO_REPAIRS == {
        "coerced:confidence", "coerced:preserved_terms", "coerced:rewrite_warnings"}


async def test_a_reply_that_is_not_an_object_has_no_mark():
    ctx = PipelineContext(user_input="Manda a fatura.", force_language="pt")
    with pytest.raises(StageParseError) as caught:
        await _stage().process(ctx, Replies(["[1, 2]"]))
    assert caught.value.mark is None


async def test_no_reply_of_the_sweep_ends_the_turn_with_anything_else():
    """The property, over the sweep that counted the 102: each field under every value, the
    outcome is a result or a ``StageParseError`` — nothing else, ever."""
    values = [None, "", "  ", "true", "false", "yes", "x" * 90, 0, 1, 7.5, -1, True, False, [],
              ["a"], ["a"] * 7, [None, 3, "ok", ""], {}, {"k": "v"}, "0.9", "NaN", "abc",
              [["a"]], [{"a": 1}], [1.5], [True], 2, 0.5, 10 ** 400]
    results = refused = 0
    for key in CLEAN:
        for value in values:
            try:
                await _run(dict(CLEAN, **{key: value}))
                results += 1
            except StageParseError:
                refused += 1
    assert results + refused == len(CLEAN) * len(values)
    assert results >= 80 and refused >= 60          # both halves were exercised


async def test_an_echo_retry_the_stage_cannot_use_keeps_the_first_answer():
    """The backstop's own promise — "never kill a turn the primary call already served" — now
    holds for a retry whose fields are unusable, not only for one that is not JSON. Before, a
    retry with a ``null`` confidence was ADOPTED and then ended the turn."""
    echo = "Yes, I would like to know how to repot the fern."     # an example of the prompt
    first = dict(CLEAN, rewritten=echo)
    retry = dict(CLEAN, rewritten="Yes, please.", confidence=None)
    ctx = PipelineContext(user_input="Sim")
    await _stage().process(ctx, Replies([json.dumps(first), json.dumps(retry)]))
    assert ctx.noumeno.rewritten == echo
    assert "FEW_SHOT_ECHO" in ctx.noumeno.rewrite_warnings
