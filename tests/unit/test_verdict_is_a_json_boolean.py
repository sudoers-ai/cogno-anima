"""Only a JSON boolean is a verdict — in the judge, in the scope guard and in the pre-judge.

Three stages ask a model a yes/no question and act on the answer. Two of them read it with
``bool(...)``, and a non-empty string is truthy:

* the JUDGE (fail-CLOSED) read ``"approved": "false"`` as APPROVED;
* the scope GUARD (fail-OPEN) read ``"blocked": "false"`` as BLOCKED, and a reply with no JSON
  in it came back ALLOW with a record identical to the one of an allow the classifier gave.

The RED TWINS — those sentences, each failing on the tree before the fix for the reason it
names — live in ``test_verdict_red_twins.py``, which imports nothing the fix added so that it
RUNS on that tree. This file is everything around them: the reader's own table, the adverse
shapes, the key written twice, what each stage's contract does with an unread verdict, the mark
each one leaves, and the callers of the tolerant ``_parse_json`` — enumerated from the code, so
the next caller shows up here.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path

import pytest

import cogno_anima
from cogno_anima import metakeys as mk
from cogno_anima.stages import scope_options
from cogno_anima.stages.proposal_judge import ProposalJudge
from cogno_anima.stages.superego import UNREADABLE_VERDICT_CRITIQUE, SuperegoStage
from cogno_anima.tools.pre_judge import PRE_APPROVED, PRE_CRITIQUE, PRE_ERROR, Proposal
from cogno_anima.types import ScopeCheckResult, StageMetrics, SuperegoResult
from cogno_anima.verdict import (
    VALID_VERDICT_READS,
    VERDICT_BOOLEAN,
    VERDICT_CALL_FAILED,
    VERDICT_DUPLICATED,
    VERDICT_MISSING,
    VERDICT_NOT_BOOLEAN,
    VERDICT_STRING_BOOL,
    VERDICT_UNPARSEABLE,
    parse_object,
    read_verdict,
)
from tests.unit.test_superego import RaisingBackend, ScriptedBackend, _ctx

_SCOPE = "You handle scheduling and study support for the school."
_USER = "what can I use to study for the exam?"

#: The reply shapes that are NOT a verdict, and the read each one is recorded as. One table for
#: the reader, the judge, the guard and the pre-judge: ``{key}`` is the stage's own key.
_NOT_A_VERDICT = [
    ('{{"{key}": "false"}}', VERDICT_STRING_BOOL),
    ('{{"{key}": "true"}}', VERDICT_STRING_BOOL),
    ('{{"{key}": "False"}}', VERDICT_STRING_BOOL),
    ('{{"{key}": "FALSE"}}', VERDICT_STRING_BOOL),
    ('{{"{key}": "TRUE"}}', VERDICT_STRING_BOOL),
    ('{{"{key}": "  false \\n"}}', VERDICT_STRING_BOOL),
    ('{{"{key}": "no"}}', VERDICT_NOT_BOOLEAN),
    ('{{"{key}": "yes"}}', VERDICT_NOT_BOOLEAN),
    ('{{"{key}": "não"}}', VERDICT_NOT_BOOLEAN),
    ('{{"{key}": "sim"}}', VERDICT_NOT_BOOLEAN),
    ('{{"{key}": ""}}', VERDICT_NOT_BOOLEAN),
    ('{{"{key}": 0}}', VERDICT_NOT_BOOLEAN),
    ('{{"{key}": 1}}', VERDICT_NOT_BOOLEAN),
    ('{{"{key}": 1.0}}', VERDICT_NOT_BOOLEAN),
    ('{{"{key}": null}}', VERDICT_NOT_BOOLEAN),
    ('{{"{key}": [false]}}', VERDICT_NOT_BOOLEAN),
    ('{{"{key}": [true]}}', VERDICT_NOT_BOOLEAN),
    ('{{"{key}": {{"value": true}}}}', VERDICT_NOT_BOOLEAN),
    ('{{"critique": "c", "refusal_message": "r"}}', VERDICT_MISSING),
    ('{{}}', VERDICT_MISSING),
    # An envelope is not opened: picking a field out of somebody else's object is guessing.
    ('{{"message": {{"{key}": true}}}}', VERDICT_MISSING),
    ('{{"message": "{{\\"{key}\\": true}}"}}', VERDICT_MISSING),
    # A different CASE of the key is a different key.
    ('{{"{KEY}": true}}', VERDICT_MISSING),
    ('{{"{key}": false, "{key}": true}}', VERDICT_DUPLICATED),
    ('{{"{key}": true, "{key}": false}}', VERDICT_DUPLICATED),
    ('{{"{key}": true, "{key}": true}}', VERDICT_DUPLICATED),
    ('{{"{key}": false, "critique": "c", "{key}": true}}', VERDICT_DUPLICATED),
    ('It all looks fine to me — {key}.', VERDICT_UNPARSEABLE),
    ('', VERDICT_UNPARSEABLE),
    ('{key}: true', VERDICT_UNPARSEABLE),
    ('{{"{key}": true', VERDICT_UNPARSEABLE),
    ('[{{"{key}": false}}, {{"{key}": true}}]', VERDICT_UNPARSEABLE),
    ('[true]', VERDICT_UNPARSEABLE),
    # Two objects side by side: the extraction is first ``{{`` to last ``}}`` and is not salvaged.
    ('{{"{key}": false}} {{"{key}": true}}', VERDICT_UNPARSEABLE),
    ("{{'{key}': True}}", VERDICT_UNPARSEABLE),
]


def _shape(template: str, key: str) -> str:
    return template.format(key=key, KEY=key.upper())


def _ids(rows):
    return [f"{read}:{template[:28]}" for template, read in rows]


# ── the reader ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, value", [
    ('{"approved": true}', True),
    ('{"approved": false}', False),
    ('{"approved": false, "critique": "did X, asked Y"}', False),
    ('Here is my verdict:\n```json\n{"approved": true, "critique": ""}\n```\nDone.', True),
    ('  {"critique": "", "approved"  :  true }  ', True),
    # The extraction is the first ``{`` to the last ``}`` and always was: ONE object wrapped in
    # anything — prose, a fence, a one-element list — is that object. Declared, not accidental.
    ('[{"approved": false, "critique": "c"}]', False),
])
def test_a_json_boolean_is_a_verdict(raw, value):
    read = read_verdict(raw, "approved")
    assert read.value is value and read.read == VERDICT_BOOLEAN and read.ok
    assert "approved" in read.data


@pytest.mark.parametrize("raw, value", [
    # the words of an error, in the fields BESIDE a verdict that is a proper boolean
    ('{"approved": true, "critique": "false"}', True),
    ('{"approved": false, "critique": "true"}', False),
    ('{"approved": false, "critique": "approved: true"}', False),
    (r'{"approved": true, "critique": "it would be wrong to answer \"approved\": \"false\""}',
     True),
    (r'{"approved": false, "critique": "the draft says {\"approved\": true}"}', False),
    ('{"approved": true, "critique": null, "confidence": "false", "ok": 0}', True),
    ('{"critique": "", "approved": true, "blocked": "false"}', True),
    ('{"approved": false, "notes": ["approved", "approved"], "critique": ""}', False),
    # the verdict's name as a VALUE or as part of another key is not the verdict
    ('{"approved": true, "status": "approved", "approved_at": "false"}', True),
    ('{"APPROVED": "false", "approved": true}', True),
])
def test_adverse_a_boolean_verdict_is_not_disturbed_by_what_sits_beside_it(raw, value):
    """The other direction of strictness: what is NOT the case must not fire. A verdict that
    IS a JSON boolean reads as itself whatever strings, numbers or look-alike keys travel
    beside it — none of them makes it a string, a duplicate or an error."""
    read = read_verdict(raw, "approved")
    assert (read.value, read.read) == (value, VERDICT_BOOLEAN)


@pytest.mark.parametrize("template, expected", _NOT_A_VERDICT, ids=_ids(_NOT_A_VERDICT))
@pytest.mark.parametrize("key", ["approved", "blocked"])
def test_everything_else_is_an_error_with_a_name(template, expected, key):
    read = read_verdict(_shape(template, key), key)
    assert read.value is None, "an unread verdict must never carry a value"
    assert read.read == expected and not read.ok


def test_the_alphabet_is_closed_and_the_reader_stays_inside_it():
    returned = {read_verdict(_shape(t, "approved"), "approved").read for t, _ in _NOT_A_VERDICT}
    returned.add(read_verdict('{"approved": true}', "approved").read)
    assert returned <= set(VALID_VERDICT_READS)
    # ...and it uses every row but the one that is the STAGE's to record: there is no reply to
    # read when the call raised, so the reader can never say so.
    assert returned == set(VALID_VERDICT_READS) - {VERDICT_CALL_FAILED}
    assert len(set(VALID_VERDICT_READS)) == len(VALID_VERDICT_READS)
    assert "" not in VALID_VERDICT_READS, "'' means no verdict was ASKED — it is not a read"


def test_python_truthiness_is_not_a_verdict():
    """``True == 1`` and ``isinstance(True, int)`` in Python; the reader decides by TYPE, and
    the type it accepts is the one JSON ``true``/``false`` decode to."""
    assert read_verdict('{"approved": 1}', "approved").value is None
    assert read_verdict('{"approved": 0}', "approved").value is None
    assert read_verdict('{"approved": true}', "approved").value is True


# ── a key written twice ───────────────────────────────────────────────────────────────────

def test_a_verdict_key_written_twice_is_refused_whatever_the_two_values_are():
    """``json.loads`` keeps the LAST value in silence — measured on the tree before the fix:
    ``{"approved": false, "approved": true}`` APPROVED, in the judge and in the pre-judge."""
    for first, second in (("false", "true"), ("true", "false"), ("true", "true"),
                          ("false", "false")):
        raw = f'{{"approved": {first}, "critique": "c", "approved": {second}}}'
        assert json.loads(raw)["approved"] is (second == "true"), "the premise: last wins"
        read = read_verdict(raw, "approved")
        assert (read.value, read.read) == (None, VERDICT_DUPLICATED)


def test_control_only_the_verdict_key_and_only_the_top_level_are_refused():
    """The controls of the duplicate rule — without them it could be refusing everything."""
    # another key repeated: read as JSON always read it, and the verdict is still ONE
    read = read_verdict('{"critique": "a", "approved": false, "critique": "b"}', "approved")
    assert (read.value, read.read) == (False, VERDICT_BOOLEAN)
    assert read.data["critique"] == "b"
    # the verdict key repeated inside a NESTED object: the top level still says it once
    read = read_verdict('{"approved": true, "meta": {"approved": false, "approved": true}}',
                        "approved")
    assert (read.value, read.read) == (True, VERDICT_BOOLEAN)
    # the same key once at each of two levels is not a repetition at all
    read = read_verdict('{"approved": false, "meta": {"approved": true}}', "approved")
    assert (read.value, read.read) == (False, VERDICT_BOOLEAN)
    # ...and a key that merely CONTAINS the verdict's name is another key
    read = read_verdict('{"approved": true, "approved_by": "x", "not_approved": false}',
                        "approved")
    assert (read.value, read.read) == (True, VERDICT_BOOLEAN)


def test_parse_object_reports_the_top_level_repeats_and_keeps_json_semantics():
    data, repeated = parse_object('{"a": 1, "b": {"a": 2, "a": 3}, "a": 4, "c": 5, "c": 6}')
    assert data == {"a": 4, "b": {"a": 3}, "c": 6}, "the object is what json.loads gives"
    assert repeated == frozenset({"a", "c"}), "top level only — the nested repeat is not listed"
    assert parse_object('{"a": 1}') == ({"a": 1}, frozenset())
    assert parse_object("no json here") == (None, frozenset())
    assert parse_object("[1, 2]") == (None, frozenset())
    assert parse_object('{"a": [1, {"b": 2}]}') == ({"a": [1, {"b": 2}]}, frozenset())


def test_a_reply_nested_past_the_decoder_is_unreadable_and_never_raises():
    deep = '{"approved": ' + "[" * 100_000 + "]" * 100_000 + "}"
    assert parse_object(deep) == (None, frozenset())
    assert read_verdict(deep, "approved").read == VERDICT_UNPARSEABLE


# ── the judge: fail-CLOSED, and it says why ───────────────────────────────────────────────

async def test_control_the_judge_still_approves_and_rejects_on_a_json_boolean():
    stage = SuperegoStage()
    yes = await stage.evaluate(_ctx(), ScriptedBackend(['{"approved": true, "critique": ""}']),
                               limits_prompt="")
    assert (yes.approved, yes.critique, yes.verdict_read) == (True, None, VERDICT_BOOLEAN)
    no = await stage.evaluate(
        _ctx(), ScriptedBackend(['{"approved": false, "critique": "booked the wrong day"}']),
        limits_prompt="")
    assert (no.approved, no.critique, no.verdict_read) == (
        False, "booked the wrong day", VERDICT_BOOLEAN)
    # a JSON false with no critique keeps the default it always had — and it is NOT the
    # unreadable sentence: the judge DID say no.
    bare = await stage.evaluate(_ctx(), ScriptedBackend(['{"approved": false}']),
                                limits_prompt="")
    assert (bare.approved, bare.critique, bare.verdict_read) == (
        False, "execution rejected", VERDICT_BOOLEAN)


@pytest.mark.parametrize("template, expected", _NOT_A_VERDICT, ids=_ids(_NOT_A_VERDICT))
async def test_the_judge_fails_closed_on_every_unread_verdict_and_says_which(template, expected):
    backend = ScriptedBackend([_shape(template, "approved")], ti=41, to=9)
    result = await SuperegoStage().evaluate(_ctx(), backend, limits_prompt="")
    assert result.approved is False
    assert result.critique == UNREADABLE_VERDICT_CRITIQUE
    assert result.verdict_read == expected
    # asked ONCE: the judge does not re-ask by itself. What an unread verdict costs the
    # correction loop is the orchestrator's to decide, and it can only decide what it can see.
    assert len(backend.calls) == 1
    # the call COMPLETED: what it cost and what it was asked are this attempt's, unlike the
    # path where the call raised
    assert (result.metrics.tokens_in, result.metrics.tokens_out) == (41, 9)
    assert result.prompt_blocks and result.judge_branch


async def test_a_string_true_is_not_an_approval_either():
    """The declared cost of strictness: "true" in a string used to approve — by the same
    accident that made "false" approve. Neither is a verdict."""
    result = await SuperegoStage().evaluate(
        _ctx(), ScriptedBackend(['{"approved": "true", "critique": ""}']), limits_prompt="")
    assert (result.approved, result.verdict_read) == (False, VERDICT_STRING_BOOL)


async def test_nothing_of_an_unread_reply_is_used_not_its_critique_either():
    """A critique beside a verdict that could not be read is guidance from a reply we did not
    read. The retry gets the fixed sentence, which names no defect."""
    planted = "SENTINEL-CRITIQUE-7391"
    for raw in (f'{{"approved": "false", "critique": "{planted}"}}',
                f'{{"critique": "{planted}"}}',
                f'{{"approved": false, "approved": true, "critique": "{planted}"}}'):
        result = await SuperegoStage().evaluate(_ctx(), ScriptedBackend([raw]), limits_prompt="")
        assert result.approved is False and planted not in (result.critique or "")


def test_the_unreadable_critique_is_its_own_sentence_and_fits_a_ledger():
    assert UNREADABLE_VERDICT_CRITIQUE.strip() == UNREADABLE_VERDICT_CRITIQUE
    assert UNREADABLE_VERDICT_CRITIQUE not in ("execution rejected",
                                               "could not verify the execution; please retry")
    # an orchestrator keeps a critique per attempt, CUT (cogno-soma: 400 characters): the
    # sentence has to survive that cut whole to be countable by equality.
    assert len(UNREADABLE_VERDICT_CRITIQUE) <= 200
    # the voice reads a critique for its VALUES (figures, addresses) — this one must carry none
    assert not any(ch.isdigit() for ch in UNREADABLE_VERDICT_CRITIQUE)
    assert "@" not in UNREADABLE_VERDICT_CRITIQUE and "http" not in UNREADABLE_VERDICT_CRITIQUE


async def test_the_judge_marks_the_call_that_failed_apart_from_the_reply_it_could_not_read():
    result = await SuperegoStage().evaluate(_ctx(), RaisingBackend(), limits_prompt="")
    assert result.approved is False
    assert result.critique == "could not verify the execution; please retry"
    assert result.verdict_read == VERDICT_CALL_FAILED
    assert (result.metrics.tokens_in, result.metrics.tokens_out) == (0, 0)


async def test_the_judge_asked_nothing_says_nothing():
    """No execution → no call → no read. ``""`` is 'no verdict was asked', not a read."""
    backend = ScriptedBackend([])
    result = await SuperegoStage().evaluate(_ctx(with_ego=False), backend, limits_prompt="")
    assert (result.approved, result.verdict_read, backend.calls) == (True, "", [])


async def test_the_unread_judge_verdict_is_logged_as_a_warning_with_its_read(caplog):
    with caplog.at_level(logging.WARNING, logger="cogno_anima.superego"):
        await SuperegoStage().evaluate(
            _ctx(), ScriptedBackend(['{"approved": "false"}']), limits_prompt="")
    lines = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("event=judge_verdict_unreadable" in m and "read=string_bool" in m for m in lines)
    # ...and not as the judge's own rejection: that line is for a verdict the judge gave
    assert not any("event=judge approved=false" in m for m in lines)


# ── the guard: fail-OPEN, and it leaves a mark ────────────────────────────────────────────

async def _guard(raw=None, *, backend=None, ctx=None, slot=_SCOPE):
    backend = backend or ScriptedBackend([raw], ti=23, to=5)
    ctx = ctx or _ctx(user=_USER, intent_class="INFORMATION_REQUEST")
    result = await SuperegoStage().check_input_scope(ctx, backend, scope_prompt=slot)
    return result, ctx, backend


async def test_control_the_guard_still_blocks_and_allows_on_a_json_boolean():
    blocked, ctx, _ = await _guard('{"blocked": true, "refusal_message": "Out of scope."}')
    assert (blocked.blocked, blocked.refusal_message, blocked.verdict_read) == (
        True, "Out of scope.", VERDICT_BOOLEAN)
    assert ctx.metadata[mk.SCOPE_VERDICT_READ] == VERDICT_BOOLEAN
    allowed, ctx, _ = await _guard('{"blocked": false, "refusal_message": ""}')
    assert (allowed.blocked, allowed.refusal_message, allowed.verdict_read) == (
        False, "", VERDICT_BOOLEAN)
    assert ctx.metadata[mk.SCOPE_VERDICT_READ] == VERDICT_BOOLEAN


@pytest.mark.parametrize("template, expected", _NOT_A_VERDICT, ids=_ids(_NOT_A_VERDICT))
async def test_the_guard_fails_open_on_every_unread_verdict_and_marks_it(template, expected):
    result, ctx, backend = await _guard(_shape(template, "blocked"))
    assert len(backend.calls) == 1
    assert result.blocked is False and result.refusal_message == ""
    assert result.verdict_read == expected
    assert ctx.metadata[mk.SCOPE_VERDICT_READ] == expected
    # the call completed and the prompt was built: cost, inventory and digest are recorded
    assert (result.metrics.tokens_in, result.metrics.tokens_out) == (23, 5)
    assert result.prompt_blocks and result.prompt_sha
    assert ctx.metadata[mk.SCOPE_PROMPT_SHA] == result.prompt_sha


async def test_a_string_true_does_not_block_and_the_mark_says_so():
    """The declared cost on the guard's side: a block spelled as a string no longer blocks. It
    is fail-OPEN by contract, and the turn is counted instead of silently refused."""
    result, ctx, _ = await _guard('{"blocked": "true", "refusal_message": "Out of scope."}')
    assert (result.blocked, result.refusal_message) == (False, "")
    assert ctx.metadata[mk.SCOPE_VERDICT_READ] == VERDICT_STRING_BOOL


async def test_the_refusal_of_an_unread_reply_is_never_shown():
    result, _, _ = await _guard('{"blocked": "true", "refusal_message": "SENTINEL-REFUSAL"}')
    assert "SENTINEL" not in result.refusal_message


async def test_the_guard_marks_the_call_that_failed_apart_from_both():
    result, ctx, _ = await _guard(backend=RaisingBackend())
    assert result.blocked is False
    assert result.verdict_read == VERDICT_CALL_FAILED
    assert ctx.metadata[mk.SCOPE_VERDICT_READ] == VERDICT_CALL_FAILED
    assert result.prompt_blocks and result.prompt_sha, "the prompt WAS built"
    assert (result.metrics.tokens_in, result.metrics.tokens_out) == (0, 0)


async def test_the_three_allows_are_three_records():
    """The sentence of the contract, as one assertion: read-and-allowed, answered-but-unread
    and call-failed all come back ``blocked=False`` and are pairwise distinguishable."""
    read, _, _ = await _guard('{"blocked": false}')
    unread, _, _ = await _guard("not json")
    failed, _, _ = await _guard(backend=RaisingBackend())
    assert {read.blocked, unread.blocked, failed.blocked} == {False}
    assert len({read.verdict_read, unread.verdict_read, failed.verdict_read}) == 3


@pytest.mark.parametrize("build", [
    lambda: (_ctx(user=_USER, intent_class="INFORMATION_REQUEST"), ""),           # no rules
    lambda: (_ctx(user="hi there", intent_class="SOCIAL"), _SCOPE),               # NER bypass
    lambda: (_ctx(user="at 3pm", intent_class="UNKNOWN", goal_status="ONGOING"), _SCOPE),
], ids=["no_scope_rules", "social", "ongoing_unknown"])
async def test_a_bypass_asked_no_verdict_and_removes_a_stale_mark(build):
    """ABSENT means no verdict was asked, and absence is only true if nothing stale sits there:
    a carrier that holds metadata between turns must not make a bypassed turn wear the read of
    an earlier one — the rule ``SCOPE_PROMPT_SHA`` follows, for the same reason."""
    ctx, slot = build()
    ctx.metadata[mk.SCOPE_VERDICT_READ] = VERDICT_STRING_BOOL      # left over from another turn
    backend = ScriptedBackend(['{"blocked": true}'])
    result = await SuperegoStage().check_input_scope(ctx, backend, scope_prompt=slot)
    assert backend.calls == [], "the premise: this path asks the model nothing"
    assert (result.blocked, result.verdict_read) == (False, "")
    assert mk.SCOPE_VERDICT_READ not in ctx.metadata


async def test_the_mark_is_safe_to_persist():
    """A closed alphabet with nothing of the reply in it: the unread reply's own text must not
    reach the record through the mark."""
    result, ctx, _ = await _guard('{"blocked": "SENTINEL-VALUE"}')
    assert ctx.metadata[mk.SCOPE_VERDICT_READ] in VALID_VERDICT_READS
    dumped = json.dumps({"r": result.model_dump(), "m": ctx.metadata}, default=str)
    assert "SENTINEL" not in dumped


async def test_the_unread_guard_verdict_is_logged_as_a_warning_with_its_read(caplog):
    with caplog.at_level(logging.WARNING, logger="cogno_anima.superego"):
        await _guard('{"blocked": "false"}')
    lines = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("event=scope_verdict_unreadable" in m and "read=string_bool" in m for m in lines)


# ── the defaults a host of today inherits ─────────────────────────────────────────────────

def test_the_new_fields_default_to_no_verdict_asked():
    """A caller that builds a result without the field — every stand-in stage, every host
    double, the voice — gets ``""``: nothing was asked. It is never a member of the alphabet,
    so a default can not be mistaken for a read."""
    metrics = StageMetrics(stage="x", elapsed_ms=0.0, tokens_in=0, tokens_out=0, model="m")
    assert ScopeCheckResult(metrics=metrics).verdict_read == ""
    assert SuperegoResult(metrics=metrics).verdict_read == ""
    assert SuperegoStage()._blocked_response(_ctx()).verdict_read == ""
    assert mk.SCOPE_VERDICT_READ == "scope_verdict_read"


# ── the pre-judge reads through the same reader ───────────────────────────────────────────

async def _pre(raw: str) -> str:
    judge = ProposalJudge(ScriptedBackend([raw]), request="book it for tomorrow at ten")
    return (await judge(Proposal("book_slot", {"day": "tomorrow", "time": "10:00"}))).verdict


async def test_the_pre_judge_refuses_a_verdict_key_written_twice():
    """It was already strict about the TYPE and still read the last of two keys: measured on
    the tree before the fix, ``false`` then ``true`` came back ``approved``."""
    assert await _pre('{"approved": false, "critique": "wrong day", "approved": true}') == PRE_ERROR
    assert await _pre('{"approved": true, "approved": false}') == PRE_ERROR


async def test_control_the_pre_judge_reads_one_boolean_as_before():
    assert await _pre('{"approved": true, "critique": ""}') == PRE_APPROVED
    assert await _pre('{"approved": false, "critique": "wrong day"}') == PRE_CRITIQUE


@pytest.mark.parametrize("template, expected", _NOT_A_VERDICT, ids=_ids(_NOT_A_VERDICT))
async def test_the_pre_judge_answers_error_on_every_unread_verdict(template, expected):
    assert await _pre(_shape(template, "approved")) == PRE_ERROR


# ── the tolerant ``_parse_json`` answers what it always answered — for every caller ───────

def _parse_json_before(raw: str) -> dict:
    """``SuperegoStage._parse_json`` as it was before it was expressed over ``parse_object`` —
    copied here literally, as the reference the new one is held to."""
    import re
    match = re.compile(r"\{.*\}", re.DOTALL).search(raw or "")
    if not match:
        return {}
    try:
        data = json.loads(match.group())
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


_REPLIES = [_shape(t, k) for t, _ in _NOT_A_VERDICT for k in ("approved", "blocked")] + [
    '{"approved": true}', '{"blocked": false, "refusal_message": ""}',
    '{"answers": ["Fees"], "maybe": [], "asked": "the fees"}',
    '{"answers": ["Fees"], "answers": ["Calendar"], "maybe": []}',
    '{"answers": "Fees", "maybe": null}', '{"maybe": ["Calendar"]}',
    'prose {"answers": [], "maybe": ["Calendar"], "asked": ""} more prose',
    '{"a": {"b": {"c": [1, 2, {"d": null}]}}}', '{"a": 1} trailing }', '{{"a": 1}}',
    '{"a": NaN}', '{"a": 1e400}', '{"a": "\\u00e9"}', "{}", "{ }", "}{", "{", "}", "null",
    "[]", '"text"', "42", "\n\n", '{"a": 1}\n{"b": 2}', "<think>{}</think>",
]


def test_parse_json_is_unchanged_over_every_reply_shape():
    assert len(_REPLIES) > 80
    for raw in _REPLIES:
        assert SuperegoStage._parse_json(raw) == _parse_json_before(raw), raw
    assert SuperegoStage._parse_json(None) == {}  # type: ignore[arg-type]


def _references(name: str, *, owner: str = "") -> "set[tuple[str, str]]":
    """Every ``(module, enclosing function)`` in the package that REFERENCES ``name`` — read
    off the AST, so a caller added tomorrow is listed here without anyone remembering to.

    With ``owner`` the reference must be the attribute ``<owner>.<name>`` (or ``self``/``cls``
    inside the owner's own module): another class's method of the same name is another
    function — the NOUMENO has a ``_parse_json`` of its own."""
    root = Path(cogno_anima.__file__).parent
    found: "set[tuple[str, str]]" = set()
    for path in sorted(root.rglob("*.py")):
        module = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if func.name == name:
                continue
            for node in ast.walk(func):
                if owner:
                    hit = (isinstance(node, ast.Attribute) and node.attr == name
                           and isinstance(node.value, ast.Name)
                           and (node.value.id == owner
                                or (node.value.id in ("self", "cls")
                                    and module == "stages/superego.py")))
                else:
                    hit = isinstance(node, ast.Name) and node.id == name
                if hit:
                    found.add((module, func.name))
    return found


def test_the_callers_of_each_reader_are_the_ones_this_file_tests():
    """ONE test per caller, and the list of callers is derived: a new one fails here until it
    is named — and until somebody decides which of the two readings it is entitled to."""
    # the TOLERANT read: a caller whose own parser decides what an empty object means
    assert _references("_parse_json", owner="SuperegoStage") == {
        ("stages/scope_options.py", "select_options")}
    # the STRICT read: every stage where a boolean a model wrote decides something
    assert _references("read_verdict") == {
        ("stages/superego.py", "evaluate"),
        ("stages/superego.py", "check_input_scope"),
        ("stages/proposal_judge.py", "__call__"),
    }


@pytest.mark.parametrize("raw", _REPLIES)
async def test_caller_the_selector_decides_what_it_decided(raw):
    """``select_options`` — the one caller left on the tolerant read. Its outcome over each
    reply is the outcome of its own parser over the OLD extraction."""
    options = ["Fees", "Calendar"]
    message = "what are the fees this term?"
    expected = scope_options.parse_selection(_parse_json_before(raw), options, message)
    got = await scope_options.select_options(message, options, ScriptedBackend([raw]))
    assert (got.outcome, got.covered, got.suggested, got.asked, got.discarded,
            got.covered_unsupported) == expected


def test_the_selector_reads_no_boolean_at_all():
    """Why the selector is not on the strict reader: it has no boolean to read. Its verdict is
    two LISTS, and its own parser is already strict about them (a non-list is ``error``)."""
    for value in ('"true"', "true", '"false"', "1"):
        outcome = scope_options.parse_selection(
            json.loads(f'{{"answers": {value}, "maybe": []}}'), ["Fees"], "the fees")[0]
        assert outcome == scope_options.OUTCOME_ERROR


# ── the defect CLASS: no ``bool(...)`` over a model's field where it decides ──────────────

#: ``bool(<x>.get("<field>"...))`` over a model's reply that is KNOWN and left as it is, with
#: the reason. A signal, not a decision — and widening or tightening its coercion is a change
#: to measure, not to slip into this one.
_DECLARED_SIGNALS = {
    ("stages/noumeno.py", "changed"):
        "NoumenoResult.changed is recorded and reconciled against the measured drift; nothing "
        "in this library branches on it",
}


def test_no_stage_reads_a_models_boolean_with_bool_except_the_declared_signals():
    """The two defects were the SAME line: ``bool(data.get("<verdict>", False))``. This walks
    the stages for that shape, so the third one cannot be written without being declared."""
    root = Path(cogno_anima.__file__).parent
    found: "set[tuple[str, str]]" = set()
    for path in sorted((root / "stages").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "bool" and node.args):
                continue
            inner = node.args[0]
            if (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "get" and inner.args
                    and isinstance(inner.args[0], ast.Constant)
                    and isinstance(inner.args[0].value, str)):
                found.add((path.relative_to(root).as_posix(), inner.args[0].value))
    assert found == set(_DECLARED_SIGNALS), (
        "a model's field is being read with bool(...): use cogno_anima.verdict.read_verdict "
        "if it decides something, or declare it above with the reason")
