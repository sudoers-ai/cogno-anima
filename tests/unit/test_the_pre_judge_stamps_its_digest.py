"""The pre-judge stamps the digest of its own template on the rows it writes.

Every stage that authors a prompt names it with ``cogno_anima.prompts.prompt_digest`` on its
metrics row (NOUMENO and NER stamp their templates; the host labels the slots it authors).
``ProposalJudge`` authored one and stamped nothing, so a ``judge_pre`` row said which model
answered and never under which prompt. Measured on a downstream host's stage rows: 27
``judge_pre`` rows that RAN (tokens above zero) and not one with a digest — 22 carrying an empty
label and 5 with no key at all — plus one ``judge_pre:estimated`` row in the same state. Those
rows are filed by the host and never pass through an orchestrator's stamp, so nobody downstream
can add the label: the layer that authors the text has to.

The digest is of the TEMPLATE — the system line and the ``# Decide`` block with the rules this
construction turns on. Nothing of a turn is in it.

The ESTIMATED row (a judgement cut after its prompt was handed over) is the cost of a request
that was sent under that same configuration, so it carries the same label — handed to the
wrapper through the hook the callback already calls before it awaits. A judgement cut before the
callback noted anything was never sent: no estimated row, and no label.
"""

from __future__ import annotations

import time

import pytest

from cogno_anima.prompts import prompt_digest
from cogno_anima.stages import proposal_judge as pj
from cogno_anima.stages.proposal_judge import PersonaCard
from cogno_anima.tools import JUDGE_PRE_ESTIMATED_STAGE, JUDGE_PRE_STAGE, Proposal
from cogno_anima.tools.pre_judge import _Entry
from tests.unit.test_pre_judge_shadow import (_ASKED, _WRITE, _Backend, _Judge, _pj,
                                              _SlowBackend, _wrap)


# ── the twin: a row that ran carries the digest of the template ────────────────────────────

@pytest.mark.parametrize("reply, verdict", [
    ('{"approved": true, "critique": ""}', "approved"),
    ('{"approved": false, "critique": "wrong day"}', "critique"),
    ('{"approved": "false"}', "error"),
    ("not json", "error"),
])
async def test_twin_every_row_the_judge_builds_carries_the_digest_of_its_template(reply, verdict):
    """Approved, critique and error alike: the call ran under this prompt whatever came back.
    Before the fix ``prompt_sha`` was ``""`` on all of them."""
    judge = _pj(_Backend(reply))
    judged = await judge(Proposal(_WRITE, dict(_ASKED)))
    assert judged.verdict == verdict, "the premise"
    expected = prompt_digest(pj._SYSTEM, judge._decide())
    assert expected and judged.metrics.prompt_sha == expected == judge.prompt_sha
    assert (judged.metrics.stage, judged.metrics.tokens_in) == (JUDGE_PRE_STAGE, 210)


async def test_a_backend_that_raised_is_labelled_too_the_prompt_was_built():
    judge = _pj(_Backend(exc=RuntimeError("down")))
    judged = await judge(Proposal(_WRITE, dict(_ASKED)))
    assert judged.verdict == "error" and judged.metrics.tokens_in == 0
    assert judged.metrics.prompt_sha == judge.prompt_sha


async def test_the_digest_reaches_the_row_the_wrapper_files():
    """Through ``PreJudgeDispatcher``: the wrapper forces the stage label and must keep the
    digest — it copies the callback's row, it does not rebuild it."""
    judge = _pj(_Backend())
    _, _, sink, d = _wrap(judge=judge)
    await d.execute(_WRITE, dict(_ASKED))
    await sink.settle(grace_s=1.0)
    [row] = sink.metrics
    assert (row.stage, row.prompt_sha) == (JUDGE_PRE_STAGE, judge.prompt_sha)


# ── what the digest IS: the template, and nothing of a turn ────────────────────────────────

async def test_two_contacts_and_two_calls_under_one_configuration_share_the_label():
    """The rendered prompt differs in every block that belongs to a turn — the contact's
    words, the reply they answer, the tool, the arguments. The label does not."""
    first = _pj(_Backend(), request="sim, pode marcar", previous_reply="Posso marcar quinta?")
    second = _pj(_Backend(), request="cancela a de sexta, por favor", previous_reply="")
    a = await first(Proposal(_WRITE, dict(_ASKED)))
    b = await second(Proposal("cancel_slot", {"date": "2026-03-20"}))
    assert first.render(Proposal(_WRITE, dict(_ASKED))) != second.render(
        Proposal("cancel_slot", {"date": "2026-03-20"})), "the premise: the bytes sent differ"
    assert a.metrics.prompt_sha == b.metrics.prompt_sha != ""


def test_the_VALUES_of_the_context_do_not_move_the_label_only_which_rules_are_on():
    """A clock's value, a persona's name or purpose, the roster's entries are the TENANT's data
    and never enter the digest. Whether a clock, a persona or a roster was GIVEN turns a rule
    on — that is configuration, and it moves the label."""
    def label(**kw):
        return _pj(_Backend(), **kw).prompt_sha

    assert label(now="2026-03-12 10:00") == label(now="2031-01-01 23:59")
    assert label(persona=PersonaCard("sec", "Front desk", "books classes")) == label(
        persona=PersonaCard("fin", "Ledger", "keeps the books"))
    assert label(personas=[PersonaCard("a", "A", "x")]) == label(
        personas=[PersonaCard("b", "B", "y"), PersonaCard("c", "C", "z")])
    bare = label()
    switched = {label(now="2026-03-12 10:00"), label(persona=PersonaCard("s", "S", "p")),
                label(personas=[PersonaCard("a", "A", "x")]), label(facts_not_wording=True)}
    assert bare not in switched and len(switched) == 4, "four rules, four configurations"


def test_the_digest_is_of_the_template_this_module_authors():
    """Sensitivity, by mutation of the two texts the digest names: change either and the label
    changes; change nothing and it is stable across instances."""
    judge = _pj(_Backend())
    assert judge.prompt_sha == _pj(_Backend()).prompt_sha
    assert judge.prompt_sha == prompt_digest(pj._SYSTEM, pj._DECIDE)
    assert judge.prompt_sha != prompt_digest(pj._SYSTEM + " ", pj._DECIDE)
    assert judge.prompt_sha != prompt_digest(pj._SYSTEM, pj._DECIDE + " ")


# ── the ESTIMATED row: sent under that configuration, labelled with it ─────────────────────

async def test_twin_a_judgement_cut_after_sending_carries_the_label_on_its_estimated_row():
    backend = _SlowBackend()
    judge = _pj(backend)
    _, _, sink, d = _wrap(judge=judge, timeout_s=0.05)
    await d.execute(_WRITE, dict(_ASKED))
    await sink.settle(grace_s=1.0)
    assert len(backend.prompts) == 1, "the premise: the prompt WAS handed over"
    [row] = sink.metrics
    assert row.stage == JUDGE_PRE_ESTIMATED_STAGE and row.tokens_in > 0
    assert row.prompt_sha == judge.prompt_sha != ""


async def test_control_a_callback_that_notes_only_tokens_has_an_unlabelled_estimate():
    """The hook's second argument is optional: a callback written before it still gets its
    estimate charged, with no label — never an invented one."""
    class _TokensOnly:
        model = "old"

        async def __call__(self, proposal):
            proposal.note_prompt(321)
            import asyncio
            await asyncio.sleep(5.0)

    _, _, sink, d = _wrap(judge=_TokensOnly(), timeout_s=0.05)
    await d.execute(_WRITE, dict(_ASKED))
    await sink.settle(grace_s=1.0)
    [row] = sink.metrics
    assert (row.stage, row.tokens_in, row.prompt_sha) == (JUDGE_PRE_ESTIMATED_STAGE, 321, "")


async def test_control_a_judgement_cut_before_anything_was_noted_has_no_row_to_label():
    """Nothing was sent: the unmarked zero row, with no digest — absence is the right value."""
    _, _, sink, d = _wrap(judge=_Judge(sleep_s=5.0), timeout_s=0.05)
    await d.execute(_WRITE, dict(_ASKED))
    await sink.settle(grace_s=1.0)
    [row] = sink.metrics
    assert (row.stage, row.tokens_in, row.prompt_sha) == (JUDGE_PRE_STAGE, 0, "")


@pytest.mark.parametrize("junk", [
    "the whole rendered prompt, with the contact's sentence in it", "ABCDEF123456",
    "abc", "g" * 12, "0123456789ab ", 123456789012, None, b"0123456789ab", ["0123456789ab"],
    "a" * 65,
])
def test_the_hook_takes_only_something_shaped_like_a_digest_and_keeps_the_estimate(junk):
    """The hook is called by foreign code and its value rides into a ledger: anything that is
    not lower-case hex of a digest's length is dropped. The estimate beside it is kept — a
    label is never worth losing the cost it labels."""
    entry = _Entry(tool=_WRITE, model="m", started=time.perf_counter())
    entry.note_prompt(500, junk)  # type: ignore[arg-type]
    entry.finish("timeout", None)
    assert (entry.metrics.stage, entry.metrics.tokens_in, entry.metrics.prompt_sha) == (
        JUDGE_PRE_ESTIMATED_STAGE, 500, "")


def test_control_a_digest_is_taken_and_a_closed_record_takes_nothing():
    entry = _Entry(tool=_WRITE, model="m", started=time.perf_counter())
    entry.note_prompt(500, "e8b9344aed58")
    entry.finish("timeout", None)
    assert entry.metrics.prompt_sha == "e8b9344aed58"
    closed = _Entry(tool=_WRITE, model="m", started=time.perf_counter())
    closed.finish("timeout", None)
    closed.note_prompt(500, "e8b9344aed58")
    assert (closed.metrics.tokens_in, closed.metrics.prompt_sha) == (0, "")
    # …and not only on the row already built: the record itself took nothing
    assert (closed.prompt_tokens_estimated, closed.prompt_sha_noted) == (None, "")


def test_a_label_without_a_valid_estimate_is_not_taken():
    """The digest rides WITH the estimate: junk tokens record nothing at all, label included."""
    entry = _Entry(tool=_WRITE, model="m", started=time.perf_counter())
    entry.note_prompt(-3, "e8b9344aed58")
    # asserted on the RECORD, before any row is built: today the label is only ever read off a
    # row that has an estimate, so a label taken alone would be invisible there — and would
    # start labelling rows the day a second reader of the field appears.
    assert (entry.prompt_tokens_estimated, entry.prompt_sha_noted) == (None, "")
    entry.finish("timeout", None)
    assert (entry.metrics.stage, entry.metrics.prompt_sha) == (JUDGE_PRE_STAGE, "")


async def test_a_hook_that_takes_only_the_tokens_still_gets_them():
    """A ``note_prompt`` written before the second argument existed: the judge falls back to
    the one-argument call, so the estimate is not lost to a ``TypeError``."""
    seen: list = []
    judged = await _pj(_Backend())(Proposal(_WRITE, dict(_ASKED), note_prompt=seen.append))
    assert judged.verdict == "approved" and len(seen) == 1 and isinstance(seen[0], int)


# ── no prompt moved ────────────────────────────────────────────────────────────────────────

def test_the_prompt_the_judge_sends_is_untouched():
    """The label is ABOUT the prompt; computing it sends nothing new. The rendering is the
    one the context tests already pin by digest — asserted here by its two halves."""
    judge = _pj(_Backend())
    system, prompt = judge.render(Proposal(_WRITE, dict(_ASKED)))
    assert system == pj._SYSTEM and prompt.endswith(pj._DECIDE)
    assert judge.prompt_sha not in system + prompt
