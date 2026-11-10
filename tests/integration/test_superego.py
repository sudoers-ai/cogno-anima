"""
Integration tests for the SUPEREGO stage (Stage 5) against a real model.

Scope guard + judge consume JSON → use a json-constrained backend; voice writes
free text → plain backend. Auto-skipped when the configured model is unreachable.
temperature=0.0.
"""

import re

import pytest

from cogno_anima import metakeys as mk
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.types import (
    PipelineContext, NoumenoResult, IntentResult, StageMetrics,
    EgoResult, EgoStep, ToolExecution,
)
from tests.integration import backends

# The judge is a reasoning role, and the model has to be able to do it. Measured on the
# goal↔execution case below (user asked to record an EXPENSE, the EGO recorded INCOME),
# running the real stage n=3 per model: mistral:latest APPROVED the mismatch 3/3 with an
# empty critique; qwen3:8b rejected 3/3 with the correct critique, as did gpt-4o-mini.
# The judge is fail-closed by design — never approve unverified — so on mistral it was the
# exact inverse, a false-pass machine. The prompt was never at fault.
# The model is whatever COGNO_TEST_MODEL names — see tests/integration/backends.py.


#: Numbers the reply STATES, as values — not as the substring the model happened to type.
#:
#: The grounding assertions here ask "is the figure the tool returned in the answer", and a
#: substring test answers a different question: whether it is there IN THE WRITER'S FORMAT.
#: Measured on the nightly canary (run 35207468672, 2026-09-17, qwen3:8b):
#: `test_voice_writes_grounded_response` failed on the reply *"Seu saldo atual é de
#: R$ 1.000,00."* — a perfectly grounded answer to a tool that returned `1000 BRL`, written in
#: the pt-BR the turn asked for. The defect was the instrument's, not the voice's: `"1000" in
#: response` is false for `1.000,00` and would be false for `1,000.00` too, so the test was
#: pinning a LOCALE while claiming to pin GROUNDING.
#:
#: The decimal separator is taken to be the LAST `.` or `,` that is followed by one or two
#: digits at the end of the token; every other separator is a thousands mark. That reads
#: `1.000,00`, `1,000.00`, `1000` and `1000.00` all as 1000.0, which is the point — and it
#: deliberately does NOT strip separators blindly, because that turns `1.000,00` into `100000`
#: and would let a reply stating the wrong figure pass on a prefix match.
_NUM = re.compile(r"\d[\d.,]*\d|\d")


def stated_values(text: str) -> set[float]:
    """Every number in ``text``, locale-normalised to its VALUE."""
    out: set[float] = set()
    for token in _NUM.findall(text or ""):
        decimal = re.search(r"[.,](\d{1,2})$", token)
        if decimal:
            whole = token[:decimal.start()].replace(".", "").replace(",", "")
            token = f"{whole or '0'}.{decimal.group(1)}"
        else:
            token = token.replace(".", "").replace(",", "")
        try:
            out.add(float(token))
        except ValueError:      # a bare separator run, e.g. "1..2" — not a number
            continue
    return out


def _json_backend():
    return backends.json_backend()


def _text_backend():
    return backends.text_backend()


def _m(s):
    return StageMetrics(stage=s, elapsed_ms=1.0, tokens_in=1, tokens_out=1, model="t")


def _ctx(user, intent_class="ACTION_REQUEST", goal="", tool=None, args=None, result="",
         side_effect=False, mutating=None, draft="done", preserved=()):
    """``side_effect``/``mutating`` were omitted here until the READ-ONLY judge branch existed,
    because nothing read them — and a `record_expense` that answers "Recorded expense of 50 BRL"
    was being described to the judge as a call that WROTE NOTHING. That is a defect in the
    double, not an accommodation of the new branch: the two tests below are about a turn that
    acted, and a double that says otherwise sends them down the wrong criteria."""
    noumeno = NoumenoResult(
        original=user, rewritten=user, context_turn="", language="pt",
        canonical_language="en", drift_score=0.0, drift_tag="PASS_THROUGH", changed=False,
        confidence=1.0, change_subject=False, subject_similarity=1.0, context_used=False,
        preserved_terms=list(preserved), rewrite_warnings=[], metrics=_m("noumeno"),
    )
    intent = IntentResult(
        intent_class=intent_class, sentiment="NEUTRAL", confidence=1.0,
        temporal_class="TIMELESS", triad_signal="EGO", goal=goal or user, domains=["FINANCE"],
        metrics=_m("ner"),
    )
    ctx = PipelineContext(user_input=user, noumeno=noumeno, intent=intent)
    if tool:
        ctx.ego_result = EgoResult(steps=[EgoStep(
            index=0, path="native", assistant_text=draft,
            tool_calls=[ToolExecution(tool=tool, arguments=args or {}, result=result, ok=True,
                                      side_effect=side_effect, tool_mutating=mutating)],
        )], metrics=_m("ego"))
    return ctx


SCOPE = "You are a personal finance assistant. You only help with money, expenses, income, budgets."


@pytest.mark.asyncio
async def test_scope_blocks_off_topic():
    await backends.skip_unless_available()
    r = await SuperegoStage().check_input_scope(
        _ctx("Como faço um bolo de chocolate?", intent_class="INFORMATION_REQUEST"),
        _json_backend(), scope_prompt=SCOPE)
    assert r.blocked is True, f"expected BLOCK, got allow; msg={r.refusal_message!r}"
    assert r.refusal_message


@pytest.mark.asyncio
async def test_scope_allows_in_scope():
    await backends.skip_unless_available()
    r = await SuperegoStage().check_input_scope(
        _ctx("Quanto gastei esse mês?", intent_class="INFORMATION_REQUEST"),
        _json_backend(), scope_prompt=SCOPE)
    assert r.blocked is False


@pytest.mark.asyncio
async def test_judge_approves_correct_execution():
    await backends.skip_unless_available()
    ctx = _ctx("registra uma despesa de 50 do almoço", goal="record an expense of 50 for lunch",
               tool="record_expense", args={"amount": 50, "description": "lunch"},
               result="Recorded expense of 50 BRL", side_effect=True, mutating=True)
    r = await SuperegoStage().evaluate(ctx, _json_backend(), limits_prompt="")
    assert r.approved is True, f"expected approve, got reject: {r.critique!r}"


@pytest.mark.asyncio
async def test_judge_rejects_goal_execution_mismatch():
    await backends.skip_unless_available()
    # asked to record an EXPENSE, but the EGO recorded INCOME → goal↔execution miss
    ctx = _ctx("registra uma despesa de 50 do almoço", goal="record an expense of 50 for lunch",
               tool="record_income", args={"amount": 50, "description": "lunch"},
               result="Recorded income of 50 BRL", side_effect=True, mutating=True)
    r = await SuperegoStage().evaluate(ctx, _json_backend(), limits_prompt="")
    assert r.approved is False, "judge should catch income-instead-of-expense"
    assert r.critique


@pytest.mark.asyncio
async def test_the_judge_and_the_guard_answer_in_json_booleans():
    """The model half of the strict verdict reader (`cogno_anima.verdict`, 0.1.3).

    Only a JSON boolean is a verdict: a judge answering ``"approved": "true"`` in a STRING is
    rejected (fail-closed) and a guard answering ``"blocked": "true"`` does not block
    (fail-open). So whether THIS model spells its booleans as booleans is a property the
    deployment depends on, and the four tests above cannot name it — under a model that does
    not, they fail saying "expected approve" and "expected BLOCK", which reads as a judgement
    defect and is a FORMAT one. This test asks the same four questions and asserts only the
    READ, so the failure says which it is.

    No skip per provider: the property is wanted of every model the suite is pointed at.
    UNMEASURED when it was written — the change was made without a model, by design; the
    first run of this test on each spec IS the measurement (four calls).
    """
    await backends.skip_unless_available()
    stage = SuperegoStage()
    reads = {}
    allowed = await stage.check_input_scope(
        _ctx("Quanto gastei esse mês?", intent_class="INFORMATION_REQUEST"),
        _json_backend(), scope_prompt=SCOPE)
    reads["guard, in scope"] = allowed.verdict_read
    blocked = await stage.check_input_scope(
        _ctx("Como faço um bolo de chocolate?", intent_class="INFORMATION_REQUEST"),
        _json_backend(), scope_prompt=SCOPE)
    reads["guard, off topic"] = blocked.verdict_read
    right = _ctx("registra uma despesa de 50 do almoço", goal="record an expense of 50 for lunch",
                 tool="record_expense", args={"amount": 50, "description": "lunch"},
                 result="Recorded expense of 50 BRL", side_effect=True, mutating=True)
    reads["judge, correct execution"] = (
        await stage.evaluate(right, _json_backend(), limits_prompt="")).verdict_read
    wrong = _ctx("registra uma despesa de 50 do almoço", goal="record an expense of 50 for lunch",
                 tool="record_income", args={"amount": 50, "description": "lunch"},
                 result="Recorded income of 50 BRL", side_effect=True, mutating=True)
    reads["judge, wrong execution"] = (
        await stage.evaluate(wrong, _json_backend(), limits_prompt="")).verdict_read
    assert set(reads.values()) == {"boolean"}, (
        f"a verdict was not a JSON boolean on {backends.model_spec()}: {reads}")


@pytest.mark.asyncio
async def test_voice_writes_grounded_response():
    await backends.skip_unless_available()
    ctx = _ctx("qual meu saldo?", intent_class="INFORMATION_REQUEST", goal="get balance",
               tool="get_balance", args={}, result="Current balance: 1000 BRL")
    r = await SuperegoStage().voice(ctx, _text_backend(), voice_prompt="You are a friendly finance assistant.")
    assert r.response, "the voice wrote nothing"
    # The VALUE, not the rendering — `R$ 1.000,00` is this figure, correctly grounded and
    # correctly localised. See `stated_values`.
    assert 1000 in stated_values(r.response), \
        f"expected the grounded figure 1000; got {r.response!r}"
    assert r.metrics.stage == "superego_voice"
    assert r.metrics.tokens_in > 0 and r.metrics.tokens_out > 0


# ── the READ-ONLY judge branch (model half) ──────────────────────────────────────────
# The deterministic half — which branch is chosen and which criteria render — is pinned in
# `tests/unit/test_judge_readonly_branch.py` and needs no model. These two are the half only a
# model can answer, and they are a PAIR on purpose: the first says the branch stops rejecting a
# correct read, the second says it did not become a free pass. Shipping only the first would be
# shipping a rubber stamp.


_EMPTY_READ_TODAY = "[TODAY] 2026-09-15 (terça-feira)"


@pytest.mark.asyncio
async def test_judge_approves_a_truthful_empty_read():
    """The measured defect: the read succeeded, returned nothing, and the draft said so.

    Under the execution criteria this is judged as "the goal was not met" — 730 production
    traces put a turn that only READ at 65.9% rejected against 14.1% for one that WROTE, and
    one critique conceded "the execution CORRECTLY stated ... but it did not fully meet".
    """
    ctx = _ctx("e as aulas do mês que vem?", intent_class="INFORMATION_REQUEST",
               goal="list the classes scheduled for next month",
               tool="get_schedule", args={"month": "2026-10"}, result="No classes found.",
               draft="I found no classes scheduled for next month (October 2026).")
    # The CLOCK is fixed: «next month» is October only if today is in September. Without it the
    # judge reads the real date and rejects a correct draft for querying the wrong month — the
    # fixture, not the judge (measured on the production judge model: every rejection said so).
    ctx.metadata[mk.EGO_CONTEXT] = _EMPTY_READ_TODAY
    await backends.skip_unless_available()
    r = await SuperegoStage().evaluate(ctx, _json_backend(), limits_prompt="")
    assert r.approved is True, f"expected approve, got reject: {r.critique!r}"


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Unstable on the qwen3:8b runner — the MODEL, not the runner: over the six nightlies "
        "2026-10-15 → 10-20 on main 6c90f67 it passed 3 and failed 3, and the runner's `bind: "
        "address already in use` (a second `ollama serve` beside the installer's service, fixed in "
        "the workflow in 0.1.14) does not decide it — with the bind error it passed 2 of 3, without "
        "it 1 of 3. The assertion stays valid and this goes back to pass when the model meets it. "
        "The n=3 cloud-judge measurement stays a queued item."
    ),
)
@pytest.mark.asyncio
async def test_judge_still_rejects_a_read_whose_draft_invents():
    """The twin that keeps the branch honest — and the one that must die if it goes lax.

    Same clean read, same empty result, but the draft fills the emptiness with classes nobody
    returned. GROUNDING is criterion #1 in this branch precisely so this stays a rejection:
    an empty read grounds a NEGATIVE answer and nothing else.
    """
    ctx = _ctx("e as aulas do mês que vem?", intent_class="INFORMATION_REQUEST",
               goal="list the classes scheduled for next month",
               tool="get_schedule", args={"month": "2026-10"}, result="No classes found.",
               draft="You have Algebra on 5 October at 09:00 and Physics on 12 October at 14:00.")
    ctx.metadata[mk.EGO_CONTEXT] = _EMPTY_READ_TODAY      # the same clock as its twin
    await backends.skip_unless_available()
    r = await SuperegoStage().evaluate(ctx, _json_backend(), limits_prompt="")
    assert r.approved is False, "an empty read grounds a negative answer, never a listing"
    assert r.critique


# ── a preserved term is a VALUE, not a spelling (model half) ──────────────────────────
# The deterministic half — which terms render, and what criterion #4 asks — is pinned in
# `tests/unit/test_judge_preserved_is_a_value.py` and needs no model. These two are the half
# only a model can answer, and they are a PAIR for the same reason the read-only pair above is:
# the first says the judge stops rejecting a correct English draft, the second says the
# narrowing did not blind the grounding gate. Shipping only the first would be shipping a
# rubber stamp, and the fabrication twin directly above (`..._whose_draft_invents`) is the
# third control on the same axis.


@pytest.mark.asyncio
async def test_judge_approves_a_draft_that_TRANSLATES_a_preserved_term():
    """THE p0, measured in the rehearsal tenant on 2026-09-18 against the served code.

    The contact asked for a course syllabus by its Portuguese name; the read returned it; the
    draft answered correctly, in the English the whole pipeline works in. The judge rejected it
    with ONE reason — the preserved term "should have been reproduced exactly" — the draft was
    dropped, and the contact was told the syllabus could not be accessed. Judging the
    TRANSLATION is not judging the grounding.
    """
    await backends.skip_unless_available()
    ctx = _ctx("qual e a ementa de Modelagem de Dados?", intent_class="INFORMATION_REQUEST",
               goal="get the syllabus of the Modelagem de Dados course",
               tool="consult_material", args={"course": "Modelagem de Dados"},
               result="Modelagem de Dados - 60 hours. Units: relational model, normalisation.",
               draft="The Data Modeling syllabus is 60 hours and covers the relational model "
                     "and normalisation.",
               preserved=["Modelagem de Dados"])
    r = await SuperegoStage().evaluate(ctx, _json_backend(), limits_prompt="")
    assert r.approved is True, f"expected approve, got reject: {r.critique!r}"


@pytest.mark.asyncio
async def test_judge_still_rejects_a_draft_that_MANGLES_a_preserved_figure():
    """The twin that keeps the narrowing from being a removal — and the one that must die if it
    becomes one. Same shape, but the preserved term is a FIGURE and the draft states a
    different one. A mangled figure, email or URL is what the criterion is FOR.
    """
    await backends.skip_unless_available()
    ctx = _ctx("transfere 1234.56 para a conta do curso",
               goal="transfer 1234.56 to the course account",
               tool="transfer", args={"amount": "1234.56"},
               result="Transferred 1234.56 BRL to the course account",
               side_effect=True, mutating=True,
               draft="Done - I transferred 1234.65 BRL to the course account.",
               preserved=["1234.56"])
    r = await SuperegoStage().evaluate(ctx, _json_backend(), limits_prompt="")
    assert r.approved is False, "a preserved FIGURE stated differently is the defect this keeps"
    assert r.critique


# ── criterion (e): the held message's LANGUAGE (model half) ──────────────────────────


def _skip_on_ollama_for_criterion_e() -> None:
    """Measured on anima #204's own CI (qwen3:8b, temperature 0): the local judge REJECTED the
    correct held message — the critique conceded it was right and failed the turn because the
    tool "returned needs_confirmation, which means the message was not sent" (the hold read as an
    incomplete execution). A judge that rejects the right message cannot make the rejection twin
    mean anything, so the pair runs only against a cloud spec; the production judge's
    measurement is the host's, before landing."""
    spec = backends.model_spec()
    if backends.is_ollama(spec):
        pytest.skip(f"{spec}: qwen3:8b rejects the CORRECT held message for being held "
                    "(needs_confirmation read as not executed), so this pair does not "
                    "discriminate there. Point COGNO_TEST_MODEL at a cloud spec to run it.")


def _held_notify_ctx(message: str):
    """A proposal turn holding one ``notify_user``, the delivered text declared and the turn's
    language declared (``force_language``) as the host declares it. Names and texts invented."""
    ctx = _ctx("avisa os Docentes Bloco 17 que a reunião passou para quinta às 15h",
               goal="notify the Block 17 teachers that the meeting moved to Thursday at 3 pm")
    held = ToolExecution(tool="notify_user",
                         arguments={"target": "Docentes Bloco 17", "message": message},
                         ok=False, error="needs_confirmation", result="", tool_mutating=True)
    ctx.ego_result = EgoResult(
        steps=[EgoStep(index=0, path="native", assistant_text="", tool_calls=[held])],
        pending_confirmation=[held], metrics=_m("ego"))
    ctx.metadata[mk.HELD_DELIVERED_TEXT] = {"notify_user": "message"}
    ctx.force_language = "pt-BR"
    return ctx


@pytest.mark.asyncio
async def test_judge_rejects_a_held_message_written_in_the_rewrite_english():
    """The defect criterion (e) exists for: the executor read only the English rewrite and wrote
    the message in English for a contact who writes in Portuguese. Judged on what it SAID, it was
    approved; the recipient reads those exact words."""
    _skip_on_ollama_for_criterion_e()
    await backends.skip_unless_available()
    ctx = _held_notify_ctx("Hello! The coordination meeting was moved to Thursday at 3 pm.")
    r = await SuperegoStage().evaluate(ctx, _json_backend(), limits_prompt="")
    assert r.approved is False, "an English message for a pt-BR contact must fail"
    assert r.critique


@pytest.mark.asyncio
async def test_judge_approves_the_same_held_message_in_the_contacts_language():
    """The twin: the same proposal, written in the contact's language. Without it the rejection
    above could be a judge that rejects every held message."""
    _skip_on_ollama_for_criterion_e()
    await backends.skip_unless_available()
    ctx = _held_notify_ctx("Olá! A reunião de coordenação passou para quinta-feira às 15h.")
    r = await SuperegoStage().evaluate(ctx, _json_backend(), limits_prompt="")
    assert r.approved is True, f"expected approve, got reject: {r.critique!r}"


# ── the voice does not deny a read that worked (model half) ───────────────────────────


@pytest.mark.asyncio
async def test_voice_does_not_tell_the_contact_the_lookup_failed():
    """The other end of the same p0: holding a critique about WORDING, the voice wrote "I could
    not access the syllabus" over a read that had returned it.

    Asserted as an ABSENCE of the denial, and only of its narrowest forms. The carve-outs are
    deliberately not asserted here: their model half would have to pin a particular Portuguese
    phrasing for "I found nothing", which is a LOCALE pin wearing the clothes of a property —
    the defect `stated_values` at the top of this file exists to record. Those twins are
    deterministic, over the rendered prompt, in
    `tests/unit/test_voice_does_not_deny_a_read_that_worked.py`.
    """
    await backends.skip_unless_available()
    ctx = _ctx("qual e a ementa de Modelagem de Dados?", intent_class="INFORMATION_REQUEST",
               goal="get the syllabus of the Modelagem de Dados course",
               tool="consult_material", args={"course": "Modelagem de Dados"},
               result="Modelagem de Dados - 60 hours. Units: relational model, normalisation.",
               draft="The Data Modeling syllabus is 60 hours.")
    ctx.metadata[mk.VOICE_CORRECTION] = {
        "reason": 'the preserved term "Modelagem de Dados" should have been reproduced exactly'}
    r = await SuperegoStage().voice(ctx, _text_backend(),
                                    voice_prompt="You are a friendly course assistant.")
    assert r.response, "the voice wrote nothing"
    denials = ("nao consegui acess", "não consegui acess", "nao consegui obter",
               "não consegui obter", "could not access", "was unable to access")
    low = r.response.lower()
    assert not any(d in low for d in denials), (
        f"the read returned the syllabus; the reply claims it could not be reached: {r.response!r}")


# ── the review verdict after a read that worked (model half) ──────────────────────────
#
# The PAIR for the condition on the ``unverified_claim`` variant. Its deterministic half is
# `tests/unit/test_voice_review_verdict_after_a_read.py`; what needs a model here is whether the
# new instruction actually produces the reply the contact should have had, and whether the old
# world still behaves.


@pytest.mark.asyncio
async def test_voice_says_what_the_read_returned_when_review_flags_a_claim():
    """The measured turn: a read returned the class times, a downstream net flagged the reply
    anyway and re-voiced it as ``unverified_claim`` — and the contact was told the information
    was unavailable.

    Asserted on VALUES (``stated_values``), never on a format: "19h", "19:00" and "19.00" are
    the same answer, and pinning one of them would be the locale defect recorded at the top of
    this file. Whether the reply also states the DURATION is asserted neither way — the core
    takes no position on a derived value, and the net that judged this one is the host's.
    """
    await backends.skip_unless_available()
    ctx = _ctx("quanto tempo dura a aula?", intent_class="INFORMATION_REQUEST",
               goal="how long the class lasts",
               tool="consult_material", args={"course": "data modelling"},
               result="Class schedule: Wednesday 19:00-22:30; Tuesday 08:00-11:30.",
               draft="The class runs Wednesday 19:00-22:30 and Tuesday 08:00-11:30.")
    ctx.metadata[mk.VOICE_CORRECTION] = {
        "kind": "unverified_claim",
        "reason": "the reply states class times that no scheduling read confirmed"}
    r = await SuperegoStage().voice(ctx, _text_backend(),
                                    voice_prompt="You are a friendly course assistant.")
    assert r.response, "the voice wrote nothing"
    assert {19.0, 22.0} <= stated_values(r.response), (
        f"the read returned the class times and the reply does not state them: {r.response!r}")
    low = r.response.lower()
    unavailable = ("nao tenho essa informacao", "não tenho essa informação",
                   "nao possuo essa informacao", "não possuo essa informação",
                   "do not have that information", "don't have that information")
    assert not any(u in low for u in unavailable), (
        f"the data holds the answer and the reply denies having it: {r.response!r}")


@pytest.mark.asyncio
async def test_voice_states_the_figure_the_data_holds_even_when_review_flagged_it():
    """The 60h twin — the measured live failure behind the code-review fix. The read returned
    the workload, the draft stated it, and a downstream net flagged the reply by tool NAME.
    The data SUPPORTS the flagged claim; the section now says the data is the only authority
    and that what it holds is stated even when flagged, so the figure must reach the contact.

    Asserted on VALUES (``stated_values``) — "60h", "60 horas" and "60 hours" are one answer.
    """
    await backends.skip_unless_available()
    ctx = _ctx("qual a carga horaria total da disciplina?", intent_class="INFORMATION_REQUEST",
               goal="the total workload of the course",
               tool="consult_material", args={"course": "data modelling"},
               result="Data Modeling - total workload: 60h.",
               draft="A carga horaria total da disciplina e de 60 horas.")
    ctx.metadata[mk.VOICE_CORRECTION] = {
        "kind": "unverified_claim",
        "reason": "the reply states a workload that no scheduling read confirmed"}
    r = await SuperegoStage().voice(ctx, _text_backend(),
                                    voice_prompt="You are a friendly course assistant.")
    assert r.response, "the voice wrote nothing"
    assert 60.0 in stated_values(r.response), (
        f"the read returned 60h and the reply does not state it: {r.response!r}")
    low = r.response.lower()
    unavailable = ("nao tenho essa informacao", "não tenho essa informação",
                   "nao possuo essa informacao", "não possuo essa informação",
                   "do not have that information", "don't have that information")
    assert not any(u in low for u in unavailable), (
        f"the data holds the answer and the reply denies having it: {r.response!r}")


@pytest.mark.asyncio
async def test_voice_still_drops_an_unverified_claim_when_nothing_executed():
    """The control, and the world this kind was written for: a persona that executed NOTHING
    and whose draft asserts an integration. The reply must not carry that claim.

    Narrow by design — an affirmative "yes … integrates" and nothing else. A hedged mention
    ("I cannot confirm that it integrates with …") is a CORRECT reply and must not fail here.
    """
    await backends.skip_unless_available()
    ctx = _ctx("voces integram com o ACME ERP?", intent_class="INFORMATION_REQUEST",
               goal="does the product integrate with ACME ERP")
    ctx.ego_result = EgoResult(
        steps=[EgoStep(index=0, path="native",
                       assistant_text="Yes, the product integrates with ACME ERP.")],
        metrics=_m("ego"))
    ctx.metadata[mk.VOICE_CORRECTION] = {
        "kind": "unverified_claim",
        "reason": "no source confirms an ACME ERP integration"}
    r = await SuperegoStage().voice(ctx, _text_backend(),
                                    voice_prompt="You are a product assistant.")
    assert r.response, "the voice wrote nothing"
    affirms = re.search(r"\b(sim|yes)\b[^.?!]{0,80}(integra|integrat)", r.response.lower())
    assert not affirms, (
        f"nothing executed and nothing verified the claim; the reply asserts it: {r.response!r}")


# ── on exhaustion, nothing is reconstructed (model half) ──────────────────────────────
#
# The PAIR for the unconditional clause on both exhaustion verdicts
# (`superego._NOTHING_BEYOND_WHAT_WAS_READ`). Its deterministic half is
# `tests/unit/test_voice_on_exhaustion_does_not_reconstruct.py`, which also holds the account
# of the measured turn; what needs a model here is whether a voice that holds October's REAL
# list in the Context, November's estimate in the data and a critique naming the list nobody
# read still writes only what was read.
#
# CLOUD-ONLY, AND THE REASON IS A MEASUREMENT. On qwen3:8b (temperature 0, local) the clause
# does not move the reply: 0/4 on 2026-09-22 — three wordings of the clause and then the
# clause plus the `nothing_tried` gate, every run the same "Aulas de novembro" list with
# `11/11` on each line above a faithfully reproduced estimate (11.6 s, 30.5 s, 31.7 s, 31.3 s
# of GPU). On gpt-4o-mini — the voice that shipped the measured turn — the canary
# DISCRIMINATES: the Director ran it n=3 on both trees the same day, `main` 0/3 (the reply
# says it could not find the November classes and DROPS the estimate — the muzzle, the other
# face of the same class) and the head 3/3. So the test skips itself on an Ollama spec with
# that reason in the skip text, and the three nightly Ollama shards stay green AND honest: a
# skip that says why, never an empty green. It resolves the spec through `backends` (the
# one place that reads `COGNO_TEST_MODEL`) rather than re-deciding the grammar here.
# The lesson, written where the instrument is: an instruction is not a guarantee. The
# guarantee is the host's deterministic net (`unread_date_claim`).

# A November date in any form a reply could carry — `01/11`, `1/11`, `07/11/2026`,
# `2026-11-01` — and deliberately NOT the estimate's own month header `11/2026`.
_NOVEMBER_DATE = re.compile(r"(?<!\d)\d{1,2}/11(?!\d)|2026-11-\d{2}")

# The only tool that ran, assumed WHOLE by its form: the persisted result is cut at 240
# characters, and the block is per discipline — no amount of missing characters holds a list
# of days.
_NOVEMBER_ESTIMATE = (
    "*Remuneração estimada — November 2026*\n"
    "Valor/hora declarado nas regras: R$ 120,00\n"
    "Horas por aula declaradas nas regras: 4 h\n\n"
    "*Turma DE_09 — 11/2026*\n"
    "Foundations of Machine Learning · 2 aulas · 8 h · R$ 960,00\n\n"
    "*Turma DE_10 — 11/2026*\n"
    "Fundamentals of Data Engineering · 1 aula · 4 h · R$ 480,00\n"
    "Advanced Data Modeling and SQL · 3 aulas · 12 h · R$ 1.440,00\n\n"
    "*Turma DSA_34 — 11/2026*\n"
    "Advanced Data Modeling and SQL · 2 aulas · 8 h · R$ 960,00\n\n"
    "*Base:* 32 h · R$ 3.840,00\n\n"
    "*Bônus por avaliação da turma — RESULTADO NÃO ENCONTRADO*\n"
    "Sem bônus: R$ 3.840,00\n"
    "Avaliação de 80% a 89%: R$ 4.800,00\n"
    "Avaliação de 90% ou mais: R$ 5.120,00\n"
    "O bônus só é devido se pelo menos 30% da turma tiver respondido à avaliação."
)

# The previous turn, as history: October's list was READ, by the schedule tool, and reaches
# this turn inside `# Context (memories/history)` — the material for the analogy.
_OCTOBER_HISTORY = (
    "[HISTORY]\n"
    "user: traga todas as aula de outubro e os valores que irei receber\n"
    "assistant: **Aulas de outubro de 2026**\n"
    "- 01/10 · Turma DSA_34 · Workshop de Abertura\n"
    "- 05/10 · Turma DSA_33 · Fundamentals of Machine Learning\n"
    "- 06/10 · Turma DSA_34 · Fundamentals of Data Science\n"
    "- 07/10 · Turma DE_10 · Fundamentals of Data Engineering\n"
    "- 08/10 · Turma DSA_34 · Fundamentals of Data Science\n"
    "- 14/10 · Turma DSA_33 · Fundamentals of Machine Learning\n"
    "- 19/10 · Turma DE_10 · Fundamentals of Data Engineering\n"
    "- 20/10 · Turma DSA_34 · Fundamentals of Data Science\n"
    "- 21/10 · Turma DSA_33 · Fundamentals of Machine Learning\n"
    "- 22/10 · Turma DSA_34 · Advanced Data Modeling and SQL\n"
    "- 26/10 · Turma DE_10 · Fundamentals of Data Engineering\n"
    "- 27/10 · Turma DSA_34 · Fundamentals of Data Science\n"
    "- 28/10 · Turma DSA_33 · Fundamentals of Machine Learning\n"
    "- 29/10 · Turma DSA_34 · Advanced Data Modeling and SQL\n\n"
    "**Remuneração estimada — outubro de 2026**\n"
    "- Valor/hora: R$ 120,00 · Horas por aula: 4 h\n"
    "- Base: 56 h · R$ 6.720,00"
)


@pytest.mark.asyncio
async def test_voice_on_exhaustion_does_not_reconstruct_a_list_nobody_read():
    """Turn 107, anonymised (2026-09-22): «e de novembro?» one turn after October's classes
    were listed. The only read was November's PAY ESTIMATE, per discipline; the judge rejected
    on the read-only branch; the host declared the exhaustion (``kind="not_executed"``, a tool
    ran); and the delivered reply carried the estimate PLUS seven November dates that no tool
    read — completed by analogy with October's list in the Context. 01/11/2026 is a Sunday.

    The owner's principle, verbatim: «quem escreve não pode escrever coisas que não sabe».

    Asserted as an ABSENCE of any November date — the prompt carries none, so one in the reply
    is an invention by construction (the unit half proves that presence-first) — and as the
    PRESENCE of the estimate, on VALUES (``stated_values``, never a locale): the base total or
    the per-discipline amounts. Whether the reply also SAYS the list was not read is asserted
    neither way: its Portuguese phrasing would be a locale pin wearing the clothes of a
    property. Both assertions are needed, because each wording that failed failed one of them:
    the first cut passed the absence and lost the estimate (the muzzle), the next ones kept the
    estimate and listed `11/11`.

    Runs only against a CLOUD spec — see the block comment above for the measurement that
    decided it (qwen3:8b 0/4; gpt-4o-mini 0/3 on `main` -> 3/3 on this head, n=3).
    """
    spec = backends.model_spec()
    if backends.is_ollama(spec):
        pytest.skip(f"{spec}: qwen3:8b does not follow the clause — 0/4 on 2026-09-22; the "
                    "canary discriminates on gpt-4o-mini, 0/3 -> 3/3 (n=3, Director's run). "
                    "Point COGNO_TEST_MODEL at a cloud spec to run it.")
    await backends.skip_unless_available()
    ctx = _ctx("e de novembro?", intent_class="INFORMATION_REQUEST",
               goal="the contact's classes and pay for November",
               tool="estimate_professor_pay", args={"period": "2026-11"},
               result=_NOVEMBER_ESTIMATE,
               draft=("Estimated pay for November 2026 — DE_09: 2 classes, 8 h, R$ 960,00; "
                      "DE_10: 1 class, 4 h, R$ 480,00 and 3 classes, 12 h, R$ 1.440,00; "
                      "DSA_34: 2 classes, 8 h, R$ 960,00. Base: 32 h, R$ 3.840,00. The class "
                      "survey result was not found, so the bonus cannot be computed."))
    ctx.metadata[mk.EGO_CONTEXT] = _OCTOBER_HISTORY
    ctx.metadata[mk.VOICE_CORRECTION] = {
        "kind": "not_executed",
        "reason": ("The draft gives only the pay estimate. The contact's follow-up continues "
                   "the previous request, which also asked for the classes of the month, and "
                   "no schedule read for November was made.")}
    r = await SuperegoStage().voice(ctx, _text_backend(),
                                    voice_prompt="You are a friendly assistant for a school's "
                                                 "teaching staff.")
    assert r.response, "the voice wrote nothing"
    invented = _NOVEMBER_DATE.findall(r.response)
    assert not invented, (
        f"no tool read a November date this turn and the reply lists {invented}: {r.response!r}")
    values = stated_values(r.response)
    assert 3840.0 in values or {960.0, 480.0, 1440.0} <= values, (
        f"the estimate was read and the reply does not state it: {r.response!r}")


# ── ON A RE-VOICE, WHAT THE DATA HOLDS STAYS (2026-09-30) ─────────────────────────────
#
# The model half of `tests/unit/test_voice_revoice_keeps_what_the_data_holds.py`: the unit
# file pins that the list RENDERS, this asks whether the voice OBEYS it. The shape is the
# rehearsal tenant's (two traces, invented values): three rules read from a document, one of
# them framed wrongly (the institution's payment day given as a deadline of the professor's),
# a rejection for that framing, and a re-voice that fixed it and dropped the invoice line too.
#
# Cloud-only, like the canary above, and for a weaker reason stated as such: this one is NOT
# measured on qwen3:8b, and the voice that shipped the defect is a cloud model. A red canary
# on a model that does not ship would block a deterministic change. Point COGNO_TEST_MODEL at a
# cloud spec to run it; the A/B over the real traces is the consultant's replay.
_PROFESSOR_RULES = (
    "Valores e prazos do professor\n"
    "- Notas e faltas: lançar em até 3 semanas após o fim da disciplina; é também condição "
    "para receber o bônus.\n"
    "- Nota fiscal (NF): enviar até o dia 10 de cada mês.\n"
    "- Pagamento: a instituição paga no dia 28 de cada mês.\n"
    "- Bônus: devido se pelo menos 40% da turma responder à avaliação."
)


@pytest.mark.asyncio
async def test_voice_revoice_keeps_the_values_the_data_holds():
    """Every value of the rejected draft is in the document (3 weeks, day 10, day 28), and the
    critique refused how the payment day (28, which it cites) was FRAMED. The reply must still
    carry the invoice day (10) and the grades deadline (3), which the critique does not cite;
    asserted on VALUES (``stated_values``), never on a Portuguese phrasing. Nothing is asserted
    about 28: the payment line is off the list, and the voice may reframe or drop it."""
    spec = backends.model_spec()
    if backends.is_ollama(spec):
        pytest.skip(f"{spec}: unmeasured on qwen3:8b, and the voice that ships this is a cloud "
                    "model. Point COGNO_TEST_MODEL at a cloud spec to run it.")
    await backends.skip_unless_available()
    ctx = _ctx("Quais são os prazos que o professor precisa cumprir?",
               intent_class="INFORMATION_REQUEST",
               goal="the deadlines a professor must meet",
               tool="consult_documents", args={"query": "prazos do professor"},
               result=_PROFESSOR_RULES,
               draft=("Os prazos que o professor precisa cumprir são:\n"
                      "- Notas e faltas: até 3 semanas após o fim da disciplina.\n"
                      "- Nota fiscal: enviar até o dia 10.\n"
                      "- Pagamento: previsto para o dia 28.\n"
                      "Se precisar de mais alguma coisa, estou à disposição."))
    ctx.metadata[mk.VOICE_CORRECTION] = {
        "kind": "not_executed",
        "reason": ("O dia 28 é a data de pagamento da instituição; separar essa informação dos "
                   "prazos de responsabilidade do professor (lançamento de notas e faltas e "
                   "envio da nota fiscal).")}
    r = await SuperegoStage().voice(ctx, _text_backend(),
                                    voice_prompt="You are a friendly assistant for a school's "
                                                 "teaching staff.")
    assert "voice:kept_values" in r.adjustments, "the list did not render"
    assert r.response, "the voice wrote nothing"
    values = stated_values(r.response)
    assert {3.0, 10.0} <= values, (
        f"the document holds the grades deadline and the invoice day and the reply dropped "
        f"one: {r.response!r}")


# ── a person's NAME is a value too (model half) ───────────────────────────────────────
# The deterministic half — the rule renders only when the request named somebody, and a prompt
# with no name is sent the bytes of before — is `tests/unit/test_judge_person_name_is_a_value.py`.
# These two are a PAIR: the first says the judge stops rejecting the name the USER used for the
# person the execution acted on; the second says it did not become a licence to name somebody
# else. Cloud-only: unmeasured on qwen3:8b.
#
# The NAME is the only axis. The clock is fixed in the context and every date is absolute and
# the same one: the request, the goal, the call, the tool's answer and the draft all say
# 2026-10-22. The first version said «amanhã» with no clock, and its first cloud run
# (gpt-4o-mini) rejected the approve twin on the DATE, not the name: "booked an appointment
# for 2026-10-22 instead of the requested date of tomorrow (2026-10-21)". Both drafts are
# `_DRAFT` with a different name, so the two prompts differ in that name and nothing else.
_TODAY = "[TODAY] 2026-10-21 (Wednesday)"
_BOOKED = "Booked: Dr. Ruivo Tavares (Cardiologia), 2026-10-22 09:00, appointment id A-1."
_DRAFT = "Booked your consultation with {name} on 2026-10-22 at 09:00."


def _booking_ctx(draft: str):
    ctx = _ctx("marca dia 22/10 às 9 com o Dr. Ruivo", intent_class="ACTION_REQUEST",
               goal="book a consultation on 2026-10-22 at 09:00 with Dr. Ruivo",
               tool="book_appointment",
               args={"host_id": "dr_ruivo", "date": "2026-10-22", "time": "09:00"},
               result=_BOOKED, side_effect=True, draft=draft)
    ctx.intent.entities_people = ["Dr. Ruivo"]
    ctx.metadata[mk.EGO_CONTEXT] = _TODAY
    return ctx


def _skip_on_ollama():
    spec = backends.model_spec()
    if backends.is_ollama(spec):
        pytest.skip(f"{spec}: unmeasured on qwen3:8b. Point COGNO_TEST_MODEL at a cloud spec to "
                    "run it.")


@pytest.mark.asyncio
async def test_judge_approves_the_name_the_user_used_for_the_same_person():
    _skip_on_ollama()
    await backends.skip_unless_available()
    ctx = _booking_ctx(_DRAFT.format(name="Dr. Ruivo"))
    r = await SuperegoStage().evaluate(ctx, _json_backend(), limits_prompt="")
    assert r.approved is True, f"expected approve, got reject: {r.critique!r}"


@pytest.mark.asyncio
async def test_judge_still_rejects_a_name_of_another_person():
    _skip_on_ollama()
    await backends.skip_unless_available()
    ctx = _booking_ctx(_DRAFT.format(name="Dr. Bento"))
    r = await SuperegoStage().evaluate(ctx, _json_backend(), limits_prompt="")
    assert r.approved is False, "a name the evidence ties to nobody is not the booked person"
    assert r.critique


# ── a WRITE made with another value than the one the user stated (0.1.15) ───────────────
#
# The deterministic half — the comparison is computed and rendered only on a mismatch, every
# other prompt the bytes of before — is `tests/unit/test_judge_write_values.py`. These four are
# the model half, and they are TWO PAIRS on the same booking:
#   * the wrong hour is rejected / the right hour is approved;
#   * the paragraph does not become a reflex: a value the request's own words DERIVE («1h antes
#     da consulta das 15») or MOVE TO («às 15 não dá, marca o primeiro livre depois») is still
#     approved, though the paragraph renders on both.
# The shape is the measured demo: the hour had moved before («e às 14?» → «às 14h está livre»),
# the contact answered «pode ser às 15» and the executor booked the earlier hour. The clock is
# fixed in the context and every date is absolute. Cloud-only: unmeasured on qwen3:8b.
_WV_TODAY = ("[TODAY] 2026-11-10 (terça-feira)\n[HISTORY]\nuser: e às 14?\n"
             "assistant: Às 14h o Dr. Quostral está livre amanhã, 11/11. Quer marcar?")


def _write_value_ctx(user, calls, draft, goal):
    ctx = _ctx(user, intent_class="ACTION_REQUEST", goal=goal)
    ctx.ego_result = EgoResult(steps=[EgoStep(index=0, path="native", tool_calls=list(calls)),
                                      EgoStep(index=1, path="native", assistant_text=draft)],
                               metrics=_m("ego"))
    ctx.intent.entities_people = ["Dr. Quostral"]
    ctx.metadata[mk.EGO_CONTEXT] = _WV_TODAY
    return ctx


def _booked(hour):
    return ToolExecution(tool="book_appointment",
                         arguments={"host_id": "dr_quostral", "date": "2026-11-11",
                                    "time": f"{hour}:00"},
                         result=f"Booked: Dr. Quostral, 2026-11-11 {hour}:00, appointment id A-7.",
                         ok=True, side_effect=True)


_WV_GOAL = "book the appointment with Dr. Quostral tomorrow at the time the user chose"


@pytest.mark.asyncio
async def test_judge_rejects_a_booking_at_another_hour_than_the_one_stated():
    _skip_on_ollama()
    await backends.skip_unless_available()
    ctx = _write_value_ctx("pode ser às 15", [_booked(14)],
                           "Prontinho! Sua consulta com o Dr. Quostral está marcada para "
                           "amanhã, 11/11, às 14h.", _WV_GOAL)
    r = await SuperegoStage().evaluate(ctx, _json_backend(), limits_prompt="")
    assert r.approved is False, "the user said 15:00 and the booking is at 14:00"
    assert r.critique and "15" in r.critique


@pytest.mark.asyncio
async def test_judge_approves_the_booking_at_the_hour_stated():
    _skip_on_ollama()
    await backends.skip_unless_available()
    ctx = _write_value_ctx("pode ser às 15", [_booked(15)],
                           "Prontinho! Sua consulta com o Dr. Quostral está marcada para "
                           "amanhã, 11/11, às 15h.", _WV_GOAL)
    r = await SuperegoStage().evaluate(ctx, _json_backend(), limits_prompt="")
    assert r.approved is True, f"expected approve, got reject: {r.critique!r}"


@pytest.mark.asyncio
async def test_judge_approves_a_value_the_request_itself_derives():
    _skip_on_ollama()
    await backends.skip_unless_available()
    remind = ToolExecution(tool="remind_me",
                           arguments={"at": "2026-11-11T14:00:00-03:00",
                                      "note": "consulta com o Dr. Quostral"},
                           result="Reminder set for 2026-11-11 14:00.", ok=True,
                           side_effect=True)
    ctx = _write_value_ctx("me lembra 1h antes da consulta das 15 amanhã", [remind],
                           "Combinado! Vou te lembrar amanhã às 14h, uma hora antes da "
                           "consulta das 15h.",
                           "set a reminder one hour before tomorrow's 15:00 appointment")
    r = await SuperegoStage().evaluate(ctx, _json_backend(), limits_prompt="")
    assert r.approved is True, f"expected approve, got reject: {r.critique!r}"


@pytest.mark.asyncio
async def test_judge_approves_a_value_the_request_moves_to():
    _skip_on_ollama()
    await backends.skip_unless_available()
    free = ToolExecution(tool="check_availability",
                         arguments={"host_id": "dr_quostral", "date": "2026-11-11"},
                         result="Free on 2026-11-11: 16:00, 17:00.", ok=True, side_effect=False)
    ctx = _write_value_ctx("às 15 não dá pra mim, marca o primeiro horário livre depois disso",
                           [free, _booked(16)],
                           "Feito! Marquei sua consulta com o Dr. Quostral para amanhã, 11/11, "
                           "às 16h — o primeiro horário livre depois das 15h.",
                           "book the first free slot after 15:00 tomorrow with Dr. Quostral")
    r = await SuperegoStage().evaluate(ctx, _json_backend(), limits_prompt="")
    assert r.approved is True, f"expected approve, got reject: {r.critique!r}"


# ── NOTHING listed without a read: the voice after a failed availability read (0.1.15) ──
#
# The deterministic half is `tests/unit/test_voice_lists_no_time_nobody_read.py`: whatever the
# model writes, no time nobody read reaches the contact. This is the model half — does the ONE
# re-voice, told which times and why, obey, so the floor that takes statements out is not what
# shipped? The guarantee is asserted; the floor is reported, not failed on. Cloud-only.
_VOICE_SECRETARY = ("Você é a secretária da Clínica Vendrim. Responda em português do Brasil, "
                    "com frases curtas e cordiais.")


@pytest.mark.asyncio
async def test_voice_lists_no_time_after_the_availability_read_failed():
    _skip_on_ollama()
    await backends.skip_unless_available()
    from cogno_anima.explicit_values import clock_times
    failed = ToolExecution(tool="check_availability", arguments={"host_id": "1",
                                                                 "date": "2026-11-11"},
                           result="", error="unknown host_id", ok=False, side_effect=False)
    date = ToolExecution(tool="resolve_date", arguments={"text": "amanhã"},
                         result="2026-11-11 (quarta-feira)", ok=True, side_effect=False)
    user = "quais horários o Dr. Quostral tem amanhã?"
    ctx = _ctx(user, intent_class="INFORMATION_REQUEST",
               goal="list Dr. Quostral's free times tomorrow")
    ctx.ego_result = EgoResult(steps=[
        EgoStep(index=0, path="native", tool_calls=[failed, date]),
        EgoStep(index=1, path="native",
                assistant_text="Tomorrow Dr. Quostral has 09:00, 10:30 and 14:00 free.")],
        metrics=_m("ego"))
    ctx.turn_executions = [failed, date]
    ctx.metadata[mk.EGO_CONTEXT] = "[TODAY] 2026-11-10 (terça-feira)"
    ctx.metadata[mk.VOICE_CORRECTION] = {
        "reason": "check_availability failed (unknown host_id) and the draft lists 09:00, 10:30 "
                  "and 14:00, which no successful call returned.",
        "kind": "not_executed"}
    ctx.metadata[mk.JUDGE_VERDICT] = {"approved": False, "attempts": 1}
    r = await SuperegoStage().voice(ctx, _text_backend(), voice_prompt=_VOICE_SECRETARY)
    assert clock_times(r.response) == [], (r.response, r.adjustments)


# ── the stance follows the branch; a hold is HELD (0.1.15) ──────────────────────────────
#
# The deterministic half — the read-only and conversational system messages no longer say
# "Default to NOT approving", the execution branch is the base's bytes, a held call renders HELD,
# criterion (d) points at the GROUNDING criterion — is
# `tests/unit/test_judge_stance_follows_the_branch.py`. These are the model half, in PAIRS: the
# correct reply approved, the invented one still rejected. The read-only pair is the one above
# (`test_judge_approves_a_truthful_empty_read` / `..._whose_draft_invents`). Cloud-only.

@pytest.mark.asyncio
async def test_judge_approves_a_read_that_asks_before_acting():
    """An ACTION request, a turn that only READ the availability, a draft that offers the slot
    and asks before booking: the read-only branch, MID-FLOW, nothing claimed done."""
    _skip_on_ollama()
    await backends.skip_unless_available()
    ctx = _ctx("marca amanhã às 15 com o Dr. Quostral", intent_class="ACTION_REQUEST",
               goal="book an appointment with Dr. Quostral tomorrow at 15:00",
               tool="check_availability", args={"host_id": "dr_quostral", "date": "2026-11-11"},
               result="Free on 2026-11-11: 15:00, 16:00.",
               draft="Amanhã, 11/11, às 15h o Dr. Quostral está livre. Quer que eu marque?")
    ctx.metadata[mk.EGO_CONTEXT] = "[TODAY] 2026-11-10 (terça-feira)"
    r = await SuperegoStage().evaluate(ctx, _json_backend(), limits_prompt="")
    assert r.judge_branch == "readonly"
    assert r.approved is True, f"expected approve, got reject: {r.critique!r}"


_CONV_LIMITS = ("You are the reception of Clínica Vendrim. Opening hours: Monday to Friday, "
                "08:00-18:00. Closed on weekends.")


def _conversational_ctx(draft):
    ctx = _ctx("vocês abrem no sábado?", intent_class="INFORMATION_REQUEST",
               goal="know whether the clinic opens on Saturday")
    ctx.ego_result = EgoResult(steps=[EgoStep(index=0, path="native", assistant_text=draft)],
                               metrics=_m("ego"))
    ctx.metadata[mk.JUDGE_CONVERSATIONAL] = True
    return ctx


@pytest.mark.asyncio
async def test_judge_approves_a_conversational_reply_the_limits_support():
    _skip_on_ollama()
    await backends.skip_unless_available()
    ctx = _conversational_ctx("Não abrimos aos sábados: atendemos de segunda a sexta, das 8h às 18h.")
    r = await SuperegoStage().evaluate(ctx, _json_backend(), limits_prompt=_CONV_LIMITS)
    assert r.approved is True, f"expected approve, got reject: {r.critique!r}"


@pytest.mark.asyncio
async def test_judge_still_rejects_a_conversational_reply_that_invents():
    _skip_on_ollama()
    await backends.skip_unless_available()
    ctx = _conversational_ctx("Sim! No sábado abrimos das 9h às 13h.")
    r = await SuperegoStage().evaluate(ctx, _json_backend(), limits_prompt=_CONV_LIMITS)
    assert r.approved is False, "the limits say closed on weekends"


# The held message, judged before it is sent: values the ADMIN gave earlier in the conversation
# (the `# Context` history) and a total whose parts the message shows are grounded; a value in no
# source is not; and an affirmative with the message HELD is not "the message was not sent".
_HELD_HISTORY = ("[TODAY] 2026-11-10 (terça-feira)\n[HISTORY]\n"
                 "user: o workshop da Belmira Quostral Vendrim é dia 20/11, às 19h, na sala 4\n"
                 "assistant: Anotado.\n"
                 "user: são 2 turmas e cada uma paga R$ 50,00\n"
                 "assistant: Certo.")


def _held_ctx(message, user="manda para a Belmira o resumo do workshop"):
    ctx = _ctx(user, intent_class="ACTION_REQUEST",
               goal="send Belmira the workshop summary the user dictated")
    held = ToolExecution(tool="notify_user",
                         arguments={"to": "Belmira Quostral Vendrim", "message": message},
                         ok=False, error="needs_confirmation", tool_mutating=True,
                         result=("[PENDING CONFIRMATION] 'notify_user' is destructive and was "
                                 "NOT executed; it needs explicit user confirmation first."))
    ctx.ego_result = EgoResult(steps=[EgoStep(index=0, path="native", assistant_text="",
                                              tool_calls=[held])],
                               pending_confirmation=[held], metrics=_m("ego"))
    ctx.metadata[mk.HELD_DELIVERED_TEXT] = {"notify_user": "message"}
    ctx.metadata[mk.EGO_CONTEXT] = _HELD_HISTORY
    ctx.force_language = "pt-BR"
    return ctx


_HELD_RIGHT = ("Olá, Belmira! Seu workshop é no dia 20/11, às 19h, na sala 4. São 2 turmas de "
               "R$ 50,00 cada: 2 x R$ 50,00 = R$ 100,00 no total.")


@pytest.mark.asyncio
async def test_judge_approves_a_held_message_grounded_in_the_conversation():
    _skip_on_ollama()
    await backends.skip_unless_available()
    r = await SuperegoStage().evaluate(_held_ctx(_HELD_RIGHT), _json_backend(), limits_prompt="")
    assert r.approved is True, f"expected approve, got reject: {r.critique!r}"


@pytest.mark.asyncio
async def test_judge_still_rejects_a_held_message_with_a_value_in_no_source():
    _skip_on_ollama()
    await backends.skip_unless_available()
    wrong = _HELD_RIGHT.replace("sala 4", "sala 9")
    r = await SuperegoStage().evaluate(_held_ctx(wrong), _json_backend(), limits_prompt="")
    assert r.approved is False, "the room the user dictated is 4"


@pytest.mark.asyncio
async def test_judge_approves_a_held_message_on_a_bare_affirmative():
    _skip_on_ollama()
    await backends.skip_unless_available()
    r = await SuperegoStage().evaluate(_held_ctx(_HELD_RIGHT, user="pode enviar"),
                                       _json_backend(), limits_prompt="")
    assert r.approved is True, f"expected approve, got reject: {r.critique!r}"


# The read-only shape that dominates the exhausted turns downstream: the reads came back with
# data, the reply is right, and what the judge asked for was a confirmation or a detail the turn
# had no business doing — a read-only turn asking before it writes is MID-FLOW. And the held
# message in the shape it was rejected in: several values the user dictated over earlier turns
# (in the Context history, none from a tool this turn) and a total of parts the message shows —
# with its twin, the same message whose total does not come out.
_APPOINTMENT = ("Appointment A-7: Dr. Quostral, 2026-11-11 09:00, status PENDING.")


def _appointment_ctx(user, draft, goal, intent_class):
    ctx = _ctx(user, intent_class=intent_class, goal=goal, tool="list_appointments",
               args={"date": "2026-11-11"}, result=_APPOINTMENT, draft=draft)
    ctx.metadata[mk.EGO_CONTEXT] = "[TODAY] 2026-11-10 (terça-feira)"
    return ctx


@pytest.mark.asyncio
@pytest.mark.parametrize("user,draft,goal,intent_class", [
    ("quero confirmar minha consulta de amanhã",
     "Encontrei sua consulta com o Dr. Quostral amanhã, 11/11, às 9h — ainda pendente. Quer que "
     "eu a confirme?", "confirm tomorrow's appointment", "ACTION_REQUEST"),
    ("cancela minha consulta de amanhã",
     "Encontrei sua consulta com o Dr. Quostral amanhã, 11/11, às 9h. Confirma que quer cancelar?",
     "cancel tomorrow's appointment", "ACTION_REQUEST"),
    ("minha consulta de amanhã já está confirmada?",
     "Ainda não: sua consulta com o Dr. Quostral amanhã, 11/11, às 9h está pendente de "
     "confirmação.", "know whether tomorrow's appointment is confirmed", "INFORMATION_REQUEST"),
], ids=["confirm_asks_first", "cancel_asks_first", "status_answered"])
async def test_judge_approves_a_read_with_data_whose_reply_is_right(user, draft, goal,
                                                                    intent_class):
    _skip_on_ollama()
    await backends.skip_unless_available()
    ctx = _appointment_ctx(user, draft, goal, intent_class)
    r = await SuperegoStage().evaluate(ctx, _json_backend(), limits_prompt="")
    assert r.judge_branch == "readonly"
    assert r.approved is True, f"expected approve, got reject: {r.critique!r}"


_DICTATED = ("[TODAY] 2026-11-10 (terça-feira)\n[HISTORY]\n"
             "user: vamos organizar o workshop da Turma A\nassistant: Claro.\n"
             "user: vai ser dia 20/11, às 19h\nassistant: Anotado.\n"
             "user: na sala 4\nassistant: Certo.\n"
             "user: a inscrição custa R$ 120,00\nassistant: Anotado.\n"
             "user: o material sai R$ 80,00 e o certificado R$ 50,00\nassistant: Certo.")


def _summary_ctx(total):
    message = ("Olá, Belmira! Resumo do workshop: dia 20/11, às 19h, na sala 4. Inscrição: "
               "R$ 120,00; material: R$ 80,00; certificado: R$ 50,00. Total por aluno: "
               f"{total}.")
    ctx = _held_ctx(message, user="manda para a Belmira o resumo do workshop com os valores")
    read = ToolExecution(tool="list_classes", arguments={"term": "2026-2"},
                         result="Turma A — Workshop de Abertura; Turma B — Oficina de Escrita.",
                         ok=True, side_effect=False)
    held = ctx.ego_result.pending_confirmation[0]
    ctx.ego_result = EgoResult(steps=[EgoStep(index=0, path="native", assistant_text="",
                                              tool_calls=[read, held])],
                               pending_confirmation=[held], metrics=_m("ego"))
    ctx.metadata[mk.EGO_CONTEXT] = _DICTATED
    return ctx


@pytest.mark.asyncio
async def test_judge_approves_a_held_summary_of_dictated_values_and_their_sum():
    _skip_on_ollama()
    await backends.skip_unless_available()
    r = await SuperegoStage().evaluate(_summary_ctx("R$ 250,00"), _json_backend(),
                                       limits_prompt="")
    assert r.approved is True, f"expected approve, got reject: {r.critique!r}"


@pytest.mark.asyncio
async def test_judge_still_rejects_a_held_summary_whose_sum_does_not_come_out():
    _skip_on_ollama()
    await backends.skip_unless_available()
    r = await SuperegoStage().evaluate(_summary_ctx("R$ 260,00"), _json_backend(),
                                       limits_prompt="")
    assert r.approved is False, "120 + 80 + 50 is 250"
