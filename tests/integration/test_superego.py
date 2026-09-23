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
    await backends.skip_unless_available()
    r = await SuperegoStage().evaluate(ctx, _json_backend(), limits_prompt="")
    assert r.approved is True, f"expected approve, got reject: {r.critique!r}"


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
