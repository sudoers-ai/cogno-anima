"""On exhaustion the reply is the draft and the data — NOTHING beyond them.

The owner's principle, verbatim: «quem escreve não pode escrever coisas que não sabe» — who
writes cannot write what they do not know.

## The measured turn (`turn_traces` 1970, turn 107, 2026-09-22; host ``2f0d2cd6``, this
## library ``d34822a5``) — anonymised: no person, tenant or identity below

The contact asked «e de novembro?» one turn after asking for October's classes and October's
pay. The ONLY tool that ran was the pay estimate for November — a block PER DISCIPLINE, with no
list of days in it — and it returned ``ok=True``. The judge rejected 1/1 on the read-only
branch, the host declared ``judge_rejected_all`` → ``last_draft_voiced`` (``kind="not_executed"``,
since a tool ran), and `# Execution verdict (HARD RULE)` rendered.

The voice delivered the estimate faithfully AND, above it, «Aulas de novembro de 2026» with
SEVEN DATES, 01/11 to 07/11, each with a class code and a discipline. No tool read them this
turn. They contradict the estimate they sit next to (7 dates against 8 classes; 2 against 3 for
one discipline), and 01/11/2026 is a Sunday. The previous turn's October list — read for real,
by the schedule tool — was in `# Context (memories/history)`: the voice completed November by
ANALOGY with October.

**The persisted tool result is CUT** (`…[cortado, faltam 794 chars]` — the trace bounds a
stored result at 240 characters), so ``ESTIMATE`` below is the whole block ASSUMED BY ITS
FORM: it is per discipline, and no amount of missing characters of it holds a list of days.
The reply's own per-discipline lines say what the cut part contained.

## What is pinned here, and how

* PRESENCE FIRST. A test that asserts an absence has to prove it can produce the presence
  (`test_voice_does_not_adopt_the_critiques_remedy` learned that the hard way): the rendered
  prompt CARRIES October's fourteen dates in the Context and the November estimate in the
  executor data, and carries NO November date anywhere — so a November date in a reply is
  invented, by construction.
* THE CLAUSE. Unconditional, on both variants (`# Execution verdict`, both branches of
  `# Review verdict`), the section's LAST word; written once, spliced by reference.
* THE CONTROLS. Every rendering WITHOUT an exhaustion is byte-identical to `main` — whole-prompt
  digests per kind, measured on ``d34822a5`` with this exact fixture — and the persisted
  inventory does not move (no new header).
* MUTATION: delete the splice — this file goes red, the controls stay green. On `main` this
  file fails by ASSERTION, never by ImportError: nothing new is imported from the library.

Deterministic: every assertion is on the RENDERED voice prompt. The model half is
`tests/integration/test_superego.py::test_voice_on_exhaustion_does_not_reconstruct_a_list_nobody_read`.
"""

from __future__ import annotations

import hashlib
import inspect
import re

import pytest

from cogno_anima import metakeys as mk
from cogno_anima.stages import superego as _se
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.types import EgoResult, EgoStep, ToolExecution
from tests.unit.test_superego import ScriptedBackend, _ctx, _m

# ── the fixture: turn 107, anonymised ─────────────────────────────────────────────────

USER = "e de novembro?"

# The only tool that ran. Whole block assumed by its FORM (see the module docstring).
ESTIMATE = (
    "*Remuneração estimada — November 2026*\n"
    "Valor/hora declarado nas regras: R$ 120,00\n"
    "Horas por aula declaradas nas regras: 4 h\n"
    "\n"
    "*Turma DE_09 — 11/2026*\n"
    "Foundations of Machine Learning · 2 aulas · 8 h · R$ 960,00\n"
    "\n"
    "*Turma DE_10 — 11/2026*\n"
    "Fundamentals of Data Engineering · 1 aula · 4 h · R$ 480,00\n"
    "Advanced Data Modeling and SQL · 3 aulas · 12 h · R$ 1.440,00\n"
    "\n"
    "*Turma DSA_34 — 11/2026*\n"
    "Advanced Data Modeling and SQL · 2 aulas · 8 h · R$ 960,00\n"
    "\n"
    "*Base:* 32 h · R$ 3.840,00\n"
    "\n"
    "*Bônus por avaliação da turma — RESULTADO NÃO ENCONTRADO*\n"
    "Sem bônus: R$ 3.840,00\n"
    "Avaliação de 80% a 89%: R$ 4.800,00\n"
    "Avaliação de 90% ou mais: R$ 5.120,00\n"
    "O bônus só é devido se pelo menos 30% da turma tiver respondido à avaliação."
)

# The previous turn (`turn_traces` 1969, turn 106): the October list was READ, by the
# schedule tool, and every date of that reply is in that read. It reaches this turn's voice
# as history, inside `# Context (memories/history)` — the material for the analogy.
OCTOBER_LIST = (
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
    "- 29/10 · Turma DSA_34 · Advanced Data Modeling and SQL"
)
CONTEXT = (
    "[HISTORY]\n"
    "user: traga todas as aula de outubro e os valores que irei receber\n"
    "assistant: **Aulas de outubro de 2026**\n"
    f"{OCTOBER_LIST}\n"
    "\n"
    "**Remuneração estimada — outubro de 2026**\n"
    "- Valor/hora: R$ 120,00 · Horas por aula: 4 h\n"
    "- Turma DE_10 — Fundamentals of Data Engineering · 3 aulas · 12 h · R$ 1.440,00\n"
    "- Turma DSA_33 — Fundamentals of Machine Learning · 4 aulas · 16 h · R$ 1.920,00\n"
    "- Turma DSA_34 — Workshop de Abertura · 1 aula · 4 h · R$ 480,00; Fundamentals of Data "
    "Science · 4 aulas · 16 h · R$ 1.920,00; Advanced Data Modeling and SQL · 2 aulas · 8 h · "
    "R$ 960,00\n"
    "- Base: 56 h · R$ 6.720,00"
)

# The executor's draft — English by design (the NOUMENO rewrites, the EGO drafts there).
DRAFT = (
    "Estimated pay for November 2026 — DE_09: Foundations of Machine Learning, 2 classes, "
    "8 h, R$ 960,00. DE_10: Fundamentals of Data Engineering, 1 class, 4 h, R$ 480,00; "
    "Advanced Data Modeling and SQL, 3 classes, 12 h, R$ 1.440,00. DSA_34: Advanced Data "
    "Modeling and SQL, 2 classes, 8 h, R$ 960,00. Base: 32 h, R$ 3.840,00. The class survey "
    "result was not found, so the bonus cannot be computed."
)

# A read-only judge's rejection of the shape that pushes hardest towards the analogy: the
# critique names the list nobody read. The clause says the critique is not a source, and
# that what was not read is said to be not read.
CRITIQUE = ("The draft gives only the pay estimate. The contact's follow-up continues the "
            "previous request, which also asked for the classes of the month, and no schedule "
            "read for November was made.")

# ── what the clause says, asserted on the PROMPT ──────────────────────────────────────

OPENING = "NOTHING IS RECONSTRUCTED."
WHEN_ONE_IS_SHOWN = "when one is shown, the executor's answer"
NOTHING_BEYOND = ("NOTHING beyond them: no list, date, item, name or value that is not "
                  "written in them")
NOT_THIS_TURNS_READ = "an EARLIER turn read is not what this turn read"
BY_ANALOGY = "a list completed by analogy, by pattern, or from memory is INVENTED"
STATE_WHAT_WAS_READ = ("state it, reproducing every figure, date, name and identifier in it "
                       "exactly as written there")
NOT_YOURS_TO_WRITE = ("What the request asked for and NO tool read this turn is not yours to "
                      "write")
SAY_NOT_READ = "it is reported as not read"
BESIDE_NEVER_INSTEAD = "in one sentence BESIDE what was read — never instead of it"
NO_DERIVED_DATE = "a DATE is never derived — not from a month, a period, a count or a pattern"
NOT_A_TEMPLATE = "an earlier reply in the Context is not a template"
LAST_WORD = "however plausible it looks."

# A November date in any of the forms a reply could carry: `01/11`, `1/11`, `01/11/2026`,
# `2026-11-01`. Deliberately NOT `11/2026` (the estimate's own month header) — the regex is
# self-tested below so the presence proof cannot pass on a pattern that matches nothing.
NOVEMBER_DATE = re.compile(r"(?<!\d)\d{1,2}/11(?!\d)|2026-11-\d{2}")
OCTOBER_DATE = re.compile(r"(?<!\d)\d{2}/10(?!\d)")


def _read(tool: str = "estimate_professor_pay", result: str = ESTIMATE) -> ToolExecution:
    return ToolExecution(tool=tool, arguments={"period": "2026-11"}, result=result, ok=True,
                         side_effect=False, tool_mutating=False)


def _failed() -> ToolExecution:
    return ToolExecution(tool="get_professor_schedule", arguments={}, result="", ok=False,
                         error="upstream timeout", side_effect=False, tool_mutating=False)


def _write() -> ToolExecution:
    return ToolExecution(tool="reschedule_class", arguments={}, result="Moved", ok=True,
                         side_effect=True, tool_mutating=True)


def _ego(*calls: ToolExecution, draft: str = DRAFT) -> EgoResult:
    return EgoResult(
        steps=[EgoStep(index=0, path="native", assistant_text="", tool_calls=list(calls)),
               EgoStep(index=1, path="native", assistant_text=draft)],
        metrics=_m("ego"))


def _turn(*calls: ToolExecution):
    """Turn 107: the estimate read, the draft built from it, October's list in the Context."""
    calls = calls or (_read(),)
    ctx = _ctx(user=USER, intent_class="INFORMATION_REQUEST", with_ego=False)
    ctx.ego_result = _ego(*calls)
    ctx.metadata[mk.EGO_CONTEXT] = CONTEXT
    return ctx


def _bare(*calls: ToolExecution):
    """The same turn with NO execution at all (the review verdict's legacy branch)."""
    ctx = _turn(*calls)
    ctx.ego_result = _ego(draft=DRAFT)
    return ctx


def _render(ctx, *, kind: "str | None" = "not_executed", reason: str = CRITIQUE) -> str:
    """The prompt the voice receives, over the payload `voice()` itself would build."""
    if kind is not None:
        ctx.metadata[mk.VOICE_CORRECTION] = {"reason": reason, "kind": kind}
    payload = SuperegoStage._tool_payload(ctx)
    return SuperegoStage()._build_voice_prompt(ctx, payload, ["general:review"])


def _section(prompt: str, slug: str) -> str:
    return SuperegoStage.voice_prompt_block(prompt, slug)


# ── PRESENCE FIRST ────────────────────────────────────────────────────────────────────

def test_the_regex_tells_a_november_date_from_the_estimates_month_header():
    """The presence proof below rests on this pattern; a pattern that matched nothing would
    let it pass over a prompt full of dates."""
    assert NOVEMBER_DATE.search("- 01/11 · Turma DE_09")
    assert NOVEMBER_DATE.search("1/11") and NOVEMBER_DATE.search("07/11/2026")
    assert NOVEMBER_DATE.search("2026-11-01")
    assert NOVEMBER_DATE.search("*Turma DE_09 — 11/2026*") is None
    assert NOVEMBER_DATE.search("R$ 1.440,00 · 12 h") is None


def test_presence_first_the_prompt_holds_the_material_for_the_analogy_and_no_november_date():
    """The temptation is REAL in the rendered prompt — fourteen October dates, one per class,
    in the Context — and the November estimate is there too. What is NOT there, anywhere, is
    a November date: whatever the reply lists for November, it did not read it here."""
    prompt = _render(_turn())
    context = _section(prompt, "context")
    assert len(OCTOBER_DATE.findall(context)) == 14, "the October list did not reach the prompt"
    data = _section(prompt, "executor_data")
    assert "R$ 3.840,00" in data and "2 aulas · 8 h · R$ 960,00" in data
    assert NOVEMBER_DATE.search(prompt) is None, (
        "the prompt carries a November date — then a reply listing one is not an invention "
        "and this file is asserting the wrong property")


# ── THE MEASURED TURN ─────────────────────────────────────────────────────────────────

def test_the_measured_turn_the_execution_verdict_ends_by_forbidding_what_was_not_read():
    """The prompt turn 107 received, plus the clause. It forbids the list by name — no list,
    date, item, name or value not written in the data or the answer — says why the Context
    is not a source (an earlier turn's read), and says what to do instead: say it was not
    read. And it is the section's LAST word, after the critique.

    MUTATION: remove the splice from `# Execution verdict` — red here.
    """
    section = _section(_render(_turn()), "execution_verdict")
    assert CRITIQUE in section, "the scaffold did not even render the critique"
    for sentence in (OPENING, NOTHING_BEYOND, NOT_THIS_TURNS_READ, BY_ANALOGY, SAY_NOT_READ):
        assert sentence in section, f"missing from the execution verdict: {sentence!r}"
    assert section.index(CRITIQUE) < section.index(OPENING)
    assert section.rstrip().endswith(LAST_WORD), "the clause is not the section's last word"


def test_the_critique_that_names_the_unread_list_is_still_not_a_source():
    """The critique above asks, in as many words, for the classes of the month. The clause
    that says a critique is not evidence stays, and the new one follows it."""
    section = _section(_render(_turn()), "execution_verdict")
    assert _se._CRITIQUE_IS_NOT_EVIDENCE.strip() in section
    assert OPENING in section, "the clause did not render"     # assert, never str.index
    assert section.index(_se._CRITIQUE_IS_NOT_EVIDENCE.strip()) < section.index(OPENING)


def test_the_mirror_the_estimate_that_WAS_read_is_ordered_into_the_reply_beside_the_limit():
    """Measured on the model-backed canary, first cut (qwen3:8b, temperature 0, this turn): the
    clause said "the reply IS the executor data … say that it was not read" — positive half
    descriptive, negative half an order — and the reply obeyed the order and dropped the data:
    no November date, and no estimate either ("Ainda não há um calendário de aulas para
    novembro de 2026 disponível"). A muzzle is the mirror of the defect.

    So the reproduction is an ORDER, it comes FIRST, and the limit stands BESIDE what was read,
    never instead of it — asserted on one rendered section, by order.

    MUTATION: make the positive half descriptive again, or drop "never instead of it" — red.
    """
    section = _section(_render(_turn()), "execution_verdict")
    assert STATE_WHAT_WAS_READ in section, "the positive half is not an order"
    assert BESIDE_NEVER_INSTEAD in section, "the limit may again replace what was read"
    assert section.index(STATE_WHAT_WAS_READ) < section.index(NOTHING_BEYOND) \
        < section.index(NOT_YOURS_TO_WRITE) < section.index(SAY_NOT_READ) \
        < section.index(BESIDE_NEVER_INSTEAD)


def test_the_second_mirror_a_derived_date_and_the_previous_reply_as_a_template():
    """Measured on the same canary, second cut: with the order in place the estimate came
    back whole — and above it, again, "Aulas de novembro de 2026", four lines with `11/11`
    on every one: a date DERIVED from the block's own `11/2026` header, in a section copied
    from the SHAPE of the previous reply in the Context. "No list not written in them" did
    not reach it: every ITEM was in the data and only the date column was made up.

    So the two mechanisms are named, and named LAST: a date is never derived, and an earlier
    reply is not a template — a section it had that this turn did not read does not exist.

    MUTATION: drop either sentence — red.
    """
    section = _section(_render(_turn()), "execution_verdict")
    assert NO_DERIVED_DATE in section, "a date may again be derived from a month or a count"
    assert "character for character" in section
    assert NOT_A_TEMPLATE in section, "the previous reply may again be copied as a shape"
    assert "a section it had that this turn did not read" in section
    assert section.index(SAY_NOT_READ) < section.index(NO_DERIVED_DATE) \
        < section.index(NOT_A_TEMPLATE) < section.index(NOT_THIS_TURNS_READ)


# ── BOTH VARIANTS, EVERY BRANCH ───────────────────────────────────────────────────────

def test_the_review_verdict_carries_the_clause_in_both_of_its_branches():
    """`# Review verdict (HARD RULE)` renders two ways — after a read that worked, and for a
    persona that executed nothing — and the clause is the last word of both."""
    after_a_read = _section(_render(_turn(), kind="unverified_claim"), "review_verdict")
    assert _se._EVERY_TOOL_SUCCEEDED in after_a_read
    nothing_ran = _section(_render(_bare(), kind="unverified_claim"), "review_verdict")
    assert "nothing was executed this turn" in nothing_ran
    for section in (after_a_read, nothing_ran):
        assert OPENING in section and NOTHING_BEYOND in section and SAY_NOT_READ in section
        assert section.rstrip().endswith(LAST_WORD)


_SHAPES = {
    "no_exec": lambda: _bare(),
    "read_ok": lambda: _turn(_read()),
    "read_plus_failed": lambda: _turn(_read(), _failed()),
    "write": lambda: _turn(_write()),
}
_KIND_TO_SLUG = {"not_executed": "execution_verdict", "unverified_claim": "review_verdict"}


@pytest.mark.parametrize("kind", sorted(_KIND_TO_SLUG))
@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_unconditional_every_rendering_of_both_headers_carries_it(shape, kind):
    """No gate. Whatever ran — nothing, a read, a read beside a failure, a write — and
    whichever verdict the host stamped, the clause is there and it is the last word. A gate
    would have to name the shape of the reconstruction; the point is that the voice is never a
    source, of anything, in any shape."""
    section = _section(_render(_SHAPES[shape](), kind=kind), _KIND_TO_SLUG[kind])
    assert section, f"{shape}/{kind}: the section did not render at all"
    assert OPENING in section and NOTHING_BEYOND in section and SAY_NOT_READ in section
    assert section.rstrip().endswith(LAST_WORD)


def test_the_clause_is_written_once_and_spliced_by_reference():
    """One definition in the module, three renderings — the rule `_EVERY_TOOL_SUCCEEDED` and
    `_ADMITTING_A_LIMIT` already follow. A hand-written second copy diverges the first time
    either is reworded."""
    src = inspect.getsource(_se)
    assert src.count(OPENING) == 1, "the clause's opening appears more than once in the source"
    # a phrase that sits inside ONE string literal of the constant and in no comment
    assert src.count("item, name or value that is not written in them") == 1


# ── THE DRAFT IS NOT SHOWN ON THIS PATH, AND THE CLAUSE SAYS SO ───────────────────────

def test_the_draft_stays_withheld_on_exhaustion_so_the_clause_says_when_one_is_shown():
    """"Voice the last draft on exhaustion" was measured and REFUSED (4 of 11 exhausted turns
    were rejected BECAUSE the draft invented its figures — `test_voice_does_not_adopt_the_
    critiques_remedy`). So the draft is NOT in this prompt, and a clause that said "the
    reply is the draft and the data" would name a source the voice cannot see. It says
    "when one is shown" instead, and this test is what keeps that sentence honest.

    With its own CONTROL: the scaffold can show the draft (a conversational turn with no
    rejection), so the absence below discriminates.
    """
    control = _turn()
    control.metadata[mk.JUDGE_CONVERSATIONAL] = True
    assert DRAFT in _render(control, kind=None), "the scaffold cannot show the draft at all"
    prompt = _render(_turn())
    assert DRAFT not in prompt, "the rejected draft came back into the exhaustion prompt"
    assert WHEN_ONE_IS_SHOWN in _section(prompt, "execution_verdict")


# ── THE PATH IS REALLY REACHED ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_clause_reaches_the_prompt_voice_actually_sends():
    """`_build_voice_prompt` is a helper; what ships is what `voice()` hands the backend."""
    ctx = _turn()
    ctx.metadata[mk.VOICE_CORRECTION] = {"reason": CRITIQUE, "kind": "not_executed"}
    backend = ScriptedBackend(["Remuneração estimada de novembro: R$ 3.840,00 (32 h)."])
    result = await SuperegoStage().voice(ctx, backend, voice_prompt="persona")
    sent = backend.calls[0]["prompt"]
    assert OPENING in sent and SAY_NOT_READ in sent
    assert OPENING in (result.prompt_text or "")
    assert "execution_verdict" in [b["block"] for b in result.prompt_blocks]


# ── THE CONTROLS: WITHOUT AN EXHAUSTION, BYTE-IDENTICAL TO MAIN ───────────────────────
#
# Whole-prompt digests, measured on `main` (d34822a5) with THIS fixture. The change is an
# unconditional clause on the two exhaustion verdicts, so every rendering that carries NO
# verdict — no rejection at all, the anti-repeat guard, an approved draft, a conversational
# turn — must not move by one byte. Recompute by rendering the same fixtures on the revision
# you are comparing against.

_MAIN_PROMPTS = {
    "no_rejection": "095ac3e3c99ea942",
    "repeated_reply": "f21de0752b5858d4",
    "approved_draft": "8d34dbc31c934279",
    "conversational": "957f447768a89f05",
}
# …and the two cells that DID move, so the table above cannot be satisfied by a no-op.
_MAIN_EXHAUSTION = {
    "not_executed": "c2257f38b5039055",
    "unverified_claim": "8dc0177ab59d631a",
}


def _controls() -> "dict[str, str]":
    no_rejection = _render(_turn(), kind=None)
    repeated = _render(_turn(), kind="repeated_reply", reason="already said")
    approved = _turn()
    approved.metadata[mk.JUDGE_VERDICT] = {"approved": True, "attempts": 1}
    conversational = _turn()
    conversational.metadata[mk.JUDGE_CONVERSATIONAL] = True
    return {"no_rejection": no_rejection, "repeated_reply": repeated,
            "approved_draft": _render(approved, kind=None),
            "conversational": _render(conversational, kind=None)}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@pytest.mark.parametrize("label", sorted(_MAIN_PROMPTS))
def test_without_an_exhaustion_the_prompt_is_exactly_mains(label):
    prompt = _controls()[label]
    assert OPENING not in prompt, f"{label}: the clause leaked into a prompt with no verdict"
    assert _digest(prompt) == _MAIN_PROMPTS[label], f"{label}: the prompt moved"


def test_the_approved_draft_control_really_shows_the_draft():
    """The control is only a control if it exercises the path it stands for."""
    assert DRAFT in _controls()["approved_draft"]
    assert DRAFT in _controls()["conversational"]


@pytest.mark.parametrize("kind", sorted(_MAIN_EXHAUSTION))
def test_the_two_exhaustion_renderings_are_NOT_mains(kind):
    assert _digest(_render(_turn(), kind=kind)) != _MAIN_EXHAUSTION[kind]


# ── NO NEW HEADER ─────────────────────────────────────────────────────────────────────

def test_no_new_header_the_persisted_inventory_does_not_move():
    """A clause inside a section that exists, not a section: `_VOICE_BLOCKS` and the inventory
    the host persists are untouched — the same shape `nothing_tried` and the critique clause
    took. The slug list is `main`'s for this fixture."""
    slugs = [b["block"] for b in SuperegoStage.voice_prompt_inventory(_render(_turn()))]
    assert slugs == ["user_request", "context", "executor_data", "execution_verdict",
                     "signals", "task"]
    alphabet = {slug for _, slug in SuperegoStage._VOICE_BLOCKS}
    assert set(slugs) <= alphabet
