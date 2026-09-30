"""The "did you mean…?" selector against a REAL model — the two shapes that decide it.

`tests/unit/test_scope_options.py` pins the MECHANISM with a stub (the closed alphabet, covered
wins, the error path). None of that says what a model does with the prompt, and the prompt is where
the two outcomes part: a false refusal must come back ``covered`` (the orchestrator then lets the
turn through) and NEVER ``suggested`` (which the host turns into a question that would hide the
false refusal), while a request nothing on the list names must come back with no option at all.

`n=3` per case, because a selector's answer is a sample. The fixture is invented.

**Point it at the model the host runs the selector on** (the scope guard's, ``gpt-4o-mini`` in the
preset); the downstream A/B over labelled real refusals is the consultant's, not this file's::

    COGNO_TEST_MODEL=openai:gpt-4o-mini pytest tests/integration/test_scope_options.py
"""

import pytest

from cogno_anima.stages import scope_options as so
from tests.integration import backends

N = 3

#: A school's guide, as the host would list it: document titles and section headings this reader
#: may read, then the capability names on the turn's table.
OPTIONS = (
    "Guia do Estudante 2026",
    "Calendário do semestre",
    "Oficina de Fotografia",
    "Mensalidade e descontos",
    "Biblioteca e empréstimos",
    "consult_documents",
    "book_class",
)


async def _run(message: str) -> "list[so.OptionSelection]":
    return [await so.select_options(message, OPTIONS, backends.json_backend()) for _ in range(N)]


@pytest.mark.asyncio
async def test_a_false_refusal_comes_back_covered_never_as_a_question():
    """The 1a shape: the guide HAS the semester calendar. The answer the host acts on is
    ``covered`` (let it through); ``suggested`` here would be "did you mean <what you asked>?"."""
    await backends.skip_unless_available()
    got = await _run("Quando começa o semestre?")
    assert [g.outcome for g in got] == [so.OUTCOME_COVERED] * N, [g.record() for g in got]
    assert all("Calendário do semestre" in g.covered for g in got)


@pytest.mark.asyncio
async def test_the_wifi_password_draws_no_option():
    """THE NEGATIVE. A selector that finds "something close" for every refusal is the free
    selector the strict prompt replaced — and it would pass the case above."""
    await backends.skip_unless_available()
    got = await _run("Qual é a senha do Wi-Fi da escola?")
    assert all(not g.covered and not g.suggested for g in got), [g.record() for g in got]


@pytest.mark.asyncio
async def test_whatever_the_model_says_only_listed_options_come_out():
    await backends.skip_unless_available()
    for message in ("Tem curso de fotografia avançada?", "Posso levar meu cachorro?"):
        for g in await _run(message):
            assert set(g.covered) | set(g.suggested) <= set(OPTIONS), g.record()
