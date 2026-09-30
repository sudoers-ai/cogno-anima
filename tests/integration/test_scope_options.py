"""The "did you mean…?" selector against a REAL model — the two shapes that decide it.

`tests/unit/test_scope_options.py` pins the MECHANISM with a stub (the closed alphabet, covered
wins, the error path). None of that says what a model does with the prompt, and the prompt is where
the two outcomes part: a false refusal must come back ``covered`` (the orchestrator then lets the
turn through) and NEVER ``suggested`` (which the host turns into a question that would hide the
false refusal), while a request nothing on the list names must come back with no option at all.

`n=3` per case, because a selector's answer is a sample. The fixture is invented.

**Two models, every run** (2026-09-30): the suite's own spec (``qwen3:8b`` on the CI runner — its
nightly is where the Wi-Fi came back ``covered`` 3/3 and the code-side evidence rule was born) AND
the model the host runs the selector on (``openai:gpt-4o-mini``, the scope guard's in the preset),
which skips itself naming the missing key when there is none. The Wi-Fi must draw NOTHING on both.
The downstream A/B over labelled real refusals is the consultant's, not this file's.
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


#: The host's selector model. Parametrised beside the suite's own spec, never instead of it.
CLOUD_SPEC = "openai:gpt-4o-mini"
SPECS = pytest.mark.parametrize("spec", [None, CLOUD_SPEC], ids=["suite_model", "cloud"])


async def _skip_unless(spec):
    reason = await backends.backend_unavailable_reason(spec)
    if reason:
        pytest.skip(reason)


async def _run(message: str, spec=None) -> "list[so.OptionSelection]":
    return [await so.select_options(message, OPTIONS, backends.json_backend(spec))
            for _ in range(N)]


@SPECS
@pytest.mark.asyncio
async def test_a_false_refusal_comes_back_covered_never_as_a_question(spec):
    """The 1a shape: the guide HAS the semester calendar. The answer the host acts on is
    ``covered`` (let it through); ``suggested`` here would be "did you mean <what you asked>?"."""
    await _skip_unless(spec)
    got = await _run("Quando começa o semestre?", spec)
    assert [g.outcome for g in got] == [so.OUTCOME_COVERED] * N, [g.record() for g in got]
    assert all("Calendário do semestre" in g.covered for g in got)


@SPECS
@pytest.mark.asyncio
async def test_the_wifi_password_draws_no_option(spec):
    """THE NEGATIVE. A selector that finds "something close" for every refusal is the free
    selector the strict prompt replaced — and it would pass the case above."""
    await _skip_unless(spec)
    got = await _run("Qual é a senha do Wi-Fi da escola?", spec)
    assert all(not g.covered and not g.suggested for g in got), [g.record() for g in got]


@SPECS
@pytest.mark.asyncio
async def test_whatever_the_model_says_only_listed_options_come_out(spec):
    await _skip_unless(spec)
    for message in ("Tem curso de fotografia avançada?", "Posso levar meu cachorro?"):
        for g in await _run(message, spec):
            assert set(g.covered) | set(g.suggested) <= set(OPTIONS), g.record()
