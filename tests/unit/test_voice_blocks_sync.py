"""`_VOICE_BLOCKS` must keep up with the prompt it claims to describe.

The inventory (`voice_prompt_inventory`) reports which sections the voice prompt carried, and
the host persists that per turn. Its accuracy rests on a hand-written table, `_VOICE_BLOCKS`,
sitting in a different part of the file from `_build_voice_prompt` — the classic duplicated
contract, and the repo already pins the sibling case
(`test_pipeline.py::test_code_domains_match_prompt_domains_exactly`).

Nothing pinned this one. Add a section to the voice prompt and no test fails: the inventory
simply under-reports from then on, in silence — and the open investigation into the JSON
envelope depends on that inventory being complete, so a silent gap costs exactly the question
it was built to answer.

These tests check the EFFECT (headers in a rendered prompt), not the form (string literals in
the source), so a section that is renamed or moved is caught the same way a new one is.
"""

from __future__ import annotations

import re

import pytest

from cogno_anima import metakeys as mk
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.types import EgoResult, EgoStep
from tests.unit.test_superego import _ctx

# Every top-level header line the rendered prompt may carry.
_HEADER = re.compile(r"^# .+$", re.MULTILINE)


def _known(header_line: str) -> bool:
    return any(header_line.startswith(known) for known, _ in SuperegoStage._VOICE_BLOCKS)


def _configs():
    """Contexts chosen to light up every OPTIONAL block, not just the always-on ones."""
    plain = _ctx()

    with_context = _ctx()
    with_context.metadata[mk.EGO_CONTEXT] = "[MEMORIES]\nmora em Santos"

    with_traits = _ctx()
    with_traits.metadata[mk.VOICE_TRAITS] = ["warm", "direct"]

    # A CONVERSATIONAL turn is the only one that renders the executor's draft as its own
    # section (`_draft_section` returns "" on an execution turn, where tool data is the only
    # grounding). Leaving it out of this matrix is what the reverse-direction test caught the
    # first time it ran — which is the whole reason that test exists.
    conversational = _ctx()
    conversational.metadata[mk.JUDGE_CONVERSATIONAL] = True
    # `draft` is DERIVED from the steps (the trace is the source of truth), so the draft is set
    # by giving the turn a step whose assistant text is it — not by assigning the property.
    conversational.ego_result = EgoResult(
        steps=[EgoStep(index=0, path="text",
                       assistant_text="posso explicar como funciona o atendimento")],
        metrics=conversational.ego_result.metrics)

    out = [("plain", plain), ("context", with_context), ("traits", with_traits),
           ("conversational", conversational)]
    # The three rejection variants are mutually exclusive — each needs its own render.
    for kind in ("repeated_reply", "unverified_claim", "execution_rejected"):
        c = _ctx()
        c.metadata[mk.VOICE_CORRECTION] = {"kind": kind, "reason": "because"}
        out.append((f"rejection:{kind}", c))
    return out


@pytest.mark.parametrize("name,ctx", _configs(), ids=lambda v: v if isinstance(v, str) else "")
def test_every_rendered_header_is_in_the_table(name, ctx):
    prompt = SuperegoStage()._build_voice_prompt(
        ctx, payload="check_availability: 09:00, 10:00",
        adjustments=list(SuperegoStage().detect_adjustments(ctx)),
        traits=ctx.metadata.get(mk.VOICE_TRAITS, ()))
    unknown = [h for h in _HEADER.findall(prompt) if not _known(h)]
    assert not unknown, (
        f"[{name}] the voice prompt renders {unknown} and `_VOICE_BLOCKS` does not list it — "
        f"the inventory would silently under-report this section on every turn.")


def test_the_table_does_not_list_sections_the_prompt_never_renders():
    """The other direction. A slug for a header that no longer exists is a row that can never
    appear — dead weight that reads like coverage."""
    rendered = set()
    for _, ctx in _configs():
        prompt = SuperegoStage()._build_voice_prompt(
            ctx, payload="p", adjustments=[], traits=ctx.metadata.get(mk.VOICE_TRAITS, ()))
        rendered.update(_HEADER.findall(prompt))
    for known, slug in SuperegoStage._VOICE_BLOCKS:
        assert any(h.startswith(known) for h in rendered), (
            f"`{slug}` maps to {known!r}, which none of the rendered configurations produce — "
            f"either the header changed or this row is dead.")


def test_the_check_would_actually_catch_a_new_section():
    """A guard that cannot fail is a guard that passes for the wrong reason: pin that an
    unlisted header IS rejected by the same predicate the tests above use."""
    assert not _known("# Something Nobody Listed")
    assert _known("# Task")
    assert _known("# Data gathered by the executor (ground figures/dates ONLY in this)")
