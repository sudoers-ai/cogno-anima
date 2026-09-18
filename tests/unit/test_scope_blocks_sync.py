"""`_SCOPE_BLOCKS` must keep up with the prompt it claims to describe.

`scope_prompt_inventory` reports which sections the scope guard's prompt carried, and a host
persists that per turn. Its accuracy rests on a hand-written table sitting in a different part
of the file from `_build_scope_prompt` — the classic duplicated contract. The repo pins the two
siblings the same way (`test_voice_blocks_sync.py`, and
`test_pipeline.py::test_code_domains_match_prompt_domains_exactly`), and the voice one exists
because the table had already rotted once with nobody noticing: a new section simply makes the
inventory under-report, in silence, from then on.

**Why one row is checked against an ARGUMENT and not against the skeleton.** `tool_table` is
rendered by the HOST, into the `scope_prompt` string it hands in — the guard takes no
dispatcher, so what a turn can DO is knowledge only the host has. What the core owns is the
NAME (`SCOPE_TOOL_TABLE_HEADER`) and the counting; so the row is proved by rendering a
`scope_prompt` that carries that header, which is exactly the shape production passes.

**Why the forward direction scans a NEUTRAL scope slot.** The slot is a tenant's free prose and
may contain any `#` line it likes. That is not an unlisted section of this contract — the
inventory reports a forged header as a duplicate ROW and never as a byte of its own — so the
headers under test are the ones the CORE renders plus the one the core NAMES.
"""

from __future__ import annotations

import re

import pytest

from cogno_anima import SCOPE_TOOL_TABLE_HEADER
from cogno_anima.stages.superego import SuperegoStage

# Every header line a rendered guard prompt may carry, `#` or `##` (the table is a sub-section
# of `# Scope Definition` — see the constant's own docstring).
_HEADER = re.compile(r"^#{1,2} .+$", re.MULTILINE)

# A scope slot with no markdown of its own, so the forward direction reads the SKELETON.
_NEUTRAL = "You handle scheduling for the clinic and nothing else."

_TABLE = (f"{SCOPE_TOOL_TABLE_HEADER}\n"
          "A request one of these serves IS in scope.\n"
          "- consult_material — reads the published corpus")


def _known(header_line: str) -> bool:
    return any(header_line.startswith(known) for known, _ in SuperegoStage._SCOPE_BLOCKS)


def _configs():
    """Renders chosen to light up every row, including the host-rendered one."""
    return [
        ("skeleton", _NEUTRAL, "que materiais posso usar para estudar?", "pt-BR"),
        ("no_language", _NEUTRAL, "quanto custa?", ""),
        ("with_table", f"{_NEUTRAL}\n\n{_TABLE}", "que materiais posso usar?", "pt-BR"),
        # The host's own no-op: an empty table leaves the slot byte-identical.
        ("table_absent", _NEUTRAL, "olá", "en"),
    ]


@pytest.mark.parametrize("name,scope,user,lang", _configs(),
                         ids=[c[0] for c in _configs()])
def test_every_rendered_header_is_in_the_table(name, scope, user, lang):
    prompt = SuperegoStage._build_scope_prompt(scope, user, lang)
    unknown = [h for h in _HEADER.findall(prompt) if not _known(h)]
    assert not unknown, (
        f"[{name}] the guard prompt renders {unknown} and `_SCOPE_BLOCKS` does not list it — "
        f"the inventory would silently under-report this section on every turn.")


def test_the_table_does_not_list_sections_the_prompt_never_renders():
    """The other direction. A slug for a header that no longer exists is a row that can never
    appear — dead weight that reads like coverage."""
    rendered = set()
    for _, scope, user, lang in _configs():
        rendered.update(_HEADER.findall(SuperegoStage._build_scope_prompt(scope, user, lang)))
    for known, slug in SuperegoStage._SCOPE_BLOCKS:
        assert any(h.startswith(known) for h in rendered), (
            f"`{slug}` maps to {known!r}, which none of the rendered configurations produce — "
            f"either the header changed or this row is dead.")


def test_the_check_would_actually_catch_a_new_section():
    """A guard that cannot fail is a guard that passes for the wrong reason: pin that an
    unlisted header IS rejected by the same predicate the tests above use."""
    assert not _known("# Something Nobody Listed")
    assert not _known("## Tools this persona could once run")
    assert _known("# Task")
    assert _known(SCOPE_TOOL_TABLE_HEADER)


def test_the_header_a_host_renders_is_the_header_this_table_counts():
    """One definition, not two. A host that spelled its own header would render a section the
    inventory cannot see — under-reporting in silence, which is the single failure this whole
    record exists to end. The row and the exported constant are therefore the same object."""
    assert dict(SuperegoStage._SCOPE_BLOCKS)[SCOPE_TOOL_TABLE_HEADER] == "tool_table"
    assert SCOPE_TOOL_TABLE_HEADER.startswith("## "), (
        "the table is a SUB-section of `# Scope Definition`; a top-level header here would "
        "read as a fourth section competing with `# User Input`.")
