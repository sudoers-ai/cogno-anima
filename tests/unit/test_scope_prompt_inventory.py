"""The scope guard must leave a record of what it was ASKED.

It is the only stage that can end a turn on its own: it BLOCKS before the EGO runs, so a turn
it ends carries no execution, no judge verdict and no voice inventory — the canned
`refusal_message` is the entire trace, and it reads the same whatever the cause.

Measured 2026-09-17: a GUEST asking «que materiais posso usar para estudar?» was refused by
this guard on a turn whose persona held the tool whose entire job is that question. Four
readings fit every byte the trace held, and they have different fixes — the turn's path never
appended the tool table; the dispatcher probe answered empty; the identity's RBAC emptied it;
or all of it arrived and the classifier blocked anyway. A `tool_table` row separates the last
(a prompt defect) from the first three (wiring), which is the same split the judge's inventory
exists for, and the split this house has already paid for reading backwards.

What makes the record safe to persist is the same property its two siblings rest on: the
OUTPUT ALPHABET IS CLOSED. Slugs come from `_SCOPE_BLOCKS`, lengths are integers, and nothing
that was matched is ever copied out. That matters more here than for either sibling — this
prompt is the one whose every block is written by somebody else (the tenant's rules, the
contact's own sentence) — so the canary test below is the load-bearing one.
"""

from __future__ import annotations

import json

import pytest

from cogno_anima import SCOPE_TOOL_TABLE_HEADER, metakeys as mk
from cogno_anima.stages.superego import SuperegoStage
from tests.unit.test_superego import RaisingBackend, ScriptedBackend, _ctx

_BUILD = SuperegoStage._build_scope_prompt
_INV = SuperegoStage.scope_prompt_inventory

# A neutral slot: the tenant's own prose, carrying no markdown header of its own.
_SCOPE = "You handle scheduling and study support for the school."

_TABLE = (f"{SCOPE_TOOL_TABLE_HEADER}\n"
          "A request one of these serves IS in scope.\n"
          "- consult_material — reads the tenant's published corpus")

# A table that RENDERED and names nothing. The host this was written for renders the empty case
# as the empty string, so today this shape arrives from nobody — which is exactly why the
# property is pinned here rather than assumed: the distinction has to survive in the RECORD
# whatever a host chooses to put in the prompt.
_EMPTY_TABLE = SCOPE_TOOL_TABLE_HEADER


def _slugs(prompt):
    return [b["block"] for b in _INV(prompt)]


# ── the alphabet is closed: no byte of content ───────────────────────

def test_not_one_byte_of_the_tenants_rules_or_the_contacts_words_reaches_the_record():
    """The canary. Both halves of this prompt are somebody else's text — the tenant's scope
    rules and the contact's sentence, quoted verbatim — and a record that echoed what it
    matched would be a store of contact messages under a name nobody thinks of as a message
    log, with no purge path and no obligation anybody declared.
    """
    canary_rules = "CANARY_TENANT_RULE_7f3a"
    canary_contact = "CANARY_CONTACT_SENTENCE_91b2"
    canary_tool = "CANARY_TOOL_NAME_c5d8"
    prompt = _BUILD(f"{_SCOPE} {canary_rules}\n\n{SCOPE_TOOL_TABLE_HEADER}\n- {canary_tool}",
                    f"que materiais posso usar, {canary_contact}?", "pt-BR")
    # The canaries really are in the prompt — an absence test over an input that never carried
    # the value proves nothing at all.
    for canary in (canary_rules, canary_contact, canary_tool):
        assert canary in prompt

    inventory = _INV(prompt)
    blob = json.dumps(inventory, sort_keys=True)
    for canary in (canary_rules, canary_contact, canary_tool):
        assert canary not in blob, f"{canary} reached the inventory"

    # Stronger than "the canary is absent": the record can hold NOTHING but a known slug and an
    # int, so a value this test did not think to plant has nowhere to sit either.
    alphabet = {slug for _, slug in SuperegoStage._SCOPE_BLOCKS}
    for row in inventory:
        assert set(row) == {"block", "chars"}
        assert row["block"] in alphabet
        assert type(row["chars"]) is int


def test_a_forged_header_in_the_tenants_rules_adds_a_visible_row_and_never_a_column():
    """The tenant writes `# Scope Definition`'s content, so it can write a line that looks like
    one of these headers. The worst it buys is a DUPLICATE row (reported as a duplicate, never
    merged) and a shifted length — the same bound `_JUDGE_BLOCKS` documents for a tenant's
    persona limits."""
    forged = _BUILD(f"{_SCOPE}\n\n# Examples\nsomething the tenant wrote", "olá", "pt-BR")
    assert _slugs(forged).count("examples") == 2
    assert "something the tenant wrote" not in json.dumps(_INV(forged))


# ── the table: rendered, rendered-empty, and absent are three states ──

def test_a_table_that_rendered_is_distinguishable_from_one_that_did_not():
    """The question that motivated the whole record."""
    with_table = _BUILD(f"{_SCOPE}\n\n{_TABLE}", "que materiais posso usar?", "pt-BR")
    without = _BUILD(_SCOPE, "que materiais posso usar?", "pt-BR")

    rows = {b["block"]: b["chars"] for b in _INV(with_table)}
    assert rows["tool_table"] == len(_TABLE) + 2      # the block plus its trailing blank line
    assert "tool_table" not in _slugs(without)


def test_an_empty_table_and_an_absent_table_are_not_the_same_record():
    """Two different facts: "this turn could run nothing" and "nobody told the guard anything".
    The first is a persona/RBAC answer, the second is a wiring answer, and a record that spells
    them the same way sends the next reader down the wrong one. The anima's own judge prompt
    already had to learn this — a consulted specialist that ran no tool renders its header with
    an explicit empty line, so that "there was no second executor" stays legible.

    The record keeps them apart because it keys on the HEADER and never on what follows it.
    """
    empty = _BUILD(f"{_SCOPE}\n\n{_EMPTY_TABLE}", "olá", "pt-BR")
    absent = _BUILD(_SCOPE, "olá", "pt-BR")

    assert "tool_table" in _slugs(empty)
    assert "tool_table" not in _slugs(absent)
    assert _INV(empty) != _INV(absent)

    # And the empty one is legible AS empty: a header and nothing under it.
    chars = {b["block"]: b["chars"] for b in _INV(empty)}["tool_table"]
    assert chars == len(SCOPE_TOOL_TABLE_HEADER) + 2


def test_order_is_as_rendered_so_two_turns_diff_as_lists():
    """As RENDERED, which is why this list changed when the layout did: a turn carrying a
    table now opens with the decision rule and the table, and the definition follows them.
    `test_scope_capability_outranks_the_definition.py` is where that order is the PROPERTY;
    here it is the demonstration that the inventory reports position rather than a fixed
    table order."""
    prompt = _BUILD(f"{_SCOPE}\n\n{_TABLE}", "quanto custa?", "pt-BR")
    assert _slugs(prompt) == ["decision_rule", "tool_table", "scope_definition",
                              "user_input", "task", "examples"]
    assert _slugs(_BUILD(_SCOPE, "quanto custa?", "")) == [
        "scope_definition", "user_input", "task", "examples"]


def test_the_lengths_are_the_sections_and_they_cover_the_prompt():
    """A length that does not measure its own section is a number that reads like evidence."""
    prompt = _BUILD(f"{_SCOPE}\n\n{_TABLE}", "quanto custa?", "pt-BR")
    assert sum(b["chars"] for b in _INV(prompt)) == len(prompt)   # starts at char 0


# ── what the guard records, per path ─────────────────────────────────

@pytest.mark.asyncio
async def test_a_consulted_guard_records_what_it_was_asked_on_both_verdicts():
    for reply, blocked in (('{"blocked": true, "refusal_message": "Fora do escopo."}', True),
                           ('{"blocked": false, "refusal_message": ""}', False)):
        ctx = _ctx(user="que materiais posso usar para estudar?",
                   intent_class="INFORMATION_REQUEST", language="pt-BR")
        r = await SuperegoStage().check_input_scope(
            ctx, ScriptedBackend([reply]), scope_prompt=f"{_SCOPE}\n\n{_TABLE}")
        assert r.blocked is blocked
        # The ALLOWED turn is the comparison half — a capture that only fires on the anomaly has
        # nothing to compare the anomaly against.
        assert [b["block"] for b in r.prompt_blocks] == [
            "decision_rule", "tool_table", "scope_definition",
            "user_input", "task", "examples"]


@pytest.mark.asyncio
async def test_a_bypassed_guard_records_nothing_because_it_built_nothing():
    """Empty means NO PROMPT WAS BUILT — the model was never consulted. It cannot mean "a
    prompt with no sections": a built prompt always carries at least four."""
    for kwargs, scope in ((dict(intent_class="SOCIAL"), _SCOPE),
                          (dict(intent_class="UNKNOWN", goal_status="ONGOING"), _SCOPE),
                          (dict(intent_class="INFORMATION_REQUEST"), "")):
        b = ScriptedBackend([])
        ctx = _ctx(user="oi", **kwargs)
        r = await SuperegoStage().check_input_scope(ctx, b, scope_prompt=scope)
        assert b.calls == [] and r.prompt_blocks == []
        assert ctx.metadata[mk.SCOPE_PROMPT_BLOCKS] == []


@pytest.mark.asyncio
async def test_the_fail_open_path_records_the_prompt_it_had_already_built():
    """"The classifier call blew up" and "the classifier read this and allowed" are different
    facts, and only the second is about the prompt. Both come back `blocked=False`, so without
    the inventory the fail-open branch is indistinguishable from a clean ALLOW — the judge's
    fail-CLOSED twin records for the same reason."""
    ctx = _ctx(user="como faço bolo?", intent_class="INFORMATION_REQUEST")
    r = await SuperegoStage().check_input_scope(
        ctx, RaisingBackend(), scope_prompt=f"{_SCOPE}\n\n{_TABLE}")
    assert r.blocked is False
    assert [b["block"] for b in r.prompt_blocks] == [
        "decision_rule", "tool_table", "scope_definition",
        "user_input", "task", "examples"]


@pytest.mark.asyncio
async def test_the_record_reaches_a_trace_writer_and_not_only_the_result():
    """`ScopeCheckResult` is consumed by the orchestrator and dropped — its metrics are
    appended and its refusal becomes a `SuperegoResult` — so a record living only on the result
    is a record with no reader. It is stamped where the turn's trace is written from."""
    ctx = _ctx(user="que materiais posso usar?", intent_class="INFORMATION_REQUEST")
    r = await SuperegoStage().check_input_scope(
        ctx, ScriptedBackend(['{"blocked": true, "refusal_message": "Não posso."}']),
        scope_prompt=f"{_SCOPE}\n\n{_TABLE}")
    assert ctx.metadata[mk.SCOPE_PROMPT_BLOCKS] == r.prompt_blocks
    # It is a per-turn fact, so it must survive JSON — the carrier is a serializable dict that
    # a host writes into JSONB.
    assert json.loads(json.dumps(ctx.metadata[mk.SCOPE_PROMPT_BLOCKS])) == r.prompt_blocks
