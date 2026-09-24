"""The judge prompt opens with what does NOT change between turns — the provider's cache (F1.3).

A provider's prompt cache serves only an IDENTICAL PREFIX, and OpenAI's only from 1024 tokens up.
The judge's system message carries nothing of the turn (`_judge_system`: the fixed instruction,
then the business rules when declared), so the first bytes of the USER half decide how far two
turns of the same (persona, role) keep sharing a prefix. That half used to open with the
contact's own sentence, so the shared prefix ended there — 36 tokens on every persona a host
registers, and the stage billed 0% cached. `# Persona limits` is the one section that belongs to
the persona, not to the turn, and it now comes first; nothing else moves and no byte changes.

Three things are pinned here, each with the check that would catch its opposite:

* **the twin** — two different turns of one (persona, role) share the whole system message and
  the whole limits slot up to the host's clock line, and that is at least 1024 tokens;
* **the control** — a slot with no limits renders exactly what it always rendered;
* **the multiset** — the sections of the new prompt, digest by digest, are the sections of the
  legacy one: nothing entered, nothing left, one moved.

The token count is a LOWER BOUND with no dependency: GPT-family tokenizers (cl100k/o200k)
pre-split text on whitespace before any merge, so no token spans two whitespace-separated words
and ``len(text.split())`` never exceeds the token count. Measured against o200k_base on the
twin below, the real count is higher still — the bound is what makes the assertion true for
any of them.
"""

from __future__ import annotations

import hashlib
from collections import Counter

import pytest

from cogno_anima import metakeys as mk
from cogno_anima.stages.superego import _JUDGE_SYSTEM, SuperegoStage
from tests.unit.test_judge_blocks_sync import _configs
from tests.unit.test_superego import _ctx

#: The provider's minimum: below this OpenAI caches nothing at all.
CACHE_MIN_TOKENS = 1024

# A persona's limits slot of realistic size, then the host's environment block at its END —
# where `cogno-host` puts it, because it carries the minute. Invented rules, invented names.
_RULES = "\n".join(
    f"- Rule {i}: never confirm a booking, a payment or a refund that the tools did not "
    f"return; the clinic closes at 18h on weekdays and the reception desk answers calls."
    for i in range(1, 60))
_STABLE = f"You judge the replies of the reception persona Lia.\n\n{_RULES}"


def _limits(minute: str) -> str:
    return (f"{_STABLE}\n\n[AMBIENTE] — cita estes valores; nunca os calcules.\n"
            f"Persona: Lia (RECEPTION)\n[TODAY] 2026-09-24 (Thursday)\n"
            f"Hora: {minute} (America/Sao_Paulo — relógio do servidor, não do contacto)")


def _common(a: str, b: str) -> str:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return a[:n]


def _asked(ctx, limits: str) -> tuple[str, str]:
    """(system, user) — what `evaluate` hands `backend.generate`, in that order."""
    return SuperegoStage._judge_system(ctx), SuperegoStage()._build_judge_prompt(ctx, limits)


def _turn(user: str, goal: str, memory: str, draft_rows: str):
    ctx = _ctx(user=user, goal=goal, intent_class="INFORMATION_REQUEST")
    ctx.metadata[mk.EGO_CONTEXT] = f"[TODAY] 2026-09-24\n{memory}"
    ctx.ego_result.steps[0].tool_calls[0].result = draft_rows
    ctx.ego_result.steps[0].assistant_text = f"Here is what I found: {draft_rows}"
    return ctx


def _shared_prefix(rules: str = "") -> tuple[str, str, str]:
    a = _turn("quais os horários de amanhã?", "list tomorrow's slots",
              "Ana prefers mornings.", "09:00, 14:00")
    b = _turn("e na sexta, tem vaga à tarde?", "find a Friday afternoon slot",
              "Asked for a receipt in August.", "15:30")
    for ctx in (a, b):
        if rules:
            ctx.metadata[mk.PERSONA_RULES] = rules
    sa, ua = _asked(a, _limits("10:02"))
    sb, ub = _asked(b, _limits("10:19"))
    assert sa == sb, "the judge's system message carries something of the turn"
    return sa, _common(ua, ub), ua


# ── the twin ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rules", ["", "Aula avulsa: R$ 120,00 por hora."],
                         ids=["no_rules", "with_rules"])
def test_two_turns_of_one_persona_share_the_limits_slot_and_it_is_cacheable(rules):
    system, shared, user = _shared_prefix(rules)
    # The whole persona text is inside the shared prefix, and it ends at the host's clock line.
    assert shared.startswith(f"# Persona limits\n{_STABLE}\n\n[AMBIENTE]"), shared[:80]
    assert shared.endswith("Hora: 10:"), shared[-40:]
    words = len(system.split()) + len(shared.split())
    assert words >= CACHE_MIN_TOKENS, (
        f"the identical prefix is at most {words} words — below the {CACHE_MIN_TOKENS}-token "
        f"floor a provider caches from")


def test_what_follows_the_prefix_is_the_turn():
    """The prefix ends where the turn begins — the clock, then the contact's own sentence."""
    _, shared, user = _shared_prefix()
    rest = user[len(shared):]
    assert rest.startswith("02 (America/Sao_Paulo")
    assert rest.index('# User request\n"quais os horários de amanhã?"') < 200


def test_the_system_message_is_still_only_the_fixed_instruction_without_rules():
    """#184's promise, re-checked rather than assumed: no rules → exactly `_JUDGE_SYSTEM`."""
    system, _, _ = _shared_prefix()
    assert system == _JUDGE_SYSTEM


# ── the control ───────────────────────────────────────────────────────────────────────────

#: The order this prompt's sections had until F1.3, as the inventory names them. The limits
#: section sat between the preserved terms and the execution; everything else is unchanged.
_LEGACY_ORDER = ("user_request", "context", "active_goal", "user_constraints", "unavailable",
                 "preserved_terms", "persona_limits", "executed", "held_messages", "draft",
                 "criteria_execution", "criteria_conversational", "criteria_readonly")


def _order(prompt: str) -> list[str]:
    return [b["block"] for b in SuperegoStage.judge_prompt_inventory(prompt)]


def _legacy_rank(slugs: list[str]) -> list[int]:
    return [_LEGACY_ORDER.index(s) for s in slugs]


@pytest.mark.parametrize("name,ctx", _configs(), ids=lambda v: v if isinstance(v, str) else "")
def test_a_turn_with_no_limits_slot_keeps_the_legacy_order(name, ctx):
    """The CONTROL: a persona with no limits slot has nothing to move, and its prompt is the
    legacy prompt — the same first section, the same sequence."""
    prompt = SuperegoStage()._build_judge_prompt(ctx, "")
    assert prompt.startswith("# User request\n"), prompt[:40]
    order = _order(prompt)
    assert "persona_limits" not in order
    assert _legacy_rank(order) == sorted(_legacy_rank(order)), order


def test_a_blank_limits_slot_is_no_section():
    stage = SuperegoStage()
    ctx = _configs()[0][1]
    assert stage._build_judge_prompt(ctx, "   \n") == stage._build_judge_prompt(ctx, "")


@pytest.mark.parametrize("name,ctx", _configs(), ids=lambda v: v if isinstance(v, str) else "")
def test_the_limits_section_is_first_and_the_rest_keep_their_order(name, ctx):
    prompt = SuperegoStage()._build_judge_prompt(ctx, _STABLE)
    assert prompt.startswith(f"# Persona limits\n{_STABLE}\n\n# User request\n"), prompt[:40]
    rest = [s for s in _order(prompt) if s != "persona_limits"]
    assert _order(prompt)[0] == "persona_limits"
    assert _legacy_rank(rest) == sorted(_legacy_rank(rest)), rest


# ── the multiset, digest by digest ────────────────────────────────────────────────────────

def _legacy(new: str, limits: str) -> str:
    """The same prompt in the order it had until F1.3: the limits section put back between the
    preserved terms and `# What the EGO executed`, with the blank line it had there. Built from
    the NEW rendering, so a test comparing the two can only ever see a difference of ORDER; the
    byte-for-byte identity of this reconstruction with the pre-F1.3 code was checked across
    trees when the change was made (the PR carries the per-section digests of both)."""
    section = f"# Persona limits\n{limits}\n"
    assert new.startswith(section + "\n")
    rest = new[len(section) + 1:]
    at = "\n# What the EGO executed\n"
    assert rest.count(at) == 1
    return rest.replace(at, f"\n{section}{at}", 1)


def _sections(prompt: str) -> list[str]:
    found = SuperegoStage._block_positions(prompt, SuperegoStage._JUDGE_BLOCKS)
    assert found and found[0][0] == 0, "text before the first known section would go uncounted"
    return [prompt[at:(found[n + 1][0] if n + 1 < len(found) else len(prompt))]
            for n, (at, _) in enumerate(found)]


def _digests(prompt: str) -> Counter:
    return Counter(hashlib.sha256(s.encode()).hexdigest()[:16] for s in _sections(prompt))


@pytest.mark.parametrize("name,ctx", _configs(), ids=lambda v: v if isinstance(v, str) else "")
def test_the_sections_are_the_same_multiset_as_before_only_the_order_moved(name, ctx):
    stage = SuperegoStage()
    new = stage._build_judge_prompt(ctx, _STABLE)
    old = _legacy(new, _STABLE)
    assert new != old
    assert _digests(new) == _digests(old), (
        f"[{name}] a section's BYTES changed, or one entered or left — the reorder was meant "
        f"to move one section and nothing else")
    inv_new = SuperegoStage.judge_prompt_inventory(new)
    inv_old = SuperegoStage.judge_prompt_inventory(old)
    assert sorted(map(repr, inv_new)) == sorted(map(repr, inv_old))
    # …and the ONE difference is the declared one: the limits section moved to the front, and
    # every other section kept its relative order.
    assert inv_new[0]["block"] == "persona_limits"
    rest_new = [b["block"] for b in inv_new if b["block"] != "persona_limits"]
    rest_old = [b["block"] for b in inv_old if b["block"] != "persona_limits"]
    assert rest_new == rest_old


def test_every_sentence_that_places_the_limits_still_reads_true():
    """The criteria say the limits section is ABOVE them. It still is — the criteria stay last."""
    ctx = dict(_configs())["execution"]
    prompt = SuperegoStage()._build_judge_prompt(ctx, _STABLE)
    claim = "'# Persona limits' section above"
    assert claim in prompt
    assert prompt.index("# Persona limits\n") < prompt.index(claim)
    assert prompt.index("# Persona limits\n") < prompt.index("# Judge the EXECUTION")
