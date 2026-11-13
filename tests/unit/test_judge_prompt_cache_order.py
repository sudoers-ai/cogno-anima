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
  legacy one: nothing entered, nothing left, one moved;
* **origin/main, byte for byte** — the legacy order rebuilt from the new prompt, and the prompt
  of a turn with no limits slot, hash to what `origin/main` (c0d6bb9) rendered for the same
  fixtures (`_ORIGIN_MAIN`, `_ORIGIN_MAIN_NO_LIMITS`);
* **the mutation, as a twin** — the same predicates reject the legacy order.

What this file measures is a PREFIX: how many bytes two turns share. Whether the provider then
serves them from its cache also depends on its time-to-live (OpenAI: minutes of inactivity) and
on how the traffic falls inside it, so the real cached rate is a number only the ledger gives.

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
from cogno_anima.stages.superego import _HELD_ASKING_IS_THE_HOSTS, _JUDGE_SYSTEM, SuperegoStage
from tests.unit._judge_0115 import as_0114
from tests.unit._judge_context_clauses import with_the_context_clauses
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

@pytest.mark.parametrize("rules", ["", "Aula avulsa: R$ 135,00 por hora."],
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
_LEGACY_ORDER = ("user_request", "context", "contact_memo", "active_goal", "user_constraints", "unavailable",
                 "preserved_terms", "persona_limits", "executed", "consulted", "held_messages",
                 "draft",
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


#: sha256 of `_build_judge_prompt(ctx, _STABLE)` for each `_configs()` configuration, rendered
#: by `origin/main` at c0d6bb9 — the base F1.3 merged — i.e. the prompt in the order it had
#: BEFORE this change. Computed by rendering these very fixtures with that tree's code, not with
#: this one's. (ed32560, the base F1.3 was first measured against, gives the same eight digests;
#: `memo` is the configuration #187 added.)
#:
#: This is a LANDING-TIME proof and it is meant to go stale: a later, deliberate change of the
#: judge's TEXT changes these digests legitimately. When that happens, regenerate them from the
#: new base (render each configuration with the base tree and hash it) in the same PR that
#: changed the text; never edit one to make a red go away.
#:
#: Regenerated 2026-11-12 (figures privacy scrub): the WORKED EXAMPLE inside
#: `_DERIVED_FROM_EVIDENCE` — a real production rate/hours/total that this public repo must not
#: carry — was replaced with a synthetic set. Every configuration that renders that constant
#: moved; `conversational`, which reads neither grounding constant, did not and keeps its digest.
#: The deliberate delta was confirmed to be ONLY that example: reverting the new example back to
#: the old inside the rendered prompt reproduced every digest below, so nothing else moved. The
#: old (real) example bytes are deliberately NOT stored here — that is the whole point of the
#: scrub — so this is a true regeneration, not a `_since_f13`-style undo.
def _since_f13(prompt: str) -> str:
    """The prompt with the judge's DELIBERATE later text changes removed by name, so the digests
    below can keep being c0d6bb9's and keep proving that nothing ELSE moved. Each entry is a
    change that has its own test pinning it; today two: `_HELD_ASKING_IS_THE_HOSTS` (2026-09-30,
    rendered only on a turn with a declared held message — the `held` configuration here), and
    the two "the Context above" clauses, which since 0.1.6 render only with the `# Context`
    section and are put BACK here on the configurations that have none (ten of the eleven).
    And the three deliberate changes of 0.1.15 (`_judge_0115.as_0114`), which reach the `held`
    configuration: a hold by name rendered `ERROR`, criterion (d) and the hold-is-the-outcome
    sentence."""
    return with_the_context_clauses(as_0114(prompt).replace(_HELD_ASKING_IS_THE_HOSTS, ""))


_ORIGIN_MAIN = {
    "readonly": "96da1b256f5578a4507844508cadd14f2ca2f2d11ec5f3260aacebe6630b96f7",
    "execution": "0d94f0018539335b6bddec0b201bdb8a88a7103126c5befa54b086a1b7bb05f8",
    "held": "0f592e05cfd6bdc0f92cb57dcf5d5ddb28a21f29d9f1c7ff7af8414079c5f3ff",
    "conversational": "e26a84569bbd6fae8770320b08cf7a4cc51db2c5e69ff3c68a529f7919cacb2b",
    "context": "99c9f13bbee2ee8a6fc836478afacc7ba14e73bebf75a8ce4b0e8269a3d12872",
    "unavailable": "943389b067a2adbe1d69795aba09f7da356534a131dff8d8abf7b9a05f864687",
    "constraints": "dbb5805af2dec42543ed9234aff3ffeb3cc8d2cde1cb27f2a4ecdc136ede540d",
    "preserved": "5add613ce4181206cf50ab6c6f0555fff5a82a6daa9267a749b528a76daec97b",
    "rules": "721e75e9d582afd31570e07c6997a8a04f7f7edb5f0bac4bdd45567c9376c005",
    "memo": "addfef671cb462c98c8eafb5de193dc0bf55ba8d0d17b7fa823be06eadd7d704",
    # Joined with the `consulted` row of `_JUDGE_BLOCKS`: the configuration is new, the prompt
    # is not — rendered by the tree before that row existed (55788c8), like the ten above.
    "consulted": "ca09a941d82abbfb4fd2dc3b2a3da9027d29ff557471332dbd8afddaf4476198",
}


@pytest.mark.parametrize("name,ctx", _configs(), ids=lambda v: v if isinstance(v, str) else "")
def test_the_legacy_reconstruction_is_origin_main_byte_for_byte(name, ctx):
    """The reconstruction the multiset test compares against is not a guess about the old
    order: put back where it was, the new prompt is — byte for byte — what `origin/main`
    rendered. So the two prompts differ by the position of one section and by nothing else."""
    new = SuperegoStage()._build_judge_prompt(ctx, _STABLE)
    legacy = hashlib.sha256(_since_f13(_legacy(new, _STABLE)).encode()).hexdigest()
    assert legacy == _ORIGIN_MAIN[name], (
        f"[{name}] the legacy order rebuilt from this prompt is not what origin/main rendered "
        f"— either a byte changed, or the judge's text changed after F1.3 and `_ORIGIN_MAIN` is "
        f"stale (see its comment)")


#: The same, for the CONTROL: `_build_judge_prompt(ctx, "")` — no limits slot at all — as
#: `origin/main` at c0d6bb9 rendered it. Same staleness rule as `_ORIGIN_MAIN`.
_ORIGIN_MAIN_NO_LIMITS = {
    "readonly": "7814f63c22908c23364eb9359a1f2bac0e927e43147c5f395bfd91f7edd89752",
    "execution": "779e81f0ae8022a6b63b59c8518defa363c1aa4c9d0a64668e5fa1a256ecbec6",
    "held": "a958a358bea6ad67f697bb122d461ab11fd79f19c4260a27b419a34e4cf88936",
    "conversational": "fb161a6011018dcb9700287d25061a4e234af7ecebcafc9fa00d7fedacba2d25",
    "context": "6cfdbfeeeda932d1871f2745d95ffa0a7eb017cfd8ff984741bceea26356caf8",
    "unavailable": "ce66ef049219631d9d54a6d9a83e871000a7d7c060e4c2cf65157e5387ca4e88",
    "constraints": "d155e63d8bfd6458cc4a483bfbbcf4e69e47479414e8819e96177f460e00b04e",
    "preserved": "a93ba2193ea80cbc8af243a9c60ad30e16d5fc6a4135e04c1e7d32f4ec767677",
    "rules": "3c8af0dd970d71211bbccb476fcc9ae657add1762db42a6f59a465759061afee",
    "memo": "d34e645fba08cfc85487de3631ae602ea70e3ca07a6bf33e022d5146b800b3b1",
    "consulted": "c5e5648439fb765232b0c00f74e25c0a76be3be7ee0b906eb0c782893572af8d",   # 55788c8
}


@pytest.mark.parametrize("name,ctx", _configs(), ids=lambda v: v if isinstance(v, str) else "")
def test_a_turn_with_no_limits_slot_is_origin_main_byte_for_byte(name, ctx):
    """The control, byte for byte and not only by order: with nothing to move, the prompt is
    the one `origin/main` rendered."""
    got = hashlib.sha256(_since_f13(SuperegoStage()._build_judge_prompt(ctx, "")).encode()).hexdigest()
    assert got == _ORIGIN_MAIN_NO_LIMITS[name], (
        f"[{name}] a prompt with no limits slot changed — or `_ORIGIN_MAIN_NO_LIMITS` is stale")


# ── the mutation, written as a twin ───────────────────────────────────────────────────────

def test_the_twin_rejects_the_legacy_order():
    """A gate that cannot fail passes for the wrong reason. Given the LEGACY rendering of the
    same two turns — the order `origin/main` had, rebuilt as above — the twin's own predicates
    must reject it: the shared prefix ends inside the contact's sentence, far below the floor."""
    a = _turn("quais os horários de amanhã?", "list tomorrow's slots",
              "Ana prefers mornings.", "09:00, 14:00")
    b = _turn("e na sexta, tem vaga à tarde?", "find a Friday afternoon slot",
              "Asked for a receipt in August.", "15:30")
    stage = SuperegoStage()
    old_a = _legacy(stage._build_judge_prompt(a, _limits("10:02")), _limits("10:02"))
    old_b = _legacy(stage._build_judge_prompt(b, _limits("10:19")), _limits("10:19"))
    shared = _common(old_a, old_b)
    assert not shared.startswith("# Persona limits")
    assert shared == '# User request\n"'
    assert len(stage._judge_system(a).split()) + len(shared.split()) < CACHE_MIN_TOKENS


def test_every_sentence_that_places_the_limits_still_reads_true():
    """The criteria say the limits section is ABOVE them. It still is — the criteria stay last."""
    ctx = dict(_configs())["execution"]
    prompt = SuperegoStage()._build_judge_prompt(ctx, _STABLE)
    claim = "'# Persona limits' section above"
    assert claim in prompt
    assert prompt.index("# Persona limits\n") < prompt.index(claim)
    assert prompt.index("# Persona limits\n") < prompt.index("# Judge the EXECUTION")
