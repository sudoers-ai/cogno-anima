"""Several contacts answer to one name: every summarised fact belongs to the block it came from.

A conversation digest a host returns for a name can match more than one contact record — one
identity per CHANNEL, so one person on two channels is two records, and two people who share a
name are two records too. Measured on a downstream host: the digest held two such blocks, the
executor summarised one contact's conversation as the other's, and the judge APPROVED it, because
nothing in its criteria asked WHOSE block a fact came from.

The host's digest now opens with ``N contacts match this name: …`` and labels each block by its
channel. That header is the CONDITION for the judge's attribution rule — read off the bytes of a
successful result, never guessed — so:

* the TWIN: a result with two (or three) contact blocks under the header → the rule renders;
* the CONTROL: one block, a failed call, the header only quoted mid-line, ``1 contacts …``, a
  turn with no digest at all → system + prompt byte for byte the parent tree's (anima
  ``94263aa``), by digest — and the two-block turn with the rule taken out is ALSO the parent
  tree's bytes, so the rule is the only thing that moved.

Whether a model then rejects the misattributing draft is a model-backed measurement this file
does not make (no cloud, no GPU in this suite) — said, not implied.

Every name, channel id and message is invented. The digests below are the host's REAL renderer
output (``EngramConversationAuditor.digest``, cogno-host ``e4ae59b9``) over invented identities.
"""

from __future__ import annotations

import hashlib

from cogno_anima.stages.superego import (_HOMONYM_ATTRIBUTION_RULE, HOMONYM_DIGEST_RE,
                                         SuperegoStage)
from cogno_anima.types import EgoResult, EgoStep, ToolExecution
from tests.unit.test_superego import _ctx, _m

USER = "me faz um resumo da conversa com o Prof Dorival Quintanilha"

TWO = (
    "2 contacts match this name: Dorival Quintanilha (tg-4410, Telegram; exact name match) and "
    "Prof. Dorival Quintanilha (wa-8820, WhatsApp; exact name match). They are SEPARATE contact "
    "records: summarise EACH one on its own, labelled by its channel, and present them all — "
    "never merge them into one conversation, and never pick one of them silently.\n\n"
    "Recent conversations (4 turns, 2 contacts):\n"
    "- Dorival Quintanilha (tg-4410, Telegram): 2 turns; last: “a minha aula de quinta "
    "passou para a sala 9?”\n"
    "   User: preciso de um projector na sala 9\n"
    "   User: a minha aula de quinta passou para a sala 9?\n"
    "- Prof. Dorival Quintanilha (wa-8820, WhatsApp): 2 turns; last: “vou faltar na "
    "segunda por consulta médica”\n"
    "   User: quem me substitui na turma B?\n"
    "   User: vou faltar na segunda por consulta médica"
)

THREE = TWO.replace("2 contacts match this name: Dorival Quintanilha (tg-4410, Telegram; exact "
                    "name match) and",
                    "3 contacts match this name: Dorival Quintanilha (tg-4410, Telegram; exact "
                    "name match), Dorival Ramos (wa-8822, WhatsApp; no recent conversation) and")

ONE = ("Recent conversations (2 turns, 1 contacts):\n"
       "- Prof. Dorival Quintanilha (wa-8820): 2 turns; last: “vou faltar na segunda por "
       "consulta médica”\n"
       "   User: quem me substitui na turma B?\n"
       "   User: vou faltar na segunda por consulta médica")

#: The header quoted INSIDE a contact's message: the host renders it indented, mid-line.
QUOTED = ONE + "\n   User: o sistema disse 2 contacts match this name: é isso?"

#: The draft that did what the measured turn did: the Telegram block's facts given to the
#: WhatsApp contact.
WRONG = ("Resumo da conversa com o Prof. Dorival Quintanilha (WhatsApp): ele perguntou se a aula "
         "de quinta passou para a sala 9 e pediu um projector.")


def _audit(result: str, ok: bool = True) -> ToolExecution:
    return ToolExecution(tool="audit_conversations", arguments={"query": "Prof Dorival Quintanilha"},
                         ok=ok, result=result, error=None if ok else "upstream timeout")


def _turn(*calls: ToolExecution, consulted: "tuple[ToolExecution, ...]" = ()):
    ctx = _ctx(user=USER, intent_class="INFORMATION_REQUEST",
               goal="summarise the conversation with the named contact", with_ego=False)
    ctx.ego_result = EgoResult(
        steps=[EgoStep(index=0, path="native", assistant_text=WRONG, tool_calls=list(calls))],
        metrics=_m("ego"))
    if consulted:
        ctx.consult_result = EgoResult(
            steps=[EgoStep(index=0, path="native", assistant_text="", tool_calls=list(consulted))],
            persona="secretary", metrics=_m("ego"))
    return ctx


def _prompt(ctx) -> str:
    return SuperegoStage()._build_judge_prompt(ctx, "")


def _sha_of(system: str, prompt: str) -> str:
    return hashlib.sha256((system + "\x00" + prompt).encode()).hexdigest()


def _sha(ctx) -> str:
    """The digest of BOTH halves the judge sends: its system message and its prompt."""
    return _sha_of(SuperegoStage()._judge_system(ctx), _prompt(ctx))


# (name, context builder, digest of system + NUL + prompt on anima 94263aa — the parent tree)
_BASE = [
    ("one contact block", lambda: _turn(_audit(ONE)),
     "ea9686c7cf65c45e23a29775ae8b88625107069265f498f4b28db0a18d5b3904"),
    ("the header only quoted mid-line, inside a message", lambda: _turn(_audit(QUOTED)),
     "3d3283c2f0da97b00971e7c6266e62f6e0aa46f143b76783c9e3e50e8f5d3681"),
    ("a header that says ONE contact", lambda: _turn(_audit(TWO.replace("2 contacts", "1 contacts", 1))),
     "360dc9b7b552b6ff0a36203287a1514ff41ec437e246f97120e64b5367d09991"),
    ("the two-block digest on a call that FAILED", lambda: _turn(_audit(TWO, ok=False)),
     "a65f26bee1d767c78e85da1266a0ace38afc1b710a737d33cd4bdd02a164decd"),
    ("a turn with no digest at all", lambda: _ctx(user=USER),
     "d30090a5e63b937dc6704904b37b40439517349d98d62d16e6ef09b0fbf04918"),
]

#: The TWIN's turn, digested on the parent tree: the rule taken out, these bytes must come back.
_BASE_TWO = "03323f49e7e4c3d7f9988236f669901ab12589fc8269839f08775bfeb631276e"


# ── the condition ────────────────────────────────────────────────────────────────────


def test_the_condition_is_the_hosts_header_at_the_start_of_a_line_saying_at_least_two():
    assert HOMONYM_DIGEST_RE.search(TWO) and HOMONYM_DIGEST_RE.search(THREE)
    assert HOMONYM_DIGEST_RE.search("[source]\n12 contacts match this name: a and b")
    for absent in (ONE, QUOTED, TWO.replace("2 contacts", "1 contacts", 1),
                   TWO.replace("2 contacts", "02 contacts", 1), "", "Recent conversations"):
        assert not HOMONYM_DIGEST_RE.search(absent), absent


# ── the twin ─────────────────────────────────────────────────────────────────────────


def test_the_twin_two_contact_blocks_render_the_attribution_rule():
    prompt = _prompt(_turn(_audit(TWO)))
    assert _HOMONYM_ATTRIBUTION_RULE in prompt
    # a clean read is judged by the READ-ONLY criteria (the measured shape): the rule comes after
    # them, before the fixed rules
    assert SuperegoStage._judge_branch(_turn(_audit(TWO))) == "readonly"
    assert prompt.index("# This turn executed READS ONLY") < prompt.index(_HOMONYM_ATTRIBUTION_RULE) \
        < prompt.index("TRUST THE TOOLS")
    assert ("REJECT a draft that attributes to one contact anything that appears only in "
            "ANOTHER contact's block") in _HOMONYM_ATTRIBUTION_RULE
    assert "asking the user which of them they meant, is CORRECT" in _HOMONYM_ATTRIBUTION_RULE


def test_three_contact_blocks_render_it_too():
    assert _HOMONYM_ATTRIBUTION_RULE in _prompt(_turn(_audit(THREE)))


def test_a_digest_the_CONSULTED_specialist_read_renders_it_too():
    """Her results are rendered as this turn's evidence, so they carry the condition too."""
    ctx = _turn(consulted=(_audit(TWO),))
    prompt = _prompt(ctx)
    assert "# What the CONSULTED specialist executed" in prompt
    assert _HOMONYM_ATTRIBUTION_RULE in prompt


def test_the_twin_with_the_rule_taken_out_is_the_parent_trees_bytes():
    """Only the rule moved: the evidence, the criteria, the order — all as before."""
    ctx = _turn(_audit(TWO))
    stripped = _prompt(ctx).replace(_HOMONYM_ATTRIBUTION_RULE, "", 1)
    assert _HOMONYM_ATTRIBUTION_RULE not in stripped
    assert _sha_of(SuperegoStage()._judge_system(ctx), stripped) == _BASE_TWO


# ── the control ──────────────────────────────────────────────────────────────────────


def test_the_control_without_the_header_the_judge_is_byte_for_byte_the_parent_tree():
    for name, build, digest in _BASE:
        ctx = build()
        assert _HOMONYM_ATTRIBUTION_RULE not in _prompt(ctx), name
        assert _sha(ctx) == digest, name


def test_the_control_can_produce_the_presence_first():
    """A control that asserts an absence proves first that the same scaffold yields the rule."""
    assert _HOMONYM_ATTRIBUTION_RULE in _prompt(_turn(_audit(TWO)))
    assert _sha(_turn(_audit(TWO))) != _BASE_TWO


def test_the_rule_adds_no_row_to_the_inventory_and_does_not_touch_the_system_message():
    def slugs(prompt: str) -> list:
        return [row["block"] for row in SuperegoStage.judge_prompt_inventory(prompt)]
    one, two = _turn(_audit(ONE)), _turn(_audit(TWO))
    assert slugs(_prompt(one)) == slugs(_prompt(two))
    stage = SuperegoStage()
    assert stage._judge_system(one) == stage._judge_system(two)


def test_an_unreadable_record_contributes_nothing():
    class _Exploding:
        tool = "audit_conversations"
        arguments: dict = {}
        error = None

        @property
        def ok(self):
            raise RuntimeError("boom")
    from cogno_anima.stages.superego import _results_name_several_contacts
    assert _results_name_several_contacts([_Exploding()]) is False
    assert _results_name_several_contacts([_Exploding(), _audit(TWO)]) is True


def test_the_header_reader_is_exported_for_the_host_to_pin_its_own_header_against():
    """The header is the HOST's bytes and the condition is read here: a host test that renders
    its digest and asserts this matches is what keeps the two from drifting apart in silence."""
    import cogno_anima
    assert cogno_anima.HOMONYM_DIGEST_RE is HOMONYM_DIGEST_RE
    assert "HOMONYM_DIGEST_RE" in cogno_anima.__all__
