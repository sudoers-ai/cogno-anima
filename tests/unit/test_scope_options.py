"""The "did you mean…?" selector over a CLOSED list (``cogno_anima.stages.scope_options``).

Every name here is invented. The shapes are the ones measured downstream: a refused message whose
answer the persona HELD (the false refusal → ``covered`` → let it through, never a question), a
message about a neighbouring thing (``suggested`` → the closed question, rendered by the host), a
message about something nobody offers ("the Wi-Fi password" → nothing), and a model that returns an
option it was never shown (discarded by the code).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from cogno_anima import metakeys as mk
from cogno_anima.stages import scope_options as so
from cogno_anima.types import PipelineContext

OPTIONS = ("Calendário acadêmico", "Oficina de Fotografia", "Mensalidade e descontos",
           "consult_material")


class Selector:
    """A stub selector backend: answers ``reply`` (a dict is JSON-encoded), counts its calls and
    keeps the prompts it was shown."""

    def __init__(self, reply, *, tokens=(40, 9), raises: "Exception | None" = None):
        self.reply = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        self.tokens = tokens
        self.raises = raises
        self.calls: list[tuple[str, str]] = []
        self.model = "stub-selector"

    async def generate(self, system: str, prompt: str):
        self.calls.append((system, prompt))
        if self.raises is not None:
            raise self.raises
        return self.reply, *self.tokens


def run(coro):
    return asyncio.run(coro)


# ── the twin: a neighbouring option → SUGGESTED, the closed question ───────────────────────

def test_twin_a_neighbouring_option_is_suggested_from_the_list():
    sel = Selector({"answers": [], "maybe": ["Oficina de Fotografia"], "asked": "curso de Fotografia Digital"})
    got = run(so.select_options("Quando começa o curso de Fotografia Digital?", OPTIONS, sel))
    assert got.outcome == so.OUTCOME_SUGGESTED
    assert got.suggested == ("Oficina de Fotografia",)
    assert got.covered == ()
    assert got.asked == "curso de Fotografia Digital"
    assert got.discarded == 0 and got.offered == len(OPTIONS) and got.called


def test_twin_control_the_same_call_with_nothing_picked_is_none():
    # The twin's other world: same message, same list, the selector finds nothing.
    sel = Selector({"answers": [], "maybe": [], "asked": ""})
    got = run(so.select_options("Quando começa o curso de Fotografia Digital?", OPTIONS, sel))
    assert got.outcome == so.OUTCOME_NONE
    assert got.suggested == () and got.covered == ()


# ── the closed alphabet is ENFORCED ──────────────────────────────────────────────────────

def test_an_option_outside_the_list_is_discarded_and_counted():
    sel = Selector({"answers": [], "maybe": ["Fotografia Avançada"], "asked": ""})
    got = run(so.select_options("Tem curso de fotografia avançada?", OPTIONS, sel))
    assert got.outcome == so.OUTCOME_NONE
    assert got.suggested == ()
    assert got.discarded == 1


def test_an_outside_option_beside_a_valid_one_leaves_only_the_valid_one():
    sel = Selector({"answers": [], "maybe": ["Fotografia Avançada", "Oficina de Fotografia", 7]})
    got = run(so.select_options("Tem curso de fotografia avançada?", OPTIONS, sel))
    assert got.outcome == so.OUTCOME_SUGGESTED
    assert got.suggested == ("Oficina de Fotografia",)
    assert got.discarded == 2  # the invented title and the non-string


def test_a_near_miss_spelling_is_not_the_option():
    # Case or one letter different is ANOTHER string: the host renders exactly what it offered.
    sel = Selector({"answers": ["calendário acadêmico"], "maybe": []})
    got = run(so.select_options("Quando começam as aulas?", OPTIONS, sel))
    assert got.outcome == so.OUTCOME_NONE and got.discarded == 1


def test_surrounding_whitespace_is_not_a_difference():
    sel = Selector({"answers": ["  Calendário   acadêmico "], "maybe": []})
    got = run(so.select_options("Onde vejo o calendário acadêmico?", OPTIONS, sel))
    assert got.covered == ("Calendário acadêmico",)


# ── a false refusal passes, it is never turned into a question (1a) ─────────────────────

def test_an_option_about_the_very_thing_asked_is_covered():
    sel = Selector({"answers": ["Calendário acadêmico"], "maybe": []})
    got = run(so.select_options("Qual é o calendário acadêmico do semestre?", OPTIONS, sel))
    assert got.outcome == so.OUTCOME_COVERED
    assert got.covered == ("Calendário acadêmico",)
    assert got.suggested == ()


def test_covered_wins_over_every_suggestion():
    sel = Selector({"answers": ["Calendário acadêmico"], "maybe": ["Oficina de Fotografia"]})
    got = run(so.select_options("Qual é o calendário acadêmico do semestre?", OPTIONS, sel))
    assert got.outcome == so.OUTCOME_COVERED
    assert got.suggested == ()  # no question rides beside a pass


def test_an_option_in_both_lists_is_covered():
    sel = Selector({"answers": ["Oficina de Fotografia"], "maybe": ["Oficina de Fotografia"]})
    got = run(so.select_options("Como funciona a oficina de fotografia?", OPTIONS, sel))
    assert got.outcome == so.OUTCOME_COVERED
    assert got.covered == ("Oficina de Fotografia",) and got.suggested == ()


def test_at_most_two_picks_covered_first():
    sel = Selector({"answers": [], "maybe": list(OPTIONS)})
    got = run(so.select_options("Quero saber tudo", OPTIONS, sel))
    assert len(got.suggested) == so.MAX_PICKS == 2


# ── the negative: something nobody offers draws nothing ─────────────────────────────────

def test_the_wifi_password_draws_no_option():
    sel = Selector({"answers": [], "maybe": [], "asked": "senha do Wi-Fi"})
    got = run(so.select_options("Qual é a senha do Wi-Fi da escola?", OPTIONS, sel))
    assert got.outcome == so.OUTCOME_NONE
    assert got.covered == () and got.suggested == ()


# ── ``asked`` can only ever be the contact's own words ───────────────────────────────────

@pytest.mark.parametrize("asked, kept", [
    ("curso de Fotografia Digital", "curso de Fotografia Digital"),
    ("curso de fotografia digital", "curso de fotografia digital"),   # case folded
    ("curso de Fotografía  Digital", "curso de Fotografía Digital"),  # accents + spaces folded
    ("Fotografia Digital para iniciantes", ""),                        # not in the message
    ("x" * (so.MAX_ASKED_CHARS + 1), ""),                              # too long
    (["curso"], ""),                                                  # not a string
])
def test_asked_survives_only_when_it_is_in_the_message(asked, kept):
    message = "Quando começa o curso de Fotografia Digital?"
    sel = Selector({"answers": [], "maybe": ["Oficina de Fotografia"], "asked": asked})
    got = run(so.select_options(message, OPTIONS, sel))
    assert got.asked == kept


# ── errors are the refusal of today (every caller treats ``error`` like ``none``) ──────────

@pytest.mark.parametrize("reply", [
    "I think the answer is the photography workshop.",       # no JSON
    json.dumps({"options": ["Oficina de Fotografia"]}),       # the one-list shape: not this schema
    json.dumps({"answers": "Oficina de Fotografia"}),         # a list that is not a list
    json.dumps(["Oficina de Fotografia"]),                    # not an object
])
def test_an_unusable_answer_is_error_with_no_option(reply):
    got = run(so.select_options("Tem oficina de fotografia?", OPTIONS, Selector(reply)))
    assert got.outcome == so.OUTCOME_ERROR
    assert got.covered == () and got.suggested == ()


def test_a_backend_that_raises_is_error_and_does_not_raise():
    sel = Selector({}, raises=RuntimeError("provider down"))
    got = run(so.select_options("Tem oficina de fotografia?", OPTIONS, sel))
    assert got.outcome == so.OUTCOME_ERROR and got.called
    assert got.metrics.tokens_in == 0


def test_cancellation_still_unwinds():
    sel = Selector({}, raises=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        run(so.select_options("Tem oficina de fotografia?", OPTIONS, sel))


# ── nothing to select from → no call at all ─────────────────────────────────────────────

@pytest.mark.parametrize("options, message", [((), "Tem oficina?"), (None, "Tem oficina?"),
                                              (OPTIONS, "   ")])
def test_no_options_or_no_message_makes_no_call(options, message):
    sel = Selector({"answers": ["Oficina de Fotografia"], "maybe": []})
    got = run(so.select_options(message, options, sel))
    assert got.outcome == so.OUTCOME_NONE and not got.called
    assert sel.calls == []


# ── the prompt and the ledger line ──────────────────────────────────────────────────────

def test_the_user_half_is_the_measured_shape():
    sel = Selector({"answers": [], "maybe": []})
    run(so.select_options("Tem oficina de fotografia?", OPTIONS, sel))
    (system, prompt), = sel.calls
    assert prompt == ("Message: Tem oficina de fotografia?\n\nOptions:\n"
                      + "\n".join(f"- {o}" for o in OPTIONS))
    assert "STRICT" in system and "Never invent an option" in system
    assert '"answers"' in system and '"maybe"' in system


def test_the_call_has_its_own_ledger_line():
    sel = Selector({"answers": [], "maybe": []}, tokens=(123, 11))
    got = run(so.select_options("Tem oficina de fotografia?", OPTIONS, sel))
    assert got.metrics.stage == so.SELECT_STAGE == "superego_select"
    assert (got.metrics.tokens_in, got.metrics.tokens_out) == (123, 11)
    assert got.metrics.model == "stub-selector"


# ── the host's list, made safe ──────────────────────────────────────────────────────────

def test_closed_options_flattens_dedupes_and_drops_what_it_cannot_offer_whole():
    long = "y" * (so.MAX_OPTION_CHARS + 1)
    got = so.closed_options(["Oficina de\nFotografia", "Oficina de Fotografia", "", "  ", 3,
                             None, long, "Mensalidade"])
    assert got == ("Oficina de Fotografia", "Mensalidade")
    assert so.closed_options("Um título só") == ("Um título só",)
    assert so.closed_options(None) == () and so.closed_options(42) == ()


# ── the per-turn stamp ──────────────────────────────────────────────────────────────────

def test_select_scope_options_reads_the_contacts_message_and_stamps_the_record():
    ctx = PipelineContext(user_input="Quando começa o curso de Fotografia Digital?")
    sel = Selector({"answers": [], "maybe": ["Oficina de Fotografia", "Inventado"],
                    "asked": "curso de Fotografia Digital"})
    got = run(so.select_scope_options(ctx, sel, options=OPTIONS))
    assert "Fotografia Digital" in sel.calls[0][1]
    assert ctx.metadata[mk.SCOPE_OPTIONS_SELECTION] == {
        "outcome": "suggested", "covered": [], "suggested": ["Oficina de Fotografia"],
        "asked": "curso de Fotografia Digital", "offered": len(OPTIONS), "discarded": 1,
        "covered_unsupported": 0}
    assert got.record() == ctx.metadata[mk.SCOPE_OPTIONS_SELECTION]
    assert set(got.record()["suggested"]) <= set(OPTIONS)


def test_at_most_two_covered_and_then_no_room_for_a_suggestion():
    sel = Selector({"answers": list(OPTIONS[:3]), "maybe": [OPTIONS[3]]})
    got = run(so.select_options("Calendário, oficina de fotografia e mensalidade?", OPTIONS, sel))
    assert got.covered == OPTIONS[:2]
    assert got.suggested == ()


# ── ``covered`` needs EVIDENCE in the code (the nightly: qwen3 covered the Wi-Fi 3/3) ────────

WIFI = "Qual é a senha do Wi-Fi da escola?"


def test_twin_a_covered_wifi_without_evidence_is_the_refusal_of_today():
    """The measured shape: the local model answered the Wi-Fi with a capability as ``covered``.
    The capability shares no term with the question, so the pick is dropped and counted."""
    sel = Selector({"answers": ["consult_material"], "maybe": [], "asked": "senha do Wi-Fi"})
    got = run(so.select_options(WIFI, OPTIONS, sel))
    assert got.outcome == so.OUTCOME_NONE
    assert got.covered == () and got.suggested == ()
    assert got.covered_unsupported == 1
    assert got.record()["covered_unsupported"] == 1


def test_control_a_false_refusal_whose_section_shares_a_term_passes():
    sel = Selector({"answers": ["Oficina de Fotografia"], "maybe": []})
    got = run(so.select_options("Quando começa a oficina de fotografia?", OPTIONS, sel))
    assert got.outcome == so.OUTCOME_COVERED and got.covered_unsupported == 0


def test_control_a_section_sharing_only_generic_words_is_the_refusal():
    options = OPTIONS + ("Alunos da faculdade",)
    sel = Selector({"answers": ["Alunos da faculdade"], "maybe": []})
    got = run(so.select_options("A faculdade tem academia de ginástica para os alunos?",
                                options, sel))
    assert got.outcome == so.OUTCOME_NONE and got.covered_unsupported == 1


def test_an_unsupported_covered_pick_is_never_turned_into_a_question():
    sel = Selector({"answers": ["consult_material"], "maybe": ["Oficina de Fotografia"]})
    got = run(so.select_options(WIFI, OPTIONS, sel))
    assert got.outcome == so.OUTCOME_NONE and got.suggested == ()


def test_a_supported_covered_pick_survives_beside_an_unsupported_one():
    sel = Selector({"answers": ["consult_material", "Oficina de Fotografia"], "maybe": []})
    got = run(so.select_options("Tem oficina de fotografia?", OPTIONS, sel))
    assert got.covered == ("Oficina de Fotografia",) and got.covered_unsupported == 1


def test_suggested_does_not_need_evidence():
    sel = Selector({"answers": [], "maybe": ["Mensalidade e descontos"]})
    got = run(so.select_options("Tem bolsa para atletas?", OPTIONS, sel))
    assert got.outcome == so.OUTCOME_SUGGESTED and got.covered_unsupported == 0


@pytest.mark.parametrize("message, option, expected", [
    ("Quando começa a oficina de fotografia?", "Oficina de Fotografia", True),
    ("Qual o valor das mensalidades?", "Mensalidade e descontos", True),      # prefix + plural
    ("Onde leio os materiais?", "consult_material", True),                     # a tool's words
    ("Quando é a prova de informática?", "Informática básica", True),          # not a frame word
    (WIFI, "consult_material", False),
    ("A escola tem aula no sábado?", "Aulas da escola", False),                 # frame words only
    ("", "Oficina de Fotografia", False),
])
def test_has_evidence(message, option, expected):
    assert so.has_evidence(message, option) is expected


def test_without_the_tokenizer_no_covered_pick_has_evidence(monkeypatch):
    """Fail-CLOSED: an environment without ``cogno_engram`` never lifts a refusal."""
    import builtins
    real = builtins.__import__

    def _no_engram(name, *a, **kw):
        if name.startswith("cogno_engram"):
            raise ImportError(name)
        return real(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", _no_engram)
    assert so.has_evidence("Quando começa a oficina de fotografia?", "Oficina de Fotografia") is False
