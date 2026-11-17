"""The voice may not DECLARE a refusal/limit an APPROVED draft does not — even with no figure.

The divergence backstop under `_draft_section`'s promise started as a FIGURE net: it compared
the delivered reply against the approved draft on decimal amounts only (`_FIGURE_RE`). Measured
afterwards: a turn whose approved draft is a text SUMMARY of what the contact asked for — no
figure in it at all — a hard rule in the prompt ("MUST NOT tell the user … could not"), review
`approved=true`, and no failed call, where the voice wrote the refusal anyway, anchored in two
earlier refusals that sat in its `# Context` block. The figure net saw nothing there (`lost=[]`)
and never fired.

This pins the TEXT twin of a lost figure: a reply that declares a refusal/limit from the closed
`_REFUSAL_LEXICON` (pt/en/es) the approved draft does NOT arms the SAME single re-voice. The
asymmetry is the whole guard — a limit the draft itself states is a TRUE limit the voice must
keep, so a reply that only repeats it is not re-voiced and no delivery the draft never promised
is ever forced.

Rare invented stand-ins throughout (this repo is public).
"""

from __future__ import annotations

import pytest

from cogno_anima.stages.superego import SuperegoStage
from tests.unit.test_voice_delivers_the_approved_answer import _turn, _voice

# An approved draft that DELIVERS an answer and carries NO figure: on the base tree the figure
# net reads nothing from it, so only the refusal dimension can arm the re-voice here.
_SUMMARY = ("Belmira Quostral is confirmed for the Vendrim consultation; the welcome pack and "
            "the parking note are included.")
_SUMMARY_PT = ("Belmira Quostral está confirmada para a consulta Vendrim; o pack de "
               "boas-vindas e a nota de estacionamento estão incluídos.")


# ── (1) a refusal over an approved draft that does NOT refuse → re-voice once ─────────

@pytest.mark.asyncio
async def test_a_refusal_with_no_figure_is_re_voiced_once():
    """RED on the base (`lost=[]` with no figure → no re-voice), GREEN here. The reply refuses;
    the approved summary does not. One re-voice, never a refusal and never a handoff."""
    result, b = await _voice(
        _turn(draft=_SUMMARY),
        ["Infelizmente, não consigo ajudar com esse pedido agora.", _SUMMARY_PT],
    )
    assert len(b.calls) == 2, "the refusal over the approved draft did not re-voice"
    assert "REWRITE" in b.calls[1]["prompt"]
    assert "could not do, provide, find, or access" in b.calls[1]["prompt"]
    assert result.response == _SUMMARY_PT
    assert "voice:diverged_from_approved_draft" in result.adjustments
    assert "voice:revoiced_from_approved_draft" in result.adjustments


@pytest.mark.asyncio
async def test_a_refusal_over_an_english_delivering_draft_is_re_voiced():
    """The positive in the REAL cross-language form: an English approved draft that DELIVERS, a
    Portuguese reply that refuses with "não consegui encontrar" — a form the first cut of the
    lexicon did not carry, so it caught neither the refusal nor, before the net existed, anything
    at all."""
    result, b = await _voice(
        _turn(draft=_SUMMARY),
        ["Infelizmente, não consegui encontrar informações sobre isso.", _SUMMARY_PT],
    )
    assert len(b.calls) == 2, "the refusal over the English draft did not re-voice"
    assert result.response == _SUMMARY_PT
    assert "voice:revoiced_from_approved_draft" in result.adjustments


# ── (2) COLLISION — a TRUE limit the draft itself states, in the REAL cross-language form ─

@pytest.mark.asyncio
async def test_a_true_limit_in_the_english_draft_kept_in_portuguese_is_not_re_voiced():
    """The asymmetry in the shape production actually has. The NOUMENO rewrites the request to
    canonical English and the executor drafts in it, so the APPROVED draft is English; the voice
    writes the contact's Portuguese. A true limit the English draft states ("I could not find…"),
    kept faithfully in the Portuguese reply ("Não encontrei…"), must NOT read as a refusal over
    the approved answer — re-voicing it would drive the voice to DROP the limit.

    With a narrow English half the draft's limit is not seen, the pt reply is read as a refusal
    over an approved answer that "does no such thing", and the net re-voices — the regression the
    wide English half closes."""
    result, b = await _voice(
        _turn(draft="I could not find any appointment for tomorrow."),
        ["Não encontrei nenhuma marcação para amanhã."],
    )
    assert len(b.calls) == 1, "a true English limit kept in Portuguese was re-voiced"
    assert not [a for a in result.adjustments if a.startswith("voice:diverged")]


@pytest.mark.asyncio
async def test_a_true_limit_the_draft_itself_states_is_not_re_voiced_same_language():
    """The same-language control: approved draft and reply both in Portuguese, both stating the
    limit. Kept as the simpler twin beside the real cross-language case above."""
    draft = ("Não consigo confirmar o reembolso até o Jardel Vexmoril aprovar; o pedido fica "
             "pendente.")
    result, b = await _voice(
        _turn(draft=draft),
        ["No momento não consigo confirmar o reembolso; aguardamos a aprovação do Jardel "
         "Vexmoril."],
    )
    assert len(b.calls) == 1, "a true limit the draft states was re-voiced"
    assert not [a for a in result.adjustments if a.startswith("voice:diverged")]


# ── (3) controls ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_reply_that_delivers_the_approved_summary_is_left_alone():
    """The discrimination twin: a faithful delivery of a figure-free draft is not re-voiced."""
    result, b = await _voice(_turn(draft=_SUMMARY), [_SUMMARY_PT])
    assert len(b.calls) == 1, "the net re-voiced a faithful reply"
    assert not [a for a in result.adjustments if a.startswith("voice:diverged")]


@pytest.mark.asyncio
async def test_the_figure_path_still_re_voices_a_lost_amount():
    """The figure dimension is untouched: a reply that drops an approved amount still re-voices,
    exactly as before this change."""
    result, b = await _voice(
        _turn(),  # the figure draft: R$ 135,00 / 35,00 / 50,00
        ["A taxa é R$ 135,00 e o bónus R$ 35,00.",
         "A taxa é R$ 135,00 por hora, com bónus de R$ 35,00 e R$ 50,00."],
    )
    assert len(b.calls) == 2
    assert "voice:revoiced_from_approved_draft" in result.adjustments
    assert "50,00" in result.response


# ── the pure predicates, exercised directly ──────────────────────────────────────────

def test_the_closed_lexicon_reads_all_three_languages():
    d = SuperegoStage._declares_refusal
    assert d("Infelizmente não consigo ajudar.")
    assert d("não tenho como fazer isso")
    assert d("I'm unable to help with that")
    assert d("I cannot share that")
    assert d("no puedo proporcionar esa información")
    assert d("no encontré ese registro")
    # the typographic apostrophe is folded to a straight one, not a new alphabet
    assert d("Sorry, I can’t provide that")
    # a plain delivery declares nothing
    assert not d("A consulta está confirmada para amanhã.")
    assert not d("Your order shipped and arrives Thursday.")


def test_every_widened_phrase_is_read():
    """Each phrase the English draft is likely to carry — the half that was too narrow — plus the
    two Portuguese additions. The English draft is the one the asymmetry reads for a limit."""
    d = SuperegoStage._declares_refusal
    english = [
        "I can't do that right now", "I cannot help with this",
        "I could not find any appointment for tomorrow", "I couldn't locate the record",
        "I did not find a matching booking", "I didn't find anything",
        "I was unable to confirm it", "I am unable to confirm it",
        "I'm not able to confirm it", "I am not able to confirm it",
        "I do not have that information", "I don't have that on file",
        "That is not possible this week", "That isn't possible this week",
    ]
    for s in english:
        assert d(s), s
    assert d("Não consegui encontrar informações sobre isso")
    assert d("Não pude confirmar a marcação")


def test_the_refusal_divergence_is_asymmetric():
    r = SuperegoStage._refuses_over_approved
    # reply refuses, draft does not → divergence
    assert r("A consulta está confirmada.", "não consigo confirmar isso") is True
    # draft refuses too → the limit is true, kept, not a divergence
    assert r("não consigo confirmar isso", "ainda não consigo confirmar") is False
    # neither refuses
    assert r("confirmado", "confirmado para amanhã") is False
    # the REAL cross-language shape: an English draft states the limit, the Portuguese reply
    # keeps it → NOT a divergence (the defect the widened English half fixes)
    assert r("I could not find any appointment for tomorrow.",
             "Não encontrei nenhuma marcação para amanhã.") is False
    # ...and an English draft that DELIVERS, refused in Portuguese → a divergence
    assert r("Your appointment is confirmed for Thursday at 9.",
             "não consegui encontrar informações") is True
