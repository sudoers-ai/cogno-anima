"""The outgoing-PII rule: "may come IN, must never go OUT" — except the contact's own.

The owner's decision (2026-08-25) has three parts and each one is a way this feature dies:

* **Redact in place, never refuse.** A refusal costs the contact their answer and hands the turn
  to the re-voicing loop, which has already been measured shipping a handoff instead of a
  booking. Every test here asserts on the SHIPPED SENTENCE, not on a verdict.
* **Provenance decides.** A value the contact themselves supplied may be said back; anything from
  configuration, memory, a third party's record or a tool result may not.
* **Per SESSION, not per turn.** Measured per SCENARIO against the merged bench's answer key
  (host #549, `a2aef70`): the per-turn variant additionally breaks
  `own_email_recalled_three_turns_later` — 1 of the 5 good-reply scenarios — and blocks exactly
  zero extra leaks. (That bench's first per-RUN headline was withdrawn by its own author as
  arithmetic; the unit is the scenario, or the decidable run.)

And one property that is not the owner's decision but the correction the director made to it: the
session allowlist carries **digests, never values**, so closing the outgoing leak does not open a
new store of personal data in `metadata`, the trace or the logs. `test_the_feature_itself_leaks_
nothing` is the negative control of that correction.

All PII here is SYNTHETIC and born in this file: `example.com` is RFC-reserved, the phone digits
are sequential, and the two CPFs are decade-old public Brazilian test fixtures
(`529.982.247-25`, `111.444.777-35`). The checksums have to be VALID or the detector does not see
them and the test would measure a clean run it never made.
"""

from __future__ import annotations

import logging
import re

import pytest

from cogno_anima import metakeys as mk
from cogno_anima import vocab
from cogno_anima.security.detector import default_detector
from cogno_anima.security.pii import VALID_PII_TYPES
from cogno_anima.security.redaction import (
    MAX_ALLOWLIST_DIGESTS,
    ProvenanceContext,
    normalize_pii_value,
    pii_digest,
    pii_digests_in,
    redact_pii,
    sanitize_digests,
)
from cogno_anima.stages.superego import SuperegoStage
from tests.unit.test_superego import ScriptedBackend, _ctx

DETECTOR = default_detector()

CONTACT_EMAIL = "ana.souza@example.com"
STAFF_CPF = "529.982.247-25"          # the doctor's — configuration, never the contact's
THIRD_PARTY_PHONE = "(11) 98888-7777"  # another patient's — from memory
CLIENT_CPF = "111.444.777-35"          # came back in a tool result


def _digests(*texts: str) -> list[str]:
    """What the HOST would accumulate for the session: digests of the contact's own words."""
    out: set[str] = set()
    for t in texts:
        out |= pii_digests_in(t, DETECTOR)
    return sorted(out)


async def _voice(user: str, reply: str, *, session=None, mode=vocab.PII_MODE_ENFORCE,
                 role=None):
    """One turn, with the reply text FIXED.

    Deterministic on purpose: the merged bench measured the model refusing to repeat the
    contact's own e-mail in 2 of 3 runs all by itself, so a test that let the model choose what
    to say would pass for the wrong reason — it would be measuring the model's reticence, not
    this rule. Every case here hands the voicer the exact sentence to produce.

    `mode` defaults to ENFORCE in this suite while the SHIPPED default is `observe`; the default
    itself is asserted separately (`test_the_shipped_default_is_observation`), so the two facts
    cannot be confused for one another.
    """
    ctx = _ctx(user=user)
    if session is not None:
        ctx.metadata[mk.PII_OUTPUT_ALLOWLIST] = session
    if mode is not None:
        ctx.metadata[mk.PII_OUTPUT_MODE] = mode
    if role is not None:
        ctx.metadata[mk.PII_READER_ROLE] = role
    res = await SuperegoStage().voice(ctx, ScriptedBackend([reply]), voice_prompt="persona")
    return ctx, res


# ── Branch 1 — the contact's own value, this turn (MUST PASS) ────────────────────────────────

@pytest.mark.asyncio
async def test_the_agent_may_confirm_back_the_email_the_contact_just_gave():
    """THE false positive that kills the feature, so it gets its own name.

    A contact types their e-mail and the agent confirms it: repeating it IS the answer. A rule
    that masks here is not a strict rule, it is a broken assistant — and the first person to see
    `[EMAIL REDACTED]` where their own address should be will report the protection as the bug.
    """
    _, res = await _voice(f"meu email e {CONTACT_EMAIL}",
                          f"Perfeito! Registrei {CONTACT_EMAIL} no seu cadastro.")
    assert res.response == f"Perfeito! Registrei {CONTACT_EMAIL} no seu cadastro."
    assert "pii:redacted_in_output" not in res.adjustments
    assert [(f.pii_type, f.provenance, f.redacted) for f in res.pii_findings] == [
        ("EMAIL", vocab.PII_PROVENANCE_TURN, False)]


@pytest.mark.asyncio
async def test_punctuation_does_not_defeat_the_contacts_own_document():
    """Comparison is on the normalised VALUE, not the raw substring.

    The contact writes `52998224725`, the agent formats it `529.982.247-25`. Same CPF, and a rule
    that matches on the substring masks the reply — the identical failure with a phone written
    `11988887777` and read back `(11) 98888-7777`.
    """
    _, res = await _voice("meu cpf e 52998224725",
                          f"Confirmado, CPF {STAFF_CPF} registrado.")
    assert STAFF_CPF in res.response
    assert not res.pii_findings[0].redacted


# ── Branch 2 — the contact's value from an EARLIER turn (MUST PASS, via digests) ──────────────

@pytest.mark.asyncio
async def test_the_email_given_at_turn_1_may_be_recalled_at_turn_3_WITH_DIGESTS():
    """The measured cost of a per-TURN rule, and the reason the host keeps a session memory.

    "Qual email ficou no meu cadastro?" — given at turn 1, asked at turn 3. Per-turn, the answer
    is `[EMAIL REDACTED]`. It is ONE scenario in the merged bench, red in all three of its runs,
    and it is the only place the two variants disagree — the per-turn rule buys no extra leak
    stopped anywhere in the suite. (Never quote that bench's first per-RUN headline: its own
    author withdrew it as arithmetic. A wrong number in a TEST is worse than in a comment — it
    reads as measured.)

    It must pass **through digests**, not only through values: if the two ends normalise
    differently the value-based version of this test would still pass and production would mask
    every recall. That is why the allowlist here is built by the same
    `pii_digests_in` the host calls, over the turn-1 text, and never contains the address.
    """
    session = _digests(f"meu email e {CONTACT_EMAIL}")
    assert CONTACT_EMAIL not in "".join(session)
    _, res = await _voice("qual email ficou no meu cadastro?",
                          f"O e-mail no seu cadastro e {CONTACT_EMAIL}.",
                          session=session)
    assert res.response == f"O e-mail no seu cadastro e {CONTACT_EMAIL}."
    assert [f.provenance for f in res.pii_findings] == [vocab.PII_PROVENANCE_SESSION]
    assert "pii:provenance_contact_session" in res.adjustments


@pytest.mark.asyncio
async def test_the_same_recall_is_masked_when_the_host_remembers_nothing():
    """The twin of the test above — without it, the one above cannot tell "the allowlist worked"
    from "nothing was ever masked". Absence of a host allowlist makes the rule STRICTER."""
    _, res = await _voice("qual email ficou no meu cadastro?",
                          f"O e-mail no seu cadastro e {CONTACT_EMAIL}.")
    assert CONTACT_EMAIL not in res.response
    assert res.response == "O e-mail no seu cadastro e [EMAIL REDACTED]."


# ── Branch 3 — everything else (MUST BE MASKED) ──────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("source,user,reply,leaked", [
    ("config",
     "quem e a responsavel tecnica?",
     f"A responsavel e a Dra. Ana, CPF {STAFF_CPF}.", STAFF_CPF),
    ("memory",
     "com quem eu falei sobre a consulta?",
     f"Voce falou com o Carlos, telefone {THIRD_PARTY_PHONE}.", THIRD_PARTY_PHONE),
    ("third_party",
     "qual o contato do outro paciente?",
     f"O contato dele e {THIRD_PARTY_PHONE}.", THIRD_PARTY_PHONE),
    ("tool_result",
     "qual o cpf do cliente do lancamento?",
     f"O lancamento e do cliente de CPF {CLIENT_CPF}.", CLIENT_CPF),
])
async def test_pii_the_contact_never_supplied_is_masked_in_place(source, user, reply, leaked):
    """One case per carrier the bench named. The contact asked; the value is somebody else's.

    The assertion is deliberately two-sided: the value is GONE **and** a sentence still ships.
    Half of the decision is "never refuse", so a test that only checked the value's absence would
    pass just as happily on a rule that returned an empty reply.
    """
    _, res = await _voice(user, reply)
    assert leaked not in res.response, source
    assert res.response and res.response != reply
    assert res.response.endswith(".") and len(res.response) > 20, "a sentence still ships"
    assert "pii:redacted_in_output" in res.adjustments
    assert "pii:provenance_not_from_contact" in res.adjustments
    assert [f.redacted for f in res.pii_findings] == [True]


@pytest.mark.asyncio
async def test_one_reply_can_carry_both_verdicts_at_once():
    """The rule is per VALUE, not per reply. A reply that confirms the contact's own e-mail and
    also names a third party's phone must keep the first and lose the second — a per-reply
    decision gets one of the two wrong whichever way it falls."""
    _, res = await _voice(f"meu email e {CONTACT_EMAIL}, quem me atendeu?",
                          f"Registrei {CONTACT_EMAIL}. Quem atendeu foi o Carlos, "
                          f"telefone {THIRD_PARTY_PHONE}.")
    assert CONTACT_EMAIL in res.response
    assert THIRD_PARTY_PHONE not in res.response
    assert sorted(f.provenance for f in res.pii_findings) == [
        vocab.PII_PROVENANCE_TURN, vocab.PII_PROVENANCE_UNKNOWN]
    assert "pii:provenance_contact_turn" in res.adjustments
    assert "pii:provenance_not_from_contact" in res.adjustments


# ── The correction: the feature must not become a PII transport itself ───────────────────────

@pytest.mark.asyncio
async def test_the_feature_itself_leaks_nothing(caplog):
    """Negative control of the digest design: no personal VALUE may appear in the trace, in the
    session allowlist, or in a log line *because of this feature*.

    The three carriers are named because all three have leaked before: `metadata` is serialised
    into the turn trace and persisted as session state, `SuperegoResult` is persisted into
    `turn_traces`, and a best-effort guard rendering its kwargs into an event is exactly how a
    third PII-carrying log line reached the security review.

    `prompt_text`/`response` are excluded on purpose — the reply is the contact's own answer and
    the rendered prompt is documented as in-memory-only. What is under test is what this feature
    ADDS.
    """
    caplog.set_level(logging.DEBUG)
    session = _digests(f"meu email e {CONTACT_EMAIL}")
    ctx, res = await _voice("qual o cpf da responsavel e o meu email?",
                            f"A responsavel tem CPF {STAFF_CPF} e o seu e-mail e "
                            f"{CONTACT_EMAIL}.", session=session)
    # The turn must actually have exercised BOTH branches, or this control proves nothing: a
    # turn that masked nothing and allowed nothing leaks nothing trivially.
    assert sorted(f.provenance for f in res.pii_findings) == [
        vocab.PII_PROVENANCE_SESSION, vocab.PII_PROVENANCE_UNKNOWN]

    # ...and the log arm has to have CAPTURED something, or `caplog.text` is "" and a
    # value-carrying line would sail through the loop below. Setting `propagate = False` on the
    # stage logger is the mutation this closes.
    assert caplog.records and "pii_redacted_in_output" in caplog.text

    record = res.model_dump(exclude={"response", "prompt_text"})
    blob = repr(record) + repr(ctx.metadata) + caplog.text
    for value in (STAFF_CPF, "52998224725", CONTACT_EMAIL, normalize_pii_value(CONTACT_EMAIL)):
        assert value not in blob, f"{value!r} escaped into trace/metadata/logs"
    # ...and the allowlist that made the e-mail shippable is digests all the way down.
    assert all(re.fullmatch(r"[0-9a-f]{64}", d)
               for d in ctx.metadata[mk.PII_OUTPUT_ALLOWLIST])


def test_a_host_that_injects_VALUES_gets_no_allowlist():
    """The metadata channel refuses to carry values at all, so a host that misreads the contract
    fails toward MASKING instead of quietly turning the whole allowlist into cleartext PII."""
    assert sanitize_digests([CONTACT_EMAIL, STAFF_CPF, "52998224725"]) == frozenset()


@pytest.mark.parametrize("garbage", [None, "not-a-list", 42, {"a": 1}, [None, 3, ""],
                                     ["ZZ" * 32], ["abc"], [pii_digest("x@y.zz", "EMAIL").upper()]])
def test_garbage_never_raises_and_never_widens(garbage):
    """Absence and garbage both mean "no earlier-session values" — stricter, never off. A typo in
    a metadata key must not be able to switch the protection off, and a malformed allowlist must
    not be able to abort a turn."""
    out = sanitize_digests(garbage)
    assert isinstance(out, frozenset)
    assert all(len(d) == 64 for d in out)


def test_the_allowlist_is_bounded():
    """It arrives from metadata, so its size is whatever a host accumulated; the per-turn cost
    must not be a function of that."""
    assert len(sanitize_digests([f"{i:064x}" for i in range(MAX_ALLOWLIST_DIGESTS * 3)])) \
        == MAX_ALLOWLIST_DIGESTS


# ── The primitives ───────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b", [
    ("529.982.247-25", "52998224725"),
    ("(11) 98888-7777", "11988887777"),
    ("Ana.Souza@Example.COM", "ana.souza@example.com"),
    ("  ana.souza@example.com  ", "ana.souza@example.com"),
])
def test_the_same_value_written_differently_digests_the_same(a, b):
    kind = "EMAIL" if "@" in a else "NATIONAL_ID"
    assert pii_digest(a, kind) == pii_digest(b, kind) != ""


@pytest.mark.parametrize("a,b", [
    ("529.982.247-25", "111.444.777-35"),
    ("ana@example.com", "ana@exemplo.com"),
    ("josé@example.com", "jose@example.com"),   # an accent must NOT be folded away
    ("ana@example.com", "an@aexample.com"),    # nor may punctuation be deleted: different hosts
    ("ana.souza@example.com", "anasouza@example.com"),
    ("(52) 99822-4725", "529.982.247-25"),     # same digits, DIFFERENT type → different digest
])
def test_different_values_digest_differently(a, b):
    """A digest collision here is a FALSE ALLOW — somebody else's value shipped because it
    normalised onto the contact's, for the rest of the session.

    Three families, all of them measured rather than imagined. Folding accents to be tidy buys
    the first. Deleting punctuation from an e-mail — which the first version did, uniformly —
    buys the next two: `ana@example.com` and `an@aexample.com` are different HOSTS. And the last
    is why the TYPE is in the digest: a contact's own phone and a stranger's CPF reduce to the
    same eleven digits, and about one BR mobile in a hundred lands on a checksum-valid CPF.
    """
    types = {"@": "EMAIL"}
    ta = next((v for k, v in types.items() if k in a), "NATIONAL_ID")
    tb = next((v for k, v in types.items() if k in b), "PHONE" if a != b and "@" not in a
              else "NATIONAL_ID")
    assert pii_digest(a, ta) != pii_digest(b, tb)


@pytest.mark.parametrize("blank", ["", "   ", "\t\n", None])
def test_the_empty_value_is_never_digested(blank):
    """One empty entry in an allowlist would otherwise match every value that normalises to
    nothing — "matches everything" is the one state an allowlist must not reach by accident."""
    assert pii_digest(blank, "EMAIL") == ""


def test_find_and_detect_are_the_same_definition():
    """`detect` is expressed over `find`; the redaction path and the flagging path can therefore
    never disagree about what counts as PII. A second set of regexes for spans is the failure
    this asserts against."""
    text = (f"CPF {STAFF_CPF}, outro {CLIENT_CPF}, mail {CONTACT_EMAIL}, "
            f"fone {THIRD_PARTY_PHONE}, protocolo 123.456.789-00")
    # Written out independently. `detect() == dict.fromkeys(find())` would be a tautology — both
    # sides call the same method, so it stays green with a `find()` that returns nothing at all,
    # or that labels every match EMAIL. A literal is the only side that cannot move with the code.
    assert DETECTOR.detect(text) == ["NATIONAL_ID", "PHONE", "EMAIL"]
    values = [m.value for m in DETECTOR.find(text)]
    assert STAFF_CPF in values and CLIENT_CPF in values
    assert "123.456.789-00" not in values, "checksum-invalid: never a document, never masked"


def test_every_value_is_decided_not_just_the_first():
    """`detect` stops at the first hit per pattern (types are all it needs). Masking one of two
    CPFs is not a protection, so `find` must not inherit that shortcut."""
    text = f"CPF {STAFF_CPF} e CPF {CLIENT_CPF}"
    out = redact_pii(text, detector=DETECTOR, mode=vocab.PII_MODE_ENFORCE)
    assert STAFF_CPF not in out.text and CLIENT_CPF not in out.text
    assert len(out.findings) == 2


# A sample per PII type the SHIPPED detector can produce. It is a DUPLICATED CONTRACT — the
# types live in the pattern packs, these strings live here — so the test below pins it in BOTH
# directions and a new country pack cannot arrive without a sample. All synthetic: RFC-reserved
# domain, sequential digits, public BR test fixtures, and a card that satisfies Luhn because the
# detector will not see it otherwise.
_SAMPLE_PER_TYPE = {
    "EMAIL":       "escreva para outra.pessoa@example.com",
    "PHONE":       "o numero dele e (11) 98888-7777",
    "NATIONAL_ID": f"o CPF dele e {STAFF_CPF}",
    "COMPANY_ID":  "o CNPJ e 11.222.333/0001-81",
    "ADDRESS":     "o CEP e 01310-100",
    "IP_ADDRESS":  "o servidor e 192.168.15.42",
    "CREDIT_CARD": "o cartao e 4539 1488 0343 6467",
    "CREDENTIAL":  "a senha: hunter2000",
    "BANK_ACCOUNT": "a chave pix e 3f2504e0-4f89-41d3-9a0c-0305e82c3301",
}


def test_every_class_the_stamp_can_carry_is_reachable_AND_named():
    """Both directions, in the mould of ``test_voice_blocks_sync`` — either alone is decorative.

    **Why the class is on the stamp at all.** Observation mode is only worth its name if the
    counts it produces can END it, and the classes have OPPOSITE verdicts: a withheld `ADDRESS`
    or `COMPANY_ID` is almost always the tenant's own CEP/CNPJ (the frequent false positive), a
    withheld `NATIONAL_ID` is almost always a person who is not the one reading (the leak).

    The company number got its OWN class on 2026-09-03, and that sharpens exactly this: it used
    to arrive as `TAX_ID`, sharing a bucket with a person's tax number, so the one class most
    likely to be a false positive and one likely to be a leak were counted together — the
    failure this docstring describes, inside the alphabet meant to prevent it.
    Counted together they say nothing, and "observing" quietly becomes "forgetting".

    * every type the SHIPPED detector can produce has a sample here and reaches a
      `pii:withheld_<type>` token — a new country pack cannot arrive uncounted;
    * every token the stage emits is in the closed alphabet — a free-form reason cannot arrive.

    Pass one alone and the pair is worthless: the first is green with an invented vocabulary, the
    second is green with a stage that emits nothing at all.
    """
    detectable = {p.pii_type for p in DETECTOR._patterns}
    assert set(_SAMPLE_PER_TYPE) == detectable, (
        "the sample table and the shipped packs disagree; add a sample for a new pack "
        f"(missing {detectable - set(_SAMPLE_PER_TYPE)}, stale {set(_SAMPLE_PER_TYPE) - detectable})")

    fixed = {"pii:flagged_in_output", "pii:redacted_in_output", "pii:would_redact_in_output"}
    alphabet = (fixed
                | {f"pii:provenance_{p}" for p in vocab.VALID_PII_PROVENANCE}
                | {f"pii:withheld_{t.lower()}" for t in VALID_PII_TYPES})
    seen: set[str] = set()
    for pii_type, sample in _SAMPLE_PER_TYPE.items():
        out = redact_pii(sample, detector=DETECTOR, mode=vocab.PII_MODE_ENFORCE)
        adj = SuperegoStage._pii_adjustments(out)
        assert f"pii:withheld_{pii_type.lower()}" in adj, f"{pii_type} unreachable: {sample!r}"
        seen |= set(adj)
    # ...and the branches the samples above cannot reach on their own: each allow reason, and
    # the observation stamp (which the enforce runs above can never produce).
    for turn, session, role, mode in ((True, False, "", vocab.PII_MODE_ENFORCE),
                                      (False, True, "", vocab.PII_MODE_ENFORCE),
                                      (False, False, "EMPLOYEE", vocab.PII_MODE_ENFORCE),
                                      (False, False, "", vocab.PII_MODE_OBSERVE)):
        out = redact_pii(f"{STAFF_CPF} e {CONTACT_EMAIL}", detector=DETECTOR, mode=mode,
                         context=ProvenanceContext(
                             turn_digests=frozenset({pii_digest(STAFF_CPF, "NATIONAL_ID")} if turn else ()),
                             session_digests=frozenset({pii_digest(STAFF_CPF, "NATIONAL_ID")} if session else ()),
                             reader_role=role))
        seen |= set(SuperegoStage._pii_adjustments(out))
    assert seen <= alphabet, seen - alphabet
    assert fixed <= seen and {f"pii:provenance_{p}" for p in vocab.VALID_PII_PROVENANCE} <= seen


def test_the_exit_criterion_is_a_number_someone_can_come_back_to():
    """A mode with no exit criterion is a switch nobody touches again, so the threshold is a
    CONSTANT rather than a sentence in a PR nobody re-reads. Rule of three: zero stamps in 200
    turns bounds the false-positive rate under 3/200 = 1.5% at ~95% confidence, and 200 is about
    an order of magnitude above the sample that exists today (9 of the box's 297 traces could
    carry a SUPEREGO block at all). Nothing here promotes anybody — graduation is per tenant and
    is a human setting `mk.PII_OUTPUT_MODE`."""
    assert vocab.PII_OBSERVATION_MIN_TURNS == 200
    assert 3 / vocab.PII_OBSERVATION_MIN_TURNS <= 0.015


def test_clean_text_stays_byte_identical_and_silent():
    """A reply with no personal data must be untouched and must produce no tokens — the rule has
    to be free on the overwhelming majority of turns or every count downstream becomes noise."""
    out = redact_pii("Agendado para terca as 10h. Ate la!", detector=DETECTOR,
                     mode=vocab.PII_MODE_ENFORCE)
    assert out.text == "Agendado para terca as 10h. Ate la!"
    assert out.findings == () and SuperegoStage._pii_adjustments(out) == []


# ── The residue, NAMED rather than discovered in production ──────────────────────────────────

@pytest.mark.parametrize("reply,masked_type", [
    ("Nosso CNPJ e 11.222.333/0001-81, pode faturar.", "COMPANY_ID"),
    ("Estamos na Rua das Flores, 123 - CEP 01310-100.", "ADDRESS"),
    ("Estamos na versao 1.2.3.4 do sistema.", "IP_ADDRESS"),
])
def test_the_known_false_positives_are_named_here_not_found_by_a_contact(reply, masked_type):
    """Three replies this rule breaks, pinned so they are a KNOWN cost and not a surprise.

    All three are the class host bench #549 isolated as scenarios `3` vs `3b`: a value the
    OPERATOR wrote in order to be given out. Provenance cannot tell it from the doctor's CPF —
    the two have the SAME source and OPPOSITE correct answers, which is why "look at the origin"
    cannot decide it and no threshold helps. The fix is a DECLARATION (a tenant-quotable set,
    one more `vocab.VALID_PII_PROVENANCE` value and one metakey), and it is not in this change
    because `TenantConfig` has no structured field to declare it FROM today: its business
    contact data lives in free prose (`custom_rules`), and digesting that would allowlist the
    doctor's CPF along with the reception's phone — the leak this exists to stop.

    The trade was made with eyes open: a clinic's CNPJ is masked so a doctor's CPF is not. When
    the declaration lands, these three flip and this test is what says so.

    (The CNPJ now arrives as `COMPANY_ID`. That changes what the observation COUNTS, not
    what the enforcing mask DOES: provenance still decides, and a number the contact never
    supplied is still withheld. The mask is about what goes OUT; the class split was about
    what the ID does with what comes IN.)
    """
    out = redact_pii(reply, detector=DETECTOR, mode=vocab.PII_MODE_ENFORCE)
    assert out.redacted and [f.pii_type for f in out.findings] == [masked_type]
    assert "REDACTED" in out.text, "loud on purpose: an operator can SEE the cost and report it"


@pytest.mark.parametrize("reply", [
    "Registrei R$ 1.234.567,89 na categoria Vendas.",
    "Seu saldo e de R$ 12.345,67 e o vencimento e 10/07/2026.",
    "Agendado para 20/07 as 10h30. Ate la!",
    "Protocolo 2026.08.26.0001 registrado.",
    "Parcelas de 12x de R$ 99,90 (total R$ 1.198,80).",
    "O horario 09:00-10:00 esta livre.",
    "Pedido 1234-5678 confirmado.",
])
def test_money_dates_and_protocol_numbers_are_never_touched(reply):
    """The other half of the residue, and the bigger one by volume. A bookkeeper reply is mostly
    digits; if figures were masked the feature would be pulled within a day. Checked against the
    real shapes those personas emit, not against invented ones."""
    assert redact_pii(reply, detector=DETECTOR,
                      mode=vocab.PII_MODE_ENFORCE).text == reply


# ── The mode: ONE switch, and the default was chosen by arithmetic ───────────────────────────

@pytest.mark.asyncio
async def test_the_shipped_default_is_observation():
    """No metakey, no configuration: the rule runs in full and masks NOTHING.

    The default is not caution, it is arithmetic. With the role bit the rule still breaks one of
    the merged bench's seven scenarios — the tenant's own reception number, the shape a
    scheduling persona says all day — while production has shown zero leaks in 297 turns.
    Enforcing on those numbers trades an unobserved harm for one that lands on a real contact.

    What observation buys is the missing measurement: `pii:would_redact_in_output` with the type,
    counted on real traffic, is what decides whether a tenant can be switched to enforcing.
    """
    _, res = await _voice("quem e a responsavel tecnica?",
                          f"A responsavel e a Dra. Ana, CPF {STAFF_CPF}.", mode=None)
    assert res.response == f"A responsavel e a Dra. Ana, CPF {STAFF_CPF}."
    assert "pii:would_redact_in_output" in res.adjustments
    assert "pii:redacted_in_output" not in res.adjustments
    assert [f.provenance for f in res.pii_findings] == [vocab.PII_PROVENANCE_UNKNOWN]
    assert [f.redacted for f in res.pii_findings] == [False], "recorded, not acted on"


@pytest.mark.asyncio
@pytest.mark.parametrize("bogus", ["ENFORCE_MAYBE", "", "block", None, 1, "  "])
async def test_an_unknown_mode_falls_to_observation_not_to_enforcement(bogus):
    """A typo must not be able to START masking a tenant's replies. The unsafe direction for a
    mode is the opposite of the unsafe direction for an allowlist, and both fall the safe way."""
    _, res = await _voice("quem e a responsavel tecnica?",
                          f"A responsavel e a Dra. Ana, CPF {STAFF_CPF}.", mode=bogus)
    assert STAFF_CPF in res.response
    assert "pii:redacted_in_output" not in res.adjustments


def test_the_two_modes_decide_identically_and_differ_only_in_the_text():
    """Observation is the SAME rule with the masking withheld — not a second, laxer rule.

    If the two ever diverged in what they DECIDE, the counts gathered under observation would be
    measuring something other than what enforcement would do, and the production evidence the
    graduation depends on would be worthless.
    """
    text = f"CPF {STAFF_CPF}, e-mail {CONTACT_EMAIL}, fone {THIRD_PARTY_PHONE}"
    ctx = ProvenanceContext(turn_digests=frozenset(pii_digests_in(CONTACT_EMAIL, DETECTOR)))
    obs = redact_pii(text, detector=DETECTOR, context=ctx, mode=vocab.PII_MODE_OBSERVE)
    enf = redact_pii(text, detector=DETECTOR, context=ctx, mode=vocab.PII_MODE_ENFORCE)
    assert [f.provenance for f in obs.findings] == [f.provenance for f in enf.findings]
    assert [f.mask for f in obs.findings] == [f.mask for f in enf.findings]
    assert len(obs.withheld) == len(enf.withheld) == 2
    assert obs.text == text and enf.text != text
    assert obs.redacted is False and enf.redacted is True


# ── The role bit — provenance alone is under-determined ──────────────────────────────────────

@pytest.mark.asyncio
async def test_the_tenants_own_bookkeeper_may_re_read_the_row_he_wrote():
    """Merged bench `tool_result_document_number` (host #549, `a2aef70`), whose answer key says
    withholding is WRONG — and no rule that looks only at the ORIGIN can agree with it.

    The value came from a tool result, exactly like a doctor's CPF handed to a patient. What
    differs is who is reading: the tenant's own accountant, about a ledger row he wrote, whose
    read the RBAC layer already authorised. `Identity.role` carries that, so the bit is available
    rather than invented.
    """
    _, res = await _voice("busca o lancamento e me mostra a descricao gravada",
                          f"A descricao gravada e: Consultoria cliente Roberto CPF {CLIENT_CPF}.",
                          role="EMPLOYEE")
    assert CLIENT_CPF in res.response
    assert [f.provenance for f in res.pii_findings] == [vocab.PII_PROVENANCE_READER_STAFF]
    assert "pii:redacted_in_output" not in res.adjustments


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["GUEST", "guest", "", None, "   "])
async def test_the_same_row_is_masked_when_a_visitor_asks(role):
    """The twin, and without it the test above only proves that nothing was ever masked. An
    absent or empty role reads as GUEST: a host that forgets the key gets MORE masking, never a
    silent widening."""
    _, res = await _voice("me mostra a descricao gravada do lancamento",
                          f"A descricao gravada e: Consultoria cliente Roberto CPF {CLIENT_CPF}.",
                          role=role)
    assert CLIENT_CPF not in res.response
    assert [f.provenance for f in res.pii_findings] == [vocab.PII_PROVENANCE_UNKNOWN]


@pytest.mark.asyncio
async def test_a_staff_turn_still_records_the_most_specific_reason():
    """Order in `decide_provenance` is about the RECORD, not the outcome. An employee repeating
    a value they typed themselves reads `contact_turn`, so `reader_is_staff` counts exactly the
    values that ONLY the role let out — which is the widening this bit buys and the number worth
    watching."""
    _, res = await _voice(f"meu email e {CONTACT_EMAIL}, confirma?",
                          f"Confirmado: {CONTACT_EMAIL}.", role="EMPLOYEE")
    assert [f.provenance for f in res.pii_findings] == [vocab.PII_PROVENANCE_TURN]


def test_the_role_bit_is_a_FIELD_not_a_rewrite():
    """The second bit — contacts the tenant DECLARED quotable — enters the same way: one more
    field on `ProvenanceContext`, one more branch in `decide_provenance`, nothing at any call
    site. This pins the shape, which is the part the next change depends on."""
    from cogno_anima.security.redaction import decide_provenance
    digest = pii_digest(STAFF_CPF, "NATIONAL_ID")
    assert decide_provenance(digest, ProvenanceContext()) == vocab.PII_PROVENANCE_UNKNOWN
    assert decide_provenance(digest, ProvenanceContext(reader_role="EMPLOYEE")) \
        == vocab.PII_PROVENANCE_READER_STAFF
    assert decide_provenance("", ProvenanceContext(turn_digests=frozenset({""}))) \
        == vocab.PII_PROVENANCE_UNKNOWN, "an empty digest is never allowlisted"


# ── overlapping patterns: the case the first cut shipped in the clear ────────────────────────

def test_a_PARTIAL_overlap_masks_the_whole_value_and_records_both():
    r"""Two patterns can overlap without either containing the other, and the first cut leaked.

    `credential_kv`'s `\S{4,}` stops at whitespace; the card pattern spans it. So
    `"senha: 4539 1488 0343 6467"` yields a CREDENTIAL over `senha: 4539` and a CREDIT_CARD over
    all sixteen digits — a PARTIAL overlap. Dropping the later match (correct for a NESTED one)
    produced three harms at once, all measured on this exact string: `"[CREDENTIAL REDACTED]
    1488 0343 6467"` — fourteen of nineteen card digits in the clear, a mask spliced into the
    middle of a value, and no CREDIT_CARD in `findings` at all, so the observation counts that
    decide graduation would have under-reported it in silence.
    """
    out = redact_pii("senha: 4539 1488 0343 6467", detector=DETECTOR,
                     mode=vocab.PII_MODE_ENFORCE)
    assert out.text == "[CREDIT_CARD REDACTED]"
    assert "4539" not in out.text and "6467" not in out.text
    assert sorted(f.pii_type for f in out.findings) == ["CREDENTIAL", "CREDIT_CARD"]
    assert all(f.redacted for f in out.findings)


def test_one_number_matched_by_two_packs_counts_ONCE():
    """The twin of the test above, and it pulls the other way: `(11) 98888-7777` matches
    `phone_ddd` whole and `phone_mobile` on its tail. Clustering must not turn one number into
    two withheld values — that figure is what decides whether a tenant leaves observation, and
    inflating it would keep everyone in it forever. Same TYPE nested in a wider span is the same
    value; different types (the pair above) are not."""
    out = redact_pii("o fone e (11) 98888-7777.", detector=DETECTOR,
                     mode=vocab.PII_MODE_ENFORCE)
    assert out.text == "o fone e [PHONE REDACTED]."
    assert [f.pii_type for f in out.findings] == ["PHONE"]


def test_a_value_that_may_leave_goes_with_the_one_that_may_not_when_they_overlap():
    """Collateral, stated rather than left to be discovered: overlapping spans cannot be split,
    so a cluster ships verbatim only if EVERY member may leave. The finding still records the
    provenance that would have allowed it, so the collateral is visible in the audit rather than
    looking like a rule that changed its mind."""
    text = "senha: 4539 1488 0343 6467"
    ctx = ProvenanceContext(turn_digests=frozenset({pii_digest("senha: 4539", "CREDENTIAL")}))
    out = redact_pii(text, detector=DETECTOR, context=ctx, mode=vocab.PII_MODE_ENFORCE)
    assert out.text == "[CREDIT_CARD REDACTED]"
    allowed = [f for f in out.findings if f.provenance == vocab.PII_PROVENANCE_TURN]
    assert allowed and allowed[0].redacted is True, "masked, but the record says why it could go"


# ── the reader role is host input, and its failure direction is the leak ─────────────────────

class _StrRole(str):
    """A `str` subclass, as a real host enum would be — this one must keep working."""


@pytest.mark.asyncio
@pytest.mark.parametrize("injected,leaks", [
    ("GUEST", False),
    (_StrRole("GUEST"), False),
    ({"role": "GUEST"}, False),
    (["GUEST"], False),
    (object(), False),
    (None, False),
    ("EMPLOYEE", True),
])
async def test_a_role_that_is_not_a_string_reads_as_GUEST(injected, leaks):
    """`str(x)` looked equivalent to a coercion and was the one that widened.

    `str({"role": "GUEST"})` is `"{'role': 'GUEST'}"` — not empty, not `"GUEST"` — so
    `reader_is_staff` said yes and a stranger's CPF shipped. Worst of all on a plain `Enum`:
    `str(Role.GUEST)` is `"Role.GUEST"`, so a host explicitly saying GUEST got staff treatment.
    `metakeys` promises the opposite ("a host that forgets this key gets more masking, never a
    silent widening"), and this is the only one of the three injected keys whose failure
    direction is the leak.
    """
    _, res = await _voice("me mostra o cadastro", f"O CPF e {CLIENT_CPF}.", role=injected)
    assert (CLIENT_CPF in res.response) is leaks
