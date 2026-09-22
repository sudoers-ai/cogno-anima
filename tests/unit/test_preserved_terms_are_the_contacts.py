"""A preserved e-mail or URL the contact never typed is the model's, not the contact's.

Measured 3 of 3, deterministic (qwen3:8b, temperature=0), while re-recording the onboarding
cassette in the host (#954): the NOUMENO listed the contact's e-mail under ``preserved_terms``
with ONE character altered — same domain, local part at edit distance 1 — and the rewrite carried
the same altered address. The judge demands the exact reproduction of every preserved VALUE
(criterion #4, #172), so a reply carrying the RIGHT address was rejected for "altering" it, the
correction loop exhausted, and the contact got the failure sentence — or the voice repeated the
wrong address. The owner will test the onboarding with his own e-mail.

The rule is narrow ON PURPOSE. An e-mail or a URL is an exact token by nature: there is no
legitimate rewrite of one, so a preserved e-mail/URL that is not in the contact's own words is
fabrication and is dropped before the judge or the voice ever sees it. A FIGURE is not an exact
token: the rewriter normalises format on its way to English ("R$ 1.000,00" may come out "1000"),
and #172 decided a preserved term is a VALUE, not a spelling — a "not verbatim" discard would eat
legitimate figures. Figures never take this path; neither does a name or a phrase.

The drop leaves a MARK, because a net nobody counts becomes the mechanism:
``vocab.PRESERVED_NOT_IN_INPUT`` on ``NoumenoResult.degradations`` (the closed alphabet
``EMBED_UNAVAILABLE`` already lives in), the COUNT on ``NoumenoResult.preserved_dropped``, and
``NOUMENO.PRESERVED_NOT_IN_INPUT`` among the drift tags. Never the value — it is PII.

The comparison is a whole-address match against ``ctx.user_input`` — the RAW text the contact
typed, never the rewrite (the rewrite carries the same mutation) — case included. Exact, because
the two errors are not symmetric: a false DROP costs only the marking (the contact's own words
are already in the judge's prompt, verbatim, under ``# User request``), while a false KEEP is the
measured defect. Case-folding cannot rescue the measured shape (a changed character); it could
only admit a normalisation the MODEL chose, and the judge would then demand the model's spelling
over the contact's. There is nothing to buy with it. And "whole address", not bare substring:
the measured family is edit distance 1, and a character dropped at either END of the local part
or of the domain leaves a SUBSTRING of what was typed that is still not the address.
"""

from __future__ import annotations

import json
import logging

import pytest

from cogno_anima import vocab
from cogno_anima.stages.drift import DriftCalculator
from cogno_anima.stages.noumeno import Noumeno
from cogno_anima.types import DriftMetrics, PipelineContext
from tests.conftest import StubBackend

from .test_noumeno import PROMPTS_DIR, FixedSimilarityEmbedder

# The VALUE is the contract (it lands in traces); the constant is checked against it below.
TOKEN = "preserved:not_in_input"
TAG = "NOUMENO.PRESERVED_NOT_IN_INPUT"

TYPED = "ana.silva@example.com"
MUTATED = "nana.silva@example.com"      # edit distance 1, same domain — the measured shape


def _rewrite(rewritten: str, *preserved: str) -> str:
    return json.dumps({"rewritten": rewritten, "context_turn": "", "confidence": 0.95,
                       "changed": True, "preserved_terms": list(preserved),
                       "rewrite_warnings": []})


async def _run(user_input: str, rewritten: str, *preserved: str) -> PipelineContext:
    ctx = PipelineContext(user_input=user_input)
    stage = Noumeno(embedder=FixedSimilarityEmbedder(0.9), prompts_dir=PROMPTS_DIR)
    return await stage.process(ctx, StubBackend(response=_rewrite(rewritten, *preserved)))


def _tags(ctx: PipelineContext) -> list[str]:
    return DriftCalculator().compute(ctx.noumeno, None).to_tags()


# ── the twin: the measured defect, inverted ──────────────────────────────────

async def test_a_mutated_email_is_dropped_and_the_drop_is_marked():
    """The rewrite carries the SAME mutated address on purpose: a filter that compared against
    the rewrite instead of the contact's words would find it there and keep it."""
    ctx = await _run(f"meu e-mail é {TYPED}", f"My e-mail is {MUTATED}.", MUTATED)
    n = ctx.noumeno

    assert MUTATED not in n.preserved_terms, "the model's address reached the judge and the voice"
    assert n.preserved_terms == []
    assert TOKEN in n.degradations
    assert n.preserved_dropped == 1
    assert TAG in _tags(ctx)


# ── control 1: the address the contact typed stays, and nothing is marked ────

async def test_the_email_the_contact_typed_stays_and_nothing_is_marked():
    ctx = await _run(f"meu e-mail é {TYPED}", f"My e-mail is {TYPED}.", TYPED)
    n = ctx.noumeno

    assert n.preserved_terms == [TYPED]
    assert n.degradations == []
    assert n.preserved_dropped == 0
    assert TAG not in _tags(ctx)


# ── control 2: the Director's narrowing — figures, names and phrases never take this path ──

async def test_a_figure_is_never_dropped_for_not_being_verbatim():
    """"R$ 1.000,00" typed, "1000" preserved: not verbatim, and CORRECT — the rewriter
    normalises format on the way to English, and a preserved term is a VALUE (#172)."""
    ctx = await _run("a mensalidade custa R$ 1.000,00", "The monthly fee is 1000.", "1000")
    n = ctx.noumeno

    assert n.preserved_terms == ["1000"]
    assert n.degradations == []
    assert n.preserved_dropped == 0


async def test_a_name_or_phrase_is_never_dropped_either():
    """Neither figure nor exact token: outside the rule entirely, whatever its spelling."""
    ctx = await _run("quero a ementa de modelagem de dados", "I want the Data Modeling syllabus.",
                     "Modelagem de Dados", "Acmee")
    n = ctx.noumeno

    assert n.preserved_terms == ["Modelagem de Dados", "Acmee"]
    assert n.degradations == []
    assert n.preserved_dropped == 0


# ── control 3: URLs — a mutated one is dropped, a typed one stays, in one list ─

async def test_a_mutated_url_is_dropped_and_a_present_one_stays():
    typed = "https://example.org/planos"
    other = "https://example.com/ajuda"
    mutated = "https://example.org/plano-s"
    ctx = await _run(f"veja {typed} e {other}", f"See {mutated} and {other}.", mutated, other)
    n = ctx.noumeno

    assert n.preserved_terms == [other]
    assert n.degradations == [TOKEN]
    assert n.preserved_dropped == 1


# ── the count is the number dropped; the token is recorded once; the value never ─

async def test_the_count_is_the_number_dropped_and_the_value_is_never_logged(caplog):
    other_mutated = "ana.silvaa@example.com"
    with caplog.at_level(logging.WARNING, logger="cogno_anima.noumeno"):
        ctx = await _run(f"meu e-mail é {TYPED}", f"My e-mail is {MUTATED}.",
                         MUTATED, TYPED, other_mutated)
    n = ctx.noumeno

    assert n.preserved_terms == [TYPED]
    assert n.preserved_dropped == 2
    assert n.degradations.count(TOKEN) == 1, "a count belongs in the counter, not in the alphabet"

    lines = [r.getMessage() for r in caplog.records if "preserved" in r.getMessage()]
    assert lines, "a drop nobody can see in the log is a net nobody counts"
    assert any("2" in line for line in lines)
    for line in lines:
        assert MUTATED not in line and other_mutated not in line and TYPED not in line, (
            "an address in a log line is PII in a log line")


# ── the comparison, pinned at its edges ──────────────────────────────────────

@pytest.mark.parametrize("typed, listed, kept", [
    pytest.param(f"escreve para {TYPED}.", TYPED, True, id="sentence-final period is not the address"),
    pytest.param(f"escreve para ({TYPED}), obrigado", TYPED, True, id="punctuation around it"),
    pytest.param(f"escreve para {TYPED}", "na.silva@example.com", False,
                 id="leading character dropped: a substring, still not the address"),
    pytest.param(f"escreve para {TYPED}.br", TYPED, False, id="domain truncated at a label boundary"),
    pytest.param(f"escreve para {TYPED}", "Ana.Silva@example.com", False, id="case: exact, see module docstring"),
    pytest.param("veja https://example.org/planos", "https://example.org/plano", False,
                 id="URL truncated inside a path segment"),
    pytest.param("veja https://example.org/planos.", "https://example.org/planos", True,
                 id="URL before a sentence-final period"),
])
async def test_the_comparison_is_a_whole_address_in_the_contacts_words(typed, listed, kept):
    ctx = await _run(typed, f"Write to {listed}.", listed)
    n = ctx.noumeno

    assert (listed in n.preserved_terms) is kept
    assert n.preserved_dropped == (0 if kept else 1)
    assert (TOKEN in n.degradations) is not kept


# ── one definition of "critical", now three readers ──────────────────────────

def test_the_stage_and_the_judge_read_one_definition_of_critical():
    """The judge's block and its output backstop already shared ``_CRITICAL_TERM_RE`` (a figure,
    an e-mail, a URL). The NOUMENO needs the e-mail/URL HALF of it; a second regex would be the
    divergence #172 closed, reopened one stage upstream."""
    from cogno_anima import preserved
    from cogno_anima.stages import superego

    assert superego._CRITICAL_TERM_RE is preserved.CRITICAL_TERM_RE

    for exact in (TYPED, "https://example.org/x", "HTTP://EXAMPLE.ORG"):
        assert preserved.EXACT_TOKEN_RE.search(exact) and preserved.CRITICAL_TERM_RE.search(exact)
    for figure in ("1000", "R$ 1.000,00", "14h"):
        assert preserved.CRITICAL_TERM_RE.search(figure)
        assert not preserved.EXACT_TOKEN_RE.search(figure), "a figure is critical but not exact"
    assert not preserved.CRITICAL_TERM_RE.search("Modelagem de Dados")


# ── the mark: closed alphabet, and the tag reads what the calculator seeded ──

def test_the_mark_is_in_the_closed_alphabet():
    assert vocab.PRESERVED_NOT_IN_INPUT == TOKEN
    assert TOKEN in vocab.VALID_NOUMENO_DEGRADATIONS
    assert vocab.EMBED_UNAVAILABLE in vocab.VALID_NOUMENO_DEGRADATIONS, "the precedent stays"


def test_the_tag_reads_the_degradation_the_calculator_seeded():
    def _mk(degradations):
        return DriftMetrics(word_count_original=5, word_count_noumeno=5, compression_ratio=1.0,
                            aristotelian_coverage=0, drift_score=0.0,
                            noumeno_degradations=degradations)

    assert TAG in _mk([TOKEN]).to_tags()
    assert TAG not in _mk([]).to_tags()
    assert TAG not in _mk([vocab.EMBED_UNAVAILABLE]).to_tags(), "one tag per fact"
