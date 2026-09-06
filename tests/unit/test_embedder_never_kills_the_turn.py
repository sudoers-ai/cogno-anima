"""The embedder may fail; the turn may not die — and the drift it could not measure
is recorded as UNKNOWN, which is neither 0.0 nor 1.0.

Measured in production 2026-09-06: 2 accepted contact messages out of 274 (0.73%) over a
23.28 h window produced no turn at all. Both landed under a second past the embedding
client's 120 s read timeout, and every embedding call that returned 200 in the same
windows finished under 86 s — so it was the timeout, not a slow box. The `httpx.ReadTimeout`
came out of the NOUMENO's similarity call, crossed `process` untouched, and killed the turn.

The half that is NOT obvious is what the drift becomes when it cannot be computed, because
the two obvious answers are the two wrong ones:

  * 0.0 reads as "verified, nothing drifted" — the silent optimism that waves through a
    turn nobody checked;
  * 1.0 reads as "total drift" — forces the DRIFT tag and can trip `self_correct` over a
    turn that was fine.

The third way was already in the chain: `DriftMetrics` documents `None` as "stage not
computed yet (distinct from 0.0 = computed, no drift)" and `compute_cumulative`
renormalizes over the components that are not `None`. The epistemological component was
the ONE that could not say it — which is exactly the component an embedder outage
destroys. These tests pin that it now can, and that the OK path did not move.
"""
import asyncio

import pytest

from cogno_anima import metakeys as mk
from cogno_anima.stages.drift import DriftCalculator
from cogno_anima.stages.noumeno import Noumeno, classify_drift
from cogno_anima.types import PipelineContext
from cogno_anima.vocab import (
    DRIFT_TAG_UNKNOWN,
    EMBED_UNAVAILABLE,
    VALID_DRIFT_TAGS,
    VALID_NOUMENO_DEGRADATIONS,
)
from tests.conftest import StubBackend, StubEmbedder

from .test_noumeno import PROMPTS_DIR, FixedSimilarityEmbedder

REWRITE = ('{"rewritten": "Book an appointment for tomorrow at 2pm.", "context_turn": "", '
           '"confidence": 0.95, "changed": true, "preserved_terms": ["2pm"], '
           '"rewrite_warnings": []}')


class TimingOutEmbedder(StubEmbedder):
    """Raises exactly what production raised, on every call."""

    def __init__(self, message: str = "timed out"):
        super().__init__()
        self.calls = 0
        self._message = message

    async def similarity(self, a: str, b: str) -> float:
        self.calls += 1
        import httpx
        raise httpx.ReadTimeout(self._message)


class HalfDeadEmbedder(StubEmbedder):
    """Fails only on the call whose right-hand side is the rewrite (the drift call),
    so the subject-continuity measurement of the same turn still succeeds."""

    def __init__(self, rewritten: str):
        super().__init__()
        self._rewritten = rewritten

    async def similarity(self, a: str, b: str) -> float:
        if b == self._rewritten:
            raise ConnectionError("embed backend down")
        return 0.9


def _noumeno(embedder, **kw) -> Noumeno:
    return Noumeno(embedder=embedder, prompts_dir=PROMPTS_DIR, **kw)


# ── Twin 1: the embedder is down → the turn is DELIVERED ─────────────────────

async def test_embedder_timeout_delivers_a_normal_turn():
    """The measured defect, inverted. Before this change the ReadTimeout propagated
    out of `process` and the turn died with no row in `turns` at all."""
    emb = TimingOutEmbedder()
    ctx = PipelineContext(user_input="marcar consulta amanha as 14h")
    ctx.metadata[mk.LAST_REWRITTEN] = "The user greeted the assistant."

    ctx = await _noumeno(emb).process(ctx, StubBackend(response=REWRITE))

    assert ctx.noumeno is not None
    assert ctx.noumeno.rewritten == "Book an appointment for tomorrow at 2pm."
    assert ctx.noumeno.preserved_terms == ["2pm"]
    assert emb.calls == 2, "both similarity calls were attempted, not short-circuited"
    assert ctx.noumeno.degradations == [EMBED_UNAVAILABLE]


# ── Twin 2: the embedder is fine → nothing moved ─────────────────────────────

async def test_embedder_ok_is_unchanged():
    """The OK path keeps every value it had, and records NO degradation. `degradations`
    is the empty list, never the absent field — a reader must be able to tell
    "measured, nothing degraded" from "this producer does not report degradations"."""
    ctx = PipelineContext(user_input="hello world")
    ctx = await _noumeno(FixedSimilarityEmbedder(0.9)).process(
        ctx, StubBackend(response=REWRITE))

    n = ctx.noumeno
    assert n.drift_score == pytest.approx(0.1)
    assert n.drift_tag == "REWRITTEN" == classify_drift(0.1)
    assert n.subject_similarity == 1.0        # no last_rewritten → never measured, as before
    assert n.change_subject is False
    assert n.degradations == []
    assert n.metrics.embedding_calls == 2


async def test_embedder_ok_high_drift_still_reconciles():
    """The reconciliation rule (drift > 0.50 forces changed=True and DRIFT) is untouched:
    it now sits behind an `is None` branch, and this is the twin that proves the branch
    did not swallow it."""
    ctx = PipelineContext(user_input="original input")
    ctx = await _noumeno(FixedSimilarityEmbedder(0.4)).process(
        ctx, StubBackend(response=REWRITE.replace('"changed": true', '"changed": false')))

    assert ctx.noumeno.drift_score > 0.50
    assert ctx.noumeno.drift_tag == "DRIFT"
    assert ctx.noumeno.changed is True
    assert ctx.noumeno.degradations == []


# ── The third way: UNKNOWN is neither of the two wrong answers ───────────────

async def test_unmeasured_drift_is_neither_zero_nor_one():
    """The whole point. 0.0 would say "verified, no drift"; 1.0 would say "total drift"
    and force the DRIFT tag. Both are claims about a measurement that never happened."""
    ctx = PipelineContext(user_input="marcar consulta amanha as 14h")
    ctx = await _noumeno(TimingOutEmbedder()).process(ctx, StubBackend(response=REWRITE))

    n = ctx.noumeno
    assert n.drift_score is None
    assert n.drift_score != 0.0 and n.drift_score != 1.0
    assert n.drift_tag == DRIFT_TAG_UNKNOWN
    assert n.drift_tag not in ("PASS_THROUGH", "DRIFT")
    assert n.drift_tag in VALID_DRIFT_TAGS


async def test_unmeasured_drift_does_not_override_the_rewriter():
    """`changed` keeps what the rewriter itself reported. Forcing it either way would be
    deciding from the absence of evidence."""
    for declared, expected in (("true", True), ("false", False)):
        ctx = PipelineContext(user_input="marcar consulta")
        body = REWRITE.replace('"changed": true', f'"changed": {declared}')
        ctx = await _noumeno(TimingOutEmbedder()).process(ctx, StubBackend(response=body))
        assert ctx.noumeno.changed is expected


# ── The drift chain: dropped from the vote, and SAID so ──────────────────────

def test_cumulative_drops_the_unmeasured_component():
    """`compute` must carry the `None` through — coercing it to 0.0 is the optimism —
    and `compute_cumulative` must renormalize without it, exactly as it already does
    for an uncomputed ontological component."""
    from cogno_anima.types import IntentResult, StageMetrics

    intent = IntentResult(
        intent_class="ACTION_REQUEST", sentiment="NEUTRAL", temporal_class="RECENT",
        triad_signal="EGO", confidence=0.9, goal="book an appointment",
        domains=["HEALTH"], verbs=["book"], langue="pt-BR",
        metrics=StageMetrics(stage="ner", elapsed_ms=1.0, tokens_in=1, tokens_out=1,
                             model="stub"))

    class _N:
        original = "marcar consulta amanha as 14h"
        rewritten = "Book an appointment for tomorrow at 2pm."
        drift_score = None
        aristotelian: dict = {}

    calc = DriftCalculator()
    drift = calc.compute(_N(), intent)
    assert drift.drift_score is None, "the None must survive compute(), not become 0.0"

    drift.situational_drift = 1.0
    drift.ontological_drift = 1.0
    calc.compute_cumulative(drift)
    # Renormalized over the two measured components only — an epistemological 0.0
    # silently voted in would have pulled this below 1.0.
    assert drift.cumulative_drift == pytest.approx(1.0)


def test_unmeasured_drift_neither_forces_self_correct_nor_reads_as_verified():
    """The twin the owner asked for, both halves at once.

    Left: a turn whose measured components are calm must NOT be escalated to
    `self_correct` just because one component is missing. Right: the record must not
    read as a verified clean turn either — `to_tags()` says NOUMENO.DRIFT_UNKNOWN, which
    is the flag a reader of tags alone needs in order not to mistake the absence of
    NOUMENO.DRIFT for a pass."""
    from cogno_anima.types import DriftMetrics

    drift = DriftMetrics(word_count_original=5, word_count_noumeno=6,
                         compression_ratio=1.2, aristotelian_coverage=0,
                         drift_score=None, ontological_drift=0.1,
                         situational_drift=0.1)
    DriftCalculator().compute_cumulative(drift)

    assert drift.drift_action != "self_correct"
    tags = drift.to_tags()
    assert "NOUMENO.DRIFT_UNKNOWN" in tags
    assert "NOUMENO.DRIFT" not in tags


def test_measured_zero_and_unmeasured_are_not_the_same_record():
    """0.0 and None must be distinguishable by a reader that only has the DriftMetrics —
    otherwise the third state buys nothing."""
    from cogno_anima.types import DriftMetrics

    def _mk(score):
        return DriftMetrics(word_count_original=5, word_count_noumeno=5,
                            compression_ratio=1.0, aristotelian_coverage=0,
                            drift_score=score)

    measured, unmeasured = _mk(0.0), _mk(None)
    assert measured.drift_score == 0.0 and unmeasured.drift_score is None
    assert "NOUMENO.DRIFT_UNKNOWN" in unmeasured.to_tags()
    assert "NOUMENO.DRIFT_UNKNOWN" not in measured.to_tags()


# ── Subject continuity degrades to the conservative side ─────────────────────

async def test_unmeasured_subject_similarity_keeps_the_thread():
    """`change_subject=True` DROPS the thread's carry-over context in the NER. On a turn
    where the similarity was never taken, the conservative answer is "assume continuity"
    — the same value a first turn carries."""
    ctx = PipelineContext(user_input="e quanto custa?")
    ctx.metadata[mk.LAST_REWRITTEN] = "Something entirely unrelated about telescopes."
    ctx = await _noumeno(TimingOutEmbedder()).process(ctx, StubBackend(response=REWRITE))

    assert ctx.noumeno.subject_similarity is None
    assert ctx.noumeno.change_subject is False


async def test_only_the_failing_half_degrades():
    """Each measurement stands alone: the subject similarity of the same turn succeeded,
    so it keeps its number while the drift says UNKNOWN. And the flag is recorded ONCE."""
    emb = HalfDeadEmbedder("Book an appointment for tomorrow at 2pm.")
    ctx = PipelineContext(user_input="marcar consulta")
    ctx.metadata[mk.LAST_REWRITTEN] = "The user greeted the assistant."
    ctx = await _noumeno(emb).process(ctx, StubBackend(response=REWRITE))

    assert ctx.noumeno.subject_similarity == pytest.approx(0.9)
    assert ctx.noumeno.drift_score is None
    assert ctx.noumeno.degradations == [EMBED_UNAVAILABLE]


# ── What must still be loud ──────────────────────────────────────────────────

async def test_cancellation_still_unwinds_the_turn():
    """`except Exception` and not `except BaseException`: a shutdown or a client hang-up
    must still tear the turn down. Catching this one would make a cancelled turn look
    like a delivered one."""
    class Cancelling(StubEmbedder):
        async def similarity(self, a: str, b: str) -> float:
            raise asyncio.CancelledError()

    ctx = PipelineContext(user_input="marcar consulta")
    with pytest.raises(asyncio.CancelledError):
        await _noumeno(Cancelling()).process(ctx, StubBackend(response=REWRITE))


async def test_llm_failure_is_still_fatal():
    """The line this change draws: the LLM PRODUCES the rewrite the pipeline consumes,
    the embedder only MEASURES it. Losing the product stays loud."""
    class FailingLLM(StubBackend):
        async def generate(self, system: str, prompt: str):
            raise RuntimeError("Fatal API Error")

    ctx = PipelineContext(user_input="marcar consulta")
    with pytest.raises(RuntimeError):
        await _noumeno(StubEmbedder()).process(ctx, FailingLLM())


# ── The alphabet is closed, and never carries the error text ─────────────────

async def test_degradation_alphabet_is_closed_and_leaks_nothing():
    """The list is written into traces, so its values come from the vocabulary and never
    from an exception message — a transport error can carry a URL, a host name, or the
    contact's own text."""
    secret = "connect to 10.0.0.7 for contact +5511999998888"
    ctx = PipelineContext(user_input="marcar consulta")
    ctx = await _noumeno(TimingOutEmbedder(secret)).process(
        ctx, StubBackend(response=REWRITE))

    assert set(ctx.noumeno.degradations) <= VALID_NOUMENO_DEGRADATIONS
    blob = " ".join(ctx.noumeno.degradations)
    assert "10.0.0.7" not in blob and "5511999998888" not in blob


async def test_failed_embedding_is_counted_as_an_attempt():
    """The round trip happened — in the measured case it burned the full 120 s — so it
    is 2 calls at 0 tokens, not hidden from the trace."""
    ctx = PipelineContext(user_input="marcar consulta")
    ctx.metadata[mk.LAST_REWRITTEN] = "previous"
    ctx = await _noumeno(TimingOutEmbedder()).process(ctx, StubBackend(response=REWRITE))

    assert ctx.noumeno.metrics.embedding_calls == 4   # subject + drift, both attempted
    assert ctx.noumeno.metrics.embedding_tokens == 0


def test_every_tag_the_stage_can_emit_is_in_the_closed_alphabet():
    """`VALID_DRIFT_TAGS` is a DUPLICATED CONTRACT — the tags are produced in
    `noumeno.py`, the set lives in `vocab.py` — so both directions are pinned, in the
    mould of `test_code_domains_match_prompt_domains_exactly`."""
    produced = {classify_drift(s) for s in (0.0, 0.1, 0.3, 0.5, 0.7, 1.0)}
    produced |= {"DRIFT", DRIFT_TAG_UNKNOWN}
    assert produced == set(VALID_DRIFT_TAGS)


# ── Where the missing stage's weight goes (measured, not assumed) ────────────

def test_the_default_that_would_have_swallowed_the_absence_is_gone():
    """The failure mode the renormalization does NOT protect against: a default put in
    FRONT of it. `compute()` used to read the score through
    `safe_float(..., default=0.0)`, so a `None` arrived at `compute_cumulative` already
    turned into a measured-looking 0.0 and the component voted after all.

    Measured on the base tree (ee383b2): `compute(drift_score=None).drift_score` was
    `0.0`, and the cumulative of two components at 1.0 came out `0.7` instead of `1.0` —
    the absent stage cast a zero worth 30% of the vote."""
    from cogno_anima.types import IntentResult, StageMetrics

    class _N:
        original = "marcar consulta"
        rewritten = "Book an appointment."
        drift_score = None
        aristotelian: dict = {}

    intent = IntentResult(
        intent_class="ACTION_REQUEST", sentiment="NEUTRAL", temporal_class="RECENT",
        triad_signal="EGO", confidence=0.9, goal="book", domains=["HEALTH"],
        verbs=["book"], langue="pt-BR",
        metrics=StageMetrics(stage="ner", elapsed_ms=1.0, tokens_in=1, tokens_out=1,
                             model="stub"))

    calc = DriftCalculator()
    drift = calc.compute(_N(), intent)
    assert drift.drift_score is None

    drift.ontological_drift = 1.0
    drift.situational_drift = 1.0
    calc.compute_cumulative(drift)
    assert drift.cumulative_drift == pytest.approx(1.0), (
        "an unmeasured component must not cast a zero: 0.7 here is the old default voting")


def test_dropping_the_component_is_symmetric_not_an_inflation():
    """The coordinator's second question, pinned. Removing a component does not push the
    drift UP for being fewer — it removes a VOTE, and the result moves toward the mean of
    what is left. Measured over the full grid (step 0.05, denominator 9261 cells, three
    stages present): 49.7% rise, 49.7% fall, 0.7% unchanged, bounded at ±0.300 — which is
    exactly the closed form `delta = 0.3 * (m - e)`.

    These two cases are the two directions, so a mutation that biases the mechanism one
    way kills one of them."""
    from cogno_anima.types import DriftMetrics

    def _cum(epist):
        d = DriftMetrics(word_count_original=5, word_count_noumeno=5, compression_ratio=1.0,
                         aristotelian_coverage=0, drift_score=epist,
                         ontological_drift=0.8, situational_drift=0.8)
        DriftCalculator().compute_cumulative(d)
        return d.cumulative_drift

    unmeasured = _cum(None)
    # The missing value would have been BELOW the others → dropping it raises. Bounded.
    assert unmeasured > _cum(0.0)
    assert unmeasured - _cum(0.0) == pytest.approx(0.3 * 0.8, abs=0.002)
    # The missing value would have been ABOVE the others → dropping it LOWERS. Same bound.
    assert unmeasured < _cum(1.0)
    assert _cum(1.0) - unmeasured == pytest.approx(0.3 * 0.2, abs=0.002)


def test_the_absence_is_never_ITSELF_the_escalation():
    """Half of the owner's twin, and ONLY the half that was measured — the name says so
    on purpose.

    What IS pinned: the absence is never a value that pushes. A 1.0 default escalates on
    evidence nobody gathered; `None` cannot, because it does not vote. Calm measured
    components stay calm however unmeasurable the NOUMENO was.

    What is NOT pinned, and must not be read into this: that `self_correct` can never
    fire on a turn with an unmeasured component. It can — measured over the full grid
    (step 0.05, denominator 9261, three stages), 4.12% of cells reach `self_correct`
    without being there with the component present. Every one of those requires the
    REMAINING measured components to average >= 0.85, i.e. the escalation is carried by
    evidence that WAS gathered, not by the gap. `test_dropping_the_component_is_symmetric`
    above is the bound on how far the gap can move anything at all (+/-0.30 here).

    The open question that belongs to the owner, not to this test: whether an incomplete
    vote should be capped below `self_correct` at all. Capping it would also change 546
    grid cells of the PRE-EXISTING `ontological_drift=None` case (greetings), which is
    not this defect — so the policy is deliberately NOT added here."""
    from cogno_anima.types import DriftMetrics

    def _mk(epist, others):
        return DriftMetrics(word_count_original=5, word_count_noumeno=5,
                            compression_ratio=1.0, aristotelian_coverage=0,
                            drift_score=epist, ontological_drift=others,
                            situational_drift=others, execution_drift=others,
                            synthesis_drift=others)

    # 0.45 is chosen so the ACTION, not only the score, separates the two: the measured
    # components sit just under `warn` and only the invented 1.0 pushes them over.
    calm = _mk(None, 0.45)
    DriftCalculator().compute_cumulative(calm)
    assert calm.cumulative_drift == pytest.approx(0.45)
    assert calm.drift_action == "none"

    # The rejected 1.0 default on the SAME components DOES push — which is what makes the
    # assertion above a measurement and not a tautology.
    forced = _mk(1.0, 0.45)
    DriftCalculator().compute_cumulative(forced)
    assert forced.cumulative_drift > calm.cumulative_drift
    assert forced.drift_action == "warn" != calm.drift_action

    # And the boundary this test does NOT claim to hold: extreme measured components
    # escalate on their own, absence or not. Pinned so nobody reads the name too widely.
    loud = _mk(None, 1.0)
    DriftCalculator().compute_cumulative(loud)
    assert loud.drift_action == "self_correct"
