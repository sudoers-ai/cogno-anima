"""A truncated model response is transient — it must buy one retry, and nothing else must.

Live failure this covers (2026-08-04, gpt-4o-mini driving NOUMENO): the payload arrived as

    {"rewritten":"Between 7 to 10 calls on average.","context_turn":"The user is providing
    information about the volume of c

— a string opened and never closed, 211 characters in, with ``max_tokens=4096`` nowhere near
reached. It raised StageParseError and killed the user's turn. Nothing on the path was built
to notice: the OpenAI backend never inspects ``finish_reason``, and neither cogno-host nor
cogno-soma catches StageParseError, so a provider cutting a stream reached the user as a dead
turn diagnosed as "bad JSON".
"""
from __future__ import annotations

import pytest

from cogno_anima.errors import StageParseError
from cogno_anima.stages.ner import IntentAnalyzer
from cogno_anima.stages.noumeno import Noumeno
from cogno_anima.types import NoumenoResult, PipelineContext, StageMetrics
from cogno_anima.utils import looks_truncated

# The exact payload that died in production.
TRUNCATED = ('{"rewritten":"Between 7 to 10 calls on average.","context_turn":'
             '"The user is providing information about the volume of c')
NER_OK = ('{"intent_class":"INFORMATION_REQUEST","sentiment":"NEUTRAL","goal":"ok",'
          '"domains":["GENERAL"]}')
NOUMENO_OK = '{"rewritten":"Ten calls a day.","context_turn":"","confidence":0.9}'
PROSE = "Desculpe, não consigo responder isso agora."


class _Scripted:
    """Returns the given responses in order; repeats the last one forever."""

    model = "scripted"

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def generate(self, system: str, prompt: str) -> tuple[str, int, int]:
        out = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return out, 100, 10


class _ConstEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    async def similarity(self, a: str, b: str) -> float:
        return 1.0


def _noumeno_result() -> NoumenoResult:
    return NoumenoResult(
        original="dez por dia", rewritten="Ten a day.", context_turn="", language="pt-BR",
        drift_score=0.0, drift_tag="PASS_THROUGH", changed=False, confidence=0.9,
        change_subject=False, subject_similarity=1.0, context_used=False,
        preserved_terms=[], rewrite_warnings=[],
        metrics=StageMetrics(stage="noumeno", elapsed_ms=0, tokens_in=0, tokens_out=0,
                             model="scripted"))


# ── the detector: severed vs merely malformed ────────────────────────────────────────────
# The distinction decides whether a retry is worth making at all. Getting it wrong in the
# permissive direction buys a second identical failure plus latency and cost.

@pytest.mark.parametrize("raw, truncated", [
    (TRUNCATED, True),                                  # the real one
    ('{"a": 1, "b": ', True),                           # cut between keys
    ('{"a": "x"}', False),                              # complete
    (PROSE, False),                                     # prose — a prompt problem, not a cut
    ('{"a":1} trailing prose', False),                  # complete object, junk after
    (r'{"a":"he said \"hi\" to her"}', False),          # escaped quotes must not read as open
    ("", False),                                        # empty is not "truncated"
])
def test_looks_truncated_separates_a_cut_stream_from_bad_output(raw, truncated):
    assert looks_truncated(raw) is truncated


# ── the retry itself ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ner_retries_a_truncated_response_once_and_recovers():
    backend = _Scripted(TRUNCATED, NER_OK)
    ctx = PipelineContext(user_input="dez por dia")
    ctx.noumeno = _noumeno_result()

    await IntentAnalyzer().process(ctx, backend)

    assert backend.calls == 2
    assert ctx.intent is not None
    assert ctx.intent.intent_class == "INFORMATION_REQUEST"


@pytest.mark.asyncio
async def test_noumeno_retries_a_truncated_response_once_and_recovers():
    backend = _Scripted(TRUNCATED, NOUMENO_OK)
    ctx = PipelineContext(user_input="dez por dia")

    await Noumeno(_ConstEmbedder()).process(ctx, backend)

    assert backend.calls == 2
    assert ctx.noumeno is not None
    assert ctx.noumeno.rewritten == "Ten calls a day."


@pytest.mark.asyncio
async def test_retry_bills_both_attempts():
    """A retry that did not show up in the token count would understate the turn.

    The stages fold these numbers straight into StageMetrics and the host meters on them, so
    a silently-free second call is a billing defect, not a rounding detail."""
    backend = _Scripted(TRUNCATED, NER_OK)
    ctx = PipelineContext(user_input="dez por dia")
    ctx.noumeno = _noumeno_result()

    await IntentAnalyzer().process(ctx, backend)

    assert ctx.intent is not None
    assert ctx.intent.metrics.tokens_in == 200      # 100 + 100, not 100
    assert ctx.intent.metrics.tokens_out == 20


@pytest.mark.asyncio
async def test_a_malformed_response_is_NOT_retried():
    """Prose instead of JSON is a prompt/model problem: a second call returns the same thing.

    This is the half of the contract that keeps the retry cheap — without it, every parse
    failure would double the stage's cost and latency to no purpose."""
    backend = _Scripted(PROSE, NER_OK)
    ctx = PipelineContext(user_input="dez por dia")
    ctx.noumeno = _noumeno_result()

    with pytest.raises(StageParseError):
        await IntentAnalyzer().process(ctx, backend)

    assert backend.calls == 1


@pytest.mark.asyncio
async def test_truncated_twice_still_raises():
    """Failure stays LOUD. No silent degradation: a NOUMENO that quietly passed the original
    through would hand the NER un-rewritten text and lose the rewrite with nobody the wiser."""
    backend = _Scripted(TRUNCATED, TRUNCATED)
    ctx = PipelineContext(user_input="dez por dia")
    ctx.noumeno = _noumeno_result()

    with pytest.raises(StageParseError):
        await IntentAnalyzer().process(ctx, backend)

    assert backend.calls == 2
