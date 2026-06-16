"""Unit tests for the NER anaphora → context_dependent deterministic fallback."""

import json
from pathlib import Path

from cogno_anima.types import StageMetrics
from cogno_anima.stages.ner import IntentAnalyzer, _has_anaphora
from tests.conftest import StubBackend

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


def _ner():
    return IntentAnalyzer(backend=StubBackend(), prompts_dir=PROMPTS_DIR)


def _m():
    return StageMetrics(stage="ner", elapsed_ms=1.0, tokens_in=1, tokens_out=1, model="test")


def _raw(context_dependent=False):
    return json.dumps({
        "intent_class": "INFORMATION_REQUEST", "sentiment": "NEUTRAL",
        "confidence": 0.9, "temporal_class": "TIMELESS", "triad_signal": "BALANCED",
        "context_dependent": context_dependent,
    })


# ── the matcher itself ───────────────────────────────────────────────

def test_has_anaphora_matches_strong_markers():
    assert _has_anaphora("deles, qual o mais usado?")
    assert _has_anaphora("of those, which is best?")
    assert _has_anaphora("", "give me the same one")
    assert not _has_anaphora("quanto tá o bitcoin?")
    assert not _has_anaphora("what is the capital of France?")


def test_no_false_match_inside_words():
    # 'modeles' must not match 'deles' (word boundary)
    assert not _has_anaphora("os modelos novos")


# ── the fallback inside _parse ───────────────────────────────────────

def test_anaphora_flips_context_dependent_when_llm_misses_it():
    r = _ner()._parse(_raw(context_dependent=False), _m(),
                      original="deles, qual o mais usado?",
                      rewritten="of them, which is the most used?")
    assert r.context_dependent is True


def test_non_anaphoric_stays_false():
    r = _ner()._parse(_raw(context_dependent=False), _m(),
                      original="quanto tá o bitcoin?", rewritten="what is bitcoin's price?")
    assert r.context_dependent is False


def test_llm_true_is_preserved():
    r = _ner()._parse(_raw(context_dependent=True), _m(),
                      original="plain text", rewritten="plain text")
    assert r.context_dependent is True


def test_anaphora_detected_in_rewritten_only():
    # user typed English; original carries the marker too, but check EN path
    r = _ner()._parse(_raw(context_dependent=False), _m(),
                      original="which of those is best", rewritten="which of those is best")
    assert r.context_dependent is True
