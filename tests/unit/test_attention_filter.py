"""Unit tests for cogno_anima.routing.attention.AttentionFilter (pure heuristic)."""

from cogno_anima.routing.attention import AttentionFilter
from tests.unit.test_drift import make_intent_result


def test_empty_candidates_returns_empty():
    af = AttentionFilter()
    assert af.focus(make_intent_result(), []) == []


def test_temporal_match_scores_highest():
    af = AttentionFilter(top_n=1)
    intent = make_intent_result(temporal_class="RECENT")
    out = af.focus(intent, ["RECENT:TECH:server", "TIMELESS:MATH:integral"])
    assert out == ["RECENT:TECH:server"]


def test_domain_overlap_scores():
    af = AttentionFilter(top_n=1)
    intent = make_intent_result(temporal_class="TIMELESS")
    intent.domains = ["TECH"]
    out = af.focus(intent, ["TECH:docker", "HEALTH:diet"])
    assert out == ["TECH:docker"]


def test_people_match_scores():
    af = AttentionFilter(top_n=1)
    intent = make_intent_result()
    intent.entities_people = ["José Manzoli"]
    out = af.focus(intent, ["note about josé", "unrelated note"])
    assert out == ["note about josé"]


def test_goal_keyword_overlap():
    af = AttentionFilter(top_n=1)
    intent = make_intent_result(goal="configure docker")
    out = af.focus(intent, ["docker setup steps", "weather forecast"])
    assert out == ["docker setup steps"]


def test_top_n_limit_and_ordering():
    af = AttentionFilter(top_n=2)
    intent = make_intent_result(temporal_class="RECENT", goal="docker")
    intent.domains = ["TECH"]
    candidates = [
        "RECENT:TECH:docker",   # temporal + domain + goal → highest
        "TECH:docker",          # domain + goal
        "HEALTH:diet",          # nothing
    ]
    out = af.focus(intent, candidates)
    assert out == ["RECENT:TECH:docker", "TECH:docker"]
    assert "HEALTH:diet" not in out
