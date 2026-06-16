"""Unit tests for cogno_anima.routing.intention.IntentionTracker (pure state)."""

from types import SimpleNamespace

from cogno_anima.routing.intention import IntentionTracker


def _intent(intent_class="ACTION_REQUEST", goal=None, concepts=None, objects=None):
    return SimpleNamespace(
        intent_class=intent_class,
        goal=goal,
        entities_concepts=concepts or [],
        entities_objects=objects or [],
    )


def test_creates_intention_on_new():
    t = IntentionTracker()
    active = t.update(_intent(goal="configure docker"), goal_status="NEW")
    assert active == ["configure docker"]


def test_infers_intention_without_goal():
    t = IntentionTracker()
    active = t.update(_intent(goal=None, concepts=["docker"]), goal_status="NEW")
    assert active == ["perform task: docker"]


def test_dedup_same_text():
    t = IntentionTracker()
    t.update(_intent(goal="configure docker"), goal_status="NEW")
    active = t.update(_intent(goal="configure docker"), goal_status="ONGOING")
    assert active == ["configure docker"]


def test_completed_closes_all():
    t = IntentionTracker()
    t.update(_intent(goal="a"), goal_status="NEW")
    t.update(_intent(goal="b", intent_class="ACTION_REQUEST"), goal_status="ABANDONED")
    active = t.update(_intent(goal="c", intent_class="SOCIAL"), goal_status="COMPLETED")
    assert active == []


def test_abandoned_closes_oldest_and_adds_new():
    t = IntentionTracker()
    t.update(_intent(goal="first"), goal_status="NEW")
    active = t.update(_intent(goal="second"), goal_status="ABANDONED")
    # oldest ("first") closed, "second" added
    assert active == ["second"]


def test_non_intention_class_does_not_create():
    t = IntentionTracker()
    active = t.update(_intent(goal="hello", intent_class="SOCIAL"), goal_status="NEW")
    assert active == []


def test_fifo_eviction_over_limit():
    t = IntentionTracker()
    for i in range(7):
        # each new goal on a NEW status creates a fresh intention
        t.update(_intent(goal=f"goal{i}"), goal_status="NEW")
    active = t.active
    assert len(active) == 5
    # oldest two evicted
    assert "goal0" not in active and "goal1" not in active
    assert "goal6" in active


def test_to_dict_from_dict_round_trip():
    t = IntentionTracker()
    t.update(_intent(goal="configure docker"), goal_status="NEW")
    snapshot = t.to_dict()

    t2 = IntentionTracker()
    t2.from_dict(snapshot)
    assert t2.active == ["configure docker"]


def test_reset_clears():
    t = IntentionTracker()
    t.update(_intent(goal="x"), goal_status="NEW")
    t.reset()
    assert t.active == []
