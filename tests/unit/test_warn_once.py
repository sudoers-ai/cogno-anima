"""``utils.WarnOnce`` — the "say it once" gate shared by the SUPEREGO's carrier warnings.

Its own tests, because it exists precisely because the same machinery, hand-rolled inside one
stage, produced a defect in three consecutive review rounds: a key that carried the offending
VALUE (unbounded cardinality), and one set shared between two features (a flood of keys from
one silently evicted the other's single entry).
"""

from cogno_anima.utils import WarnOnce


def test_says_it_once_per_key():
    w = WarnOnce()
    assert w.first("a") is True
    assert w.first("a") is False
    assert w.first("b") is True
    assert len(w) == 2


def test_the_cap_evicts_and_that_is_what_makes_a_shape_key_safe():
    """The bound is not decoration: it is the reason a shape-key is SAFE to use.

    With a well-chosen key the cardinality is small by construction and the cap is never
    reached — which is exactly why its absence would go unnoticed. If it dies silently and a
    later change reintroduces a value-smelling key (three rounds running, in this repo's
    history), the gate grows without limit in a long-lived worker and nothing catches it.
    """
    w = WarnOnce(limit=2)
    assert [w.first(k) for k in ("a", "b")] == [True, True]
    assert len(w) == 2
    w.first("c")                      # at the cap → cleared wholesale, then "c" is recorded
    assert len(w) == 1
    assert w.first("a") is True       # ...and a previously-seen key speaks again after eviction


def test_reset_is_total():
    w = WarnOnce()
    w.first("a")
    w.reset()
    assert len(w) == 0 and w.first("a") is True


def test_two_instances_do_not_share_state():
    # one gate per feature: a flood in one must never evict the other's entry
    a, b = WarnOnce(limit=2), WarnOnce(limit=2)
    a.first("kept")
    for i in range(50):
        b.first(f"flood-{i}")
    assert a.first("kept") is False    # still remembered
    assert len(a) == 1
