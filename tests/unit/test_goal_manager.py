"""Unit tests for cogno_anima.routing.goal.GoalManager (pure, async)."""

import pytest

from cogno_anima.routing.goal import GoalManager, _jaccard, _tokenize


class _SimRecorder:
    """Async similarity_fn double — records (a, b) calls and returns a fixed value."""
    def __init__(self, value: float = 0.9) -> None:
        self.value = value
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, a: str, b: str) -> float:
        self.calls.append((a, b))
        return self.value


# ── First turn / fresh start ──────────────────────────────────────────────

async def test_first_turn_is_new_similarity_one():
    gm = GoalManager()
    status, goal, sim = await gm.update("configure docker", "ACTION_REQUEST", domains=["TECH"])
    assert status == "NEW"
    assert goal == "configure docker"
    assert sim == 1.0


async def test_no_goal_first_turn_stays_new():
    gm = GoalManager()
    status, goal, sim = await gm.update(None, "SOCIAL")
    assert status == "NEW"
    assert goal is None and sim == 1.0


async def test_social_greeting_does_not_establish_a_goal_to_abandon():
    # "E aí" (SOCIAL) must NOT set a persistent goal — otherwise stating the real request next
    # turn looks like the user ABANDONED the greeting. It should read as a fresh NEW goal.
    gm = GoalManager()
    s1, g1, _ = await gm.update("greet the user", "SOCIAL", domains=[])
    assert s1 == "NEW" and g1 is None            # greeting sets no active goal
    s2, g2, _ = await gm.update("schedule an appointment", "ACTION_REQUEST", domains=["SCHEDULING"])
    assert s2 == "NEW" and g2 == "schedule an appointment"   # NOT ABANDONED


# ── Fast-paths (no embedding) ─────────────────────────────────────────────

async def test_stage0_clarification_is_ongoing():
    sim_fn = _SimRecorder(0.0)
    gm = GoalManager(similarity_fn=sim_fn)
    await gm.update("configure docker", "ACTION_REQUEST", domains=["TECH"])
    status, goal, sim = await gm.update("what?", "CLARIFICATION", domains=[])
    assert status == "ONGOING"
    assert sim == 1.0
    assert sim_fn.calls == []     # never reached Stage 2


async def test_stage1_domain_match_is_ongoing():
    sim_fn = _SimRecorder(0.0)
    gm = GoalManager(similarity_fn=sim_fn)
    await gm.update("configure docker", "ACTION_REQUEST", domains=["TECH"])
    status, _, sim = await gm.update("fix docker error", "ACTION_REQUEST", domains=["TECH"])
    assert status == "ONGOING"
    assert sim == 1.0
    assert sim_fn.calls == []


async def test_stage15_anaphoric_pii_is_ongoing():
    sim_fn = _SimRecorder(0.0)
    gm = GoalManager(similarity_fn=sim_fn)
    await gm.update("share my CPF", "ACTION_REQUEST", domains=["FINANCE"])
    status, _, sim = await gm.update(
        "who can see it?", "INFORMATION_REQUEST", domains=[], pii_session_hint=True,
    )
    assert status == "ONGOING"
    assert sim == 1.0
    assert sim_fn.calls == []


async def test_stage16_context_dependent_is_ongoing():
    sim_fn = _SimRecorder(0.0)
    gm = GoalManager(similarity_fn=sim_fn)
    await gm.update("configure docker", "ACTION_REQUEST", domains=["TECH"])
    status, _, sim = await gm.update(
        "and theirs?", "INFORMATION_REQUEST", domains=[], context_dependent=True,
    )
    assert status == "ONGOING"
    assert sim == 1.0
    assert sim_fn.calls == []


# ── Stage 2 semantic ──────────────────────────────────────────────────────

async def test_stage2_high_similarity_ongoing_returns_computed_sim():
    sim_fn = _SimRecorder(0.82)
    gm = GoalManager(similarity_fn=sim_fn)
    await gm.update("configure docker", "ACTION_REQUEST", domains=[])
    status, _, sim = await gm.update("what does daemon mean", "INFORMATION_REQUEST", domains=[])
    assert status == "ONGOING"
    assert sim == pytest.approx(0.82)
    assert len(sim_fn.calls) == 1


async def test_stage2_low_similarity_abandoned_returns_computed_sim():
    sim_fn = _SimRecorder(0.10)
    gm = GoalManager(similarity_fn=sim_fn)
    await gm.update("configure docker", "ACTION_REQUEST", domains=[])
    status, goal, sim = await gm.update("best pizza place", "INFORMATION_REQUEST", domains=[])
    assert status == "ABANDONED"
    assert goal == "best pizza place"      # new active goal took over
    assert sim == pytest.approx(0.10)


async def test_one_sided_enrichment_anchor_only():
    """Stage 2 enriches the active-goal anchor with history; the query is untouched."""
    sim_fn = _SimRecorder(0.05)   # force ABANDONED so history accumulates
    gm = GoalManager(similarity_fn=sim_fn)
    await gm.update("goal A", "ACTION_REQUEST", domains=[])           # NEW, history=[A]
    await gm.update("goal B", "ACTION_REQUEST", domains=[])           # ABANDONED, history=[B,A]
    sim_fn.value = 0.9
    await gm.update("goal C", "ACTION_REQUEST", domains=[])           # Stage 2 with history
    a, b = sim_fn.calls[-1]
    assert b == "goal C"                          # query unchanged
    assert a == "goal B | goal A"                 # anchor = active + history (excl. active)


# ── Jaccard fallback ──────────────────────────────────────────────────────

async def test_jaccard_fallback_without_similarity_fn():
    gm = GoalManager()    # no similarity_fn → Jaccard
    await gm.update("configure docker server", "ACTION_REQUEST", domains=[])
    # high lexical overlap → ONGOING under Jaccard's lower default? default threshold is
    # 0.75 (cosine-calibrated); Jaccard rarely reaches it, so expect ABANDONED here.
    status, _, sim = await gm.update("configure docker daemon", "ACTION_REQUEST", domains=[])
    assert 0.0 <= sim <= 1.0
    # with an explicit Jaccard-calibrated threshold it continues:
    gm2 = GoalManager(similarity_threshold=0.3)
    await gm2.update("configure docker server", "ACTION_REQUEST", domains=[])
    status2, _, _ = await gm2.update("configure docker daemon", "ACTION_REQUEST", domains=[])
    assert status2 == "ONGOING"


async def test_similarity_fn_error_falls_back_to_jaccard():
    async def boom(a, b):
        raise RuntimeError("embedder down")
    gm = GoalManager(similarity_fn=boom, similarity_threshold=0.3)
    await gm.update("configure docker server", "ACTION_REQUEST", domains=[])
    status, _, sim = await gm.update("configure docker daemon", "ACTION_REQUEST", domains=[])
    assert status == "ONGOING"      # degraded to Jaccard, did not crash
    assert 0.0 <= sim <= 1.0


# ── Completion / carry-over ───────────────────────────────────────────────

async def test_social_after_goal_completes():
    gm = GoalManager()
    await gm.update("configure docker", "ACTION_REQUEST", domains=["TECH"])
    status, goal, sim = await gm.update("thanks!", "SOCIAL", domains=[])
    assert status == "COMPLETED"
    assert goal is None and sim == 1.0


async def test_no_goal_carryover_keeps_ongoing():
    gm = GoalManager()
    await gm.update("configure docker", "ACTION_REQUEST", domains=["TECH"])
    status, goal, sim = await gm.update(None, "INFORMATION_REQUEST", domains=[])
    assert status == "ONGOING"
    assert goal == "configure docker" and sim == 1.0


# ── State round-trip ──────────────────────────────────────────────────────

async def test_to_dict_from_dict_round_trip():
    gm = GoalManager()
    await gm.update("goal A", "ACTION_REQUEST", domains=["TECH"])
    snapshot = gm.to_dict()

    gm2 = GoalManager()
    gm2.from_dict(snapshot)
    assert gm2.active_goal == "goal A"
    assert gm2.goal_status == "NEW"
    assert gm2.goal_history == ["goal A"]
    # continuity decision survives rehydration (same TECH domain → ONGOING)
    status, _, _ = await gm2.update("fix the TECH thing", "ACTION_REQUEST", domains=["TECH"])
    assert status == "ONGOING"


def test_helpers():
    assert _jaccard(set(), set()) == 1.0
    assert _jaccard({"a"}, set()) == 0.0
    assert _jaccard({"a", "b"}, {"a"}) == pytest.approx(0.5)
    assert _tokenize("Olá, Docker-99!") == {"olá", "docker", "99"}
