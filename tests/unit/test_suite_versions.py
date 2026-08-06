"""Suite versioning pins (plan 0.6).

Absolute counts only compare within the same suite version — the field diet
took the NER suite 127→125 checks and every percentage self-inflated. These
tests make suite drift IMPOSSIBLE to ship silently: any case addition, removal
or expected-value edit changes the canonical hash, and the suite stays
unshippable until the human bumps the SUITE_ID and re-records the pin
(`python -m cognobench.suites --update`) — the same pattern as the
domains↔prompt alignment test.
"""

from cognobench import suites


def test_every_dimension_has_a_pinned_suite_version():
    live, pinned = suites.registry(), suites.recorded()
    assert set(live) == set(pinned), (
        f"unpinned dimensions: {set(live) ^ set(pinned)} — run "
        f"`python -m cognobench.suites --update` after reviewing the change"
    )


def test_suite_hash_matches_pin_or_the_id_was_bumped():
    live, pinned = suites.registry(), suites.recorded()
    for dim, info in live.items():
        pin = pinned[dim]
        if info["suite_hash"] == pin["suite_hash"]:
            continue
        # Hash moved: the ONLY legitimate state is a bumped SUITE_ID that was
        # then re-recorded — which makes hashes equal again. Anything else is a
        # silent suite mutation.
        raise AssertionError(
            f"{dim}: case data changed (hash {pin['suite_hash']} → "
            f"{info['suite_hash']}) but the pin was not updated. Bump SUITE_ID "
            f"in the {dim} cases file (e.g. {pin['suite_id']} → next version), "
            f"then run `python -m cognobench.suites --update`, and re-baseline "
            f"published numbers on the new version."
        )


def test_suite_id_bump_requires_rerecording():
    """A bumped SUITE_ID with a stale pin is also unshippable — the pin file is
    what run artifacts and published tables are validated against."""
    live, pinned = suites.registry(), suites.recorded()
    for dim, info in live.items():
        assert info["suite_id"] == pinned[dim]["suite_id"], (
            f"{dim}: SUITE_ID is {info['suite_id']} but the pin records "
            f"{pinned[dim]['suite_id']} — run `python -m cognobench.suites --update`"
        )


def test_hash_is_sensitive_to_case_edits():
    """The canonical hash must move when ANY expected value changes — otherwise
    the whole mechanism is decorative. Mutate a copy of a case and compare."""
    import dataclasses
    from cognobench.ner_cases import NER_CASES

    baseline = suites.canonical_suite_hash(NER_CASES)
    mutated = [dataclasses.replace(NER_CASES[0], expect_intent="ACTION_REQUEST")] \
        + list(NER_CASES[1:])
    assert suites.canonical_suite_hash(mutated) != baseline

    reordered = list(NER_CASES[::-1])
    assert suites.canonical_suite_hash(reordered) != baseline, (
        "case ORDER must be hashed — --limit slices positionally"
    )
