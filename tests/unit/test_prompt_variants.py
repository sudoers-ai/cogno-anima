"""Contract guard for the prompt-slimming experiment's variant trees.

A variant is only allowed to be *smaller*, never to break the prompt↔code contract
the shipped templates are held to. Without this, a slimmed prompt that quietly drops
a domain or a vocabulary value would still run — the stage would coerce the model's
now-unteachable answers into fallbacks and the bench would score the damage as a
model result. That is the expensive failure this experiment must not produce.

Skips cleanly when no variants exist (they are generated, not committed as fixtures:
``python3 cognobench/prompt_variants/build_variants.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.test_pipeline import (
    _parse_prompt_domains,
    _parse_prompt_mandatory_tags,
    vocab_values_missing_from,
)
from cogno_anima.stages.ner import NER_KNOWLEDGE_DOMAINS, VALID_MANDATORY
from cogno_anima.prompts import load_prompt

VARIANTS_DIR = Path(__file__).parent.parent.parent / "cognobench" / "prompt_variants"

_REQUIRED = (("noumeno", "system.txt"), ("noumeno", "user.txt"),
             ("ner", "system.txt"), ("ner", "user.txt"))


def _variant_trees() -> list[Path]:
    if not VARIANTS_DIR.is_dir():
        return []
    return sorted(p for p in VARIANTS_DIR.iterdir()
                  if p.is_dir() and (p / "ner" / "system.txt").exists())


VARIANTS = _variant_trees()
requires_variants = pytest.mark.skipif(
    not VARIANTS, reason="no prompt variants generated (run build_variants.py)")


@requires_variants
@pytest.mark.parametrize("tree", VARIANTS, ids=lambda p: p.name)
def test_variant_supplies_every_required_prompt(tree: Path):
    """A variant tree must supply every prompt the pipeline loads.

    ``load_prompt`` returns "" for a missing file instead of raising, so a partial
    tree would hand a stage an empty system prompt and still score."""
    empty = [f"{stage}/{name}" for stage, name in _REQUIRED
             if not load_prompt(stage, name, prompts_dir=tree).strip()]
    assert not empty, f"{tree.name} is missing or empty: {empty}"


@requires_variants
@pytest.mark.parametrize("tree", VARIANTS, ids=lambda p: p.name)
def test_variant_keeps_the_domains_contract(tree: Path):
    """The closed `domains` list must stay byte-identical to NER_KNOWLEDGE_DOMAINS."""
    assert NER_KNOWLEDGE_DOMAINS == _parse_prompt_domains(tree)


@requires_variants
@pytest.mark.parametrize("tree", VARIANTS, ids=lambda p: p.name)
def test_variant_keeps_the_mandatory_tags_contract(tree: Path):
    assert VALID_MANDATORY == _parse_prompt_mandatory_tags(tree)


@requires_variants
@pytest.mark.parametrize("tree", VARIANTS, ids=lambda p: p.name)
def test_variant_still_teaches_every_vocab_value(tree: Path):
    """Every value the stage accepts must still be taught by the slimmed prompt.

    This is the check most likely to catch an over-eager cut: some vocabulary is
    only ever demonstrated inside the worked examples, so deleting an example can
    silently unteach a value the code still coerces against."""
    text = (tree / "ner" / "system.txt").read_text(encoding="utf-8")
    missing = vocab_values_missing_from(text)
    assert not missing, f"{tree.name} stopped teaching: {missing}"


@requires_variants
@pytest.mark.parametrize("tree", VARIANTS, ids=lambda p: p.name)
def test_variant_is_actually_smaller(tree: Path):
    """A variant that is not smaller than the baseline has no reason to exist."""
    baseline = load_prompt("ner", "system.txt")
    assert len(load_prompt("ner", "system.txt", prompts_dir=tree)) < len(baseline)
