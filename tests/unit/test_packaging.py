"""The prompt templates must survive the wheel.

``load_prompt`` reads ``cogno_anima/prompt_templates/<stage>/<name>.txt`` at runtime, and those
are plain data files: setuptools ships them only because ``[tool.setuptools.package-data]`` says
so. Nothing fails anywhere near a mistake here — the repo has the files, an editable install
reads them off disk, this whole suite passes, and only an install FROM THE WHEEL comes up with
an empty prompt.

That exact failure shipped in a sibling on 2026-08-03: `cogno-praxis` declared package data for
three verticals and not for the fourth, so the CLOSER persona installed from the wheel with an
EMPTY system prompt — in CI and in the Docker image, for as long as nobody looked. The demo box
runs from source and never showed it.

This package is one glob away from the same thing (``prompt_templates/**/*.txt``): it covers
today's two stages, and would silently miss a prompt saved under any other extension.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TEMPLATES = ROOT / "cogno_anima" / "prompt_templates"


def _declared_globs() -> list[str]:
    """The package-data globs for ``cogno_anima``.

    Regex rather than tomllib: the supported floor is 3.10, where tomllib does not exist, and
    a dev dependency on `tomli` to read one line of our own config is the worse trade.
    """
    text = (ROOT / "pyproject.toml").read_text()
    section = text.split("[tool.setuptools.package-data]", 1)
    assert len(section) == 2, "package-data section vanished — this test guards it"
    body = section[1].split("\n[", 1)[0]
    m = re.search(r"^cogno_anima\s*=\s*\[(?P<globs>[^\]]*)\]", body, re.M)
    assert m, "no package-data entry for cogno_anima"
    return re.findall(r'"([^"]+)"', m.group("globs"))


def test_every_shipped_prompt_is_covered_by_a_declared_glob():
    from fnmatch import fnmatch

    globs = _declared_globs()
    uncovered = [
        str(p.relative_to(ROOT / "cogno_anima"))
        for p in TEMPLATES.rglob("*")
        if p.is_file() and not any(fnmatch(str(p.relative_to(ROOT / "cogno_anima")), g)
                                   for g in globs)
    ]
    assert not uncovered, (
        f"{uncovered} live under prompt_templates/ and match no package-data glob {globs}, so "
        f"they would NOT be installed from a wheel. An editable install hides this; the wheel "
        f"the image builds from does not."
    )


def test_the_prompts_the_stages_load_are_actually_there():
    """The glob could be right and the files still missing. Assert the ones the code reads."""
    for stage in ("noumeno", "ner"):
        for slot in ("system", "user"):
            f = TEMPLATES / stage / f"{slot}.txt"
            assert f.is_file(), f"{stage}/{slot}.txt is missing"
            assert f.read_text().strip(), f"{stage}/{slot}.txt is EMPTY"
