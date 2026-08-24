from __future__ import annotations

from pathlib import Path
import hashlib
from typing import Optional

# Prompt templates ship INSIDE the package (cogno_anima/prompt_templates/) so a
# plain `pip install cogno-anima` can load them — they are declared as
# package-data in pyproject.toml. PROMPTS_ROOT is public for tooling/tests.
PROMPTS_ROOT = Path(__file__).resolve().parent / "prompt_templates"
_PROMPTS_ROOT = PROMPTS_ROOT  # backward-compatible alias

def _clean_prompt(text: str) -> str:
    """Strip YAML frontmatter and TODO(docs) lines."""
    if text.startswith("---\n"):
        parts = text.split("\n---\n", 1)
        if len(parts) == 2:
            text = parts[1]

    return "\n".join(
        line for line in text.splitlines()
        if not line.strip().startswith("TODO(docs)")
    ).strip()

def load_prompt(
    stage: str,
    prompt_name: str,
    prompts_dir: Optional[Path] = None,
) -> str:
    """Loads a prompt template for a stage."""
    root = prompts_dir or _PROMPTS_ROOT
    path = root / stage / prompt_name
    if path.exists():
        return _clean_prompt(path.read_text(encoding="utf-8"))
    return ""


# ── prompt identity ──────────────────────────────────────────────────────────────────
# Twelve hex chars: this rides in a per-call telemetry record, and a full digest buys
# collision headroom nobody needs to tell a handful of prompt configurations apart.
_DIGEST_CHARS = 12


def prompt_digest(*parts: Optional[str]) -> str:
    """The ONE way a prompt configuration is named, across every layer and every repo.

    Each layer digests the text IT authors — a stage its own template, the host its persona
    slots — because the layer that authors a text is the only one that has it before it is
    rendered. What must NOT vary is the algorithm: two layers naming the same text differently
    makes the labels incomparable, which is the one thing a label cannot afford.

    **What belongs in a digest, and the line is not a style preference.** Only text authored at
    DEPLOYMENT level: the shipped templates and a tenant's own business direction. Never text
    authored at IDENTITY level — a contact's memories, operator notes, the transcript. Those
    ride in the injected context, and they are excluded for two independent reasons:

      * digesting them makes the sha unique per CONTACT, so the label stops grouping anything,
        which is the entire purpose of having one;
      * and text stored behind such a label would carry personal data into whatever store
        holds it — a store that a contact's deletion request would not know to reach.

    Empty parts are skipped; all-empty returns "" — "nothing a deployment set", which is a
    different claim from "not recorded".
    """
    joined = "\n".join(p for p in parts if p)
    if not joined:
        return ""
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:_DIGEST_CHARS]
