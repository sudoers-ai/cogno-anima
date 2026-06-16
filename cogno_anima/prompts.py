from __future__ import annotations

from pathlib import Path
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
