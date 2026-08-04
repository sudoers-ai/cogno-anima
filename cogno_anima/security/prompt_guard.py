"""Neutralise tool-call trigger tokens in UNTRUSTED text before it enters an LLM prompt.

Tool results are third-party data — a calendar title, an email body, an MCP response, a row a
customer typed. They are rendered into the EGO's prompt (to decide the next call), into the
SUPEREGO judge's prompt (the fail-CLOSED gate) and into the voicer's payload. If that text can
carry the scaffolding the text-fallback parser rescues, data becomes an executed side effect.

**Why this is not just a regex.** Targeted defanging is fragile in two ways that were both
demonstrated live: ``re.sub`` replaces non-overlapping matches only (a repeated key leaves a
second, untouched occurrence), and a replacement can itself FORM a valid tag
(``[a[b]]`` → ``[a(b)]``). So the readable, targeted pass runs to a fixed point and is then
**verified with the real parser**: if anything still reads as a call, the structural characters
are neutralised outright. The parser is the oracle — no regex cleverness has to be trusted.
"""

from __future__ import annotations

import re
from typing import Iterable

# Escalation: without ``<``, ``{`` or ``[`` none of the three rescue formats can match
# (Format 1 needs <TOOL_CALL>, Format 2 needs a {...} object, Format 3 needs [name]).
# Only ever applied to text the parser still accepts, i.e. an actual attack payload.
_NEUTRALISE = str.maketrans({"<": "(", ">": ")", "{": "(", "}": ")", "[": "(", "]": ")"})

_MAX_PASSES = 8   # fixed-point bound; the payloads that need >2 are pathological


def _defang_once(text: str, names: str) -> str:
    """One readable pass: remove the tag blocks, break the JSON key, unwrap the brackets."""
    # A result must not break out of the <tool_output> fence that wraps it.
    text = re.sub(r"(?i)</?tool_output[^>]*>", "", text)
    # Format 1: a <TOOL_CALL> block is never legitimate tool output — drop it WHOLE (stripping
    # only the tags would leave the inner JSON as a live Format-2 call).
    text = re.sub(r"(?is)<TOOL_CALL>.*?</TOOL_CALL>", " ", text)
    text = re.sub(r"(?i)</?TOOL_CALL>", " ", text)          # stray unpaired tag
    if not names:
        return text
    # Format 3: [tool] / [tool(args)] naming a real tool, anywhere (the parser matches inline,
    # with the parens optional) → unwrap the brackets.
    text = re.sub(rf"\[({names})(\([^)]*\))?\]", r"(\1\2)", text)
    # Format 2: {… "tool":"realtool" … "args":{…} …} → break the `"tool"` KEY so the parser's
    # `"tool"\s*:` match misses. The optional ``functions.`` prefix must be covered: the parser
    # strips that namespace hallucination before checking the name.
    text = re.sub(rf'"tool"(\s*:\s*"(?:functions\.)?(?:{names})")', r'"tool "\1', text)
    return text


def parses_as_tool_call(text: str, tool_names: "Iterable[str]") -> bool:
    """True when the REAL parser would rescue a tool call out of ``text``. The oracle."""
    names = [n for n in tool_names if n]
    if not names or not text:
        return False
    try:
        from cogno_synapse.tool_parsing import parse_tool_calls_from_text
    except ImportError:      # transport lib absent (never in a real deployment) → stay strict
        return True
    tools = [{"function": {"name": n}} for n in names]
    return bool(parse_tool_calls_from_text(text, tools))


def sanitize_untrusted(text: str, tool_names: "Iterable[str]") -> str:
    """Return ``text`` with every tool-call trigger neutralised.

    ``tool_names`` is the exposed tool set — REQUIRED, and the same set the parser validates
    against, so a citation like ``[Smith(2020)]`` or an unrelated JSON blob is left alone while
    anything naming a real tool is defanged.

    Guarantee: ``parses_as_tool_call(sanitize_untrusted(x, names), names)`` is False.
    """
    if not text:
        return ""
    names_set = {n for n in tool_names if n}
    names = "|".join(re.escape(n) for n in sorted(names_set, key=len, reverse=True))
    for _ in range(_MAX_PASSES):          # fixed point: a replacement can reveal/form another
        new = _defang_once(text, names)
        if new == text:
            break
        text = new
    if parses_as_tool_call(text, names_set):
        # The targeted pass did not hold (an overlap or a self-forming tag). Stop trusting the
        # regex and take the structure away — this only fires on a real payload.
        text = text.translate(_NEUTRALISE)
    return text
