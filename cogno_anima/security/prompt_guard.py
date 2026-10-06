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

**The second half: the STRUCTURE of the prompt the text lands in (F4.3, 2026-10-06).** A tool
result that cannot call a tool can still pretend to be the prompt around it. Two shapes, both
measured on the rendered prompts (``tests/unit/test_injection_by_tool_data.py``):

* **a fence closed from inside.** Each fence used to strip only its OWN tag, and only
  ``<tool_output>`` went through here, so a held message carrying ``</held_message>`` closed
  the judge's fence around it and every line after it read as the judge's own prompt.
  :data:`FENCE_TAGS` is every fence this library wraps untrusted text in, and none of them
  survives inside untrusted text;
* **a forged section header.** The voice renders the executor's data UNFENCED, and the host's
  context block (``mk.EGO_CONTEXT``: a delivered message, memories, the conversation) is
  unfenced in all three prompts, so a line reading ``# Execution verdict (HARD RULE)`` or
  ``# Correction requested`` started a section indistinguishable from the real one.
  :func:`defang_headers` escapes, with a CommonMark backslash, every line that opens with a
  header this library RENDERS (:func:`reserved_headers`, derived from the stages' own tables,
  never copied). A header nobody here renders — a document's ``## Section`` line, a tenant's
  ``# MATERIAL`` — is left alone, so clean data renders byte for byte as before.
"""

from __future__ import annotations

import functools
import re
from typing import Iterable

# Escalation: without ``<``, ``{`` or ``[`` none of the three rescue formats can match
# (Format 1 needs <TOOL_CALL>, Format 2 needs a {...} object, Format 3 needs [name]).
# Only ever applied to text the parser still accepts, i.e. an actual attack payload.
_NEUTRALISE = str.maketrans({"<": "(", ">": ")", "{": "(", "}": ")", "[": "(", "]": ")"})

_MAX_PASSES = 8   # fixed-point bound; the payloads that need >2 are pathological

#: Every fence this library wraps UNTRUSTED text in, in any of its prompts: the executor's and
#: the judge's tool results, the judge's held messages and the ask recorded beside them, the
#: persona's business rules and the contact's note. Untrusted text may open or close NONE of
#: them — not only the one it happens to sit in, because the same text reaches several prompts.
#: A fence that is not this library's (a skill's own ``<excerpt>``) is that skill's to defang.
FENCE_TAGS: "tuple[str, ...]" = (
    "tool_output", "held_message", "held_ask", "business_rules", "contact_memo")

# A superset of the old ``</?tool_output[^>]*>`` — same shape, one alternation per fence.
_FENCE_TAG_RE = re.compile(r"(?i)</?(?:%s)[^>]*>" % "|".join(FENCE_TAGS))


@functools.lru_cache(maxsize=1)
def reserved_headers() -> "tuple[str, ...]":
    """Every top-level section header this library renders into a prompt — DERIVED, not listed.

    Read off the closed tables the prompts are inventoried by (the voice's, the judge's and the
    scope guard's, each pinned to its prompt by a sync test) plus the executor's own
    :data:`cogno_anima.stages.ego.PROMPT_HEADERS`. Imported lazily: this module is imported BY
    those stages, and the tables are only needed once a prompt is being built.
    """
    from cogno_anima.stages.ego import PROMPT_HEADERS
    from cogno_anima.stages.superego import SuperegoStage

    found = {h for table in (SuperegoStage._VOICE_BLOCKS, SuperegoStage._JUDGE_BLOCKS,
                             SuperegoStage._SCOPE_BLOCKS) for h, _slug in table}
    found.update(PROMPT_HEADERS)
    return tuple(sorted(h for h in found if h.startswith("#")))


@functools.lru_cache(maxsize=1)
def _reserved_line_re() -> "re.Pattern[str]":
    alts = "|".join(re.escape(h) for h in sorted(reserved_headers(), key=len, reverse=True))
    return re.compile(rf"(?m)^([ \t]*)(?=(?:{alts}))")


def defang_headers(text: str) -> str:
    """``text`` with every line that opens with a :func:`reserved_headers` header escaped.

    ``# Execution verdict (HARD RULE)`` becomes ``\\# Execution verdict (HARD RULE)``: the
    words stay readable (the data said them, and a reader may need to see that it did), but the
    line no longer starts a section — for the model reading the prompt, and for the inventories
    (``voice_prompt_inventory``/``judge_prompt_inventory``), which key on a header at the start
    of a line. Text with no such line comes back identical. Never raises; ``""`` for nothing.
    """
    if not text:
        return ""
    return _reserved_line_re().sub(r"\1\\", text)


def defang_structure(text: str) -> str:
    """The structural half alone — no fence tag of :data:`FENCE_TAGS`, no reserved header.

    For untrusted text whose tool-call triggers are not the risk: the host's context block
    (``mk.EGO_CONTEXT``), rendered into the executor, the judge and the voice. Running the
    tool-call pass there would rewrite every memory that names a tool in brackets; this changes
    only the bytes that could pretend to be the prompt. Identity on clean text.
    """
    if not text:
        return ""
    return defang_headers(_FENCE_TAG_RE.sub("", text))


def _defang_once(text: str, names: str) -> str:
    """One readable pass: remove the tag blocks, break the JSON key, unwrap the brackets."""
    # A result must not break out of the fence that wraps it — nor out of any other fence this
    # library renders, since the same text reaches several prompts (:data:`FENCE_TAGS`).
    text = _FENCE_TAG_RE.sub("", text)
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

    Also structural (:func:`defang_structure`'s half): no fence of :data:`FENCE_TAGS` can be
    opened or closed from inside, and no line opens with a :func:`reserved_headers` header.
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
    return defang_headers(text)
