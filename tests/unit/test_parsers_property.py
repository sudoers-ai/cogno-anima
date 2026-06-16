"""
Property-based tests for the text/JSON parsers — the components most exposed to
malformed LLM output in production.

Two invariants we never want to break:
  * ``parse_tool_calls_from_text`` must NEVER raise on arbitrary text, and must
    never return a tool name the dispatcher does not expose.
  * the NOUMENO/NER JSON parsers must yield a valid result OR raise
    ``StageParseError`` — never a raw AttributeError/TypeError from garbage.
"""

from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from tests.conftest import StubEmbedder
from cogno_core.llm.tool_parsing import parse_tool_calls_from_text
from cogno_core.stages.noumeno import Noumeno
from cogno_core.stages.ner import IntentAnalyzer
from cogno_core.types import StageMetrics
from cogno_core.errors import StageParseError

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

# A small fixed toolset (well-formed schema dicts) for the tool parser.
_TOOLS = [
    {"type": "function", "function": {"name": "get_balance", "parameters": {}}},
    {"type": "function", "function": {"name": "record_expense", "parameters": {}}},
    {"type": "function", "function": {"name": "book_appointment", "parameters": {}}},
]
_VALID_NAMES = {t["function"]["name"] for t in _TOOLS}


# ── parse_tool_calls_from_text ────────────────────────────────────────────────

@settings(max_examples=300)
@given(content=st.text())
def test_tool_parser_never_raises_on_arbitrary_text(content):
    """Any string is safe input — returns None or a list, never an exception."""
    result = parse_tool_calls_from_text(content, _TOOLS)
    assert result is None or isinstance(result, list)


@settings(max_examples=300)
@given(content=st.text())
def test_tool_parser_only_returns_known_tools(content):
    """A parsed call always names a tool the dispatcher exposes (no inventions)."""
    result = parse_tool_calls_from_text(content, _TOOLS)
    for call in result or []:
        assert call["function"]["name"] in _VALID_NAMES


# Fragments likely to confuse a naive parser, embedded in random noise.
_PAYLOADS = st.sampled_from([
    '<TOOL_CALL>{"tool": "get_balance", "args": {}}</TOOL_CALL>',
    '<TOOL_CALL>{not json}</TOOL_CALL>',
    '<TOOL_CALL></TOOL_CALL>',
    '{"tool": "record_expense", "args": {"amount": 5}}',
    '{"tool": "functions.get_balance", "args": {}}',
    '{"tool": "made_up_tool", "args": {}}',
    '[book_appointment(date="tomorrow")]',
    '[unknown_bracket]',
])


@settings(max_examples=300)
@given(pre=st.text(max_size=40), payload=_PAYLOADS, post=st.text(max_size=40))
def test_tool_parser_robust_with_embedded_payloads(pre, payload, post):
    """Real-ish tool fragments wrapped in noise never crash, names stay valid."""
    result = parse_tool_calls_from_text(pre + payload + post, _TOOLS)
    assert result is None or isinstance(result, list)
    for call in result or []:
        assert call["function"]["name"] in _VALID_NAMES


def test_tool_parser_extracts_valid_xml_call():
    """Sanity round-trip: a well-formed tag is actually found."""
    out = parse_tool_calls_from_text(
        'noise <TOOL_CALL>{"tool": "get_balance", "args": {}}</TOOL_CALL> tail', _TOOLS)
    assert out is not None and out[0]["function"]["name"] == "get_balance"


# ── NOUMENO / NER JSON parsers: valid result OR StageParseError ───────────────

@pytest.fixture(scope="module")
def noumeno():
    return Noumeno(embedder=StubEmbedder(), prompts_dir=PROMPTS_DIR)


@pytest.fixture(scope="module")
def ner():
    return IntentAnalyzer(prompts_dir=PROMPTS_DIR)


@settings(max_examples=300)
@given(raw=st.text())
def test_noumeno_parse_json_dict_or_stageparseerror(noumeno, raw):
    """Arbitrary text → a dict, or StageParseError. Never a raw exception."""
    try:
        data = noumeno._parse_json(raw)
    except StageParseError:
        return
    assert isinstance(data, dict)


@settings(max_examples=300)
@given(raw=st.text())
def test_ner_parse_intentresult_or_stageparseerror(ner, raw):
    """Arbitrary text → an IntentResult, or StageParseError. Never a raw exception."""
    metrics = StageMetrics(stage="ner", elapsed_ms=0.0, tokens_in=0, tokens_out=0, model="x")
    try:
        result = ner._parse(raw, metrics)
    except StageParseError:
        return
    assert result.intent_class  # populated, valid IntentResult


# Specifically the regression these guards fixed: valid JSON that is not an object.
@pytest.mark.parametrize("raw", ["5", "true", "null", "[]", '["a", "b"]', '"a string"'])
def test_non_object_json_raises_stageparseerror(noumeno, ner, raw):
    metrics = StageMetrics(stage="ner", elapsed_ms=0.0, tokens_in=0, tokens_out=0, model="x")
    with pytest.raises(StageParseError):
        noumeno._parse_json(raw)
    with pytest.raises(StageParseError):
        ner._parse(raw, metrics)
