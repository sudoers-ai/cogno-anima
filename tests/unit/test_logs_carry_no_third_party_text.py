"""No log call of this package formats text somebody else wrote — the class, by AST.

The judge's rejection logged the first 80 characters of its critique. A critique is text the
model wrote about the turn, and it quotes the turn: a name, a figure, the contact's own
sentence. That line was the one ``LOGGING.md`` admitted to. Read by AST on ``32803f5``, the
package had 15 calls like it, of 76:

* 7 with text of the MODEL — the critique, the rewrite (three lines), the tail of a reply that
  arrived cut, an out-of-vocabulary ``intent_class``, a tool name the model invented;
* 1 with a value the model wrote into a call — the id a guarded write named;
* 5 with an EXCEPTION's message — a ``StageParseError`` quotes 120 characters of the reply it
  could not read, and a provider's error may quote the request;
* 1 with values the TENANT typed — the trait values the vocabulary does not have;
* 1 with the ``repr`` of a HOST carrier that could not be used.

Every value a log call formats is named by the ``key=`` in front of its placeholder
(``tests/unit/_log_calls.py`` refuses a message where one is not). ``LOGGING.md`` gives every
key ONE class — count, closed, type, name, constant; there is no class for text — and
:data:`EXPRESSIONS` pins the expression each key is given. A new key, or a new expression
under an old key, fails here until somebody classifies it. The pin is the point: the defect was
``critique=%s`` with ``(critique or "")[:80]``, and ``critique_len=%d`` with the same argument
would be the same defect under a better name.

What the fifteen lines say at run time is the other file's:
``test_log_lines_carry_no_byte_of_the_turn.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import cogno_anima
from tests.unit import _log_calls as reader

PACKAGE = Path(cogno_anima.__file__).parent
DOC = PACKAGE.parent / "LOGGING.md"

CALLS, STRANGERS = reader.package_calls(PACKAGE)

#: The classes a value may have. ``constant`` is for a key that only ever has a literal.
VALUE_CLASSES = ("count", "closed", "type", "name")
CLASSES = VALUE_CLASSES + ("constant",)

#: What each key is GIVEN — the source of the argument, as ``ast.unparse`` writes it. A class
#: is a judgement about an expression, so the judgement is pinned to the expression it was
#: about. ``"<key>@<logger>"`` is a key whose class differs in one logger.
EXPRESSIONS: "dict[str, set[str]]" = {
    # ── count ──
    "at": {"hhmm"},
    "attempt": {"attempt_no", "self._attempt_of(ctx)", "index + 1"},
    "attempts": {"attempts"},
    "blocked": {"blocked"},
    "change_subject": {"change_subject"},
    "changed": {"changed"},
    "chars": {"len(raw)"},
    "composite": {"intent.is_composite"},
    "cot_stripped": {"cot_stripped"},
    "count": {"len(from_critique)", "len(outcome.withheld)"},
    "covered_unsupported": {"unsupported"},
    "critique_len": {"len(critique or '')"},
    "cumulative": {"drift.cumulative_drift"},
    "discarded": {"discarded"},
    "drift": {"'unknown' if drift_score is None else f'{drift_score:.2f}'"},
    "dropped": {"preserved_dropped", "len(dropped)"},
    "examples": {"len(self._example_rewrites)"},
    "id_len": {"len(wanted)"},
    "invented": {"len(invented)"},
    "kept": {"len(preserved_terms)", "len(kept)"},
    "len": {"len(response)"},
    "lost": {"len(lost)"},
    "max_steps": {"max_steps"},
    "ms": {"int(entry.elapsed_ms)"},
    "n": {"len(asked)", "int(state['n']) if state else 0"},
    "name_len": {"len(name) if isinstance(name, str) else 0"},
    "neutral": {"state['valence_ema'] if state else '?'"},
    "offered": {"len(offered)"},
    "ok": {"r.ok"},
    "retried_len": {"len(retried)"},
    "rewritten_len": {"len(rewritten)"},
    "rule_illustrations": {"len(self._rule_rewrites)"},
    "scores": {"','.join((f'{s}:{v:.2f}' for (s, v) in present.items())) or '-'"},
    "side_effect": {"r.side_effect"},
    "sim": {"goal_similarity"},
    "stages": {"len(present)"},
    "step": {"i"},
    "steps": {"len(steps)"},
    "streak": {"circling"},
    "tools": {"len(tools)", "n_tools"},
    "turn": {"turn"},
    # ── closed ──
    "action": {"drift.drift_action"},
    "adjustments": {"','.join(adjustments) or '-'"},
    "branch": {"branch"},
    "coerced": {"intent_class"},
    "complexity": {"complexity"},
    "declared": {"list(traits)"},
    "degraded": {"','.join(degradations) or '-'"},
    "derived": {"domains"},
    "domains": {"intent.domains"},
    "effective": {"out"},
    "goal_status": {"goal_status"},
    "intent": {"intent.intent_class"},
    "llm": {"'UNKNOWN' if llm_intent_class == 'UNKNOWN' else 'out_of_vocabulary'"},
    "outcome": {"outcome", "entry.outcome"},
    "path": {"path"},
    "pii_risk": {"intent.pii_risk"},
    "read": {"changed_read.read", "verdict.read", "VERDICT_BOOLEAN"},
    "reason": {"interrupt_reason"},
    "repairs": {"','.join(repairs.marks()) or '-'"},
    "route": {"triad_route"},
    "sentiment": {"intent.sentiment", "ctx.intent.sentiment if ctx.intent else '?'",
                  "sentiment"},
    "shape": {"shape"},
    "stage": {"stage"},
    "state": {"claim.state"},
    "step@cogno_anima.tools.idempotency": {"step"},
    "tag": {"drift_tag"},
    "types": {"withheld_types"},
    "verdict": {"entry.verdict"},
    # ── type ──  (the shape is held by a test of its own: `type(<name>).__name__`)
    "error": {"type(exc).__name__", "type(le).__name__"},
    "type": {"type(value).__name__", "type(raw).__name__"},
    # ── name ──  (what the HOST declared)
    "arg": {"arg"},
    "lang": {"detected_lang"},
    "name": {"name"},
    "persona": {"persona", "vocab._label(ctx.metadata.get(mk.ACTIVE_PERSONA_ID, '?'))"},
    "tool": {"name", "entry.tool"},
}


def field_classes(text: str) -> "dict[str, str]":
    """``{key: class}`` from the fields table of ``LOGGING.md``. A row is
    ``| `key` | class | what |``, or ``| `key` in `logger` | class | what |`` for a key whose
    class differs in one logger (returned as ``key@logger``)."""
    out: "dict[str, str]" = {}
    for line in text.splitlines():
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip())]
        if len(cells) != 5 or cells[0] or cells[-1]:
            continue
        key = re.fullmatch(r"`(\w+)`(?: in `(cogno_anima[\w.]*)`)?", cells[1])
        if not key or cells[2] not in CLASSES + ("text", "digest"):
            continue
        name = key.group(1) + (f"@{key.group(2)}" if key.group(2) else "")
        assert name not in out, f"{name} has two rows"
        out[name] = cells[2]
    return out


def _key(call: reader.LogCall, key: str, table: "dict[str, object]") -> str:
    scoped = f"{key}@{call.logger}"
    return scoped if scoped in table else key


def unclassified(calls: "list[reader.LogCall]", classes: "dict[str, str]",
                 expressions: "dict[str, set[str]]") -> "list[str]":
    """Every way a call's values escape the table — one line each; ``[]`` is the pass."""
    out: "list[str]" = []
    for call in calls:
        where = f"{call.module}:{call.lineno} event={call.event}"
        for key, _, source in call.values:
            cls = classes.get(_key(call, key, classes))
            if cls is None:
                out.append(f"{where}: {key}= has no class in LOGGING.md")
            elif cls not in VALUE_CLASSES:
                out.append(f"{where}: {key}= is classed {cls!r} and is given a value")
            if source not in expressions.get(_key(call, key, expressions), set()):
                out.append(f"{where}: {key}= is given {source!r}, which nobody classified")
            if cls == "type" and not re.fullmatch(r"type\(\w+\)\.__name__", source):
                out.append(f"{where}: {key}= is classed type and is given {source!r}")
        for key, _ in call.literals:
            if _key(call, key, classes) not in classes:
                out.append(f"{where}: {key}= has no class in LOGGING.md")
    return out


CLASSES_ON_THE_PAGE = field_classes(DOC.read_text(encoding="utf-8"))


# ── by AST: every value has a class, and no class is text ──────────────────────────────────

def test_every_value_a_log_call_formats_has_a_class():
    assert not STRANGERS, STRANGERS
    assert unclassified(CALLS, CLASSES_ON_THE_PAGE, EXPRESSIONS) == []


def test_no_key_is_classed_as_text():
    assert set(CLASSES_ON_THE_PAGE.values()) <= set(CLASSES)
    assert "text" not in CLASSES_ON_THE_PAGE.values()


def test_the_page_and_the_pins_list_the_keys_the_code_uses_and_no_other():
    used: "set[str]" = set()
    given: "set[str]" = set()
    for call in CALLS:
        used |= {_key(call, k, CLASSES_ON_THE_PAGE) for k in call.fields}
        given |= {_key(call, k, EXPRESSIONS) for k, _, _ in call.values}
    assert set(CLASSES_ON_THE_PAGE) == used
    assert set(EXPRESSIONS) == given
    constant = {k for k, cls in CLASSES_ON_THE_PAGE.items() if cls == "constant"}
    assert constant == used - given, "constant is for a key that never takes a value"


def test_every_pinned_expression_is_still_what_some_call_is_given():
    seen: "dict[str, set[str]]" = {}
    for call in CALLS:
        for key, _, source in call.values:
            seen.setdefault(_key(call, key, EXPRESSIONS), set()).add(source)
    assert seen == EXPRESSIONS


def test_the_tracebacks_are_the_five_the_page_exempts():
    """A traceback says what the exception says, so each one is an exemption with its reason
    written on the page. A sixth fails here until it has one."""
    text = DOC.read_text(encoding="utf-8")
    section = text[text.index("### Tracebacks"):text.index("### What is not in the log")]
    listed = set(re.findall(r"`(judge_pre_\w+_failed)`", section)) - {"judge_pre_failed"}
    carrying = {call.event for call in CALLS if call.traceback}
    assert carrying == listed and len(carrying) == 5
    assert {call.logger for call in CALLS if call.traceback} == {"cogno_anima.tools.pre_judge"}


def test_the_package_has_no_other_way_to_write_a_line():
    """A ``print``, a ``warnings.warn``, a write to a standard stream or a hand-formatted
    traceback is a log line this reader would never see."""
    import ast
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                called = ast.unparse(node.func)
                if called in ("print", "warnings.warn", "warn", "pprint", "pprint.pprint") \
                        or called.startswith(("sys.stdout", "sys.stderr", "traceback.")):
                    found.append(f"{path.name}:{node.lineno} {called}")
    assert found == []


# ── the controls of the AST half ───────────────────────────────────────────────────────────

def _planted(line: str) -> "list[reader.LogCall]":
    calls, strangers = reader.read(
        f"import logging\nlogger = logging.getLogger('cogno_anima.superego')\n{line}\n",
        "cogno_anima.stages.superego")
    assert not strangers
    return calls


def test_control_the_defect_under_a_better_name_is_caught():
    """``critique_len=%d`` given the critique's first 80 characters: the key is classified,
    the expression is not."""
    calls = _planted('logger.warning("stage=superego event=judge critique_len=%s", '
                     '(critique or "")[:80])')
    said = unclassified(calls, CLASSES_ON_THE_PAGE, EXPRESSIONS)
    assert len(said) == 1 and "nobody classified" in said[0]


def test_control_a_new_key_is_caught_twice():
    calls = _planted('logger.warning("stage=superego event=judge critique=%s", critique)')
    said = unclassified(calls, CLASSES_ON_THE_PAGE, EXPRESSIONS)
    assert len(said) == 2 and "has no class" in said[0] and "nobody classified" in said[1]


def test_control_an_exception_message_is_not_a_type():
    calls = _planted('logger.warning("stage=superego event=judge_failed error=%s", exc)')
    said = unclassified(calls, CLASSES_ON_THE_PAGE, EXPRESSIONS)
    assert len(said) == 2 and "classed type" in said[1]


def test_control_a_key_classed_as_text_is_caught():
    calls = _planted('logger.warning("stage=superego event=judge branch=%s", branch)')
    assert unclassified(calls, CLASSES_ON_THE_PAGE, EXPRESSIONS) == []
    said = unclassified(calls, dict(CLASSES_ON_THE_PAGE, branch="text"), EXPRESSIONS)
    assert len(said) == 1 and "classed 'text'" in said[0]
    assert field_classes("| `branch` | text | the words |") == {"branch": "text"}
