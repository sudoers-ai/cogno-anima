"""Every log call of the package, read from the source by AST — the one reader two tests share.

``test_logging_md_lists_what_the_code_emits.py`` compares what it finds with the table in
``LOGGING.md``; ``test_logs_carry_no_third_party_text.py`` classifies every value it finds.

**A call is found by what it is called ON, not by what the variable is named.** A reader that
looked for ``logger.<level>(…)`` would miss ``log.warning(…)`` in ``tools/confirm_args.py`` and
``_logger.warning(…)`` in ``utils.py`` and agree with a page that omits both. So the logger
objects are taken from the bindings (``<target> = logging.getLogger(<name>)``) and every
attribute call named like a level is then one of three things: a call on a bound logger (read),
a call on the ``logging`` module or on an inline ``getLogger(…)`` (refused: the library logs
through a named, module-level logger), or a call on something else (returned as a STRANGER, so a
test can say the package has none).

**The message is a grammar, not a sentence.** ::

    [stage=<word> ]event=<word>( <key>=<value>)*[ — <prose>]

``<value>`` is one ``%`` placeholder or a literal token, and every placeholder sits directly
behind its ``key=``. That is what lets a positional argument be NAMED: the n-th argument is the
value of the n-th placeholder's key, and a value with a name can be given a class. A message
outside the grammar raises here, with the line.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

LEVELS = {"debug": "DEBUG", "info": "INFO", "warning": "WARNING", "warn": "WARNING",
          "error": "ERROR", "exception": "ERROR", "critical": "CRITICAL", "fatal": "CRITICAL",
          "log": "LOG"}

_PLACEHOLDER = r"%(?:\.\d+)?[sdrf]"
_HEAD = re.compile(rf"(?:stage=(?P<stage>{_PLACEHOLDER}|\w+) )?event=(?P<event>\w+)")
_FIELD = re.compile(rf"(?P<key>\w+)=(?P<value>{_PLACEHOLDER}|[\w:.\-]+)")
PROSE_MARK = " — "


class LogGrammarError(AssertionError):
    """A log call the reader cannot name: not a literal, not in the grammar, or not on a
    module-level named logger."""


@dataclass(frozen=True)
class LogCall:
    module: str                      # the dotted module that makes the call
    logger: str                      # the NAME of the logger it is made on
    lineno: int
    level: str
    message: str
    event: str
    stage: str                       # "" when the message carries none; "%s" when it is passed
    literals: "tuple[tuple[str, str], ...]"      # (key, literal value), in order
    values: "tuple[tuple[str, str, str], ...]"   # (key, conversion, argument source), in order
    traceback: bool

    @property
    def fields(self) -> "frozenset[str]":
        """Every ``key=`` of the message after ``event=`` — and ``stage`` when it is PASSED
        (``stage=%s``), which makes it a value like any other."""
        return frozenset(k for k, _ in self.literals) | frozenset(k for k, _, _ in self.values)

    @property
    def row(self) -> "tuple[str, str, str, frozenset[str], bool]":
        """The five facts ``LOGGING.md`` lists for it."""
        return (self.logger, self.event, self.level, self.fields, self.traceback)


def module_name(path: Path, package: Path) -> str:
    rel = path.relative_to(package.parent).with_suffix("")
    return ".".join(p for p in rel.parts if p != "__init__")


def _is_get_logger(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call)
            and ((isinstance(node.func, ast.Attribute) and node.func.attr == "getLogger")
                 or (isinstance(node.func, ast.Name) and node.func.id == "getLogger")))


def _logger_name(call: ast.Call, module: str, lineno: int) -> str:
    arg = call.args[0] if call.args else None
    if isinstance(arg, ast.Name) and arg.id == "__name__":
        return module
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    raise LogGrammarError(f"{module}:{lineno}: a logger whose name is not __name__ or a literal")


def parse_message(message: str, where: str) -> "tuple[str, str, list[tuple[str, str]]]":
    """``(stage, event, [(key, value)…])`` of one message, or :class:`LogGrammarError`."""
    body, mark, prose = message.partition(PROSE_MARK)
    head = _HEAD.match(body)
    if not head:
        raise LogGrammarError(f"{where}: a log message without event=<name>: {message!r}")
    pairs: "list[tuple[str, str]]" = []
    rest = body[head.end():]
    for token in rest.split():
        field = _FIELD.fullmatch(token)
        if not field:
            raise LogGrammarError(f"{where}: {token!r} is not key=value in {message!r}")
        pairs.append((field.group("key"), field.group("value")))
    if rest and not rest.startswith(" "):
        raise LogGrammarError(f"{where}: no space after the event in {message!r}")
    if "%" in prose or re.search(r"\w=", prose):
        raise LogGrammarError(f"{where}: a value in the prose of {message!r}")
    return head.group("stage") or "", head.group("event"), pairs


def read(source: str, module: str) -> "tuple[list[LogCall], list[str]]":
    """The log calls of ``source`` and its STRANGERS (``<base>.<level>(…)`` on something that
    is not a bound logger), each stranger as ``module:line base.attr``."""
    tree = ast.parse(source)
    bound: "dict[str, str]" = {}
    for node in ast.walk(tree):
        value = getattr(node, "value", None)
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and value is not None \
                and _is_get_logger(value):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                bound[ast.unparse(target)] = _logger_name(value, module, node.lineno)
    calls: "list[LogCall]" = []
    strangers: "list[str]" = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in LEVELS):
            continue
        base = ast.unparse(node.func.value)
        where = f"{module}:{node.lineno}"
        if base == "logging" or _is_get_logger(node.func.value):
            raise LogGrammarError(f"{where}: a log call that is not on a module-level logger")
        if base not in bound:
            strangers.append(f"{where} {base}.{node.func.attr}")
            continue
        if node.func.attr == "log":
            raise LogGrammarError(f"{where}: .log(level, …) hides the level from the reader")
        first = node.args[0] if node.args else None
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            raise LogGrammarError(f"{where}: a log message that is not a string literal")
        stage, event, pairs = parse_message(first.value, where)
        holders = ([("stage", stage)] if stage.startswith("%") else []) + [
            (k, v) for k, v in pairs if v.startswith("%")]
        args = [ast.unparse(a) for a in node.args[1:]]
        if len(holders) != len(args) or first.value.count("%") != len(holders):
            raise LogGrammarError(f"{where}: {len(args)} argument(s) for {len(holders)} "
                                  f"named placeholder(s) in {first.value!r}")
        extra = sorted(kw.arg or "**" for kw in node.keywords if kw.arg != "exc_info")
        if extra:
            raise LogGrammarError(f"{where}: keyword(s) {extra} on a log call")
        traceback = node.func.attr == "exception" or any(
            kw.arg == "exc_info" and not (isinstance(kw.value, ast.Constant)
                                          and kw.value.value in (False, None))
            for kw in node.keywords)
        calls.append(LogCall(
            module=module, logger=bound[base], lineno=node.lineno,
            level=LEVELS[node.func.attr], message=first.value, event=event, stage=stage,
            literals=tuple((k, v) for k, v in pairs if not v.startswith("%")),
            values=tuple((k, v, a) for (k, v), a in zip(holders, args)),
            traceback=traceback))
    return calls, strangers


def package_calls(package: Path) -> "tuple[list[LogCall], list[str]]":
    calls: "list[LogCall]" = []
    strangers: "list[str]" = []
    for path in sorted(package.rglob("*.py")):
        found, odd = read(path.read_text(encoding="utf-8"), module_name(path, package))
        calls.extend(found)
        strangers.extend(odd)
    return calls, strangers


def modules(package: Path) -> "Iterator[str]":
    for path in sorted(package.rglob("*.py")):
        yield module_name(path, package)
