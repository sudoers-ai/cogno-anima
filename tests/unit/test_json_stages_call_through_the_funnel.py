"""Every JSON stage calls its model through ONE function — enumerated by SINK, by AST.

``cogno_anima.utils.call_json_stage`` is the funnel: it probes the backend, makes the call, reads
the schema receipt and hands it back. A stage that called ``backend.generate`` beside it would be
a stage whose schema is silently never sent — the comparison measuring nothing, in green — and a
stage that called ``backend.generate_structured`` beside it would skip the probe, which is the
dead turn on a backend that has no such method.

So the rule is about the SINKS, not about the callers that already migrated: every call of
``.generate(`` and of ``.generate_structured(`` in the whole package is found by AST, and each one
is either inside the funnel or in the explicit table below, with its reason written down. A new
call anywhere else fails here; so does a row of the table that no longer matches a call.
"""

from __future__ import annotations

import ast
from pathlib import Path

import cogno_anima

PACKAGE = Path(cogno_anima.__file__).parent

FUNNEL = ("utils.py", "call_json_stage")

#: Calls of ``.generate(`` that are NOT a JSON stage's, each with the reason it stays outside.
#: (module, enclosing function) → why.
NOT_A_JSON_STAGE = {
    ("stages/ego.py", "process"):
        "the executor's text-fallback step: it reads <TOOL_CALL> tags out of prose, not a JSON "
        "object, and tool calling is another contract",
    ("stages/superego.py", "voice"):
        "the voice writes the reply a person reads: prose, never a JSON object",
}

_METHODS = ("generate", "generate_structured")


def _calls(source_dir: Path) -> "list[tuple[str, str, str]]":
    """(module, enclosing function, method) for every ``<x>.generate(…)`` /
    ``<x>.generate_structured(…)`` call in the package."""
    found = []
    for path in sorted(source_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module = path.relative_to(source_dir).as_posix()

        def walk(node, function):
            for child in ast.iter_child_nodes(node):
                inside = function
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # the OUTERMOST function names the place: a closure belongs to its stage
                    inside = function or child.name
                if (isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                        and child.func.attr in _METHODS):
                    found.append((module, inside or "<module>", child.func.attr))
                walk(child, inside)

        # methods of a class: name the METHOD, not the class
        for top in tree.body:
            if isinstance(top, ast.ClassDef):
                for item in top.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        walk(item, item.name)
                    else:
                        walk(item, "")
            elif isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(top, top.name)
            else:
                walk(top, "")
    return found


CALLS = _calls(PACKAGE)


def test_the_walk_sees_the_calls_that_exist():
    """The control: an enumeration that found nothing would agree with any rule."""
    assert (FUNNEL[0], FUNNEL[1], "generate") in CALLS
    assert (FUNNEL[0], FUNNEL[1], "generate_structured") in CALLS
    assert ("stages/superego.py", "voice", "generate") in CALLS
    assert len(CALLS) >= 4


def test_no_json_stage_calls_generate_outside_the_funnel():
    outside = sorted({(module, function) for module, function, method in CALLS
                      if method == "generate" and (module, function) != FUNNEL})
    assert outside == sorted(NOT_A_JSON_STAGE), (
        "a call of .generate( outside the funnel that the table does not explain, or a row of "
        f"the table with no call: {sorted(set(outside) ^ set(NOT_A_JSON_STAGE))}")


def test_nothing_but_the_funnel_calls_generate_structured():
    """No exemption exists for this one: the probe and the call are never apart."""
    callers = {(module, function) for module, function, method in CALLS
               if method == "generate_structured"}
    assert callers == {FUNNEL}


def test_every_json_stage_reaches_the_funnel():
    """The other direction: the five stages that read a JSON object do call it."""
    def calls_funnel(module: str) -> "set[str]":
        tree = ast.parse((PACKAGE / module).read_text(encoding="utf-8"))
        return {node.func.id for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}

    assert "call_json_stage" in calls_funnel("utils.py")              # generate_json_resilient
    assert "generate_json_resilient" in calls_funnel("stages/noumeno.py")
    assert "generate_json_resilient" in calls_funnel("stages/ner.py")
    for module in ("stages/superego.py", "stages/scope_options.py", "stages/proposal_judge.py"):
        assert "call_json_stage" in calls_funnel(module), module
    # control: a module that calls no model does not
    assert "call_json_stage" not in calls_funnel("stages/drift.py")


def test_a_new_call_beside_the_funnel_is_caught(tmp_path):
    """The mutation, written as a test: a stage that calls the backend itself."""
    package = tmp_path / "pkg"
    (package / "stages").mkdir(parents=True)
    (package / "utils.py").write_text(
        "async def call_json_stage(backend, system, prompt, *, schema, name, form):\n"
        "    if schema is None:\n"
        "        return await backend.generate(system, prompt)\n"
        "    return await backend.generate_structured(system, prompt, schema=schema, name=name, form=form)\n")
    assert {(m, f) for m, f, _ in _calls(package)} == {FUNNEL}
    (package / "stages" / "new_stage.py").write_text(
        "class NewStage:\n"
        "    async def decide(self, backend):\n"
        "        async def inner():\n"
        "            return await backend.generate('s', 'p')\n"
        "        return await inner()\n\n"
        "async def shortcut(backend, schema):\n"
        "    return await backend.generate_structured('s', 'p', schema=schema, name='x', form='open')\n")
    found = _calls(package)
    assert ("stages/new_stage.py", "decide", "generate") in found
    assert ("stages/new_stage.py", "shortcut", "generate_structured") in found
