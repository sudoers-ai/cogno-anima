# What the library logs — the policy

Moved verbatim from `CLAUDE.md` on 2026-11-08: the contract did not change. `CLAUDE.md` keeps a short summary and points here. Update this file in the same pull request as any change to the logging policy (the event list itself is `LOGGING.md`), as `CLAUDE.md` used to be updated.

### What the library logs (`LOGGING.md`) — no line carries somebody else's text

`LOGGING.md` is the list of the log events, and the code is held to it (0.1.12).

- **The defect.** The judge's rejection logged the first 80 characters of its critique, and a
  critique quotes the turn. Read by AST, 15 of the package's 76 log calls formatted text
  somebody else wrote: the model's (the critique, the rewrite, the tail of a cut reply, an
  out-of-vocabulary `intent_class`, an invented tool name), an id the model put in a call, an
  exception's message, trait values the tenant typed, the `repr` of a host carrier.
- **The rule.** A line says how much and which kind: a length, a count, a member of a closed
  vocabulary, a class name. At every level, DEBUG included. The text stays on the result.
- **One grammar**: `[stage=<word> ]event=<name>( <key>=<value>)*[ — <prose>]`. Every placeholder
  sits behind its `key=`, so every value has a name. 16 calls had no event name; all have one.
- **One class per key** (`count`, `closed`, `type`, `name`, `constant`), in the fields table of
  `LOGGING.md`. There is no class for text. `name` is what the HOST declared (a tool, an
  argument's name, a persona label, a language tag): the one class the library does not close.
- **Five tracebacks** are the written exemption, all in `tools/pre_judge.py`, all from the
  host's own callbacks or the wrapper's bookkeeping.
- **Three test files hold it**, over a reader that finds a log call by the logger it is called
  on, not by the variable's name (`tests/unit/_log_calls.py`):
  - `test_logging_md_lists_what_the_code_emits.py`: the events table, both directions (logger,
    event, level, fields, traceback), the modules that log nothing, the count per level;
  - `test_logs_carry_no_third_party_text.py`: every value has a class, and the expression each
    key is given is pinned, so `critique_len=%d` given the critique's first 80 characters fails;
  - `test_log_lines_carry_no_byte_of_the_turn.py`: each of the fifteen lines, driven with a
    sentinel in the text it used to print.
- **`StageParseError` still quotes the reply** (120 characters in its message, all of it on
  `.raw`). The library never logs it. A host that logs the exception logs that; `stage` and
  `mark` are the fields that are safe anywhere.
