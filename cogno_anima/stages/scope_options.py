"""
cogno_anima.stages.scope_options — "did you mean…?" over a CLOSED list, on a refused turn.

When the scope guard (:meth:`~cogno_anima.stages.superego.SuperegoStage.check_input_scope`)
BLOCKS a message, one small model call — the SELECTOR — is shown the message and a closed list
the HOST built (the section titles this reader may read, the names of the capabilities on this
turn's table) and asked which options, if any, are about what the message asks. The code, not the
model, decides what the answer may contain.

**Two lists, and why they lead to OPPOSITE actions.**

* ``covered`` — an option about the very thing the message asks. The thing EXISTS for this reader,
  so the refusal was FALSE, and a false refusal is corrected by LETTING THE TURN THROUGH, never by
  answering it with a question: "did you mean <the thing you asked for>?" would hide the defect
  behind options. Measured downstream on real refused turns (labelled by hand): of the refusals
  on which a strict selector found an option at all, 5 of 6 were exactly that — a guard refusing
  something the persona held — and the sixth was a legitimate follow-up refused for lack of
  context. So a ``covered`` pick wins over every ``suggested`` one (:func:`parse_selection`).
* ``suggested`` — an option about a closely related but DIFFERENT thing the person may have meant.
  Only then does the refusal become the closed question («I did not find X. Did you mean: A / B?»,
  rendered by the HOST in the tenant's language — this module renders nothing a contact reads).

**The closed alphabet is enforced here, not requested.** Any option the model returns that is not,
character for character after trimming, one of the options it was shown is DISCARDED and counted
(``discarded``). A free (non-strict) selector measured downstream invented 6 options over 15 cases,
so the check is not optional. ``asked`` — the words naming what the message asks, which the host
may quote back — survives only when it is literally in the message (case, accents and whitespace
folded), so it can only ever be the contact's own words.

**Strict, and the negative is the point.** The prompt is the strict variant measured downstream
(gpt-4o-mini, temperature 0): the target option came back in 11 of 15 labelled cases, and "the
Wi-Fi password" drew 0 options on both reader profiles. "When in doubt, pick none."

**Never raises** (``asyncio.CancelledError`` excepted). A model error, a missing JSON object or a
JSON object with neither list is outcome ``error``, which every caller treats exactly like
``none``: the refusal of today, byte for byte. This is a courtesy on a refused turn, and a courtesy
must never cost the contact a reply.

Pure except for the one ``backend.generate`` call; :func:`select_scope_options` additionally
stamps the record on ``ctx.metadata[mk.SCOPE_OPTIONS_SELECTION]``.
"""

from __future__ import annotations

import asyncio
import logging
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from cogno_synapse import LLMBackend, cached_tokens_of, served_model_of, system_fingerprint_of

from cogno_anima import metakeys as mk
from cogno_anima.stages.superego import SuperegoStage
from cogno_anima.types import PipelineContext, StageMetrics

__all__ = [
    "OptionSelection",
    "SELECT_STAGE",
    "OUTCOME_COVERED",
    "OUTCOME_SUGGESTED",
    "OUTCOME_NONE",
    "OUTCOME_ERROR",
    "VALID_SELECTION_OUTCOMES",
    "MAX_PICKS",
    "closed_options",
    "parse_selection",
    "select_options",
    "select_scope_options",
]

logger = logging.getLogger(__name__)

#: The ``StageMetrics.stage`` of the selector's call — its own ledger line, never the guard's.
SELECT_STAGE = "superego_select"

OUTCOME_COVERED = "covered"      # an option IS what was asked → the refusal was false → let it pass
OUTCOME_SUGGESTED = "suggested"  # only related options → the closed "did you mean" question
OUTCOME_NONE = "none"            # nothing on the list → the refusal of today
OUTCOME_ERROR = "error"          # no usable answer → the refusal of today
VALID_SELECTION_OUTCOMES = frozenset({OUTCOME_COVERED, OUTCOME_SUGGESTED, OUTCOME_NONE,
                                      OUTCOME_ERROR})

#: At most this many options survive, ``covered`` first — the closed form offers "A / B".
MAX_PICKS = 2
#: Bounds on what the host's list and the contact's message may contribute to ONE small call.
MAX_OPTIONS = 120
MAX_OPTION_CHARS = 120
MAX_MESSAGE_CHARS = 600
#: ``asked`` is quoted back to the contact, so it is short and it is theirs.
MAX_ASKED_CHARS = 80

# The strict prompt measured downstream, split into the two lists this module acts on. The
# domain words in the examples ("the course, studying, the school") are generic on purpose.
_SYSTEM = (
    "You help a business assistant whose scope check refused a message. You get the message and "
    "a CLOSED list of options: the titles of sections the business published and the names of "
    "things the assistant can do. Pick at most 2 options, copying them EXACTLY as written, and "
    "put each one in ONE of two lists. \"answers\": an option about the very thing the message "
    "asks, so the assistant could have answered it from there. \"maybe\": an option about a "
    "closely related but DIFFERENT thing (another name, another level, a neighbouring item) that "
    "the person may have meant instead. When unsure between the two, use \"answers\". STRICT: an "
    "option counts only if its SUBJECT is the thing the message asks about or that close "
    "neighbour — not merely the same general area (the course, studying, the school, finances, "
    "the business). If the message asks about something no option NAMES (a facility, parking, "
    "the internet, a pet, a service not listed), leave both lists empty. When in doubt, pick "
    "none. Never invent an option. In \"asked\", copy the few words of the message that name what "
    "it asks about, exactly as they are written there, or \"\". Answer only JSON: "
    '{"answers": ["..."], "maybe": ["..."], "asked": "..."}'
)


@dataclass(frozen=True)
class OptionSelection:
    """What the selector decided, reduced to the closed alphabet.

    ``covered``/``suggested`` hold only strings that were IN the list the call was shown;
    ``offered`` is that list's length and ``discarded`` how many returned options were not in it
    (or were not strings) — the number that says whether the closed-alphabet check is doing work.
    ``called`` is False when no model call was made (nothing to select from), so no ledger line is
    owed.
    """

    outcome: str
    covered: tuple[str, ...] = ()
    suggested: tuple[str, ...] = ()
    asked: str = ""
    offered: int = 0
    discarded: int = 0
    called: bool = False
    metrics: StageMetrics = field(default_factory=lambda: _no_call("unknown"))

    def record(self) -> "dict[str, Any]":
        """The per-turn record stamped on ``ctx.metadata`` — the host reads it to render the
        question and to count the outcome. The option TEXTS are here (the host needs them to
        render); a trace writer copies the counts, never the texts."""
        return {"outcome": self.outcome, "covered": list(self.covered),
                "suggested": list(self.suggested), "asked": self.asked,
                "offered": self.offered, "discarded": self.discarded}


def _no_call(model: str) -> StageMetrics:
    return StageMetrics(stage=SELECT_STAGE, elapsed_ms=0.0, tokens_in=0, tokens_out=0,
                        model=model)


def closed_options(raw: Any) -> "tuple[str, ...]":
    """The host's list made safe to render: one line each (whitespace collapsed), non-strings and
    blanks dropped, duplicates dropped in order, each capped at :data:`MAX_OPTION_CHARS` (an option
    longer than that is DROPPED, not cut — a cut option is a string the host never offered, and the
    closed-alphabet check compares against exactly what was shown), at most :data:`MAX_OPTIONS`.
    Never raises."""
    if isinstance(raw, str) or not isinstance(raw, (list, tuple, set, frozenset)):
        items: Sequence[Any] = [raw] if isinstance(raw, str) else []
    else:
        items = list(raw)
    out: "list[str]" = []
    for item in items:
        if not isinstance(item, str):
            continue
        flat = " ".join(item.split())
        if not flat or flat in out or len(flat) > MAX_OPTION_CHARS:
            continue
        out.append(flat)
        if len(out) == MAX_OPTIONS:
            break
    return tuple(out)


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text or "")
    bare = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(bare.casefold().split())


def _picks(value: Any) -> "Optional[list[Any]]":
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return None


def parse_selection(data: Any, options: Sequence[str], message: str,
                    ) -> "tuple[str, tuple[str, ...], tuple[str, ...], str, int]":
    """``(outcome, covered, suggested, asked, discarded)`` from the model's parsed JSON. Pure.

    * only an option that IS in ``options`` (after trimming) survives; anything else is counted
      in ``discarded`` — including a non-string;
    * ``covered`` wins: an option picked in both lists is covered, and when ANY option is covered
      the outcome is ``covered`` whatever else was suggested — a false refusal is never turned
      into a question;
    * at most :data:`MAX_PICKS` in all, ``covered`` first;
    * ``asked`` survives only when its fold is a substring of the message's fold and it is at
      most :data:`MAX_ASKED_CHARS` long;
    * not a JSON object, or an object with neither list, or a list that is not a list → ``error``.
    """
    if not isinstance(data, dict) or ("answers" not in data and "maybe" not in data):
        return OUTCOME_ERROR, (), (), "", 0
    answers, maybe = _picks(data.get("answers")), _picks(data.get("maybe"))
    if answers is None or maybe is None:
        return OUTCOME_ERROR, (), (), "", 0
    allowed = set(options)
    discarded = 0
    covered: "list[str]" = []
    suggested: "list[str]" = []
    for bucket, sink in ((answers, covered), (maybe, suggested)):
        for item in bucket:
            text = " ".join(item.split()) if isinstance(item, str) else None
            if text is None or text not in allowed:
                discarded += 1
                continue
            if text in covered or text in suggested:
                continue
            sink.append(text)
    covered = covered[:MAX_PICKS]
    suggested = [s for s in suggested if s not in covered][:max(0, MAX_PICKS - len(covered))]
    asked_raw = data.get("asked")
    asked = " ".join(asked_raw.split()) if isinstance(asked_raw, str) else ""
    if not asked or len(asked) > MAX_ASKED_CHARS or _fold(asked) not in _fold(message):
        asked = ""
    if covered:
        return OUTCOME_COVERED, tuple(covered), (), asked, discarded
    if suggested:
        return OUTCOME_SUGGESTED, (), tuple(suggested), asked, discarded
    return OUTCOME_NONE, (), (), asked, discarded


def _prompt(message: str, options: Sequence[str]) -> str:
    # The user half measured downstream, unchanged: the message, then the options as a list.
    return ("Message: " + (message or "")[:MAX_MESSAGE_CHARS] + "\n\nOptions:\n"
            + "\n".join(f"- {o}" for o in options))


async def select_options(message: str, options: Any, backend: LLMBackend) -> OptionSelection:
    """Ask ``backend`` which of ``options`` (sanitised by :func:`closed_options`) the message is
    about. No options or no message → outcome ``none`` with no call. Never raises."""
    offered = closed_options(options)
    model = getattr(backend, "model", "unknown")
    if not offered or not (message or "").strip():
        return OptionSelection(outcome=OUTCOME_NONE, offered=len(offered),
                               metrics=_no_call(model))
    t0 = time.perf_counter()
    ti = to = cached = 0
    fingerprint: Optional[str] = None
    served: Optional[str] = None
    try:
        raw, ti, to = await backend.generate(_SYSTEM, _prompt(message, offered))
        # Read with NO await in between — the contract of ``cached_tokens_of``.
        cached = cached_tokens_of(backend)
        fingerprint = system_fingerprint_of(backend)
        served = served_model_of(backend)
        text, _ = SuperegoStage.strip_cot(raw)
        outcome, covered, suggested, asked, discarded = parse_selection(
            SuperegoStage._parse_json(text), offered, message)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 — a courtesy on a refused turn never costs a reply
        logger.warning("scope option selector failed (%s) — the refusal stands",
                       type(exc).__name__)
        outcome, covered, suggested, asked, discarded = OUTCOME_ERROR, (), (), "", 0
    metrics = StageMetrics(stage=SELECT_STAGE, elapsed_ms=(time.perf_counter() - t0) * 1000,
                           tokens_in=ti, tokens_out=to, cached_tokens=cached,
                           system_fingerprint=fingerprint, served_model=served, model=model)
    logger.info("SUPEREGO scope options outcome=%s offered=%d discarded=%d",
                outcome, len(offered), discarded)
    return OptionSelection(outcome=outcome, covered=covered, suggested=suggested, asked=asked,
                           offered=len(offered), discarded=discarded, called=True,
                           metrics=metrics)


async def select_scope_options(ctx: PipelineContext, backend: LLMBackend, *,
                               options: Any) -> OptionSelection:
    """:func:`select_options` over the contact's own message (``ctx.user_input``), with the
    record stamped on ``ctx.metadata[mk.SCOPE_OPTIONS_SELECTION]`` — the orchestrator consumes
    the result, and the host (which renders the question) and the trace read the stamp. A
    PER-TURN fact: the orchestrator pops the key before the guard runs, so a turn on which no
    selection ran never wears an earlier one. Never raises."""
    selection = await select_options(ctx.user_input or "", options, backend)
    ctx.metadata[mk.SCOPE_OPTIONS_SELECTION] = selection.record()
    return selection
