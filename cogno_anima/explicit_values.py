"""EXPLICIT values in a text — the clock times and calendar dates somebody wrote down.

Two readers, one definition, for the reason ``cogno_anima.preserved`` gives for its own: a second
reading of "what is an hour" would be a second rule to get wrong alone.

* The JUDGE (``superego._write_value_paragraph``): the contact's message names ONE time (or ONE
  date) and a WRITE of the turn was called with arguments carrying a different one. Measured on a
  scheduling demo: the contact answered «às 15», the executor called
  ``book_appointment(time="14:00")`` and the judge APPROVED — the generic "did it do exactly what
  was asked" never put the two values side by side. The code does that here; the model decides.
* The VOICE (``SuperegoStage._unread_times``): a reply that LISTS clock times on a turn where no
  successful call returned one. Measured on the same demo: ``check_availability`` failed, the
  judge exhausted, and the reply offered times nobody had read.

Deliberately NARROW, in the direction both readers need. A bare number is a price, a quantity, an
age or a duration, and reading one as an hour is the false positive that would make the judge
reject a correct write or the voice drop a true sentence. So a time is only one of:

* ``14:00`` / ``14:00:00`` / ``9:30`` — the colon form (an ISO offset such as ``-03:00`` is taken
  off first; it is not a time of day);
* ``14h`` / ``14h30`` / ``14 h`` / ``14hs`` / ``14 horas`` — NOT when the words around it make it
  a DURATION («dura 1h», «1h antes», «por 2h», «1h de antecedência»);
* a preposition of time and a number — «às 15», «as 15», «das 9», «até 18», «pelas 10», «at 3» —
  only when the next word is not a noun the number counts («as 3 opções», «às 15 vagas»);
* ``3 pm`` / ``3pm``; «meio-dia», «meia-noite»; and «da tarde/da noite/da manhã» after any of
  the above.

An hour from 1 to 11 written without a leading zero, a period or am/pm is AMBIGUOUS («às 3» is
three in the afternoon in a scheduling chat) and is read as BOTH readings. Every comparison here
is "do the two share a reading", so ambiguity only ever makes a value MATCH more — the lenient
side for both readers.

A date is ``dd/mm[/yyyy]``, ISO ``yyyy-mm-dd``, «dia 12 [de novembro]» or «12 de novembro». A day
written without a month matches that day in any month. English month names are NOT read — a
declared ceiling, not an oversight: the arguments a host's tools carry are ISO, and the contact's
side of the measured case is Portuguese.

Pure: no I/O, no state, never raises on a string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

__all__ = [
    "ClockTime", "CalendarDate", "WriteValueMismatch",
    "clock_times", "clock_time_spans", "calendar_dates", "same_time", "same_date",
    "argument_texts", "write_value_mismatches", "unread_times",
]


@dataclass(frozen=True)
class ClockTime:
    """One time of day as written. ``ambiguous`` = the 12-hour reading is also possible."""

    hour: int
    minute: int
    ambiguous: bool = False

    def readings(self) -> "frozenset[tuple[int, int]]":
        out = {(self.hour, self.minute)}
        if self.ambiguous:
            out.add(((self.hour + 12) % 24, self.minute))
        return frozenset(out)

    def label(self) -> str:
        """``15:00``; ``03:00 or 15:00`` for an ambiguous one. Our own format — digits and a
        colon — so a label can enter a prompt without a byte of the text it was read from."""
        return " or ".join(f"{h:02d}:{m:02d}" for h, m in sorted(self.readings()))


@dataclass(frozen=True)
class CalendarDate:
    """One calendar date as written; ``month`` is ``None`` when only the day was."""

    day: int
    month: Optional[int] = None

    def label(self) -> str:
        return f"{self.day:02d}/{self.month:02d}" if self.month else f"day {self.day}"


@dataclass(frozen=True)
class WriteValueMismatch:
    """A WRITE called with a time (or date) other than the ONE the request states.

    Labels only — never a byte of the argument or the message they were read from."""

    tool: str
    kind: str                   # "time" | "date"
    requested: str
    written: "tuple[str, ...]"


def same_time(a: ClockTime, b: ClockTime) -> bool:
    """Do the two share a reading? (An ambiguous one has two.)"""
    return bool(a.readings() & b.readings())


def same_date(a: CalendarDate, b: CalendarDate) -> bool:
    """Same day, and the same month whenever both name one."""
    return a.day == b.day and (a.month is None or b.month is None or a.month == b.month)


# ── times ────────────────────────────────────────────────────────────────────────────────

# An ISO offset after a time («T14:00:00-03:00», «14:00+01:00») is the zone, not a time of day.
_ISO_OFFSET_RE = re.compile(r"(\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)(?:[+\-]\d{2}:?\d{2}|z)(?!\d)",
                            re.IGNORECASE)
_ISO_T_RE = re.compile(r"(\d{4}-\d{2}-\d{2})t(\d)")
_AMPM = r"(?:\s?([ap])\.?\s?m\.?(?![a-z]))"
_PERIOD_RE = re.compile(r"\s+(?:da|de|of the|in the)\s+(tarde|noite|manhã|manha|madrugada|"
                        r"afternoon|evening|night|morning)\b")
_COLON_RE = re.compile(r"(?<![\w:/.,])(\d{1,2}):(\d{2})(?::\d{2}(?:\.\d+)?)?(?![\d:])" + _AMPM + "?")
_H_RE = re.compile(r"(?<![\w:/.,])(\d{1,2})\s?h(?:rs?|s|oras?)?(?:\s?(\d{2}))?(?![\w:])")
_AMPM_RE = re.compile(r"(?<![\w:/.,])(\d{1,2})" + _AMPM)
_PREP_RE = re.compile(r"(?<![\w])(?:às|as|das|dàs|até|ate|pelas|pras|at|around)\s+(\d{1,2})"
                      r"(?![\d:h/]|[,.]\d)")
_NOON_RE = re.compile(r"\bmeio[\s-]dia\b|\bnoon\b")
_MIDNIGHT_RE = re.compile(r"\bmeia[\s-]noite\b|\bmidnight\b")

# The words that may FOLLOW a bare «às N» and leave it a time: connectors, the period words, the
# hour words, «amanhã/hoje», and the short replies a contact closes a message with. Anything
# else — a noun («as 3 opções», «às 15 vagas») — means the number counts something.
_AFTER_A_BARE_HOUR = frozenset((
    "da", "de", "do", "e", "ou", "em", "no", "na", "nesse", "neste", "h", "hs", "hrs", "hr",
    "hora", "horas", "amanhã", "amanha", "hoje", "então", "entao", "mesmo", "pode", "por",
    "pf", "pfv", "ok", "certo", "sim", "tá", "ta", "beleza", "blz", "né", "ne", "certinho",
    "pm", "am", "tomorrow", "today", "o'clock", "please", "then", "on", "is", "works", "fine",
    "com", "pra", "para", "que", "se", "só", "so", "aí", "ai", "please", "sharp",
    "não", "nao", "nem", "dá", "serve", "funciona", "fica", "seria", "é", "rola", "consigo",
    "posso", "podemos", "obrigado", "obrigada", "valeu", "vlw", "perfeito", "show", "fechado",
    "combinado", "confirmo", "confirmado", "certo", "também", "tambem", "mas", "porque", "pq",
    "lá", "la", "with", "the", "would", "will", "okay",
))
_NEXT_WORD_RE = re.compile(r"\s*([a-zà-ÿ']+)")

# A DURATION, not a time of day: the words before or after an «Nh» that make it an amount.
_DURATION_BEFORE_RE = re.compile(
    r"(?:\bdura|\bduram|\bdurar|\bdurou|\bdurante|\bpor|\bem|\bhá|\bha|\bcerca\s+de|"
    r"\baproximadamente|\baprox\.?|\bmais\s+de|\bmenos\s+de|\bmais|\bmenos|\bfaz|\bleva|"
    r"\blevar|\blevam|\bfor|\bin|\babout|\btakes?|\blasts?|\bevery|\ba\s+cada|\bcada)\s*$")
_DURATION_AFTER_RE = re.compile(
    r"^\s*(?:de\s+(?:duração|duracao|antecedência|antecedencia|atraso|aula|consulta|sessão|"
    r"sessao|reunião|reuniao|intervalo|trabalho|carga|treino)|antes|depois|atrás|atras|"
    r"a\s+mais|a\s+menos|por\s+(?:dia|semana|mês|mes)|semanais|diárias|diarias|before|after|"
    r"ago|long|of\b)")
# «de 9h às 18h» is a range of times; «de 1h» alone is an amount. The connector decides.
_RANGE_AFTER_RE = re.compile(r"^\s*(?:às|as|a|até|ate|-|–|to|e)\b|^\s*[-–]")


def _period_shift(hour: int, period: str) -> "tuple[int, bool]":
    """Apply «da tarde/da noite/da manhã» — the period settles the ambiguity."""
    if period in ("tarde", "noite", "afternoon", "evening", "night"):
        return (hour + 12 if hour < 12 else hour), False
    if period in ("manhã", "manha", "madrugada", "morning"):
        return (0 if hour == 12 else hour), False
    return hour, True


def _make(hour_s: str, minute_s: Optional[str], ampm: Optional[str], rest: str,
          colon: bool = False) -> Optional[ClockTime]:
    try:
        hour = int(hour_s)
        minute = int(minute_s) if minute_s else 0
    except ValueError:      # pragma: no cover — the patterns only yield digits
        return None
    if ampm:
        if not 1 <= hour <= 12 or not 0 <= minute <= 59:
            return None
        hour = hour % 12 + (12 if ampm.lower() == "p" else 0)
        return ClockTime(hour, minute)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    # «14:00», «09:30», «10:30» are the 24-hour clock; a one-digit «3:30», «às 3», «3h» or a
    # bare «às 10» can be the afternoon.
    ambiguous = 1 <= hour <= 11 and not hour_s.startswith("0") and not (colon and len(hour_s) == 2)
    period = _PERIOD_RE.match(rest)
    if period and ambiguous:
        hour, ambiguous = _period_shift(hour, period.group(1))
    return ClockTime(hour, minute, ambiguous)


def _is_duration(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - 24):start]
    after = text[end:end + 40]
    if _DURATION_AFTER_RE.match(after):
        return True
    if _DURATION_BEFORE_RE.search(before):
        return True
    if re.search(r"\bde\s*$", before) and not _RANGE_AFTER_RE.match(after):
        return True
    return False


def _lowered(text: str) -> str:
    """``text`` lower-cased CHARACTER BY CHARACTER, so every offset still points at the same
    character of the original (a span is what the voice's floor masks)."""
    return "".join(c.lower() if len(c.lower()) == 1 else c for c in text).replace("\u00a0", " ")


def clock_time_spans(text: object) -> "list[tuple[int, int, ClockTime]]":
    """Every explicit time of day in ``text`` with its span in ``text`` — in order, every
    occurrence. The span covers the NUMBER (and its «h»/minutes), never the preposition, so a
    masked «às 15» reads «às […]». See the module docstring for what counts."""
    if not isinstance(text, str) or not text:
        return []
    low = _lowered(text)
    # Length-preserving, so the spans stay the original's: the zone becomes blanks, the ISO «T»
    # a blank.
    low = _ISO_OFFSET_RE.sub(lambda m: m.group(1) + " " * (len(m.group(0)) - len(m.group(1))), low)
    low = _ISO_T_RE.sub(lambda m: m.group(1) + " " + m.group(2), low)
    found: "list[tuple[int, int, ClockTime]]" = []
    taken: "list[tuple[int, int]]" = []

    def _free(a: int, b: int) -> bool:
        return all(b <= s or a >= e for s, e in taken)

    for m in _COLON_RE.finditer(low):
        t = _make(m.group(1), m.group(2), m.group(3), low[m.end():], colon=True)
        taken.append(m.span())
        if t is not None:
            found.append((m.start(), m.end(), t))
    for m in _AMPM_RE.finditer(low):
        if not _free(*m.span()):
            continue
        taken.append(m.span())
        t = _make(m.group(1), None, m.group(2), "")
        if t is not None:
            found.append((m.start(), m.end(), t))
    for m in _H_RE.finditer(low):
        if not _free(*m.span()):
            continue
        taken.append(m.span())
        if _is_duration(low, m.start(), m.end()):
            continue
        t = _make(m.group(1), m.group(2), None, low[m.end():])
        if t is not None:
            found.append((m.start(), m.end(), t))
    for m in _PREP_RE.finditer(low):
        span = m.span(1)
        if not _free(*span):
            continue
        rest = low[m.end():]
        nxt = _NEXT_WORD_RE.match(rest)
        if nxt and nxt.group(1) not in _AFTER_A_BARE_HOUR:
            continue
        taken.append(span)
        t = _make(m.group(1), None, None, rest)
        if t is not None:
            found.append((span[0], span[1], t))
    for m in _NOON_RE.finditer(low):
        found.append((m.start(), m.end(), ClockTime(12, 0)))
    for m in _MIDNIGHT_RE.finditer(low):
        found.append((m.start(), m.end(), ClockTime(0, 0)))
    return sorted(found, key=lambda f: f[0])


def clock_times(text: object) -> "list[ClockTime]":
    """Every explicit time of day in ``text``, in order, each once. ``[]`` for anything that is
    not a non-empty string."""
    out: "list[ClockTime]" = []
    for _a, _b, t in clock_time_spans(text):
        if t not in out:
            out.append(t)
    return out


# ── dates ────────────────────────────────────────────────────────────────────────────────

_MONTHS = {
    "janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3, "abril": 4, "maio": 5, "junho": 6,
    "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_SLASH_RE = re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?:/(?:\d{4}|\d{2}))?(?![\d/])")
_ISO_DATE_RE = re.compile(r"(?<!\d)\d{4}-(\d{2})-(\d{2})(?!\d)")
_DIA_RE = re.compile(r"\bdia\s+(\d{1,2})(?![\d:/h])(?:\s+de\s+(" + _MONTH_ALT + r"))?")
_DAY_OF_MONTH_RE = re.compile(r"(?<![\d/])(\d{1,2})\s+de\s+(" + _MONTH_ALT + r")\b")


def _date(day_s: str, month: Optional[int]) -> Optional[CalendarDate]:
    try:
        day = int(day_s)
    except ValueError:      # pragma: no cover
        return None
    if not 1 <= day <= 31 or (month is not None and not 1 <= month <= 12):
        return None
    return CalendarDate(day, month)


def calendar_dates(text: object) -> "list[CalendarDate]":
    """Every explicit calendar date in ``text``, in order, each once."""
    if not isinstance(text, str) or not text:
        return []
    low = text.lower()
    found: "list[tuple[int, Optional[CalendarDate]]]" = []
    for m in _ISO_DATE_RE.finditer(low):
        found.append((m.start(), _date(m.group(2), int(m.group(1)))))
    for m in _SLASH_RE.finditer(low):
        found.append((m.start(), _date(m.group(1), int(m.group(2)))))
    spans: "list[tuple[int, int]]" = []
    for m in _DIA_RE.finditer(low):
        spans.append(m.span())
        month = _MONTHS.get(m.group(2)) if m.group(2) else None
        found.append((m.start(), _date(m.group(1), month)))
    for m in _DAY_OF_MONTH_RE.finditer(low):
        if any(s <= m.start() < e for s, e in spans):
            continue
        found.append((m.start(), _date(m.group(1), _MONTHS[m.group(2)])))
    out: "list[CalendarDate]" = []
    for _pos, d in sorted(found, key=lambda p: p[0]):
        if d is not None and d not in out:
            out.append(d)
    return out


# ── the two readers ──────────────────────────────────────────────────────────────────────

def argument_texts(arguments: object, _depth: int = 0) -> "list[str]":
    """Every STRING value inside a call's arguments, nested dicts and lists included.

    Values only, never keys: which argument is "the time" is the host's catalogue, and the core
    does not guess it from a name. A number is not read either — ``14`` under an unknown key is
    as likely a quantity as an hour."""
    if _depth > 6:
        return []
    if isinstance(arguments, str):
        return [arguments]
    if isinstance(arguments, dict):
        return [s for v in arguments.values() for s in argument_texts(v, _depth + 1)]
    if isinstance(arguments, (list, tuple)):
        return [s for v in arguments for s in argument_texts(v, _depth + 1)]
    return []


def _distinct(values: "Sequence[Any]", same: Any) -> "list[Any]":
    out: "list[Any]" = []
    for v in values:
        if not any(same(v, o) for o in out):
            out.append(v)
    return out


def _writes(calls: "Iterable[Any]") -> "list[Any]":
    """The calls that WRITE — the family's two facts (``side_effect``, ``tool_mutating``), the
    same test ``types.write_attempted_this_turn`` applies. Failed writes included: a write
    attempted with the wrong value is the wrong action whether or not the server took it."""
    out = []
    for c in calls or ():
        try:
            if getattr(c, "side_effect", False) is True or getattr(c, "tool_mutating", None) is True:
                out.append(c)
        except Exception:      # noqa: BLE001 — an unreadable record is not evidence
            continue
    return out


def write_value_mismatches(request: object, calls: "Iterable[Any]") -> "list[WriteValueMismatch]":
    """The WRITES whose time (or date) is not the ONE the request states.

    Fires only when ``request`` names exactly one time (or exactly one date) — two named values
    («das 14 para as 15») leave the comparison to the model — and a write's arguments carry at
    least one value of that kind, NONE of which shares a reading with it. A write that carries
    the requested value anywhere in its arguments (a reschedule with the old and the new time)
    does not fire. ``[]`` on anything unreadable."""
    try:
        wanted_t = _distinct(clock_times(request), same_time)
        wanted_d = _distinct(calendar_dates(request), same_date)
        out: "list[WriteValueMismatch]" = []
        for call in _writes(calls):
            texts = argument_texts(getattr(call, "arguments", None))
            tool = " ".join(str(getattr(call, "tool", "") or "").split())[:64]
            if len(wanted_t) == 1:
                written_t = _distinct([t for s in texts for t in clock_times(s)], same_time)
                if written_t and not any(same_time(wanted_t[0], t) for t in written_t):
                    out.append(WriteValueMismatch(tool, "time", wanted_t[0].label(),
                                                  tuple(t.label() for t in written_t)))
            if len(wanted_d) == 1:
                written_d = _distinct([d for s in texts for d in calendar_dates(s)], same_date)
                if written_d and not any(same_date(wanted_d[0], d) for d in written_d):
                    out.append(WriteValueMismatch(tool, "date", wanted_d[0].label(),
                                                  tuple(d.label() for d in written_d)))
        return out
    except Exception:      # noqa: BLE001 — a computed hint must never cost the turn
        return []


def unread_times(reply: object, evidence: "Iterable[object]") -> "list[ClockTime]":
    """The times ``reply`` states that NONE of ``evidence`` states (in any reading).

    The gate — "no successful call returned a time" — is the caller's; this only compares."""
    seen = [t for e in evidence for t in clock_times(e)]
    return [t for t in clock_times(reply) if not any(same_time(t, s) for s in seen)]
