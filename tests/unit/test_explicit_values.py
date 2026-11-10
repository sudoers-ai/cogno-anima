"""What counts as an explicit time or date — the twins and the adverse probe.

`cogno_anima.explicit_values` is read by the judge (a write made with another value than the one
the user stated) and by the voice (a reply that lists times nobody read). Both readers pay for a
FALSE time: the judge is shown a mismatch that is not one, the voice re-voices a true sentence.
So the probe below is mostly phrases that are NOT times — a price, a quantity, a duration, an
ISO zone — each with the reading it must give.
"""

from __future__ import annotations

import pytest

from cogno_anima.explicit_values import (CalendarDate, ClockTime, calendar_dates,
                                         clock_time_spans, clock_times, unread_times,
                                         write_value_mismatches)
from cogno_anima.types import ToolExecution

# text -> the labels it must read, in order
_TIMES = {
    # the forms that ARE times
    "pode ser às 15": ["15:00"],
    "pode ser as 15": ["15:00"],
    "às 15h": ["15:00"],
    "15:00": ["15:00"],
    "14h30": ["14:30"],
    "14 horas": ["14:00"],
    "às 3 da tarde": ["15:00"],
    "3pm": ["15:00"],
    "at 3 pm": ["15:00"],
    "meio-dia": ["12:00"],
    "até 18h": ["18:00"],
    "Amanhã tenho 09:00, 10:30 e 14:00.": ["09:00", "10:30", "14:00"],
    "2026-11-11T14:00:00-03:00": ["14:00"],          # the zone is not a time of day
    "entre 14h e 16h": ["14:00", "16:00"],
    "de 9h às 18h": ["09:00 or 21:00", "18:00"],     # a range, not a duration
    "Não, às 15 não dá": ["15:00"],                  # read; refusing it is the model's to see
    "às 10?": ["10:00 or 22:00"],                    # ambiguous: both readings
    "1h antes da consulta das 15": ["15:00"],        # «1h antes» is an amount
    # the forms that are NOT
    "a consulta dura 1h": [],
    "consulta de 1h": [],
    "tente de novo em 1h": [],
    "1h30 de duração": [],
    "as 3 opções": [],
    "às 15 vagas": [],
    "atendimento 24h": [],
    "R$ 15,00": [],
    "11 98765-4321": [],
    "nota 9/10": [],
}


@pytest.mark.parametrize("text", sorted(_TIMES))
def test_what_reads_as_a_time(text):
    assert [t.label() for t in clock_times(text)] == _TIMES[text]


_DATES = {
    "dia 12": ["day 12"],
    "12/11": ["12/11"],
    "12/11/2026": ["12/11"],
    "12 de novembro": ["12/11"],
    "dia 12 de novembro às 15": ["12/11"],
    "2026-11-12": ["12/11"],
    "amanhã": [],
    "às 15": [],
    "R$ 1.500,00": [],
}


@pytest.mark.parametrize("text", sorted(_DATES))
def test_what_reads_as_a_date(text):
    assert [d.label() for d in calendar_dates(text)] == _DATES[text]


def test_an_ambiguous_hour_matches_both_readings_and_nothing_else():
    three = clock_times("às 3")[0]
    assert three.readings() == {(3, 0), (15, 0)}
    assert clock_times("15:00")[0].readings() == {(15, 0)}
    assert clock_times("09:00")[0].readings() == {(9, 0)}     # a leading zero is the 24 h clock


def test_a_span_is_the_number_never_the_preposition():
    text = "às 15 ou 9h30"
    assert [text[a:b] for a, b, _ in clock_time_spans(text)] == ["15", "9h30"]


def test_garbage_is_no_value():
    for junk in (None, 15, b"15:00", "", ["15:00"]):
        assert clock_times(junk) == [] and calendar_dates(junk) == []


# ── write_value_mismatches ────────────────────────────────────────────────────────────────

def _w(**args) -> ToolExecution:
    return ToolExecution(tool="book_appointment", arguments=args, result="ok", ok=True,
                         side_effect=True)


def test_the_measured_mismatch():
    found = write_value_mismatches("pode ser às 15", [_w(time="14:00")])
    assert [(f.tool, f.kind, f.requested, f.written) for f in found] \
        == [("book_appointment", "time", "15:00", ("14:00",))]


@pytest.mark.parametrize("request_, args", [
    ("pode ser às 15", {"time": "15:00"}),                    # the same hour
    ("às 3", {"time": "15:00"}),                              # an ambiguous hour, one reading
    ("muda das 14 para as 15", {"time": "14:00"}),            # two hours: the model's
    ("muda das 14 para as 15", {"time": "15:00"}),            # ...whichever of them is used
    ("marca amanhã", {"time": "14:00"}),                      # no hour stated
    ("às 15", {"old": "14:00", "new": "15:00"}),              # the stated one is among them
    ("às 15", {"amount": "50"}),                              # a write with no hour
    ("às 15", {"hour": 14}),                                  # a NUMBER is not read
])
def test_no_mismatch(request_, args):
    assert write_value_mismatches(request_, [_w(**args)]) == []


def test_a_read_is_never_compared():
    read = ToolExecution(tool="check_availability", arguments={"time": "14:00"}, result="",
                         ok=True, side_effect=False)
    assert write_value_mismatches("às 15", [read]) == []


def test_a_failed_write_declared_mutating_is_compared():
    call = ToolExecution(tool="book_appointment", arguments={"time": "14:00"}, result="",
                         error="taken", ok=False, side_effect=False, tool_mutating=True)
    assert len(write_value_mismatches("às 15", [call])) == 1


def test_the_date_half():
    found = write_value_mismatches("dia 12", [_w(date="2026-11-13")])
    assert [(f.kind, f.requested, f.written) for f in found] == [("date", "day 12", ("13/11",))]
    assert write_value_mismatches("dia 12", [_w(date="2026-11-12")]) == []


def test_an_unreadable_call_is_no_mismatch():
    class Broken:
        @property
        def side_effect(self):
            raise RuntimeError("boom")
    assert write_value_mismatches("às 15", [Broken()]) == []


# ── unread_times ──────────────────────────────────────────────────────────────────────────

def test_unread_times_compares_by_reading():
    assert unread_times("tenho 09:00 e 15h", ["pode ser às 3?"]) == [ClockTime(9, 0)]
    assert unread_times("tenho 15h", ["às 3"]) == []
    assert unread_times("", ["x"]) == []


def test_labels_are_our_own_format():
    assert ClockTime(9, 5).label() == "09:05"
    assert CalendarDate(3, 7).label() == "03/07" and CalendarDate(3).label() == "day 3"
