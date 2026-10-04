"""Dated event calendars: FOMC, CPI, employment releases, U.S. elections, mega-cap earnings.

The CSV files in ``varclock/calendars`` are committed with their sources; rebuild them with
``scripts/build_calendars.py``. An event belongs to the trading *session* whose variance it
moves: the same day if the release comes before the close, otherwise the next trading day.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time
from importlib import resources

from .tradingcal import ET, is_trading_day, next_trading_day, session_close

__all__ = [
    "EVENT_FILES",
    "SCHEDULED_KINDS",
    "Event",
    "calendar_end",
    "event_session",
    "load_events",
]

#: Kinds whose committed files are complete schedules (earnings dates are announced or projected).
SCHEDULED_KINDS = ("FOMC", "CPI", "NFP")

EVENT_FILES = {
    "FOMC": "fomc.csv",
    "CPI": "cpi.csv",
    "NFP": "employment.csv",
    "ELECTION": "elections.csv",
    "EARNINGS": "earnings.csv",
}


@dataclass(frozen=True)
class Event:
    """A scheduled or announced event with the session it moves."""

    day: date
    time_et: time
    kind: str
    label: str
    source: str

    @property
    def session(self) -> date:
        return event_session(self.day, self.time_et)

    @property
    def moment(self) -> datetime:
        return datetime.combine(self.day, self.time_et, tzinfo=ET)


def event_session(day: date, time_et: time) -> date:
    """Trading session whose variance carries an event at ``time_et`` on ``day``."""
    if is_trading_day(day) and datetime.combine(day, time_et) < session_close(day).replace(
        tzinfo=None
    ):
        return day
    return next_trading_day(day)


def _read(name: str) -> Iterable[dict[str, str]]:
    path = resources.files("varclock") / "calendars" / name
    with path.open("r", encoding="utf-8", newline="") as fh:
        yield from csv.DictReader(fh)


def calendar_end(kind: str) -> date:
    """Date of the last event in the committed calendar file of ``kind``."""
    if kind.upper() not in EVENT_FILES:
        raise ValueError(f"unknown event kind {kind!r}; choose from {sorted(EVENT_FILES)}")
    return max(date.fromisoformat(row["date"]) for row in _read(EVENT_FILES[kind.upper()]))


def load_events(
    kinds: Iterable[str] | None = None, start: date | None = None, end: date | None = None
) -> list[Event]:
    """Events of the given kinds (default all) whose session lies in ``[start, end]``."""
    wanted = list(EVENT_FILES) if kinds is None else [k.upper() for k in kinds]
    unknown = set(wanted) - set(EVENT_FILES)
    if unknown:
        raise ValueError(
            f"unknown event kinds {sorted(unknown)}; choose from {sorted(EVENT_FILES)}"
        )
    out: list[Event] = []
    for kind in wanted:
        for row in _read(EVENT_FILES[kind]):
            ev = Event(
                date.fromisoformat(row["date"]),
                time.fromisoformat(row["time_et"]),
                row["kind"],
                row["label"],
                row["source"],
            )
            if start is not None and ev.session < start:
                continue
            if end is not None and ev.session > end:
                continue
            out.append(ev)
    return sorted(out, key=lambda e: (e.session, e.kind, e.label))
