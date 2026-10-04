"""NYSE/Cboe trading calendar and the two minute conventions used by the volatility indices.

The Cboe Volatility Index counts *calendar* minutes (365 x 1,440 per year). The one-day
index counts *business* minutes inside the regular session, extended to 16:15 for index
options: 405 minutes a day and 252 x 405 = 102,060 a year.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from functools import cache
from zoneinfo import ZoneInfo

__all__ = [
    "BUSINESS_MINUTES_PER_YEAR",
    "ET",
    "MINUTES_PER_CALENDAR_YEAR",
    "SESSION_MINUTES",
    "business_minutes",
    "calendar_minutes",
    "expiry_datetime",
    "is_trading_day",
    "last_trading_day_of_month",
    "last_trading_day_of_week",
    "next_trading_day",
    "previous_trading_day",
    "resolve_asof",
    "session_close",
    "session_end",
    "third_friday",
    "trading_days",
]

ET = ZoneInfo("America/New_York")

MINUTES_PER_CALENDAR_YEAR = 365 * 1440
SESSION_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)
INDEX_TAIL = timedelta(minutes=15)
SESSION_MINUTES = 405
TRADING_DAYS_PER_YEAR = 252
BUSINESS_MINUTES_PER_YEAR = TRADING_DAYS_PER_YEAR * SESSION_MINUTES

# Unscheduled full-day closures (September 11 attacks, Sandy, national days of mourning).
_SPECIAL_CLOSURES = frozenset(
    {
        date(2001, 9, 11),
        date(2001, 9, 12),
        date(2001, 9, 13),
        date(2001, 9, 14),
        date(2004, 6, 11),
        date(2007, 1, 2),
        date(2012, 10, 29),
        date(2012, 10, 30),
        date(2018, 12, 5),
        date(2025, 1, 9),
    }
)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous algorithm, Meeus/Jones/Butcher)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month = (h + el - 7 * m + 114) // 31
    day = (h + el - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _observed(d: date, *, saturday_to_friday: bool = True) -> date | None:
    if d.weekday() == 5:
        return d - timedelta(days=1) if saturday_to_friday else None
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@cache
def _holidays(year: int) -> frozenset[date]:
    days: list[date | None] = [
        _observed(date(year, 1, 1), saturday_to_friday=False),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _easter(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed(date(year, 12, 25)),
    ]
    if year >= 2022:
        days.append(_observed(date(year, 6, 19)))
    return frozenset(d for d in days if d is not None and d.year == year)


def is_trading_day(d: date) -> bool:
    """True if the exchange holds a regular session on ``d``."""
    return d.weekday() < 5 and d not in _holidays(d.year) and d not in _SPECIAL_CLOSURES


def _is_early_close(d: date) -> bool:
    if not is_trading_day(d):
        return False
    thanksgiving = _nth_weekday(d.year, 11, 3, 4)
    if d == thanksgiving + timedelta(days=1):
        return True
    if d.month == 12 and d.day == 24:
        return True
    return d.month == 7 and d.day == 3 and date(d.year, 7, 4).weekday() < 5


def next_trading_day(d: date) -> date:
    """First trading day strictly after ``d``."""
    d += timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def previous_trading_day(d: date) -> date:
    """Last trading day strictly before ``d``."""
    d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def trading_days(start: date, end: date) -> list[date]:
    """Trading days in the closed interval ``[start, end]``."""
    out, d = [], start
    while d <= end:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def session_close(d: date) -> datetime:
    """Equity closing time on trading day ``d``: 16:00, or 13:00 on early closes."""
    if not is_trading_day(d):
        raise ValueError(f"{d} is not a trading day")
    return datetime.combine(d, EARLY_CLOSE if _is_early_close(d) else REGULAR_CLOSE, tzinfo=ET)


def session_end(d: date) -> datetime:
    """End of the index-option session: the close plus the 15-minute index tail."""
    return session_close(d) + INDEX_TAIL


def third_friday(year: int, month: int) -> date:
    """Standard monthly expiry: the third Friday, or the prior trading day if it is a holiday."""
    d = _nth_weekday(year, month, 4, 3)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def last_trading_day_of_week(d: date) -> date:
    """Last trading day of the Monday-to-Friday week containing ``d``."""
    friday = d + timedelta(days=4 - d.weekday())
    while not is_trading_day(friday):
        friday -= timedelta(days=1)
    return friday


def last_trading_day_of_month(d: date) -> date:
    """Last trading day of the calendar month containing ``d``."""
    last = date(d.year + (d.month == 12), d.month % 12 + 1, 1) - timedelta(days=1)
    while not is_trading_day(last):
        last -= timedelta(days=1)
    return last


def expiry_datetime(expiry: date, settlement: str) -> datetime:
    """Deemed expiration time: 09:30 for AM-settled SPX, the session close for PM-settled SPXW."""
    if settlement == "AM":
        return datetime.combine(expiry, SESSION_OPEN, tzinfo=ET)
    if settlement == "PM":
        return session_close(expiry)
    raise ValueError(f"settlement must be 'AM' or 'PM', got {settlement!r}")


def calendar_minutes(start: datetime, end: datetime) -> int:
    """Whole minutes of real elapsed time from ``start`` to ``end``, rounded down.

    The difference is taken between UTC instants, so a daylight-saving change between the two
    times shortens or lengthens the span by an hour; subtracting aware datetimes that share a
    ``ZoneInfo`` would count wall-clock time instead.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("datetimes must be timezone-aware")
    return int((end.astimezone(UTC) - start.astimezone(UTC)).total_seconds() // 60)


def business_minutes(start: datetime, end: datetime) -> int:
    """Minutes of index-option sessions (open to close plus 15 minutes) between two instants."""
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("datetimes must be timezone-aware")
    if end <= start:
        return 0
    start, end = start.astimezone(ET), end.astimezone(ET)
    total = 0.0
    d = start.date()
    while d <= end.date():
        if is_trading_day(d):
            lo = max(start, datetime.combine(d, SESSION_OPEN, tzinfo=ET))
            hi = min(end, session_end(d))
            if hi > lo:
                total += (hi - lo).total_seconds()
        d += timedelta(days=1)
    return int(total // 60)


def resolve_asof(fetched_at: datetime) -> datetime:
    """Valuation time implied by a download time.

    Inside an index-option session this is the download time. Outside regular hours option
    quotes are frozen at the close, so the valuation time is the end of the most recent
    index-option session (16:15 ET).
    """
    if fetched_at.tzinfo is None:
        raise ValueError("fetched_at must be timezone-aware")
    t = fetched_at.astimezone(ET).replace(second=0, microsecond=0)
    d = t.date()
    if is_trading_day(d):
        if t > session_end(d):
            return session_end(d)
        if t >= datetime.combine(d, SESSION_OPEN, tzinfo=ET):
            return t
    return session_end(previous_trading_day(d))
