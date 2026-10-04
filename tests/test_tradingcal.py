from __future__ import annotations

from datetime import date, datetime

import pytest

from varclock import tradingcal as tc

ET = tc.ET


def dt(y, m, d, hh=0, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def test_trading_days_per_year_match_published_exchange_counts():
    counts = {y: len(tc.trading_days(date(y, 1, 1), date(y, 12, 31))) for y in range(2022, 2027)}
    assert counts == {2022: 251, 2023: 250, 2024: 252, 2025: 250, 2026: 251}


def test_2026_holiday_schedule():
    holidays = [
        date(2026, 1, 1),
        date(2026, 1, 19),
        date(2026, 2, 16),
        date(2026, 4, 3),
        date(2026, 5, 25),
        date(2026, 6, 19),
        date(2026, 7, 3),
        date(2026, 9, 7),
        date(2026, 11, 26),
        date(2026, 12, 25),
    ]
    assert not any(tc.is_trading_day(d) for d in holidays)
    for d in (date(2026, 10, 12), date(2026, 11, 11), date(2026, 7, 2)):
        assert tc.is_trading_day(d)


def test_saturday_new_year_is_not_observed_on_friday():
    assert tc.is_trading_day(date(2021, 12, 31))
    assert not tc.is_trading_day(date(2022, 1, 17))


def test_special_closure_for_day_of_mourning():
    assert not tc.is_trading_day(date(2025, 1, 9))


def test_early_closes():
    assert tc.session_close(date(2025, 11, 28)).hour == 13
    assert tc.session_close(date(2025, 12, 24)).hour == 13
    assert tc.session_close(date(2025, 7, 3)).hour == 13
    assert tc.session_close(date(2025, 12, 26)).hour == 16
    with pytest.raises(ValueError, match="not a trading day"):
        tc.session_close(date(2025, 12, 25))


def test_third_friday_and_week_month_ends():
    assert tc.third_friday(2026, 10) == date(2026, 10, 16)
    assert tc.third_friday(2026, 4) == date(2026, 4, 17)
    assert tc.third_friday(2022, 4) == date(2022, 4, 14)
    assert tc.last_trading_day_of_week(date(2026, 10, 7)) == date(2026, 10, 9)
    assert tc.last_trading_day_of_week(date(2026, 4, 1)) == date(2026, 4, 2)
    assert tc.last_trading_day_of_month(date(2026, 10, 5)) == date(2026, 10, 30)
    assert tc.last_trading_day_of_month(date(2026, 12, 1)) == date(2026, 12, 31)


def test_next_and_previous_trading_day_skip_weekend_and_holiday():
    assert tc.next_trading_day(date(2026, 10, 2)) == date(2026, 10, 5)
    assert tc.next_trading_day(date(2026, 11, 25)) == date(2026, 11, 27)
    assert tc.previous_trading_day(date(2026, 10, 5)) == date(2026, 10, 2)
    assert tc.previous_trading_day(date(2026, 11, 27)) == date(2026, 11, 25)


def test_cboe_calendar_minutes_in_the_worked_example():
    calc = dt(2022, 9, 27, 10, 45)
    near = tc.expiry_datetime(date(2022, 10, 21), "AM")
    nxt = tc.expiry_datetime(date(2022, 10, 28), "PM")
    # the example calculates at 10:45:15; whole minutes are rounded down
    assert tc.calendar_minutes(calc.replace(second=15), near) == 34484
    assert tc.calendar_minutes(calc.replace(second=15), nxt) == 44954


def test_calendar_minutes_count_real_elapsed_time_across_daylight_saving():
    start, end = dt(2026, 10, 30, 16, 15), dt(2026, 11, 6, 16, 0)
    # seven calendar days less fifteen minutes, plus the extra hour of the 1 November fall-back
    assert tc.calendar_minutes(start, end) == 7 * 1440 - 15 + 60
    spring = tc.calendar_minutes(dt(2026, 3, 6, 16, 15), dt(2026, 3, 13, 16, 0))
    assert spring == 7 * 1440 - 15 - 60
    assert tc.calendar_minutes(dt(2026, 6, 1, 9, 30), dt(2026, 6, 2, 9, 30)) == 1440


def test_business_minutes_follow_the_405_minute_session():
    calc = dt(2022, 9, 27, 11, 0)
    assert tc.business_minutes(calc, tc.expiry_datetime(date(2022, 9, 27), "PM")) == 300
    assert tc.business_minutes(calc, tc.expiry_datetime(date(2022, 9, 28), "PM")) == 705
    friday_close = dt(2026, 10, 2, 16, 15)
    assert tc.business_minutes(friday_close, tc.expiry_datetime(date(2026, 10, 5), "PM")) == 390
    assert tc.business_minutes(dt(2026, 10, 2, 9, 30), dt(2026, 10, 2, 16, 15)) == 405
    assert tc.business_minutes(dt(2026, 10, 3, 9, 0), dt(2026, 10, 4, 9, 0)) == 0
    assert tc.business_minutes(dt(2026, 10, 5), dt(2026, 10, 2)) == 0


def test_a_year_of_business_minutes():
    assert tc.BUSINESS_MINUTES_PER_YEAR == 102_060
    assert tc.MINUTES_PER_CALENDAR_YEAR == 525_600


def test_naive_datetimes_are_rejected():
    naive = datetime(2026, 10, 2, 10, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        tc.calendar_minutes(naive, naive)
    with pytest.raises(ValueError, match="timezone-aware"):
        tc.business_minutes(naive, naive)
    with pytest.raises(ValueError, match="timezone-aware"):
        tc.resolve_asof(naive)


@pytest.mark.parametrize(
    ("fetched", "expected"),
    [
        (dt(2026, 10, 3, 20, 17), dt(2026, 10, 2, 16, 15)),
        (dt(2026, 10, 4, 12, 0), dt(2026, 10, 2, 16, 15)),
        (dt(2026, 10, 5, 8, 0), dt(2026, 10, 2, 16, 15)),
        (dt(2026, 10, 5, 11, 7), dt(2026, 10, 5, 11, 7)),
        (dt(2026, 10, 5, 17, 30), dt(2026, 10, 5, 16, 15)),
    ],
)
def test_resolve_asof(fetched, expected):
    assert tc.resolve_asof(fetched) == expected


def test_resolve_asof_accepts_other_time_zones():
    utc = datetime(2026, 10, 4, 0, 17, tzinfo=ET).astimezone(datetime.now().astimezone().tzinfo)
    assert tc.resolve_asof(utc) == dt(2026, 10, 2, 16, 15)


def test_expiry_settlement_must_be_known():
    with pytest.raises(ValueError, match="settlement"):
        tc.expiry_datetime(date(2026, 10, 5), "XX")
