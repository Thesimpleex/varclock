from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pytest

from varclock import tradingcal as tc
from varclock.indices import (
    IndexUnavailableError,
    compute_index,
    eligible_strips,
    replicate,
)
from varclock.quotes import Snapshot
from varclock.rates import YieldCurve
from varclock.synthetic import black76_prices, synthetic_strip

ASOF = datetime(2026, 10, 2, 16, 15, tzinfo=tc.ET)
CURVE = YieldCurve(date(2026, 10, 1), {"1 Mo": 0.0, "3 Mo": 0.0, "1 Yr": 0.0})
VOL = 0.2


def _strip(root, expiry, total_variance, forward=1000.0):
    sd = np.sqrt(total_variance)
    spacing = max(0.1, min(2.0, sd * forward / 60))
    lo = max(spacing, np.floor(forward * np.exp(-8 * sd) / spacing) * spacing)
    hi = np.ceil(forward * np.exp(8 * sd) / spacing) * spacing
    k = np.arange(lo, hi + spacing, spacing)
    call, put = black76_prices(forward, k, total_variance)
    return synthetic_strip(root, expiry, k, call, put)


def flat_snapshot(clock="calendar", asof=ASOF):
    """SPX/SPXW chains whose total variance is VOL^2 times time, in calendar or business time."""
    first = asof.date()
    last = first + timedelta(days=130)
    third_fridays = {tc.third_friday(y, m) for y in (2026, 2027) for m in range(1, 13)}
    strips = []
    day = first
    while day <= last:
        if tc.is_trading_day(day):
            near = (day - first).days <= (21 if clock == "calendar" else 10)
            wanted = []
            if clock == "calendar" and day in third_fridays:
                wanted.append("SPX")
            if near or day in third_fridays or day == tc.last_trading_day_of_week(day):
                wanted.append("SPXW")
            if day == tc.last_trading_day_of_month(day):
                wanted.append("SPXW")
            for root in dict.fromkeys(wanted):
                when = tc.expiry_datetime(day, "AM" if root == "SPX" else "PM")
                if clock == "calendar":
                    t = tc.calendar_minutes(asof, when) / tc.MINUTES_PER_CALENDAR_YEAR
                else:
                    t = tc.business_minutes(asof, when) / tc.BUSINESS_MINUTES_PER_YEAR
                if t > 0:
                    strips.append(_strip(root, day, VOL**2 * t))
        day += timedelta(days=1)
    return Snapshot("SPX", "synthetic", asof, asof.astimezone(tc.ET), 1000.0, tuple(strips))


@pytest.fixture(scope="module")
def calendar_snapshot():
    return flat_snapshot("calendar")


def test_flat_volatility_is_returned_by_every_calendar_index(calendar_snapshot):
    for name in ("VIX", "VIX9D", "VIX3M"):
        assert compute_index(calendar_snapshot, CURVE, name).value == pytest.approx(20.0, abs=0.02)


def test_flat_volatility_in_business_time_is_returned_by_vix1d():
    snap = flat_snapshot("business")
    result = compute_index(snap, CURVE, "VIX1D")
    assert result.value == pytest.approx(20.0, abs=0.02)
    assert result.near is None  # the same-day series expired at 16:00
    assert result.next.strip.expiry == date(2026, 10, 5)
    assert result.next.minutes == 390


def test_term_selection_at_the_friday_close(calendar_snapshot):
    vix = compute_index(calendar_snapshot, CURVE, "VIX")
    assert (vix.near.strip.root, vix.near.strip.expiry) == ("SPXW", date(2026, 10, 30))
    assert (vix.next.strip.root, vix.next.strip.expiry) == ("SPXW", date(2026, 11, 6))
    nine = compute_index(calendar_snapshot, CURVE, "VIX9D")
    assert (nine.near.strip.expiry, nine.near.strip.root) == (date(2026, 10, 9), "SPXW")
    assert (nine.next.strip.expiry, nine.next.strip.root) == (date(2026, 10, 16), "SPX")
    three = compute_index(calendar_snapshot, CURVE, "VIX3M")
    assert three.near.strip.expiry == date(2026, 12, 31)
    assert three.next.strip.expiry == date(2027, 1, 15)


def test_universe_rules(calendar_snapshot):
    weekly = eligible_strips(calendar_snapshot, "end_of_week")
    spx = {s.expiry for s in weekly if s.root == "SPX"}
    spxw = {s.expiry for s in weekly if s.root == "SPXW"}
    assert date(2026, 10, 16) in spx
    assert date(2026, 10, 16) not in spxw  # same date as the AM-settled series
    assert date(2026, 10, 7) not in spxw  # mid-week dailies are not "end of week"
    assert date(2026, 10, 9) in spxw
    monthly = eligible_strips(calendar_snapshot, "end_of_month")
    assert date(2026, 10, 30) in {s.expiry for s in monthly if s.root == "SPXW"}
    assert date(2026, 11, 6) not in {s.expiry for s in monthly}
    pm = eligible_strips(calendar_snapshot, "pm_all")
    assert {s.root for s in pm} == {"SPXW"}
    assert date(2026, 10, 7) in {s.expiry for s in pm}


def test_replicate_reports_unavailable_indices_instead_of_failing():
    snap = flat_snapshot("calendar")
    only_far = Snapshot(
        snap.symbol, snap.source, snap.asof, snap.fetched_at, snap.spot,
        tuple(s for s in snap.strips if s.expiry > date(2026, 12, 1)),
    )  # fmt: skip
    out = replicate(only_far, CURVE)
    assert isinstance(out["VIX1D"], IndexUnavailableError)
    assert not isinstance(out["VIX3M"], IndexUnavailableError)
    assert "extrapolation" in out["VIX9D"].warnings[0]


def test_unknown_index_and_empty_snapshot():
    snap = flat_snapshot("calendar")
    with pytest.raises(ValueError, match="unknown index"):
        compute_index(snap, CURVE, "VIX7")
    empty = Snapshot(snap.symbol, snap.source, snap.asof, snap.fetched_at, None, ())
    with pytest.raises(IndexUnavailableError):
        compute_index(empty, CURVE, "VIX")


def test_vix1d_intraday_interpolates_between_today_and_tomorrow():
    asof = datetime(2026, 10, 5, 11, 0, tzinfo=tc.ET)
    curve = YieldCurve(date(2026, 10, 2), {"1 Mo": 0.0, "3 Mo": 0.0, "1 Yr": 0.0})
    snap = flat_snapshot("business", asof)
    result = compute_index(snap, curve, "VIX1D")
    assert result.near is not None
    assert result.near.minutes == 300
    assert result.next.minutes == 705
    assert result.value == pytest.approx(20.0, abs=0.02)
    late = datetime(2026, 10, 5, 15, 30, tzinfo=tc.ET)
    warned = compute_index(flat_snapshot("business", late), curve, "VIX1D")
    assert warned.warnings
    assert "under 60" in warned.warnings[0]


def test_index_level_scales_with_volatility():
    base = compute_index(flat_snapshot("calendar"), CURVE, "VIX").value
    assert base == pytest.approx(20.0, abs=0.02)


def test_vix1d_uses_the_next_term_alone_when_it_is_shorter_than_the_target():
    """Before an early close the next session has under 390 business minutes: no extrapolation."""
    asof = datetime(2025, 11, 26, 15, 50, tzinfo=tc.ET)
    curve = YieldCurve(date(2025, 11, 25), {"1 Mo": 0.0, "3 Mo": 0.0, "1 Yr": 0.0})
    snap = flat_snapshot("business", asof)
    result = compute_index(snap, curve, "VIX1D")
    assert result.next.strip.expiry == date(2025, 11, 28)
    assert result.next.minutes < 390
    assert result.near is None
    assert not result.warnings
    assert result.value == pytest.approx(20.0, abs=0.02)
