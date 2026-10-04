from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from helpers import example_curve
from varclock.rates import CMT_DAYS, YieldCurve, bounded_spline_bey, continuous_rate

VIX1D_CURVE = {
    "1 Mo": 0.03, "2 Mo": 0.02, "3 Mo": 0.04, "6 Mo": 0.05, "1 Yr": 0.08, "2 Yr": 0.11,
    "3 Yr": 0.22, "5 Yr": 0.59, "7 Yr": 1.00, "10 Yr": 1.37, "20 Yr": 2.03, "30 Yr": 2.21,
}  # fmt: skip
AUG_2022 = {
    "1 Mo": 0.41, "2 Mo": 0.71, "3 Mo": 0.90, "6 Mo": 1.49, "1 Yr": 2.10, "2 Yr": 2.73,
    "3 Yr": 2.93, "5 Yr": 3.01, "7 Yr": 3.04, "10 Yr": 2.99, "20 Yr": 3.26, "30 Yr": 3.07,
}  # fmt: skip


def test_vix_example_rates(vix_example):
    curve = example_curve(vix_example)
    assert curve.rate(date(2022, 10, 21)) * 100 == pytest.approx(0.031664, abs=5e-7)
    assert curve.rate(date(2022, 10, 28)) * 100 == pytest.approx(0.028797, abs=5e-7)


def test_standard_monthly_30_day_index_example_rates():
    curve = YieldCurve(date(2022, 8, 1), AUG_2022)
    assert curve.rate(date(2022, 8, 19)) * 100 == pytest.approx(0.2898, abs=5e-5)
    assert curve.rate(date(2022, 9, 16)) * 100 == pytest.approx(0.5808, abs=5e-5)


def test_vix1d_example_printed_rates_correspond_to_one_day_longer_day_counts():
    # Cboe's 1-day sample prints 0.0393% and 0.0390%; they match 2 and 3 days to expiry rather
    # than the 1 and 2 days implied by the dates. The effect on the index is below 1e-9.
    curve = YieldCurve(date(2022, 9, 26), VIX1D_CURVE)
    assert curve.rate(date(2022, 9, 28)) * 100 == pytest.approx(0.0393, abs=5e-5)
    assert curve.rate(date(2022, 9, 29)) * 100 == pytest.approx(0.0390, abs=5e-5)
    assert abs(curve.rate(date(2022, 9, 27)) - curve.rate(date(2022, 9, 28))) < 1e-5


def test_spline_interpolates_the_nodes_exactly():
    days = np.array([30.0, 60.0, 91.0, 182.0, 365.0])
    cmt = np.array([0.04, 0.041, 0.043, 0.044, 0.045])
    for d, y in zip(days[1:-1], cmt[1:-1], strict=True):
        assert bounded_spline_bey(days, cmt, d) == pytest.approx(y, abs=1e-12)


def test_interpolated_yield_stays_between_adjacent_cmts():
    days = np.array([30.0, 60.0, 91.0, 182.0])
    cmt = np.array([0.01, 0.05, 0.01, 0.06])  # a zig-zag that makes a plain spline overshoot
    for t in np.linspace(31, 181, 60):
        i = np.searchsorted(days, t) - 1
        lo, hi = sorted((cmt[i], cmt[i + 1]))
        assert lo - 1e-15 <= bounded_spline_bey(days, cmt, t) <= hi + 1e-15


def test_extrapolation_below_one_month_is_bounded_by_two_lines():
    days = np.array([30.0, 60.0, 91.0])
    cmt = np.array([0.0300, 0.0200, 0.0400])
    y = bounded_spline_bey(days, cmt, 10.0)
    upper = 0.03 + (0.02 - 0.03) / 30 * (10 - 30)
    lower = 0.03 + (0.04 - 0.03) / 61 * (10 - 30)
    assert lower <= y <= upper
    flat = bounded_spline_bey(days, np.array([0.03, 0.04, 0.05]), 10.0)
    assert flat == pytest.approx(0.03 - 0.01 / 30 * 20, abs=1e-12)


def test_bey_to_continuous_rate():
    assert continuous_rate(0.0) == 0.0
    assert continuous_rate(0.05) == pytest.approx(2 * np.log(1.025))
    assert continuous_rate(0.05) < 0.05


def test_inputs_are_validated():
    days = np.array([30.0, 60.0])
    with pytest.raises(ValueError, match="at least two"):
        bounded_spline_bey(days[:1], np.array([0.01]), 30.0)
    with pytest.raises(ValueError, match="positive"):
        bounded_spline_bey(days, np.array([0.01, 0.02]), 0.0)
    with pytest.raises(ValueError, match="at least two"):
        YieldCurve(date(2026, 1, 2), {"1 Mo": 4.0})
    with pytest.raises(ValueError, match="after the curve date"):
        YieldCurve(date(2026, 1, 2), {"1 Mo": 4.0, "3 Mo": 4.1}).rate(date(2026, 1, 2))


def test_missing_and_unknown_maturities_are_ignored():
    full = YieldCurve(date(2026, 10, 1), {"1 Mo": 4.06, "3 Mo": 4.17, "1 Yr": 4.44, "2 Yr": 4.78})
    extra = YieldCurve(
        date(2026, 10, 1),
        {"1 Mo": 4.06, "1.5 Month": 9.9, "3 Mo": 4.17, "4 Mo": 9.9, "1 Yr": 4.44, "2 Yr": 4.78},
    )
    assert extra.rate(date(2026, 10, 30)) == full.rate(date(2026, 10, 30))
    assert set(CMT_DAYS) >= {"1 Mo", "30 Yr"}
