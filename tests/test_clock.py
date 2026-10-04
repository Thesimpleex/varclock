from __future__ import annotations

from datetime import date, datetime, time

import numpy as np
import pytest

from varclock import tradingcal as tc
from varclock.clock import (
    ClockBootstrap,
    ClockData,
    SessionCalendar,
    bootstrap_clock,
    event_terms,
    fit_clock,
)
from varclock.events import Event
from varclock.rates import YieldCurve
from varclock.strip import build_strip
from varclock.synthetic import synthetic_daily_snapshot

ASOF = datetime(2026, 10, 2, 16, 15, tzinfo=tc.ET)
CURVE = YieldCurve(date(2026, 10, 1), {"1 Mo": 0.0, "3 Mo": 0.0, "1 Yr": 0.0})


def sessions(n=25, start=date(2026, 10, 5)):
    out = [start]
    while len(out) < n:
        out.append(tc.next_trading_day(out[-1]))
    return out


def event(day, kind, hh=8, mm=30, label=None):
    return Event(day, time(hh, mm), kind, label or f"{kind} {day}", "test")


def model_variance(days, base, weekend, bumps):
    cal = SessionCalendar.build(ASOF, days)
    v = base * np.ones(len(days)) + weekend * cal.weekend
    index = {d: i for i, d in enumerate(days)}
    for day, j in bumps.items():
        v[index[day]] += j
    return v


def test_session_calendar_flags_weekends_and_holidays():
    days = [date(2026, 11, 23), date(2026, 11, 24), date(2026, 11, 25), date(2026, 11, 27)]
    cal = SessionCalendar.build(datetime(2026, 11, 20, 16, 15, tzinfo=tc.ET), days)
    assert cal.dates[0] == date(2026, 11, 23)
    assert list(cal.weekend) == [1.0, 0.0, 0.0, 0.0]
    assert list(cal.holidays) == [0.0, 0.0, 0.0, 1.0]
    assert list(cal.expiry_index) == [0, 1, 2, 3]
    assert cal.dates[3] == date(2026, 11, 27)
    assert cal.n == 4


def test_session_calendar_holiday_monday_gives_weekend_and_holiday():
    asof = datetime(2026, 1, 16, 16, 15, tzinfo=tc.ET)
    cal = SessionCalendar.build(asof, [date(2026, 1, 20), date(2026, 1, 21), date(2026, 1, 22)])
    assert cal.dates[0] == date(2026, 1, 20)
    assert cal.weekend[0] == 1.0
    assert cal.holidays[0] == 1.0


def test_exact_recovery_of_a_known_clock_on_analytic_totals():
    days = sessions(25)
    base, weekend = 0.8e-4, 0.15e-4
    bumps = {days[7]: 0.6e-4, days[16]: 0.9e-4}
    v = model_variance(days, base, weekend, bumps)
    events = [event(days[7], "CPI"), event(days[16], "FOMC", 14, 0)]
    data = ClockData(ASOF, tuple(days), np.cumsum(v), np.full(25, 1e-9))
    fit = fit_clock(data, events, family="meanrev", kappa=0.05)
    np.testing.assert_allclose(fit.base, base, rtol=1e-6)
    assert fit.weekend == pytest.approx(weekend, rel=1e-6)
    np.testing.assert_allclose(fit.bumps, [0.6e-4, 0.9e-4], rtol=1e-6)
    np.testing.assert_allclose(fit.session_variance, v, rtol=1e-6)
    assert fit.event_multiple(0) == pytest.approx(1 + 0.6 / 0.8)
    assert fit.weekend_ratio == pytest.approx(0.15 / 0.8, rel=1e-6)


def test_flat_base_plus_gaussian_log_jump_is_recovered_through_the_whole_pipeline():
    """Black-76 daily chains with fixed variance weights and a log-jump of known variance."""
    days = sessions(22)
    base, weekend, jump = 0.9e-4, 0.2e-4, 1.1e-4  # jump variance delta^2 on one known date
    v = model_variance(days, base, weekend, {days[10]: jump})
    snap = synthetic_daily_snapshot(ASOF, days, v, forward=1000.0, spacing=0.25)
    strip = build_strip(snap, CURVE)
    assert len(strip.points) == 22
    assert not strip.violations.any()
    # the strike grid adds the same small discretisation offset to every total, none to increments
    np.testing.assert_allclose(strip.cumulative, np.cumsum(v), atol=2e-8)
    np.testing.assert_allclose(np.diff(strip.cumulative), v[1:], rtol=1e-5)
    data = ClockData.from_strip(strip)
    fit = fit_clock(data, [event(days[10], "CPI")], family="meanrev", kappa=0.002)
    assert fit.bumps[0] == pytest.approx(jump, rel=1e-4)
    assert fit.weekend == pytest.approx(weekend, rel=1e-3)
    np.testing.assert_allclose(fit.base[:10], base, rtol=1e-4)


def test_invariants_non_negative_and_daily_variances_add_up_to_each_total():
    days = sessions(25)
    rng = np.random.default_rng(5)
    v = model_variance(days, 1e-4, 0.1e-4, {days[5]: 0.5e-4}) * rng.uniform(0.99, 1.01, 25)
    data = ClockData(ASOF, tuple(days), np.cumsum(v), 0.01 * np.cumsum(v))
    for family in ("meanrev", "spline"):
        fit = fit_clock(data, [event(days[5], "CPI")], family=family)
        assert np.all(fit.base >= 0)
        assert np.all(fit.session_variance >= 0)
        assert fit.weekend >= 0
        assert np.all(fit.bumps >= 0)
        np.testing.assert_allclose(
            np.cumsum(fit.session_variance)[fit.calendar.expiry_index], fit.fitted_cumulative
        )
        assert np.all(np.isfinite(fit.standardized_residuals))


def test_spline_family_recovers_a_rising_piecewise_linear_base_exactly():
    days = sessions(25)
    s = np.arange(25)
    base = 0.5e-4 + 0.04e-4 * s
    v = base + 0.1e-4 * SessionCalendar.build(ASOF, days).weekend
    data = ClockData(ASOF, tuple(days), np.cumsum(v), np.full(25, 1e-9))
    fit = fit_clock(data, [], family="spline", knots=6, penalty=0.0)
    np.testing.assert_allclose(fit.base, base, rtol=1e-5)
    assert fit.weekend == pytest.approx(0.1e-4, rel=1e-4)


def test_unknown_family_and_bad_inputs_are_rejected():
    days = sessions(5)
    data = ClockData(ASOF, tuple(days), np.cumsum(np.full(5, 1e-4)), np.full(5, 1e-6))
    with pytest.raises(ValueError, match="family"):
        fit_clock(data, [], family="nope")
    with pytest.raises(ValueError, match="at least three"):
        ClockData(ASOF, tuple(days[:2]), np.array([1.0, 2.0]), np.array([0.1, 0.1]))
    with pytest.raises(ValueError, match="one entry"):
        ClockData(ASOF, tuple(days), np.ones(4), np.ones(5))
    with pytest.raises(ValueError, match="increasing"):
        ClockData(ASOF, tuple(reversed(days)), np.ones(5), np.ones(5))
    with pytest.raises(ValueError, match="non-trading"):
        SessionCalendar.build(ASOF, [date(2026, 10, 10), date(2026, 10, 12)])


def test_event_terms_group_earnings_and_skip_events_already_past():
    days = sessions(10)
    cal = SessionCalendar.build(ASOF, days)
    events = [
        event(days[2], "FOMC", 14, 0),
        event(days[3], "EARNINGS", 7, 0, "AAA earnings"),
        event(days[3], "EARNINGS", 7, 30, "BBB earnings"),
        event(date(2026, 12, 1), "CPI"),
    ]
    terms = event_terms(events, cal, ASOF)
    assert [t.kind for t in terms] == ["FOMC", "EARNINGS"]
    assert terms[1].loadings[3] == 2.0
    assert terms[1].label == "AAA, BBB"
    intraday = datetime(2026, 10, 5, 15, 0, tzinfo=tc.ET)
    cal_intraday = SessionCalendar.build(intraday, days)
    today = event(date(2026, 10, 5), "CPI")  # 08:30, before the snapshot
    assert event_terms([today], cal_intraday, intraday) == []


def test_partial_first_session_scales_the_base():
    intraday = datetime(2026, 10, 5, 12, 45, tzinfo=tc.ET)
    days = sessions(12)
    cal = SessionCalendar.build(intraday, days)
    assert cal.fraction[0] == pytest.approx(195 / 390)
    assert np.all(cal.fraction[1:] == 1.0)
    base = 1e-4
    v = base * cal.fraction
    data = ClockData(intraday, tuple(days), np.cumsum(v), np.full(12, 1e-9))
    fit = fit_clock(data, [], family="meanrev", kappa=0.05)
    np.testing.assert_allclose(fit.base, base, rtol=1e-6)
    assert fit.multiple[0] == pytest.approx(1.0)


def test_bootstrap_intervals_cover_the_truth_at_a_reasonable_rate():
    days = sessions(25)
    base, weekend, jump = 0.8e-4, 0.15e-4, 0.7e-4
    truth = model_variance(days, base, weekend, {days[9]: jump})
    events = [event(days[9], "CPI")]
    rng = np.random.default_rng(11)
    cover = 0
    trials = 20
    for _ in range(trials):
        total = np.cumsum(truth) * (1 + 0.004 * rng.standard_normal(25))
        data = ClockData(ASOF, tuple(days), total, 0.004 * total)
        fit = fit_clock(data, events, family="meanrev", kappa=0.05)
        boot = bootstrap_clock(
            fit, None, n_boot=60, rng=np.random.default_rng(int(rng.integers(1e9)))
        )
        lo, hi = boot.interval("event_multiple")
        cover += lo[0] <= 1 + jump / base <= hi[0]
    assert cover >= 13  # nominal 18 of 20; allow for the small number of replicates


def test_bootstrap_validates_arguments_and_is_reproducible_given_a_seed():
    days = sessions(12)
    v = model_variance(days, 1e-4, 0.1e-4, {})
    data = ClockData(ASOF, tuple(days), np.cumsum(v), 0.01 * np.cumsum(v))
    fit = fit_clock(data, [], family="meanrev", kappa=0.05)
    a = bootstrap_clock(fit, None, n_boot=30, rng=np.random.default_rng(1))
    b = bootstrap_clock(fit, None, n_boot=30, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(a.base, b.base)
    with pytest.raises(ValueError, match="level"):
        bootstrap_clock(fit, None, n_boot=30, level=1.5)
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_clock(fit, None, n_boot=5)
    lo, hi = a.vol_interval()
    assert np.all(lo <= hi)


def test_base_vol_is_the_annualised_square_root_of_the_base_variance():
    days = sessions(25)
    v = model_variance(days, 0.8e-4, 0.1e-4, {})
    data = ClockData(ASOF, tuple(days), np.cumsum(v), np.full(25, 1e-9))
    fit = fit_clock(data, [], family="meanrev", kappa=0.05)
    np.testing.assert_allclose(fit.base_vol, np.sqrt(fit.base * 252.0), rtol=1e-14)
    assert fit.base_vol[0] == pytest.approx(np.sqrt(0.8e-4 * 252.0), rel=1e-5)


def test_mean_reverting_base_starts_at_b0_and_follows_the_exponential_form():
    days = sessions(25)
    s = np.arange(25)
    kappa, b0, b_inf = 0.05, 0.5e-4, 1.5e-4
    base = b0 * np.exp(-kappa * s) + b_inf * (1 - np.exp(-kappa * s))
    v = base + 0.1e-4 * SessionCalendar.build(ASOF, days).weekend
    data = ClockData(ASOF, tuple(days), np.cumsum(v), np.full(25, 1e-9))
    fit = fit_clock(data, [], family="meanrev", kappa=kappa)
    assert fit.base[0] == pytest.approx(b0, rel=1e-5)
    assert fit.theta[0] == pytest.approx(b0, rel=1e-5)
    np.testing.assert_allclose(fit.base, base, rtol=1e-5)


def test_bootstrap_interval_uses_the_central_quantiles_of_the_stated_level():
    draws = np.arange(101.0).reshape(-1, 1)
    boot = ClockBootstrap(0.9, draws, draws, draws, draws[:, 0], draws, draws)
    lo, hi = boot.interval("base")
    assert (lo[0], hi[0]) == pytest.approx((5.0, 95.0))
    boot80 = ClockBootstrap(0.8, draws, draws, draws, draws[:, 0], draws, draws)
    lo, hi = boot80.interval("base")
    assert (lo[0], hi[0]) == pytest.approx((10.0, 90.0))


def test_event_on_a_monday_carries_the_weekend_term_in_its_multiple():
    days = sessions(20)
    assert days[0].weekday() == 0
    base, weekend, jump = 0.8e-4, 0.2e-4, 0.5e-4
    v = model_variance(days, base, weekend, {days[0]: jump})
    data = ClockData(ASOF, tuple(days), np.cumsum(v), np.full(20, 1e-9))
    fit = fit_clock(data, [event(days[0], "CPI")], family="meanrev", kappa=0.05)
    assert fit.event_multiple(0) == pytest.approx(1 + (weekend + jump) / base, rel=1e-5)
    assert fit.event_variance(0) == pytest.approx(base + weekend + jump, rel=1e-5)


def test_event_after_a_holiday_carries_the_holiday_term():
    asof = datetime(2026, 11, 20, 16, 15, tzinfo=tc.ET)
    days = []
    d = date(2026, 11, 23)
    while len(days) < 12:
        days.append(d)
        d = tc.next_trading_day(d)
    cal = SessionCalendar.build(asof, days)
    base, holiday, jump = 1e-4, 0.3e-4, 0.4e-4
    after = days.index(date(2026, 11, 27))
    v = base + 0.2e-4 * cal.weekend + holiday * cal.holidays
    v[after] += jump
    data = ClockData(asof, tuple(days), np.cumsum(v), np.full(12, 1e-9))
    fit = fit_clock(data, [event(days[after], "CPI")], family="meanrev", kappa=0.05)
    # holiday and event share one session, so only their sum is identified
    assert fit.event_multiple(0) == pytest.approx(1 + (holiday + jump) / base, rel=1e-4)


def test_per_name_earnings_multiple_does_not_depend_on_how_many_names_report():
    days = sessions(15)
    base, jump = 1e-4, 0.3e-4
    names = {days[4]: 3.0, days[9]: 1.0}
    v = model_variance(days, base, 0.0, {})
    for day, n in names.items():
        v[days.index(day)] += n * jump
    data = ClockData(ASOF, tuple(days), np.cumsum(v), np.full(15, 1e-9))
    events = [event(days[4], "EARNINGS", 7, 0, f"N{i} earnings") for i in range(3)]
    events.append(event(days[9], "EARNINGS", 7, 0, "N9 earnings"))
    fit = fit_clock(data, events, family="meanrev", kappa=0.05)
    (term,) = fit.terms
    assert term.sessions[0] == 4
    assert fit.bumps[0] == pytest.approx(jump, rel=1e-5)
    assert fit.event_multiple(0) == pytest.approx(1 + jump / base, rel=1e-5)
    assert fit.event_variance(0) == pytest.approx(base + jump, rel=1e-5)
