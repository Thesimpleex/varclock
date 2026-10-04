from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pytest

from varclock import tradingcal as tc
from varclock.quotes import OptionStrip, Snapshot
from varclock.rates import YieldCurve
from varclock.strip import build_strip, window_total_variance
from varclock.synthetic import synthetic_daily_snapshot

ASOF = datetime(2026, 10, 2, 16, 15, tzinfo=tc.ET)
CURVE = YieldCurve(date(2026, 10, 1), {"1 Mo": 0.0, "3 Mo": 0.0, "1 Yr": 0.0})


def days(n):
    out = [tc.next_trading_day(ASOF.date())]
    while len(out) < n:
        out.append(tc.next_trading_day(out[-1]))
    return out


def variances(n, base=1e-4, ramp=0.0):
    return base + ramp * np.arange(n)


@pytest.fixture(scope="module")
def snapshot():
    d = days(12)
    return synthetic_daily_snapshot(ASOF, d, variances(12, ramp=0.05e-4), spacing=0.25)


def replace_strips(snap, strips):
    return Snapshot(snap.symbol, snap.source, snap.asof, snap.fetched_at, snap.spot, tuple(strips))


def test_forward_variances_are_recovered_from_daily_strips(snapshot):
    strip = build_strip(snapshot, CURVE)
    assert strip.expiries == days(12)
    np.testing.assert_allclose(strip.increments[1:], variances(12, ramp=0.05e-4)[1:], rtol=1e-5)
    assert strip.cumulative[-1] == pytest.approx(variances(12, ramp=0.05e-4).sum(), abs=2e-8)
    np.testing.assert_allclose(strip.forward_variance, strip.increments / strip.delta_T)
    assert strip.delta_T[0] == pytest.approx(strip.points[0].T)
    assert not strip.violations.any()
    assert not strip.significant_violations.any()


def test_daily_variances_add_up_to_each_expirys_total(snapshot):
    strip = build_strip(snapshot, CURVE)
    for point, total in zip(strip.points, strip.cumulative, strict=True):
        assert point.total_variance == pytest.approx(total, rel=1e-4)
    np.testing.assert_allclose(np.cumsum(strip.increments), strip.cumulative)


def test_calendar_arbitrage_is_flagged():
    d = days(8)
    v = variances(8)
    totals = np.cumsum(v)
    totals[4] = totals[3] - 0.2e-4  # a later expiry priced below an earlier one
    snap = synthetic_daily_snapshot(ASOF, d, np.diff(np.concatenate([[0.0], totals])), spacing=0.25)
    strip = build_strip(snap, CURVE)
    assert strip.violations[4]
    assert strip.significant_violations[4]
    assert strip.forward_variance[4] < 0


def test_common_coverage_makes_increments_independent_of_wing_coverage(snapshot):
    """Extending the wider expiry's strikes beyond the common window must not move the increment."""
    base_strips = list(snapshot.strips)
    j = 5

    def truncate(strip, lo, hi):
        m = (strip.strikes >= lo) & (strip.strikes <= hi)
        return OptionStrip(
            strip.root, strip.expiry, strip.strikes[m], strip.call_bid[m], strip.call_ask[m],
            strip.put_bid[m], strip.put_ask[m],
        )  # fmt: skip

    narrow = base_strips[j]
    lo, hi = 960.0, 1040.0
    cut_j = truncate(narrow, lo, hi)
    cut_next = truncate(base_strips[j + 1], lo, hi)
    only_j_narrow = [*base_strips[:j], cut_j, *base_strips[j + 1 :]]
    both_narrow = [*base_strips[:j], cut_j, cut_next, *base_strips[j + 2 :]]
    a = build_strip(replace_strips(snapshot, only_j_narrow), CURVE, coverage="common")
    b = build_strip(replace_strips(snapshot, both_narrow), CURVE, coverage="common")
    assert a.increments[j + 1] == pytest.approx(b.increments[j + 1], abs=1e-15)
    naive_a = a.points[j + 1].total_variance - a.points[j].total_variance
    naive_b = b.points[j + 1].total_variance - b.points[j].total_variance
    assert abs(naive_a - naive_b) > 1e-7  # the plain difference of totals does depend on coverage


def test_window_total_variance_matches_own_range_for_the_full_window(snapshot):
    strip = build_strip(snapshot, CURVE)
    term = strip.points[3].term
    assert window_total_variance(term, term.strikes[0], term.strikes[-1]) == pytest.approx(
        term.total_variance
    )
    assert window_total_variance(term, 995.0, 1005.0) < term.total_variance


def test_quote_bands_scale_with_spreads_and_resampling_stays_inside_them():
    d = days(10)
    tight = build_strip(synthetic_daily_snapshot(ASOF, d, variances(10), spacing=0.5), CURVE)
    wide = build_strip(
        synthetic_daily_snapshot(ASOF, d, variances(10), spacing=0.5, half_spread=0.05), CURVE
    )
    assert np.all(wide.cumulative_band > tight.cumulative_band)
    rng = np.random.default_rng(0)
    draws = np.array([wide.resampled_cumulative(rng) for _ in range(300)])
    assert np.all(np.abs(draws.mean(axis=0) - wide.cumulative) < 3 * wide.cumulative_band)
    assert np.all(draws.std(axis=0) < wide.cumulative_band)
    assert np.all(draws.std(axis=0) > 0)


def test_unusable_expiries_are_skipped_and_gaps_end_the_run(snapshot):
    strips = list(snapshot.strips)
    bad = strips[3]
    nan = np.full(bad.strikes.size, np.nan)
    broken = OptionStrip(bad.root, bad.expiry, bad.strikes, nan, nan, nan, nan)
    strip = build_strip(replace_strips(snapshot, [*strips[:3], broken, *strips[4:]]), CURVE)
    assert [d for d, _ in strip.skipped] == [bad.expiry]
    assert len(strip.points) == 11
    gap = build_strip(replace_strips(snapshot, strips[:6] + strips[7:]), CURVE)
    assert len(gap.points) == 6  # the run stops at the first missing trading day
    with pytest.raises(ValueError, match="at least three"):
        build_strip(replace_strips(snapshot, strips[:2]), CURVE)
    assert len(build_strip(snapshot, CURVE, max_expiries=5).points) == 5


def test_strip_starts_with_todays_expiry_during_the_session():
    asof = datetime(2026, 10, 5, 11, 0, tzinfo=tc.ET)
    d = days(10)
    assert d[0] == date(2026, 10, 5)
    snap = synthetic_daily_snapshot(asof, d, variances(len(d)), spacing=0.25)
    strip = build_strip(snap, CURVE)
    assert strip.expiries[0] == date(2026, 10, 5)
    assert strip.points[0].minutes == 5 * 60


def test_own_coverage_increments_are_differences_of_own_totals(snapshot):
    strip = build_strip(snapshot, CURVE)
    own = np.array([p.total_variance for p in strip.points])
    assert strip.coverage == "own"
    np.testing.assert_allclose(strip.cumulative, own, rtol=1e-12)
    np.testing.assert_allclose(strip.increments, np.diff(np.concatenate([[0.0], own])), atol=1e-15)
    with pytest.raises(ValueError, match="coverage"):
        build_strip(snapshot, CURVE, coverage="union")


def _truncated_truth(tick):
    from varclock.clock import ClockData, SessionCalendar, fit_clock

    days_ = days(30)
    cal = SessionCalendar.build(ASOF, days_)
    base = 0.6e-4 + 0.03e-4 * np.arange(30)
    weekend = 0.1e-4
    v = base + weekend * cal.weekend
    snap = synthetic_daily_snapshot(ASOF, days_, v, spacing=1.0, tick=tick)
    truth = np.cumsum(v)
    truth_ratio = weekend / base[cal.weekend > 0].mean()
    out = {}
    for coverage in ("own", "common"):
        strip = build_strip(snap, CURVE, coverage=coverage)
        fit = fit_clock(ClockData.from_strip(strip), ())
        out[coverage] = (np.max(np.abs(strip.cumulative / truth - 1.0)), fit.weekend_ratio)
    return out, truth_ratio


def test_truncated_chains_own_coverage_recovers_totals_and_weekend_better_than_common():
    """Zero-bid truncation with a known truth: the common window biases the weekend upward."""
    for tick in (0.05, 0.1):
        out, truth_ratio = _truncated_truth(tick)
        own_err, own_ratio = out["own"]
        common_err, common_ratio = out["common"]
        assert own_err < 0.02
        assert common_err > 2 * own_err
        assert abs(own_ratio / truth_ratio - 1.0) < 0.08
        assert common_ratio > 1.25 * truth_ratio


def test_window_weights_match_a_hand_computed_band_on_a_nonuniform_grid():
    from varclock.strip import _band

    d = days(4)
    snap = synthetic_daily_snapshot(ASOF, d, variances(4), spacing=0.5, half_spread=0.04)
    term = build_strip(snap, CURVE).points[1].term
    mask = np.ones(term.strikes.size, dtype=bool)
    mask[::3] = False
    mask[:2] = mask[-2:] = True
    k = term.strikes[mask]
    dk = np.empty_like(k)
    dk[0], dk[-1] = k[1] - k[0], k[-1] - k[-2]
    dk[1:-1] = (k[2:] - k[:-2]) / 2.0
    weights = 2.0 * dk / k**2 * np.exp(term.rate * term.T)
    total, half = _band(term, mask)
    expected_total = weights @ term.prices[mask] - (term.forward / term.k0 - 1.0) ** 2
    assert total == pytest.approx(expected_total, rel=1e-12)
    spread = (term.price_hi - term.price_lo)[mask]
    dforward = np.exp(term.rate * term.T) * (
        np.diff(term.atm_call_band)[0] / 2 + np.diff(term.atm_put_band)[0] / 2
    )
    fwd_term = abs(2.0 * (term.forward / term.k0 - 1.0) / term.k0) * dforward
    assert half == pytest.approx(weights @ spread / 2.0 + fwd_term, rel=1e-12)


def test_significant_violations_require_the_band_to_leave_the_increment_negative():
    d = days(8)
    totals = np.cumsum(variances(8))
    totals[4] = totals[3] - 0.2e-4
    snap = synthetic_daily_snapshot(
        ASOF, d, np.diff(np.concatenate([[0.0], totals])), spacing=0.25, half_spread=0.02
    )
    strip = build_strip(snap, CURVE)
    assert strip.violations[4]
    expected = strip.increments + strip.band < 0
    np.testing.assert_array_equal(strip.significant_violations, expected)
    assert strip.significant_violations[4] == (strip.increments[4] + strip.band[4] < 0)
    wide = build_strip(
        synthetic_daily_snapshot(
            ASOF, d, np.diff(np.concatenate([[0.0], totals])), spacing=0.25, half_spread=0.6
        ),
        CURVE,
    )
    assert wide.violations[4]
    assert not wide.significant_violations[4]


class _Corner:
    """Generator stand-in whose normals are +-inf, so every price sits at an edge of its band."""

    def __init__(self, sign):
        self.sign = sign

    def standard_normal(self, size=None):
        return np.full(size, self.sign * np.inf) if size is not None else self.sign * np.inf


def test_resampler_puts_every_price_at_the_bid_or_the_ask_in_the_degenerate_case():
    d = days(8)
    snap = synthetic_daily_snapshot(ASOF, d, variances(8), spacing=0.5, half_spread=0.02)
    strip = build_strip(snap, CURVE)
    for sign, pick in ((-1.0, "lo"), (1.0, "hi")):
        draw = strip.resampled_cumulative(_Corner(sign))
        inc = []
        for p in strip.points:
            t = p.term
            q = t.price_lo if pick == "lo" else t.price_hi
            c = t.atm_call_band[0 if pick == "lo" else 1]
            pu = t.atm_put_band[0 if pick == "lo" else 1]
            fwd = t.atm_strike + np.exp(t.rate * t.T) * (c - pu)
            inc.append(t.weights @ q - (fwd / t.k0 - 1.0) ** 2)
        np.testing.assert_allclose(draw, inc, rtol=1e-12)
    low = strip.resampled_cumulative(_Corner(-1.0))
    high = strip.resampled_cumulative(_Corner(1.0))
    assert np.all(high > low)


def test_quote_correlation_makes_errors_move_together_across_strikes():
    d = days(10)
    wide = build_strip(
        synthetic_daily_snapshot(ASOF, d, variances(10), spacing=0.5, half_spread=0.05), CURVE
    )
    spreads = {}
    for rho in (0.0, 0.5, 0.9):
        rng = np.random.default_rng(0)
        draws = np.array([wide.resampled_cumulative(rng, rho) for _ in range(400)])
        spreads[rho] = draws.std(axis=0)
        assert np.all(draws.std(axis=0) < wide.cumulative_band)
    assert np.all(spreads[0.5] > 2 * spreads[0.0])
    assert np.all(spreads[0.9] > spreads[0.5])
    with pytest.raises(ValueError, match="quote_correlation"):
        wide.resampled_cumulative(np.random.default_rng(0), 1.0)
