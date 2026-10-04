from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from varclock.bounds import increment_band, variance_band
from varclock.quotes import OptionStrip
from varclock.synthetic import black76_prices, synthetic_strip
from varclock.variance import single_term_variance

EXPIRY = date(2030, 1, 2)
T = 0.1


def strip_with_spread(half_spread, tick=0.0, total=0.004, forward=100.0, rate=0.0):
    k = np.arange(60.0, 160.0 + 0.5, 0.5)
    call, put = black76_prices(forward, k, total, np.exp(-rate * T))
    return synthetic_strip("SPXW", EXPIRY, k, call, put, half_spread=half_spread, tick=tick)


def test_bands_contain_the_mid_estimate_and_are_inside_the_naive_box():
    strip = strip_with_spread(0.01)
    tv = single_term_variance(strip, T, 0.0)
    band = variance_band(tv, strip, 1.0)
    assert band.feasible
    assert band.widen == 0.0
    assert band.lo < band.mid < band.hi
    assert band.naive_lo - 1e-12 <= band.lo
    assert band.hi <= band.naive_hi + 1e-12
    assert band.width <= band.naive_width + 1e-12


def noisy_strip(half_spread, noise, seed=0):
    rng = np.random.default_rng(seed)
    k = np.arange(60.0, 160.0 + 0.5, 0.5)
    call, put = black76_prices(100.0, k, 0.004)
    call = call + noise * rng.uniform(-1, 1, k.size)
    put = put + noise * rng.uniform(-1, 1, k.size)
    return synthetic_strip("SPXW", EXPIRY, k, call, put, half_spread=half_spread)


def test_no_arbitrage_constraints_tighten_the_band_when_quotes_are_noisy():
    strip = noisy_strip(0.02, 0.012)
    band = variance_band(single_term_variance(strip, T, 0.0), strip, 1.0)
    assert band.feasible
    assert band.width < 0.97 * band.naive_width


def test_zero_spread_collapses_the_band_to_the_estimate():
    strip = strip_with_spread(0.0)
    tv = single_term_variance(strip, T, 0.0)
    band = variance_band(tv, strip, 1.0)
    assert band.lo == pytest.approx(tv.total_variance, abs=2e-8)
    assert band.hi == pytest.approx(tv.total_variance, abs=2e-8)


def test_band_width_grows_with_the_spread():
    widths = []
    for h in (0.005, 0.01, 0.02):
        strip = strip_with_spread(h)
        band = variance_band(single_term_variance(strip, T, 0.0), strip, 1.0)
        widths.append(band.width)
    assert widths[0] < widths[1] < widths[2]
    assert widths[2] / widths[0] == pytest.approx(4.0, rel=0.35)


def test_discount_factor_is_used():
    strip = strip_with_spread(0.01, rate=0.05)
    tv = single_term_variance(strip, T, 0.05)
    band = variance_band(tv, strip, float(np.exp(-0.05 * T)))
    assert band.lo < tv.total_variance < band.hi


def test_quotes_that_admit_arbitrage_are_widened_minimally():
    strip = strip_with_spread(0.005)
    i = int(np.searchsorted(strip.strikes, 85.0))
    put_bid, put_ask = strip.put_bid.copy(), strip.put_ask.copy()
    put_bid[i] += 0.2  # one put much dearer than its higher-strike neighbours
    put_ask[i] += 0.2
    bad = OptionStrip(
        "SPXW", EXPIRY, strip.strikes, strip.call_bid, strip.call_ask, put_bid, put_ask
    )
    tv = single_term_variance(bad, T, 0.0)
    band = variance_band(tv, bad, 1.0)
    assert band.widen > 0.05
    assert band.widen < 0.25
    assert band.feasible
    clean = variance_band(single_term_variance(strip, T, 0.0), strip, 1.0)
    assert clean.widen == 0.0


def test_increment_band_difference_of_independent_bounds():
    s1 = strip_with_spread(0.01, total=0.004)
    s2 = strip_with_spread(0.01, total=0.006)
    b1 = variance_band(single_term_variance(s1, T, 0.0), s1, 1.0)
    b2 = variance_band(single_term_variance(s2, T, 0.0), s2, 1.0)
    lo, hi = increment_band(b1, b2)
    assert lo < 0.002 < hi
    assert lo == pytest.approx(b2.lo - b1.hi)


def test_window_restriction_only_changes_the_objective():
    strip = strip_with_spread(0.01)
    tv = single_term_variance(strip, T, 0.0)
    full = variance_band(tv, strip, 1.0)
    mask = (tv.strikes >= 85.0) & (tv.strikes <= 115.0)
    part = variance_band(tv, strip, 1.0, mask=mask)
    assert part.mid < full.mid
    assert part.lo < part.hi


def test_smooth_chain_band_equals_the_naive_all_bid_to_all_ask_box():
    """Shifting every price by the half-spread keeps a smooth curve convex and decreasing, so the
    arbitrage-free optimum is attained at the all-bid and all-ask corners."""
    for h in (0.005, 0.01):
        strip = strip_with_spread(h)
        band = variance_band(single_term_variance(strip, T, 0.0), strip, 1.0)
        assert band.lo == pytest.approx(band.naive_lo, abs=1e-12)
        assert band.hi == pytest.approx(band.naive_hi, abs=1e-12)


def test_naive_box_formula_is_the_cboe_sum_at_bids_and_asks():
    strip = strip_with_spread(0.02, rate=0.03)
    tv = single_term_variance(strip, T, 0.03)
    band = variance_band(tv, strip, float(np.exp(-0.03 * T)))
    k = tv.strikes
    dk = np.empty_like(k)
    dk[0], dk[-1] = k[1] - k[0], k[-1] - k[-2]
    dk[1:-1] = (k[2:] - k[:-2]) / 2.0
    w = 2.0 * dk / k**2 * np.exp(0.03 * T)
    penalty = (tv.forward / tv.k0 - 1.0) ** 2
    assert band.naive_lo == pytest.approx(w @ tv.price_lo - penalty, rel=1e-12)
    assert band.naive_hi == pytest.approx(w @ tv.price_hi - penalty, rel=1e-12)
    assert band.mid == pytest.approx(w @ tv.prices - penalty, rel=1e-12)
    assert band.naive_width == pytest.approx(band.naive_hi - band.naive_lo)


def test_a_three_strike_chain_has_a_hand_computed_corner_optimum():
    """Strikes 90, 100, 110 with half-spread h around a convex, decreasing call curve."""
    forward, h = 100.0, 0.01
    k = np.array([90.0, 100.0, 110.0])
    call = np.array([11.0, 4.0, 1.0])
    put = call - (forward - k)
    strip = synthetic_strip("SPXW", EXPIRY, k, call, put, half_spread=h)
    tv = single_term_variance(strip, T, 0.0)
    band = variance_band(tv, strip, 1.0)
    dk = np.array([10.0, 10.0, 10.0])
    w = 2.0 * dk / k**2
    q_mid = np.array([put[0], (call[1] + put[1]) / 2.0, call[2]])
    assert tv.total_variance == pytest.approx(w @ q_mid)
    assert band.hi == pytest.approx(w @ (q_mid + h), abs=1e-12)
    assert band.lo == pytest.approx(w @ (q_mid - h), abs=1e-12)
