from __future__ import annotations

from datetime import date
from itertools import pairwise

import numpy as np
import pytest

from varclock.quotes import OptionStrip
from varclock.synthetic import (
    black76_prices,
    heston_expected_variance,
    heston_prices,
    synthetic_strip,
)
from varclock.variance import VarianceError, interpolate_variance, single_term_variance

EXPIRY = date(2030, 1, 2)


def black_strip(total_var=0.004, forward=100.0, spacing=0.5, lo=0.4, hi=2.0, rate=0.0, T=0.1):
    k = np.arange(round(forward * lo), round(forward * hi) + spacing, spacing)
    call, put = black76_prices(forward, k, total_var, np.exp(-rate * T))
    return synthetic_strip("SPXW", EXPIRY, k, call, put), T, rate


def test_flat_smile_gives_sigma_squared_times_t():
    strip, T, rate = black_strip(total_var=0.004, rate=0.03, spacing=0.25)
    tv = single_term_variance(strip, T, rate)
    assert tv.total_variance == pytest.approx(0.004, rel=5e-4)
    assert tv.sigma2 == pytest.approx(0.004 / T, rel=5e-4)
    assert tv.forward == pytest.approx(100.0, abs=1e-6)
    assert tv.volatility == pytest.approx(np.sqrt(0.04), rel=3e-4)


def test_black_scholes_error_vanishes_with_spacing_and_range():
    spacing_errors = []
    for h in (4.0, 2.0, 1.0, 0.5):
        strip, T, r = black_strip(spacing=h, lo=0.2, hi=3.0)
        spacing_errors.append(abs(single_term_variance(strip, T, r).total_variance - 0.004))
    assert all(a > b for a, b in pairwise(spacing_errors))
    assert spacing_errors[0] / spacing_errors[-1] > 30  # second-order convergence

    wide, T, r = black_strip(spacing=0.25, lo=0.2, hi=3.0)
    reference = single_term_variance(wide, T, r).total_variance
    range_errors = []
    for width in (0.1, 0.2, 0.3, 0.4):
        strip, T, r = black_strip(spacing=0.25, lo=1 - width, hi=1 + width)
        range_errors.append(abs(single_term_variance(strip, T, r).total_variance - reference))
    assert all(a > b for a, b in pairwise(range_errors))
    assert range_errors[-1] < 1e-8


def test_heston_replicated_variance_converges_to_expected_integrated_variance():
    pars = {"v0": 0.04, "kappa": 2.0, "theta": 0.06, "xi": 0.5, "rho": -0.7}
    T = 30 / 365
    target = heston_expected_variance(T, pars["v0"], pars["kappa"], pars["theta"])
    errors = []
    for h in (2.0, 1.0, 0.5, 0.25):
        k = np.arange(20.0, 300.0 + h, h)
        call, put = heston_prices(100.0, k, T, **pars)
        strip = synthetic_strip("SPXW", EXPIRY, k, call, put)
        errors.append(single_term_variance(strip, T, 0.0).total_variance - target)
    assert all(e > 0 for e in errors)
    assert errors[-1] == pytest.approx(0, abs=2e-6)
    for coarse, fine in pairwise(errors):
        assert coarse / fine == pytest.approx(4.0, rel=0.05)


def test_heston_prices_satisfy_put_call_parity_and_martingale():
    k = np.array([80.0, 100.0, 120.0])
    call, put = heston_prices(100.0, k, 0.25, v0=0.04, kappa=1.5, theta=0.05, xi=0.4, rho=-0.5)
    np.testing.assert_allclose(call - put, 100.0 - k, atol=1e-9)
    assert np.all(call >= np.maximum(100.0 - k, 0) - 1e-9)


def test_adding_zero_bid_strikes_beyond_the_wings_changes_nothing():
    strip, T, rate = black_strip()
    base = single_term_variance(strip, T, rate)
    lo_extra = np.arange(strip.strikes[0] - 5.0, strip.strikes[0] - 0.1, 0.5)
    hi_extra = np.arange(strip.strikes[-1] + 0.5, strip.strikes[-1] + 5.0, 0.5)
    k = np.concatenate([lo_extra, strip.strikes, hi_extra])
    pad = np.zeros(lo_extra.size)
    padh = np.zeros(hi_extra.size)
    padded = OptionStrip(
        "SPXW",
        EXPIRY,
        k,
        np.concatenate([pad + 90.0, strip.call_bid, padh]),
        np.concatenate([pad + 90.1, strip.call_ask, padh + 0.05]),
        np.concatenate([pad, strip.put_bid, padh + 90.0]),
        np.concatenate([pad + 0.05, strip.put_ask, padh + 90.1]),
    )
    assert single_term_variance(padded, T, rate).total_variance == pytest.approx(
        base.total_variance, abs=1e-15
    )


def test_an_isolated_zero_bid_strike_inside_the_range_is_skipped_without_effect():
    strip, T, rate = black_strip(spacing=1.0)
    base = single_term_variance(strip, T, rate)
    i = int(np.searchsorted(strip.strikes, 70.0))
    put_bid = strip.put_bid.copy()
    put_bid[i] = 0.0
    zeroed = OptionStrip(
        "SPXW", EXPIRY, strip.strikes, strip.call_bid, strip.call_ask, put_bid, strip.put_ask
    )
    tv = single_term_variance(zeroed, T, rate)
    assert 70.0 not in tv.strikes
    assert tv.n_options == base.n_options - 1
    # dropping a strike widens its neighbours' delta-K, so only a tiny change remains
    assert tv.total_variance == pytest.approx(base.total_variance, rel=1e-4)


def test_two_consecutive_zero_bids_end_the_wing():
    strip, T, rate = black_strip(spacing=1.0)
    put_bid = strip.put_bid.copy()
    i = int(np.searchsorted(strip.strikes, 70.0))
    put_bid[i : i + 2] = 0.0
    cut = OptionStrip(
        "SPXW", EXPIRY, strip.strikes, strip.call_bid, strip.call_ask, put_bid, strip.put_ask
    )
    tv = single_term_variance(cut, T, rate)
    assert tv.strikes[0] > 71.0


def test_zero_ask_is_treated_like_zero_bid():
    strip, T, rate = black_strip(spacing=1.0)
    call_ask = strip.call_ask.copy()
    i = int(np.searchsorted(strip.strikes, 130.0))
    call_ask[i] = 0.0
    call_bid = strip.call_bid.copy()
    call_bid[i] = 0.0
    q = OptionStrip("SPXW", EXPIRY, strip.strikes, call_bid, call_ask, strip.put_bid, strip.put_ask)
    assert 130.0 not in single_term_variance(q, T, rate).strikes


def test_variance_is_linear_in_prices_given_the_selection():
    strip, T, rate = black_strip(spacing=1.0)
    tv = single_term_variance(strip, T, rate)
    rng = np.random.default_rng(3)
    bump = rng.uniform(0.9, 1.1, size=tv.prices.size)
    lhs = tv.total_variance_with(tv.prices * bump, tv.forward)
    assert lhs == pytest.approx(tv.weights @ (tv.prices * bump) - (tv.forward / tv.k0 - 1) ** 2)
    assert tv.total_variance_with(tv.prices, tv.forward) == pytest.approx(tv.total_variance)


def test_forward_and_k0_follow_the_forward_strike_rule():
    strip, T, rate = black_strip(forward=103.3, spacing=1.0, rate=0.02)
    tv = single_term_variance(strip, T, rate)
    assert tv.forward == pytest.approx(103.3, abs=1e-6)
    assert tv.k0 == 103.0
    assert tv.atm_strike in (103.0, 104.0)
    assert tv.kinds[list(tv.strikes).index(103.0)] == "B"
    assert tv.strikes.size + 1 == tv.n_options


def test_ties_for_the_forward_strike_pick_the_lowest_strike():
    strip, T, rate = black_strip(spacing=1.0)
    i99 = int(np.searchsorted(strip.strikes, 99.0))
    put_bid, put_ask = strip.put_bid.copy(), strip.put_ask.copy()
    put_bid[i99], put_ask[i99] = strip.call_bid[i99], strip.call_ask[i99]  # |call - put| = 0
    tied = OptionStrip(
        "SPXW", EXPIRY, strip.strikes, strip.call_bid, strip.call_ask, put_bid, put_ask
    )
    assert single_term_variance(strip, T, rate).atm_strike == 100.0
    assert single_term_variance(tied, T, rate).atm_strike == 99.0


def test_errors_where_cboe_says_the_index_cannot_be_calculated():
    strip, T, rate = black_strip(spacing=1.0)
    with pytest.raises(ValueError, match="positive"):
        single_term_variance(strip, 0.0, rate)
    nan = np.full(strip.strikes.size, np.nan)
    empty = OptionStrip("SPXW", EXPIRY, strip.strikes, nan, nan, nan, nan)
    with pytest.raises(VarianceError, match="two-sided"):
        single_term_variance(empty, T, rate)
    # no usable out-of-the-money puts: all put quotes below K0 removed
    put_bid, put_ask = strip.put_bid.copy(), strip.put_ask.copy()
    put_bid[strip.strikes < 100] = np.nan
    put_ask[strip.strikes < 100] = np.nan
    no_puts = OptionStrip(
        "SPXW", EXPIRY, strip.strikes, strip.call_bid, strip.call_ask, put_bid, put_ask
    )
    with pytest.raises(VarianceError, match="put"):
        single_term_variance(no_puts, T, rate)
    call_bid, call_ask = strip.call_bid.copy(), strip.call_ask.copy()
    call_bid[strip.strikes > 100] = np.nan
    call_ask[strip.strikes > 100] = np.nan
    no_calls = OptionStrip(
        "SPXW", EXPIRY, strip.strikes, call_bid, call_ask, strip.put_bid, strip.put_ask
    )
    with pytest.raises(VarianceError, match="call"):
        single_term_variance(no_calls, T, rate)


def test_k0_without_valid_quotes_raises():
    strip, T, rate = black_strip(spacing=1.0)
    i = int(np.searchsorted(strip.strikes, 100.0))
    call_bid = strip.call_bid.copy()
    call_bid[i] = np.nan
    broken = OptionStrip(
        "SPXW", EXPIRY, strip.strikes, call_bid, strip.call_ask, strip.put_bid, strip.put_ask
    )
    with pytest.raises(VarianceError, match="K0"):
        single_term_variance(broken, T, rate)


def test_crossed_quotes_are_not_used():
    strip, T, rate = black_strip(spacing=1.0)
    i = int(np.searchsorted(strip.strikes, 120.0))
    call_bid = strip.call_bid.copy()
    call_bid[i] = call_bid[i] + 5.0  # bid above ask
    crossed = OptionStrip(
        "SPXW", EXPIRY, strip.strikes, call_bid, strip.call_ask, strip.put_bid, strip.put_ask
    )
    assert 120.0 not in single_term_variance(crossed, T, rate).strikes


@pytest.mark.parametrize("seed", range(5))
def test_random_smiles_give_consistent_totals(seed):
    """Mixture of lognormals: replicated variance equals the mixture of total variances."""
    rng = np.random.default_rng(seed)
    forward = float(rng.uniform(50, 5000))
    weights = rng.dirichlet(np.ones(3))
    variances = rng.uniform(0.001, 0.02, size=3)
    # mixture of lognormals with a common forward; E[-2 ln(S/F)] = sum_i w_i * (variance_i)
    k = np.linspace(forward * 0.2, forward * 3.0, 6000)
    call = np.zeros_like(k)
    put = np.zeros_like(k)
    for w, v in zip(weights, variances, strict=True):
        c, p = black76_prices(forward, k, v)
        call += w * c
        put += w * p
    strip = synthetic_strip("SPXW", EXPIRY, k, call, put)
    tv = single_term_variance(strip, 0.05, 0.0)
    assert tv.total_variance == pytest.approx(float(weights @ variances), rel=5e-4)


def test_interpolation_formula_reproduces_flat_variance_and_weights():
    year = 525_600.0
    flat = interpolate_variance(0.04, 0.04, 30_000, 50_000, 43_200, year)
    assert flat == pytest.approx(20.0)
    # target exactly at the near term returns the near-term level
    at_near = interpolate_variance(0.04, 0.09, 43_200, 60_000, 43_200, year)
    assert at_near == pytest.approx(20.0)
    at_next = interpolate_variance(0.04, 0.09, 30_000, 43_200, 43_200, year)
    assert at_next == pytest.approx(30.0)
    assert interpolate_variance(None, 0.04, None, 50_000, 43_200, year) == pytest.approx(20.0)
    with pytest.raises(ValueError, match="after the near term"):
        interpolate_variance(0.04, 0.04, 50_000, 50_000, 43_200, year)


def test_non_finite_time_and_rate_are_rejected_with_a_clear_message():
    strip, T, rate = black_strip(spacing=1.0)
    for bad_T in (float("nan"), float("inf"), -0.1):
        with pytest.raises(ValueError, match="positive finite"):
            single_term_variance(strip, bad_T, rate)
    for bad_rate in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="rate must be finite"):
            single_term_variance(strip, T, bad_rate)


def test_interpolation_rejects_degenerate_arguments():
    year = 525_600.0
    with pytest.raises(ValueError, match="positive"):
        interpolate_variance(0.04, 0.09, 30_000, 50_000, 0, year)
    with pytest.raises(ValueError, match="positive"):
        interpolate_variance(0.04, 0.09, 30_000, 50_000, 43_200, 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        interpolate_variance(-0.01, 0.09, 30_000, 50_000, 43_200, year)
    with pytest.raises(ValueError, match="non-negative"):
        interpolate_variance(None, -0.09, None, 50_000, 43_200, year)


def test_strike_spacing_is_half_the_neighbour_distance_with_one_sided_ends():
    from varclock.variance import strike_spacing

    k = np.array([90.0, 95.0, 105.0, 106.0, 120.0])
    np.testing.assert_allclose(strike_spacing(k), [5.0, 7.5, 5.5, 7.5, 14.0])
