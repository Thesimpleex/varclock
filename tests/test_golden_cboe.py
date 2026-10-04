"""Cboe's own worked examples: the implementation must reproduce every printed number."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import example_curve, example_strip, example_time
from varclock import tradingcal as tc
from varclock.variance import interpolate_variance, single_term_variance


@pytest.fixture(scope="module")
def vix_terms(vix_example):
    calc = example_time(vix_example)
    curve = example_curve(vix_example)
    out = {}
    for key in ("near", "next"):
        strip = example_strip(vix_example[key])
        minutes = tc.calendar_minutes(calc, strip.expiry_time)
        T = minutes / tc.MINUTES_PER_CALENDAR_YEAR
        rate = curve.rate(strip.expiry)
        out[key] = (minutes, T, rate, single_term_variance(strip, T, rate))
    return out


@pytest.mark.parametrize(("key", "idx"), [("near", "1"), ("next", "2")])
def test_vix_time_rate_forward_and_k0(vix_example, vix_terms, key, idx):
    exp = vix_example["expected"]
    minutes, T, rate, tv = vix_terms[key]
    assert minutes == exp[f"M{idx}"]
    assert pytest.approx(exp[f"T{idx}"], abs=5e-8) == T
    assert rate * 100 == pytest.approx(exp[f"R{idx}_pct"], abs=5e-7)
    assert tv.forward == pytest.approx(exp[f"F{idx}"], abs=2e-5)
    assert tv.k0 == exp["K0"]


@pytest.mark.parametrize(("key", "idx"), [("near", "1"), ("next", "2")])
def test_vix_term_variance_to_printed_precision(vix_example, vix_terms, key, idx):
    exp = vix_example["expected"]
    tv = vix_terms[key][3]
    assert tv.sigma2 == pytest.approx(exp[f"sigma2_{idx}"], abs=5e-10)
    assert 2 / tv.T * tv.contributions.sum() == pytest.approx(exp[f"sum_{idx}"], abs=5e-7)
    assert tv.contributions.sum() == pytest.approx(exp[f"sum_contrib_{idx}"], abs=1e-9)


@pytest.mark.parametrize("key", ["near", "next"])
def test_vix_every_strike_contribution_matches_the_appendix(vix_example, vix_terms, key):
    tv = vix_terms[key][3]
    printed = vix_example["contributions"][key]
    assert len(printed) == tv.strikes.size
    strikes = np.array([row[0] for row in printed])
    np.testing.assert_array_equal(strikes, tv.strikes)
    np.testing.assert_allclose(tv.prices, [row[2] for row in printed], atol=1e-12)
    np.testing.assert_allclose(tv.delta_k, [row[3] for row in printed], atol=1e-12)
    np.testing.assert_allclose(tv.contributions, [row[4] for row in printed], atol=6e-11)
    kinds = {"Put": "P", "Call": "C", "Both": "B"}
    assert [kinds[row[1]] for row in printed] == list(tv.kinds)


def test_vix_index_value(vix_example, vix_terms):
    exp = vix_example["expected"]
    (m1, _, _, near), (m2, _, _, nxt) = vix_terms["near"], vix_terms["next"]
    value = interpolate_variance(
        near.sigma2, nxt.sigma2, m1, m2, 30 * 1440, tc.MINUTES_PER_CALENDAR_YEAR
    )
    assert value == pytest.approx(exp["vix_unrounded"], abs=5e-6)
    assert round(value, 2) == exp["vix"]


def test_vix_strike_selection_matches_the_narrative(vix_terms):
    near = vix_terms["near"][3]
    assert near.strikes[0] == 1370
    assert near.strikes[-1] == 2125
    assert 1350 not in near.strikes
    assert 1355 not in near.strikes
    assert 2225 not in near.strikes
    nxt = vix_terms["next"][3]
    assert nxt.strikes[0] == 1275
    assert nxt.strikes[-1] == 2200


def test_vix1d_example(vix1d_example):
    exp = vix1d_example["expected"]
    calc = example_time(vix1d_example)
    curve = example_curve(vix1d_example)
    terms = []
    for key in ("near", "next"):
        strip = example_strip(vix1d_example[key])
        minutes = tc.business_minutes(calc, strip.expiry_time)
        T = minutes / tc.BUSINESS_MINUTES_PER_YEAR
        terms.append((minutes, T, single_term_variance(strip, T, curve.rate(strip.expiry))))
    (m1, t1, near), (m2, t2, nxt) = terms
    assert (m1, m2) == (exp["M1"], exp["M2"])
    assert t1 == pytest.approx(exp["T1"], abs=5e-9)
    assert t2 == pytest.approx(exp["T2"], abs=5e-9)
    assert near.forward == pytest.approx(exp["F1"], abs=5e-6)
    assert nxt.forward == pytest.approx(exp["F2"], abs=5e-6)
    assert near.k0 == nxt.k0 == exp["K0"]
    assert near.sigma2 == pytest.approx(exp["sigma2_1"], abs=2e-8)
    assert nxt.sigma2 == pytest.approx(exp["sigma2_2"], abs=2e-8)
    value = interpolate_variance(
        near.sigma2, nxt.sigma2, m1, m2, tc.SESSION_MINUTES, tc.BUSINESS_MINUTES_PER_YEAR
    )
    assert value == pytest.approx(exp["vix1d_unrounded"], abs=2e-5)
    assert round(value, 2) == exp["vix1d"]


@pytest.mark.parametrize("key", ["near", "next"])
def test_full_strike_window_reproduces_the_cboe_total_on_the_nonuniform_grid(vix_terms, key):
    """Cboe's example grids mix 5, 10, 25 and 50 point spacings, unlike the synthetic ones."""
    from varclock.strip import window_total_variance

    tv = vix_terms[key][3]
    assert np.unique(np.diff(tv.strikes)).size > 2
    assert window_total_variance(tv, tv.strikes[0], tv.strikes[-1]) == pytest.approx(
        tv.total_variance, abs=1e-15
    )
    inner = window_total_variance(tv, 1500.0, 2000.0)
    assert 0 < inner < tv.total_variance
