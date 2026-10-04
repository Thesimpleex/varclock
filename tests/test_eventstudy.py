from __future__ import annotations

from datetime import date, time

import numpy as np
import pytest

from helpers import make_events, synthetic_history
from varclock.events import Event
from varclock.eventstudy import (
    CONTROL_BLOCK,
    block_indices,
    bootstrap_mean,
    implied_session_variance,
    run_event_study,
)


def test_implied_session_variance_uses_390_business_minutes():
    v = implied_session_variance(20.0, date(2026, 10, 1), date(2026, 10, 2))
    assert v == pytest.approx(0.04 * 390 / 102_060)
    # a Friday close prices Monday's session in the same 390 business minutes
    assert implied_session_variance(20.0, date(2026, 10, 2), date(2026, 10, 5)) == pytest.approx(v)
    # before an early close (13:00) the next session has fewer minutes
    early = implied_session_variance(20.0, date(2026, 11, 27), date(2026, 11, 30))
    assert early == pytest.approx(v)
    half = implied_session_variance(20.0, date(2025, 11, 26), date(2025, 11, 28))
    assert half < v


def test_calibrated_implied_variance_gives_unit_ratios_and_zero_premium():
    vix1d, spx, days = synthetic_history(n_days=2500, seed=1)
    events = make_events(days, every=11)
    study = run_event_study(events, vix1d, spx, start=days[5], kinds=("CPI",))
    s = study.summarise(rng=np.random.default_rng(0), n_boot=2000)
    assert s["n"] > 150
    assert s["z_squared"] == pytest.approx(1.0, abs=0.25)
    assert s["z_squared_lo"] < 1.0 < s["z_squared_hi"]
    assert s["ratio_lo"] < 1.0 < s["ratio_hi"]
    assert s["premium_var_lo"] < 0.0 < s["premium_var_hi"]
    control = study.summarise_control(rng=np.random.default_rng(0), n_boot=2000)
    assert control["ratio"] == pytest.approx(1.0, abs=0.15)
    assert s["implied_abs"] == pytest.approx(s["implied_sd_mean"] * np.sqrt(2 / np.pi), rel=0.02)


def test_overpriced_implied_variance_shows_up_as_a_positive_premium():
    vix1d, spx, days = synthetic_history(n_days=2500, seed=2, scale=0.7)
    events = make_events(days, every=11)
    study = run_event_study(events, vix1d, spx, start=days[5], kinds=("CPI",))
    s = study.summarise(rng=np.random.default_rng(0), n_boot=2000)
    assert s["premium_var"] > 0
    assert s["premium_var_lo"] > 0
    assert s["ratio_hi"] < 0.75
    assert s["share_exceeding_implied_sd"] < 0.2


def test_events_of_other_kinds_are_ignored_and_controls_exclude_event_days():
    vix1d, spx, days = synthetic_history(n_days=300, seed=3)
    events = make_events(days, every=20)
    wrong_kind = Event(days[65], time(8, 30), "ELECTION", "ignored", "test")
    study = run_event_study([*events, wrong_kind], vix1d, spx, start=days[2], kinds=("CPI",))
    event_days = {r.session for r in study.rows}
    assert wrong_kind.session not in event_days
    assert len(study.control_implied) == len(study.control_return)
    in_range = [d for d in days[2:] if d <= study.end]
    assert len(study.control_implied) + len(event_days) == len(in_range)
    assert study.end == max(days)
    assert all(r.gap_days >= 1 for r in study.rows)
    assert study.by_kind("CPI") == list(study.rows)
    assert study.by_kind(("CPI", "FOMC")) == list(study.rows)


def test_days_without_a_previous_close_are_skipped():
    vix1d, spx, days = synthetic_history(n_days=100, seed=4)
    del vix1d[days[9]]
    event = Event(days[10], time(8, 30), "CPI", "no previous VIX1D close", "test")
    study = run_event_study([event], vix1d, spx, start=days[5], kinds=("CPI",))
    assert study.rows == ()
    assert len(study.control_implied) == len([d for d in days[5:] if d != days[10]])


def test_bootstrap_mean_interval_and_validation():
    x = np.random.default_rng(4).normal(2.0, 1.0, size=400)
    m, lo, hi = bootstrap_mean(x, np.random.default_rng(5), n_boot=4000)
    assert lo < m < hi
    assert hi - lo == pytest.approx(2 * 1.96 / 20, rel=0.2)
    with pytest.raises(ValueError, match="at least two"):
        bootstrap_mean(np.array([1.0]), np.random.default_rng(0))


def test_row_properties():
    vix1d, spx, days = synthetic_history(n_days=200, seed=6)
    study = run_event_study(make_events(days), vix1d, spx, start=days[2], kinds=("CPI",))
    r = study.rows[0]
    assert r.realised_var == r.log_return**2
    assert r.implied_sd == pytest.approx(np.sqrt(r.implied_var))
    assert r.z == pytest.approx(r.log_return / r.implied_sd)


def test_bootstrap_mean_quantiles_follow_the_level_for_a_fixed_seed():
    x = np.random.default_rng(10).normal(1.0, 2.0, size=50)
    for level in (0.8, 0.95):
        m, lo, hi = bootstrap_mean(x, np.random.default_rng(3), n_boot=500, level=level)
        draws = np.random.default_rng(3).choice(x, size=(500, 50), replace=True).mean(axis=1)
        a = (1 - level) / 2
        assert m == pytest.approx(x.mean())
        assert lo == pytest.approx(np.quantile(draws, a))
        assert hi == pytest.approx(np.quantile(draws, 1 - a))


def test_block_bootstrap_widens_the_interval_for_serially_dependent_data():
    rng = np.random.default_rng(1)
    shocks = rng.standard_normal(1200)
    x = np.convolve(shocks, np.ones(40) / 40, mode="valid")  # strongly autocorrelated series
    _, lo1, hi1 = bootstrap_mean(x, np.random.default_rng(2), n_boot=2000, block=1)
    _, lo2, hi2 = bootstrap_mean(x, np.random.default_rng(2), n_boot=2000, block=60)
    assert hi2 - lo2 > 2.0 * (hi1 - lo1)
    with pytest.raises(ValueError, match="block"):
        bootstrap_mean(x, np.random.default_rng(0), block=0)


def test_block_indices_are_circular_runs_of_consecutive_positions():
    idx = block_indices(10, np.random.default_rng(0), 5, 4)
    assert idx.shape == (5, 10)
    assert idx.min() >= 0
    assert idx.max() < 10
    runs = idx[:, :4]
    assert np.all((np.diff(runs, axis=1) % 10) == 1)


def test_control_sample_uses_the_block_bootstrap():
    vix1d, spx, days = synthetic_history(n_days=600, seed=8)
    study = run_event_study(make_events(days, every=15), vix1d, spx, start=days[2], kinds=("CPI",))
    s = study.summarise_control(rng=np.random.default_rng(0), n_boot=1000)
    imp, ret = study.control_implied, study.control_return
    expected = bootstrap_mean(imp - ret**2, np.random.default_rng(0), 1000, block=CONTROL_BLOCK)
    # the first statistic drawn by summarise_control is the implied variance, so compare the
    # premium of a fresh generator run through the same sequence of calls
    rng = np.random.default_rng(0)
    for values in (imp, ret**2):
        bootstrap_mean(values, rng, 1000, block=CONTROL_BLOCK)
    premium = bootstrap_mean(imp - ret**2, rng, 1000, block=CONTROL_BLOCK)
    assert s["premium_var_lo"] == pytest.approx(premium[1])
    assert s["premium_var_hi"] == pytest.approx(premium[2])
    assert expected[0] == pytest.approx(s["premium_var"])


def test_summarise_interval_levels_and_ratio_quantiles():
    vix1d, spx, days = synthetic_history(n_days=800, seed=9)
    study = run_event_study(make_events(days, every=9), vix1d, spx, start=days[2], kinds=("CPI",))
    rows = study.by_kind("CPI")
    s = study.summarise(rows, rng=np.random.default_rng(4), n_boot=800)
    rng = np.random.default_rng(4)
    names = ("implied_var", "realised_var", "premium_var", "z_squared", "realised_abs")
    for name in names:
        values = {
            "implied_var": np.array([r.implied_var for r in rows]),
            "realised_var": np.array([r.realised_var for r in rows]),
            "premium_var": np.array([r.implied_var - r.realised_var for r in rows]),
            "z_squared": np.array([r.z**2 for r in rows]),
            "realised_abs": np.array([abs(r.log_return) for r in rows]),
        }[name]
        _, lo, hi = bootstrap_mean(values, rng, 800)
        assert s[f"{name}_lo"] == pytest.approx(lo)
        assert s[f"{name}_hi"] == pytest.approx(hi)
    assert s["ratio_lo"] < s["ratio"] < s["ratio_hi"]
