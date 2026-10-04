"""Experiment: numerical validation against Cboe's worked examples and closed forms.

Writes the headline accuracy numbers quoted in the README: agreement with the printed values of
both Cboe samples, convergence of the replication to Black-Scholes and Heston variances, and
exact recovery of a known variance clock.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time
from itertools import pairwise

import numpy as np
from common import ROOT, write_json

from varclock import tradingcal as tc
from varclock.clock import ClockData, SessionCalendar, fit_clock
from varclock.events import Event
from varclock.quotes import OptionStrip
from varclock.rates import YieldCurve
from varclock.strip import build_strip
from varclock.synthetic import (
    black76_prices,
    heston_expected_variance,
    heston_prices,
    synthetic_daily_snapshot,
    synthetic_strip,
)
from varclock.variance import interpolate_variance, single_term_variance

FIXTURES = ROOT / "tests" / "data"


def _strip(block: dict) -> OptionStrip:
    q = np.array(block["quotes"], dtype=float)
    return OptionStrip(
        block["root"],
        date.fromisoformat(block["expiry"]),
        q[:, 0],
        q[:, 1],
        q[:, 2],
        q[:, 3],
        q[:, 4],
    )


def _golden(name: str, business: bool) -> dict:
    ex = json.loads((FIXTURES / name).read_text())
    calc = datetime.fromisoformat(ex["calc_time"])
    curve = YieldCurve(date.fromisoformat(ex["curve_date"]), ex["cmt_pct"])
    terms = []
    for key in ("near", "next"):
        strip = _strip(ex[key])
        minutes = (
            tc.business_minutes(calc, strip.expiry_time)
            if business
            else tc.calendar_minutes(calc, strip.expiry_time)
        )
        year = tc.BUSINESS_MINUTES_PER_YEAR if business else tc.MINUTES_PER_CALENDAR_YEAR
        T = minutes / year
        terms.append((minutes, single_term_variance(strip, T, curve.rate(strip.expiry))))
    (m1, a), (m2, b) = terms
    e = ex["expected"]
    target = tc.SESSION_MINUTES if business else 30 * 1440
    year = tc.BUSINESS_MINUTES_PER_YEAR if business else tc.MINUTES_PER_CALENDAR_YEAR
    value = interpolate_variance(a.sigma2, b.sigma2, m1, m2, target, year)
    printed = e["vix1d_unrounded" if business else "vix_unrounded"]
    out = {
        "minutes": [m1, m2],
        "sigma2_near": a.sigma2,
        "sigma2_next": b.sigma2,
        "abs_error_sigma2": [abs(a.sigma2 - e["sigma2_1"]), abs(b.sigma2 - e["sigma2_2"])],
        "forward": [a.forward, b.forward],
        "abs_error_forward": [abs(a.forward - e["F1"]), abs(b.forward - e["F2"])],
        "index_value": value,
        "printed_index_value": printed,
        "abs_error_index": abs(value - printed),
    }
    if not business:
        diffs = []
        for key, tv in (("near", a), ("next", b)):
            printed = np.array([r[4] for r in ex["contributions"][key]])
            diffs.append(float(np.max(np.abs(tv.contributions - printed))))
        out["max_abs_error_strike_contribution"] = diffs
        out["strikes_matched"] = [
            len(ex["contributions"]["near"]),
            len(ex["contributions"]["next"]),
        ]
    return out


def _black_scholes() -> dict:
    rows = []
    for h in (4.0, 2.0, 1.0, 0.5, 0.25):
        k = np.arange(20.0, 300.0 + h, h)
        call, put = black76_prices(100.0, k, 0.004)
        strip = synthetic_strip("SPXW", date(2030, 1, 2), k, call, put)
        tv = single_term_variance(strip, 0.1, 0.0)
        rows.append({"strike_spacing": h, "error": tv.total_variance - 0.004})
    ratios = [a["error"] / b["error"] for a, b in pairwise(rows)]
    return {"true_total_variance": 0.004, "by_spacing": rows, "error_ratio_per_halving": ratios}


def _heston() -> dict:
    pars = {"v0": 0.04, "kappa": 2.0, "theta": 0.06, "xi": 0.5, "rho": -0.7}
    T = 30 / 365
    target = heston_expected_variance(T, pars["v0"], pars["kappa"], pars["theta"])
    rows = []
    for h in (2.0, 1.0, 0.5, 0.25):
        k = np.arange(20.0, 300.0 + h, h)
        call, put = heston_prices(100.0, k, T, **pars)
        strip = synthetic_strip("SPXW", date(2030, 1, 2), k, call, put)
        rows.append(
            {
                "strike_spacing": h,
                "replicated": single_term_variance(strip, T, 0.0).total_variance,
                "error": single_term_variance(strip, T, 0.0).total_variance - target,
            }
        )
    return {
        "parameters": pars,
        "T_years": T,
        "expected_integrated_variance": target,
        "by_spacing": rows,
        "error_ratio_per_halving": [a["error"] / b["error"] for a, b in pairwise(rows)],
    }


def _clock_recovery() -> dict:
    asof = datetime(2026, 10, 2, 16, 15, tzinfo=tc.ET)
    curve = YieldCurve(date(2026, 10, 1), {"1 Mo": 0.0, "3 Mo": 0.0, "1 Yr": 0.0})
    days = [tc.next_trading_day(asof.date())]
    while len(days) < 22:
        days.append(tc.next_trading_day(days[-1]))
    base, weekend, jump = 0.9e-4, 0.2e-4, 1.1e-4
    cal = SessionCalendar.build(asof, days)
    v = base + weekend * cal.weekend
    v[10] += jump
    events = [Event(days[10], time(8, 30), "CPI", "synthetic event", "synthetic")]
    exact = fit_clock(
        ClockData(asof, tuple(days), np.cumsum(v), np.full(22, 1e-9)),
        events,
        family="meanrev",
        kappa=0.002,
    )
    snap = synthetic_daily_snapshot(asof, days, v, forward=1000.0, spacing=0.25)
    via = fit_clock(
        ClockData.from_strip(build_strip(snap, curve)), events, family="meanrev", kappa=0.002
    )
    return {
        "truth": {"base": base, "weekend": weekend, "event_variance": jump},
        "from_exact_totals": {
            "max_rel_error_session_variance": float(np.max(np.abs(exact.session_variance / v - 1))),
            "event_variance_rel_error": float(abs(exact.bumps[0] / jump - 1)),
            "weekend_rel_error": float(abs(exact.weekend / weekend - 1)),
        },
        "from_black76_chains": {
            "event_variance_rel_error": float(abs(via.bumps[0] / jump - 1)),
            "weekend_rel_error": float(abs(via.weekend / weekend - 1)),
            "base_rel_error_first_ten": float(np.max(np.abs(via.base[:10] / base - 1))),
        },
    }


def _truncated_chains() -> dict:
    """Black-76 chains with zero-bid truncation: own-range totals against the common window."""
    asof = datetime(2026, 10, 2, 16, 15, tzinfo=tc.ET)
    curve = YieldCurve(date(2026, 10, 1), {"1 Mo": 0.0, "3 Mo": 0.0, "1 Yr": 0.0})
    days = [tc.next_trading_day(asof.date())]
    while len(days) < 30:
        days.append(tc.next_trading_day(days[-1]))
    cal = SessionCalendar.build(asof, days)
    base = 0.6e-4 + 0.03e-4 * np.arange(30)
    weekend = 0.1e-4
    v = base + weekend * cal.weekend
    truth_ratio = weekend / float(base[cal.weekend > 0].mean())
    out: dict = {"truth_weekend_ratio": truth_ratio, "by_tick": {}}
    for tick in (0.05, 0.1, 0.3):
        snap = synthetic_daily_snapshot(asof, days, v, spacing=1.0, tick=tick)
        rec = {}
        for coverage in ("own", "common"):
            strip = build_strip(snap, curve, coverage=coverage)
            fit = fit_clock(ClockData.from_strip(strip), ())
            rec[coverage] = {
                "max_rel_error_cumulative_variance": float(
                    np.max(np.abs(strip.cumulative / np.cumsum(v) - 1.0))
                ),
                "weekend_ratio": float(fit.weekend_ratio),
                "weekend_ratio_rel_error": float(fit.weekend_ratio / truth_ratio - 1.0),
            }
        out["by_tick"][str(tick)] = rec
    return out


def run() -> dict:
    out = {
        "golden_vix_sample": _golden("cboe_vix_example.json", business=False),
        "golden_vix1d_sample": _golden("cboe_vix1d_example.json", business=True),
        "black_scholes_limit": _black_scholes(),
        "heston_convergence": _heston(),
        "clock_recovery": _clock_recovery(),
        "truncated_chains": _truncated_chains(),
    }
    write_json("validation", out)
    return out


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: v for k, v in r.items() if k != "black_scholes_limit"}, indent=1)[:2500])
