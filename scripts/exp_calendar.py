"""Experiment: the variance calendar of the latest snapshot, per adapter, with sensitivity runs."""

from __future__ import annotations

import numpy as np
from common import FIGURES, notify, snapshot_meta, write_json

from varclock.analysis import CalendarResult, build_calendar
from varclock.clock import ClockData, bootstrap_clock, fit_clock
from varclock.figures import build_calendar_figure, build_robustness_figure
from varclock.loaders import obtain_curve, obtain_snapshot
from varclock.strip import QUOTE_CORRELATION, build_strip
from varclock.viz import save_figure

N_BOOT = 1000
SEED = 0
KEY_EVENTS = ("CPI", "FOMC", "ELECTION")


def _calendar_payload(result: CalendarResult) -> dict:
    strip = result.strip
    rel = strip.band / np.maximum(np.abs(strip.increments), 1e-30)
    return {
        "snapshot": snapshot_meta(result.snapshot),
        "treasury_curve_date": result.curve.curve_date.isoformat(),
        "n_expiries": len(strip.points),
        "first_expiry": strip.expiries[0].isoformat(),
        "last_expiry": strip.expiries[-1].isoformat(),
        "skipped_expiries": [{"expiry": d.isoformat(), "reason": r} for d, r in strip.skipped],
        "calendar_arbitrage": {
            "negative_forward_variances": int(strip.violations.sum()),
            "negative_beyond_quote_band": int(strip.significant_violations.sum()),
        },
        "strip": [
            {
                "expiry": p.expiry.isoformat(),
                "minutes": p.minutes,
                "rate": p.rate,
                "forward": p.term.forward,
                "total_variance": p.total_variance,
                "cumulative": float(c),
                "cumulative_band": float(cb),
                "increment": float(i),
                "increment_band": float(b),
                "relative_band": float(r),
                "options_used": p.term.n_options,
            }
            for p, c, cb, i, b, r in zip(
                strip.points,
                strip.cumulative,
                strip.cumulative_band,
                strip.increments,
                strip.band,
                rel,
                strict=True,
            )
        ],
        "fit": {
            "family": result.fit.family,
            "kappa_per_session": result.fit.kappa,
            "chi2_per_dof": result.fit.chi2_dof,
            "base_vol_first_session": float(result.fit.base_vol[0]),
            "base_vol_last_session": float(result.fit.base_vol[-1]),
            "weekend_in_base_sessions": result.fit.weekend_ratio,
            "weekend_in_base_sessions_90pct": [
                float(x) for x in result.boot.interval("weekend_ratio")
            ],
        },
        "events": [
            {
                "key": e.key,
                "label": e.label,
                "session": e.session.isoformat(),
                "extra_variance": e.bump_var,
                "extra_variance_90pct": [e.bump_lo, e.bump_hi],
                "extra_variance_pct_sq": e.bump_var * 1e4,
                "extra_variance_pct_sq_90pct": [e.bump_lo * 1e4, e.bump_hi * 1e4],
                "multiple_of_base": e.multiple,
                "multiple_90pct": [e.multiple_lo, e.multiple_hi],
                "implied_one_sd_move": e.implied_sd,
            }
            for e in result.event_rows
        ],
        "sessions": [
            {
                "session": r.session.isoformat(),
                "fwd_vol": r.fwd_vol,
                "fwd_vol_90pct": [r.fwd_vol_lo, r.fwd_vol_hi],
                "base_vol": r.base_vol,
                "multiple": r.multiple,
                "multiple_90pct": [r.multiple_lo, r.multiple_hi],
                "strip_vol": r.observed_vol,
                "weekend": r.weekend,
                "events": list(r.events),
            }
            for r in result.rows
        ],
        "bootstrap": {
            "draws": result.n_boot,
            "seed": result.seed,
            "level": 0.9,
            "quote_correlation": QUOTE_CORRELATION,
        },
        "strike_coverage": result.strip.coverage,
        "notes": list(result.notes),
    }


def _sensitivity(result: CalendarResult) -> dict:
    """Event multiples under other base and coverage specifications (same data and bootstrap)."""
    common = build_strip(result.snapshot, result.curve, coverage="common")
    configs = {
        "meanrev (default)": ({"family": "meanrev"}, result.strip),
        "spline, 8 knots, penalty 1": ({"family": "spline", "knots": 8, "penalty": 1.0}, None),
        "spline, 6 knots, penalty 0.1": ({"family": "spline", "knots": 6, "penalty": 0.1}, None),
        "spline, 10 knots, penalty 10": ({"family": "spline", "knots": 10, "penalty": 10.0}, None),
        "meanrev, common strike window": ({"family": "meanrev"}, common),
    }
    out = {}
    for name, (cfg, strip) in configs.items():
        strip = result.strip if strip is None else strip
        fit = fit_clock(ClockData.from_strip(strip), result.events, **cfg)
        boot = bootstrap_clock(fit, strip, n_boot=400, rng=np.random.default_rng(SEED), level=0.9)
        lo, hi = boot.interval("event_multiple")
        wk_lo, wk_hi = boot.interval("weekend_ratio")
        rec = {
            "chi2_per_dof": fit.chi2_dof,
            "weekend_in_base_sessions": [fit.weekend_ratio, float(wk_lo), float(wk_hi)],
            "events": {},
        }
        for i, term in enumerate(fit.terms):
            rec["events"][term.key] = [fit.event_multiple(i), float(lo[i]), float(hi[i])]
        out[name] = rec
    return out


def _weekend_range(sensitivity: dict) -> dict:
    """Weekend weight (in base sessions) across every specification and adapter."""
    points = {
        f"{source}: {name}": rec["weekend_in_base_sessions"][0]
        for source, runs in sensitivity.items()
        for name, rec in runs.items()
    }
    lows = [
        r["weekend_in_base_sessions"][1] for runs in sensitivity.values() for r in runs.values()
    ]
    highs = [
        r["weekend_in_base_sessions"][2] for runs in sensitivity.values() for r in runs.values()
    ]
    return {
        "point_estimates": points,
        "point_min": min(points.values()),
        "point_max": max(points.values()),
        "interval_min": min(lows),
        "interval_max": max(highs),
    }


def _noise_components(result: CalendarResult) -> dict:
    """Which part of the bootstrap drives the interval widths, and how large quote noise is."""
    strip, fit = result.strip, result.fit
    widths = {}
    for label, kwargs in (
        ("quotes_only", {"resample_residuals": False}),
        ("residuals_only", {"resample_quotes": False}),
        ("both", {}),
    ):
        boot = bootstrap_clock(
            fit, strip, n_boot=400, rng=np.random.default_rng(SEED), level=0.9, **kwargs
        )
        lo, hi = boot.interval("event_multiple")
        wlo, whi = boot.interval("weekend_ratio")
        widths[label] = {
            "event_multiple_interval_width": {
                t.key: float(hi[i] - lo[i]) for i, t in enumerate(fit.terms)
            },
            "weekend_ratio_interval_width": float(whi - wlo),
        }
    rng = np.random.default_rng(SEED)
    sd = {}
    for rho in (0.0, QUOTE_CORRELATION):
        draws = np.array([strip.resampled_cumulative(rng, rho) for _ in range(400)])
        sd[str(rho)] = (draws.std(axis=0) / strip.cumulative).tolist()
    band = (strip.cumulative_band / strip.cumulative).tolist()
    return {
        "draws": 400,
        "quote_correlation": QUOTE_CORRELATION,
        "interval_widths": widths,
        "resampled_sd_over_total": {
            "independent_quotes": _span(sd["0.0"]),
            f"correlation_{QUOTE_CORRELATION}": _span(sd[str(QUOTE_CORRELATION)]),
        },
        "quote_band_half_width_over_total": _span(band),
        "chi2_per_dof": fit.chi2_dof,
    }


def _span(values: list[float]) -> dict:
    return {
        "min": float(min(values)),
        "median": float(np.median(values)),
        "max": float(max(values)),
    }


def _robustness_table(sens: dict) -> dict:
    keys = [k for k in next(iter(sens.values()))["events"] if k.split()[0] in KEY_EVENTS]
    families = {
        "meanrev (default)": "Mean-reverting base",
        "spline, 8 knots, penalty 1": "Penalised spline, 8 knots",
    }
    pretty = {"CPI": "CPI", "FOMC": "FOMC", "ELECTION": "Election"}
    names = {k: f"{pretty[k.split()[0]]} {k.split()[1][5:]}" for k in keys}
    return {
        label: {names[k]: tuple(sens[fam]["events"][k]) for k in keys}
        for fam, label in families.items()
    }


def run(refresh: bool = False) -> dict:
    payload: dict = {}
    results: dict[str, CalendarResult] = {}
    for source in ("cboe", "yahoo"):
        snap = obtain_snapshot(source, notify=notify, refresh=refresh)
        curve = obtain_curve(snap)
        results[source] = build_calendar(snap, curve, n_boot=N_BOOT, seed=SEED)
        payload[source] = _calendar_payload(results[source])
    payload["sensitivity"] = {s: _sensitivity(results[s]) for s in results}
    payload["weekend_range"] = _weekend_range(payload["sensitivity"])
    payload["bootstrap_components"] = {s: _noise_components(results[s]) for s in results}
    common = [
        e["key"]
        for e in payload["cboe"]["events"]
        if e["key"] in {y["key"] for y in payload["yahoo"]["events"]}
    ]
    cboe = {e["key"]: e["multiple_of_base"] for e in payload["cboe"]["events"]}
    yahoo = {e["key"]: e["multiple_of_base"] for e in payload["yahoo"]["events"]}
    payload["adapter_agreement"] = {
        "events_in_both": common,
        "multiple_difference_yahoo_minus_cboe": {k: yahoo[k] - cboe[k] for k in common},
    }
    rel = [r["relative_band"] for r in payload["cboe"]["strip"]]
    payload["daily_resolution"] = {
        "first_session_with_band_over_50pct_of_increment": next(
            (i + 1 for i, r in enumerate(rel) if r > 0.5), None
        ),
        "first_session_with_band_over_100pct_of_increment": next(
            (i + 1 for i, r in enumerate(rel) if r > 1.0), None
        ),
        "relative_band_by_session": rel,
    }
    write_json("calendar", payload)
    save_figure(build_calendar_figure(results["cboe"]), FIGURES / "calendar")
    save_figure(
        build_robustness_figure(_robustness_table(payload["sensitivity"]["cboe"])),
        FIGURES / "robustness",
    )
    return payload


if __name__ == "__main__":
    out = run()
    for e in out["cboe"]["events"]:
        print(e["key"], round(e["multiple_of_base"], 2), [round(x, 2) for x in e["multiple_90pct"]])
