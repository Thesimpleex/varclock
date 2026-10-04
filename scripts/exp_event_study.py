"""Experiment: implied versus realised variance on CPI, FOMC and employment-report days.

Implied: (VIX1D / 100)^2 x 390 / 102,060 at the previous 16:15 close. Realised: squared
S&P 500 close-to-close log return. Free data: Cboe index histories and the committed calendars.
"""

from __future__ import annotations

import csv

import numpy as np
from common import FIGURES, RESULTS, write_json

from varclock.events import load_events
from varclock.eventstudy import (
    CONTROL_BLOCK,
    EventStudy,
    StudyRow,
    block_indices,
    bootstrap_mean,
    run_event_study,
)
from varclock.figures import build_study_figure
from varclock.sources import load_index_history
from varclock.viz import save_figure

KINDS = ("CPI", "FOMC", "NFP")
N_BOOT = 10_000
SEED = 0


def _multiples(study: EventStudy, rows: list[StudyRow], rng: np.random.Generator) -> dict:
    """Event-day variance as a multiple of the ordinary-day average, implied and realised."""
    imp = np.array([r.implied_var for r in rows])
    real = np.array([r.realised_var for r in rows])
    c_imp, c_real = study.control_implied, study.control_return**2
    ei = rng.integers(0, len(rows), size=(N_BOOT, len(rows)))
    ci = block_indices(c_imp.size, rng, N_BOOT, CONTROL_BLOCK)
    implied = imp[ei].mean(axis=1) / c_imp[ci].mean(axis=1)
    realised = real[ei].mean(axis=1) / c_real[ci].mean(axis=1)
    q = (0.025, 0.975)
    return {
        "implied_multiple": float(imp.mean() / c_imp.mean()),
        "implied_multiple_95pct": [float(x) for x in np.quantile(implied, q)],
        "realised_multiple": float(real.mean() / c_real.mean()),
        "realised_multiple_95pct": [float(x) for x in np.quantile(realised, q)],
    }


def _add_percent_squared(s: dict) -> None:
    """Variance premium in squared percent (decimal variance times 10,000)."""
    s["premium_pct_sq"] = s["premium_var"] * 1e4
    s["premium_pct_sq_95pct"] = [s["premium_var_lo"] * 1e4, s["premium_var_hi"] * 1e4]


def _control_sensitivity(study: EventStudy) -> dict:
    """Ordinary-day premium interval for several block lengths, and its serial dependence."""
    premium = study.control_implied - study.control_return**2
    lag1 = float(np.corrcoef(premium[:-1], premium[1:])[0, 1])
    intervals = {}
    for block in (1, 5, CONTROL_BLOCK, 60):
        _, lo, hi = bootstrap_mean(premium, np.random.default_rng(SEED), N_BOOT, 0.95, block)
        intervals[f"block_{block}"] = [lo * 1e4, hi * 1e4]
    return {
        "premium_pct_sq_95pct_by_block_length": intervals,
        "premium_lag1_autocorrelation": lag1,
        "note": "block_1 is the plain bootstrap of days; the premium series is negatively "
        "autocorrelated (implied variance reacts to the previous day's realised move), so "
        "resampling blocks gives a narrower interval than resampling single days",
    }


def run() -> dict:
    events = load_events(list(KINDS))
    vix1d, spx = load_index_history("VIX1D"), load_index_history("SPX")
    study = run_event_study(events, vix1d, spx, kinds=KINDS)
    macro: dict = {}
    for row in sorted(study.rows, key=lambda r: r.session):
        if row.kind in ("CPI", "FOMC"):
            macro.setdefault(row.session, row)
    samples = {k: study.by_kind(k) for k in KINDS}
    samples["CPI or FOMC (unique days)"] = list(macro.values())
    out: dict = {
        "period": [study.start.isoformat(), study.end.isoformat()],
        "last_vix1d_close": max(vix1d).isoformat(),
        "last_spx_close": max(spx).isoformat(),
        "bootstrap": {
            "draws": N_BOOT,
            "seed": SEED,
            "interval": "95% percentile",
            "event_days": "independent resampling of event days",
            "ordinary_days": f"circular block bootstrap, blocks of {CONTROL_BLOCK} sessions",
        },
        "events": {},
    }
    for name, rows in samples.items():
        rng = np.random.default_rng(SEED)
        s = study.summarise(rows, rng=rng, n_boot=N_BOOT)
        s.update(_multiples(study, rows, np.random.default_rng(SEED)))
        s["after_weekend_or_holiday"] = int(sum(r.gap_days > 1 for r in rows))
        _add_percent_squared(s)
        out["events"][name] = s
    ctrl = study.summarise_control(rng=np.random.default_rng(SEED), n_boot=N_BOOT)
    _add_percent_squared(ctrl)
    ctrl["bootstrap_sensitivity"] = _control_sensitivity(study)
    out["ordinary_days"] = ctrl
    write_json("event_study", out)

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "event_study_rows.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "kind", "implied_sd_pct", "log_return_pct", "z", "gap_days"])
        for r in sorted(study.rows, key=lambda r: (r.session, r.kind)):
            w.writerow(
                [
                    r.session.isoformat(),
                    r.kind,
                    f"{r.implied_sd * 100:.4f}",
                    f"{r.log_return * 100:.4f}",
                    f"{r.z:.4f}",
                    r.gap_days,
                ]
            )
    save_figure(build_study_figure(study, KINDS), FIGURES / "event_study")
    return out


if __name__ == "__main__":
    result = run()
    for name, s in result["events"].items():
        print(
            name,
            s["n"],
            f"ratio {s['ratio']:.2f} [{s['ratio_lo']:.2f},{s['ratio_hi']:.2f}]",
            f"implied x{s['implied_multiple']:.2f} realised x{s['realised_multiple']:.2f}",
        )
