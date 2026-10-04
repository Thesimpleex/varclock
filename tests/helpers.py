"""Shared helpers for the test suite."""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np

from varclock import tradingcal as tc
from varclock.events import Event
from varclock.quotes import OptionStrip
from varclock.rates import YieldCurve

DATA = Path(__file__).parent / "data"


def load_example(name: str) -> dict:
    with (DATA / name).open() as fh:
        return json.load(fh)


def example_strip(block: dict) -> OptionStrip:
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


def example_curve(example: dict) -> YieldCurve:
    return YieldCurve(date.fromisoformat(example["curve_date"]), example["cmt_pct"])


def example_time(example: dict) -> datetime:
    return datetime.fromisoformat(example["calc_time"])


ASOF = datetime(2026, 10, 2, 16, 15, tzinfo=tc.ET)
CURVE = YieldCurve(date(2026, 10, 1), {"1 Mo": 0.0, "3 Mo": 0.0, "1 Yr": 0.0})
N_SESSIONS = 30


def synthetic_history(n_days=900, vix1d=20.0, scale=1.0, seed=0, start=date(2022, 5, 12)):
    """Constant VIX1D and S&P 500 levels with returns at the implied standard deviation."""
    rng = np.random.default_rng(seed)
    days = tc.trading_days(start, start + timedelta(days=int(n_days * 1.5)))[:n_days]
    var = (vix1d / 100) ** 2 * 390 / tc.BUSINESS_MINUTES_PER_YEAR
    sd = np.sqrt(var) * scale
    level = 4000.0
    spx = {}
    for d in days:
        level *= np.exp(sd * rng.standard_normal())
        spx[d] = level
    return dict.fromkeys(days, vix1d), spx, days


def make_events(days, kind="CPI", every=17):
    return [
        Event(d, time(8, 30), kind, f"{kind} {d}", "test")
        for d in days[40::every]
        if d.weekday() < 5
    ]
