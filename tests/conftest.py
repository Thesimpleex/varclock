from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from helpers import ASOF, CURVE, N_SESSIONS, load_example
from varclock import tradingcal as tc
from varclock.analysis import build_calendar
from varclock.clock import SessionCalendar
from varclock.synthetic import synthetic_daily_snapshot


def _example(name: str) -> dict:
    """Full Cboe fixture when available locally, otherwise the printed results only."""
    try:
        return load_example("cboe_full/" + name)
    except FileNotFoundError:
        return load_example(name)


@pytest.fixture(scope="session")
def vix_example() -> dict:
    return _example("cboe_vix_example.json")


@pytest.fixture(scope="session")
def vix1d_example() -> dict:
    return _example("cboe_vix1d_example.json")


def sessions():
    out = [tc.next_trading_day(ASOF.date())]
    while len(out) < N_SESSIONS:
        out.append(tc.next_trading_day(out[-1]))
    return out


@pytest.fixture(scope="session")
def truth():
    """Known session variances: rising base, weekend term, a CPI and an FOMC bump."""
    days = sessions()
    cal = SessionCalendar.build(ASOF, days)
    s = np.arange(N_SESSIONS)
    base = 0.6e-4 + 0.03e-4 * s
    v = base + 0.1e-4 * cal.weekend
    v[days.index(date(2026, 10, 14))] += 0.5e-4
    v[days.index(date(2026, 10, 28))] += 0.7e-4
    return days, v, base


@pytest.fixture(scope="session")
def snapshot(truth):
    days, v, _ = truth
    return synthetic_daily_snapshot(ASOF, days, v, forward=1000.0, spacing=0.25)


@pytest.fixture(scope="session")
def result(snapshot):
    return build_calendar(snapshot, CURVE, n_boot=60, seed=3)
