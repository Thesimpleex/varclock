"""Offline regression test of the headline calendar numbers.

The expected values were produced by this implementation (``scripts/freeze_clock_input.py``) from
the committed strip totals: the test pins the fit against unintended changes, it is not an
independent reference. Independent references are the closed-form and truncated-chain truth tests
in ``test_clock.py`` and ``test_strip.py``.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pytest

from helpers import load_example
from varclock.clock import ClockData, fit_clock
from varclock.events import load_events


@pytest.fixture(scope="module", params=["cboe", "yahoo"])
def frozen(request):
    return load_example("clock_input_2026-10-02.json")[request.param]


def test_fit_reproduces_the_frozen_event_multiples(frozen):
    asof = datetime.fromisoformat(frozen["asof"])
    expiries = tuple(date.fromisoformat(d) for d in frozen["expiries"])
    data = ClockData(asof, expiries, np.array(frozen["cumulative"]), np.array(frozen["band"]))
    events = load_events(start=asof.date(), end=expiries[-1])
    fit = fit_clock(data, events, family="meanrev")
    for i, term in enumerate(fit.terms):
        assert fit.event_multiple(i) == pytest.approx(
            frozen["expected_multiples"][term.key], abs=1e-9
        )
    assert fit.weekend_ratio == pytest.approx(frozen["expected_weekend_ratio"], abs=1e-9)
    assert fit.kappa == pytest.approx(frozen["expected_kappa"], rel=1e-12)


def test_the_headline_pattern_holds_on_both_adapters():
    fx = load_example("clock_input_2026-10-02.json")
    for source in ("cboe", "yahoo"):
        m = fx[source]["expected_multiples"]
        assert 1.5 < m["CPI 2026-10-14"] < 1.9
        assert 1.4 < m["FOMC 2026-10-28"] < 1.9
        assert m["ELECTION 2026-11-04"] > 1.5
        assert 0.05 < fx[source]["expected_weekend_ratio"] < 0.25
