"""Rewrite ``tests/data/clock_input_2026-10-02.json`` from the cached 2026-10-02 snapshots.

The file holds derived strip totals (no quotes) and the fit frozen from them, so that the headline
event multiples are regression-tested offline. Run it after a deliberate change of the strip or
the fit and review the diff of the expected values.
"""

from __future__ import annotations

import json

from common import ROOT, notify

from varclock.clock import ClockData, fit_clock
from varclock.events import load_events
from varclock.loaders import obtain_curve, obtain_snapshot
from varclock.strip import build_strip

OUT = ROOT / "tests" / "data" / "clock_input_2026-10-02.json"


def main() -> None:
    payload: dict = {}
    for source in ("cboe", "yahoo"):
        snap = obtain_snapshot(source, notify=notify)
        strip = build_strip(snap, obtain_curve(snap))
        data = ClockData.from_strip(strip)
        events = load_events(start=data.asof.date(), end=data.expiries[-1])
        fit = fit_clock(data, events, family="meanrev")
        payload[source] = {
            "asof": data.asof.isoformat(),
            "expiries": [d.isoformat() for d in data.expiries],
            "cumulative": [float(x) for x in data.cumulative],
            "band": [float(x) for x in data.band],
            "expected_multiples": {
                t.key: float(fit.event_multiple(i)) for i, t in enumerate(fit.terms)
            },
            "expected_weekend_ratio": float(fit.weekend_ratio),
            "expected_kappa": float(fit.kappa),
        }
    payload["description"] = (
        "Derived strip totals (cumulative variance on each expiry's own strike range, and quote "
        "half-widths) of the 2026-10-02 close snapshots and the fit results frozen from them; "
        "no raw quotes. Regression data produced by scripts/freeze_clock_input.py from this "
        "implementation, not an independent reference."
    )
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    notify(f"wrote {OUT}")


if __name__ == "__main__":
    main()
