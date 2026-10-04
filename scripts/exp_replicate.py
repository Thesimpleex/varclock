"""Experiment: rebuild VIX, VIX9D, VIX3M and VIX1D from one snapshot per adapter.

Compares the rebuilt closes with the official ones, separates the error caused by strike
coverage from the error caused by different quotes, and shows the effect of the valuation time.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime

import numpy as np
from common import FIGURES, notify, snapshot_meta, write_json

from varclock.figures import build_replication_figure
from varclock.indices import IndexResult, replicate
from varclock.loaders import obtain_curve, obtain_snapshot, official_closes
from varclock.quotes import OptionStrip, Snapshot
from varclock.tradingcal import ET
from varclock.viz import save_figure

NAMES = ("VIX", "VIX9D", "VIX3M", "VIX1D")


def _values(snap: Snapshot, curve) -> dict[str, float]:
    out = {}
    for name, r in replicate(snap, curve, NAMES).items():
        if isinstance(r, IndexResult):
            out[name] = r.value
    return out


def _restrict_to(strips_from: Snapshot, strips_to: Snapshot) -> Snapshot:
    other = {(s.root, s.expiry): s for s in strips_to.strips}
    kept = []
    for s in strips_from.strips:
        o = other.get((s.root, s.expiry))
        if o is None:
            continue
        m = np.isin(s.strikes, o.strikes)
        kept.append(
            OptionStrip(
                s.root,
                s.expiry,
                s.strikes[m],
                s.call_bid[m],
                s.call_ask[m],
                s.put_bid[m],
                s.put_ask[m],
            )
        )
    return dataclasses.replace(
        strips_from, source="cboe quotes on yahoo strikes", strips=tuple(kept)
    )


def run(refresh: bool = False) -> dict:
    snaps = {s: obtain_snapshot(s, notify=notify, refresh=refresh) for s in ("cboe", "yahoo")}
    curve = obtain_curve(snaps["cboe"])
    official = official_closes(snaps["cboe"], NAMES)
    rebuilt = {}
    for source, snap in snaps.items():
        results = replicate(snap, curve, NAMES)
        rebuilt[source] = {}
        for name, r in results.items():
            assert isinstance(r, IndexResult), f"{source} {name}: {r}"
            rebuilt[source][name] = {
                "value": r.value,
                "official": official[name],
                "error": r.value - official[name],
                "near": None if r.near is None else f"{r.near.strip.root} {r.near.strip.expiry}",
                "next": f"{r.next.strip.root} {r.next.strip.expiry}",
                "options_in_next_term": r.next.variance.n_options,
                "forward_next": r.next.variance.forward,
            }
    on_yahoo_strikes = _values(_restrict_to(snaps["cboe"], snaps["yahoo"]), curve)
    yahoo_1615 = dataclasses.replace(
        snaps["yahoo"], asof=datetime.fromisoformat(snaps["cboe"].asof.isoformat())
    )
    decomposition = {
        "cboe_quotes_on_yahoo_strike_set": {
            n: {"value": v, "error": v - official[n]} for n, v in on_yahoo_strikes.items()
        },
        "yahoo_valued_at_1615": {
            n: {"value": v, "error": v - official[n]} for n, v in _values(yahoo_1615, curve).items()
        },
    }
    first = snaps["cboe"].by_root("SPXW")[0].expiry
    strikes_first = {s: snaps[s].strip("SPXW", first).strikes.size for s in snaps}
    payload = {
        "snapshots": {s: snapshot_meta(snap) for s, snap in snaps.items()},
        "treasury_curve_date": curve.curve_date.isoformat(),
        "official_closes": official,
        "rebuilt": rebuilt,
        "decomposition": decomposition,
        "strikes_in_first_spxw_expiry": strikes_first,
        "max_abs_error": {s: max(abs(v["error"]) for v in r.values()) for s, r in rebuilt.items()},
    }
    write_json("replication", payload)
    label = (
        f"Snapshots of {snaps['cboe'].asof:%Y-%m-%d}, valued at "
        f"{snaps['cboe'].asof.astimezone(ET):%H:%M} ET (Cboe feed) and "
        f"{snaps['yahoo'].asof.astimezone(ET):%H:%M} ET (Yahoo); official closes from Cboe."
    )
    errors = {s: {n: rebuilt[s][n]["error"] for n in NAMES} for s in ("cboe", "yahoo")}
    save_figure(build_replication_figure(official, errors, label), FIGURES / "replication")
    return payload


if __name__ == "__main__":
    out = run()
    print(out["max_abs_error"])
