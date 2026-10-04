"""Experiment: arbitrage-free LP bands for each expiry's variance and for forward variances."""

from __future__ import annotations

import numpy as np
from common import notify, snapshot_meta, write_json

from varclock.bounds import increment_band, strip_bands
from varclock.loaders import obtain_curve, obtain_snapshot
from varclock.strip import build_strip

MAX_EXPIRIES = 15


def run() -> dict:
    out: dict = {}
    for source in ("cboe", "yahoo"):
        snap = obtain_snapshot(source, notify=notify)
        strip = build_strip(snap, obtain_curve(snap), max_expiries=MAX_EXPIRIES)
        bands = strip_bands(snap, strip)
        rows, prev = [], None
        for p, b in zip(strip.points, bands, strict=True):
            t = p.T

            def vol(w: float, t: float = t) -> float:
                return float(np.sqrt(max(w, 0.0) / t))

            row = {
                "expiry": p.expiry.isoformat(),
                "vol_mid": vol(b.mid),
                "vol_lp": [vol(b.lo), vol(b.hi)],
                "vol_bid_ask": [vol(b.naive_lo), vol(b.naive_hi)],
                "width_ratio_lp_to_bid_ask": b.width / b.naive_width,
                "widen_needed_index_points": b.widen,
            }
            if prev is not None:
                lo, hi = increment_band(prev, b)
                row["forward_variance_step"] = b.mid - prev.mid
                row["forward_variance_step_lp"] = [lo, hi]
                row["forward_variance_step_bid_ask"] = [
                    b.naive_lo - prev.naive_hi,
                    b.naive_hi - prev.naive_lo,
                ]
            rows.append(row)
            prev = b
        ratios = [r["width_ratio_lp_to_bid_ask"] for r in rows]
        out[source] = {
            "snapshot": snapshot_meta(snap),
            "expiries": rows,
            "mean_width_ratio": float(np.mean(ratios)),
            "min_width_ratio": float(np.min(ratios)),
            "expiries_needing_widening": int(sum(r["widen_needed_index_points"] > 0 for r in rows)),
            "max_widening_index_points": float(max(r["widen_needed_index_points"] for r in rows)),
        }
    write_json("bounds", out)
    return out


if __name__ == "__main__":
    for s, v in run().items():
        print(
            s, v["mean_width_ratio"], v["expiries_needing_widening"], v["max_widening_index_points"]
        )
