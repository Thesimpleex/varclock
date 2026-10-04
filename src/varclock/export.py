"""Machine-readable (JSON-ready) forms of the calendar, the event list and the event study."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

import numpy as np

from .analysis import CalendarResult
from .eventstudy import EventStudy

__all__ = ["calendar_records", "study_records", "upcoming_records"]


def calendar_records(result: CalendarResult) -> dict[str, Any]:
    """Snapshot metadata, one record per session and one per fitted event.

    Volatilities are annualised over 252 sessions, in decimal units; ``extra_variance`` is the
    event variance in decimal total-variance units and ``extra_variance_pct_sq`` the same in
    squared percent.
    """
    snap = result.snapshot
    return {
        "snapshot": {
            "symbol": snap.symbol,
            "source": snap.source,
            "valuation_time_et": snap.asof.isoformat(),
            "downloaded_utc": snap.fetched_at.isoformat(),
            "treasury_curve_date": result.curve.curve_date.isoformat(),
            "daily_expiries": len(result.strip.points),
            "bootstrap_draws": result.n_boot,
            "seed": result.seed,
            "base_family": result.fit.family,
            "strike_coverage": result.strip.coverage,
        },
        "sessions": [
            {
                "session": r.session.isoformat(),
                "weekend": r.weekend,
                "events": list(r.events),
                "fwd_vol": r.fwd_vol,
                "fwd_vol_90pct": [r.fwd_vol_lo, r.fwd_vol_hi],
                "base_vol": r.base_vol,
                "multiple_of_base": r.multiple,
                "multiple_90pct": [r.multiple_lo, r.multiple_hi],
                "strip_vol": r.observed_vol,
            }
            for r in result.rows
        ],
        "events": [
            {
                "key": e.key,
                "kind": e.kind,
                "label": e.label,
                "session": e.session.isoformat(),
                "extra_variance": e.bump_var,
                "extra_variance_pct_sq": e.bump_var * 1e4,
                "extra_variance_90pct": [e.bump_lo, e.bump_hi],
                "multiple_of_base": e.multiple,
                "multiple_90pct": [e.multiple_lo, e.multiple_hi],
                "one_sd_move": e.implied_sd,
            }
            for e in result.event_rows
        ],
        "notes": list(result.notes),
    }


def upcoming_records(events: Sequence[Any], asof: date, horizon: int) -> list[dict[str, Any]]:
    """Events whose session lies within ``horizon`` days after ``asof``."""
    return [
        {
            "date": e.day.isoformat(),
            "session": e.session.isoformat(),
            "time_et": e.time_et.isoformat(timespec="minutes"),
            "kind": e.kind,
            "label": e.label,
        }
        for e in events
        if 0 <= (e.session - asof).days <= horizon
    ]


def study_records(study: EventStudy, kinds: Sequence[str], seed: int = 0) -> dict[str, Any]:
    """Summary statistics of the event study per kind and for ordinary days."""

    def clean(summary: dict[str, float | int]) -> dict[str, float | int]:
        return {k: (int(v) if isinstance(v, int) else float(v)) for k, v in summary.items()}

    out: dict[str, Any] = {"period": [study.start.isoformat(), study.end.isoformat()]}
    for kind in kinds:
        rows = study.by_kind(kind)
        if rows:
            out[kind] = clean(study.summarise(rows, np.random.default_rng(seed)))
    out["ordinary_days"] = clean(study.summarise_control(np.random.default_rng(seed)))
    return out
