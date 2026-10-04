"""End-to-end calendar analysis: snapshot -> strip -> clock fit -> bootstrap -> tables."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np

from .clock import ClockBootstrap, ClockData, ClockFit, bootstrap_clock, fit_clock
from .events import SCHEDULED_KINDS, Event, calendar_end, load_events
from .quotes import Snapshot
from .rates import YieldCurve
from .strip import ForwardVarianceStrip, build_strip
from .tradingcal import TRADING_DAYS_PER_YEAR

__all__ = ["CalendarResult", "CalendarRow", "EventRow", "build_calendar"]


@dataclass(frozen=True)
class CalendarRow:
    """One trading session of the variance calendar (volatilities annualised, 252 sessions)."""

    session: date
    fraction: float
    fwd_vol: float
    fwd_vol_lo: float
    fwd_vol_hi: float
    base_vol: float
    multiple: float
    multiple_lo: float
    multiple_hi: float
    observed_vol: float | None
    observed_vol_band: float | None
    weekend: bool
    events: tuple[str, ...]


@dataclass(frozen=True)
class EventRow:
    """Fitted extra variance of one event (or the per-name earnings term)."""

    key: str
    kind: str
    label: str
    session: date
    bump_var: float
    bump_lo: float
    bump_hi: float
    multiple: float
    multiple_lo: float
    multiple_hi: float
    implied_sd: float


@dataclass(frozen=True)
class CalendarResult:
    """Everything shown by ``varclock calendar`` and the HTML report."""

    snapshot: Snapshot
    curve: YieldCurve
    strip: ForwardVarianceStrip
    fit: ClockFit
    boot: ClockBootstrap
    events: tuple[Event, ...]
    rows: tuple[CalendarRow, ...]
    event_rows: tuple[EventRow, ...]
    n_boot: int
    seed: int
    notes: tuple[str, ...] = ()

    @property
    def asof(self) -> datetime:
        return self.snapshot.asof


_SHORT = {
    "NFP": "Jobs report",
    "ELECTION": "Election reaction",
    "FOMC": "FOMC",
    "CPI": "CPI",
}


def _session_labels(events: tuple[Event, ...] | list[Event]) -> dict[date, list[str]]:
    """Short labels per session; earnings reports are counted, not listed."""
    out: dict[date, list[str]] = {}
    earnings: dict[date, int] = {}
    for ev in events:
        if ev.kind == "EARNINGS":
            earnings[ev.session] = earnings.get(ev.session, 0) + 1
        else:
            out.setdefault(ev.session, []).append(_SHORT.get(ev.kind, ev.kind))
    for day, n in earnings.items():
        out.setdefault(day, []).append(f"earnings x{n}")
    return out


def build_calendar(
    snapshot: Snapshot,
    curve: YieldCurve,
    *,
    family: str = "meanrev",
    n_boot: int = 400,
    seed: int = 0,
    level: float = 0.9,
    max_expiries: int | None = None,
    coverage: str = "own",
) -> CalendarResult:
    """Run the full pipeline on one snapshot with an explicit random seed.

    ``coverage`` selects the strike windows of the strip (``build_strip``); the calendar warns
    in ``CalendarResult.notes`` when an event calendar file ends before the last expiry.
    """
    strip = build_strip(snapshot, curve, max_expiries=max_expiries, coverage=coverage)
    data = ClockData.from_strip(strip)
    start = data.asof.date()
    events = load_events(start=start, end=data.expiries[-1])
    fit = fit_clock(data, events, family=family)
    boot = bootstrap_clock(fit, strip, n_boot=n_boot, rng=np.random.default_rng(seed), level=level)

    cal = fit.calendar
    vol_lo, vol_hi = boot.vol_interval("session_variance")
    m_lo, m_hi = boot.interval("multiple")
    obs_index = {d: j for j, d in enumerate(data.expiries)}
    by_session = _session_labels(events)
    rows = []
    for s, d in enumerate(cal.dates):
        observed = observed_band = None
        j = obs_index.get(d)
        contiguous = j is not None and (
            j == 0 or cal.expiry_index[j] - cal.expiry_index[j - 1] == 1
        )
        if j is not None and contiguous and (j > 0 or cal.fraction[s] == 1.0):
            prev = strip.increments[j]
            observed = float(np.sqrt(max(prev, 0.0) * TRADING_DAYS_PER_YEAR))
            observed_band = float(
                np.sqrt(max(prev + strip.band[j], 0.0) * TRADING_DAYS_PER_YEAR) - observed
            )
        labels = list(dict.fromkeys(by_session.get(d, [])))
        rows.append(
            CalendarRow(
                session=d,
                fraction=float(cal.fraction[s]),
                fwd_vol=float(fit.session_vol[s]),
                fwd_vol_lo=float(vol_lo[s]),
                fwd_vol_hi=float(vol_hi[s]),
                base_vol=float(fit.base_vol[s]),
                multiple=float(fit.multiple[s]),
                multiple_lo=float(m_lo[s]),
                multiple_hi=float(m_hi[s]),
                observed_vol=observed,
                observed_vol_band=observed_band,
                weekend=bool(cal.weekend[s]),
                events=tuple(labels),
            )
        )
    b_lo, b_hi = boot.interval("bumps")
    em_lo, em_hi = boot.interval("event_multiple")
    event_rows = []
    for i, term in enumerate(fit.terms):
        s = term.sessions[0]
        total = fit.event_variance(i) if term.kind == "EARNINGS" else fit.session_variance[s]
        event_rows.append(
            EventRow(
                key=term.key,
                kind=term.kind,
                label=term.label,
                session=cal.dates[s],
                bump_var=float(fit.bumps[i]),
                bump_lo=float(b_lo[i]),
                bump_hi=float(b_hi[i]),
                multiple=fit.event_multiple(i),
                multiple_lo=float(em_lo[i]),
                multiple_hi=float(em_hi[i]),
                implied_sd=float(np.sqrt(total)),
            )
        )
    last = data.expiries[-1]
    notes = tuple(
        f"The committed {kind} calendar ends {calendar_end(kind)}, before the last expiry {last}: "
        "later events of that kind are not in the calendar."
        for kind in SCHEDULED_KINDS
        if calendar_end(kind) < last
    )
    return CalendarResult(
        snapshot,
        curve,
        strip,
        fit,
        boot,
        tuple(events),
        tuple(rows),
        tuple(event_rows),
        n_boot,
        seed,
        notes,
    )
