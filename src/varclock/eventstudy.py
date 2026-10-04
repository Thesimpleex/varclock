"""Event history study: next-session implied variance versus the realised move.

The one-day index (VIX1D), read at the 16:15 close of the previous session, prices the variance
of the next session in business minutes:

    implied variance = (VIX1D / 100)^2 * M / 102,060,

with ``M`` the business minutes from 16:15 to the next close (390 on an ordinary session). The
realised counterpart is the squared close-to-close log return of the S&P 500 on the event day.
The difference ``implied - realised`` is the event variance risk premium; with only a few
dozen events per type, every figure comes with a bootstrap interval and a count.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date

import numpy as np

from .events import Event
from .tradingcal import (
    BUSINESS_MINUTES_PER_YEAR,
    business_minutes,
    expiry_datetime,
    is_trading_day,
    previous_trading_day,
    session_end,
)

__all__ = [
    "CONTROL_BLOCK",
    "EventStudy",
    "StudyRow",
    "block_indices",
    "bootstrap_mean",
    "implied_session_variance",
    "run_event_study",
]


def implied_session_variance(vix1d_close: float, previous_day: date, session: date) -> float:
    """Implied variance of ``session`` from the VIX1D close of ``previous_day`` (decimal units)."""
    minutes = business_minutes(session_end(previous_day), expiry_datetime(session, "PM"))
    return (vix1d_close / 100.0) ** 2 * minutes / BUSINESS_MINUTES_PER_YEAR


@dataclass(frozen=True)
class StudyRow:
    """One event day: implied and realised session variance."""

    session: date
    kind: str
    label: str
    vix1d_prev: float
    implied_var: float
    log_return: float
    gap_days: int

    @property
    def realised_var(self) -> float:
        return self.log_return**2

    @property
    def implied_sd(self) -> float:
        return float(np.sqrt(self.implied_var))

    @property
    def z(self) -> float:
        return self.log_return / self.implied_sd


#: Block length (sessions) of the circular block bootstrap used for the ordinary-day sample,
#: which is a run of consecutive days and shows volatility clustering.
CONTROL_BLOCK = 20


def block_indices(n: int, rng: np.random.Generator, n_boot: int, block: int) -> np.ndarray:
    """Indices ``(n_boot, n)`` of a circular block bootstrap with blocks of ``block`` values."""
    n_blocks = -(-n // block)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    index = (starts[:, :, None] + np.arange(block)) % n
    return index.reshape(n_boot, -1)[:, :n]


def bootstrap_mean(
    values: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 10_000,
    level: float = 0.95,
    block: int = 1,
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap interval of the mean.

    ``block > 1`` resamples circular blocks of that many consecutive values, which keeps serial
    dependence (volatility clustering) in the resampled series; ``block = 1`` is the plain
    bootstrap for observations that are not adjacent in time.
    """
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        raise ValueError("need at least two observations")
    if block < 1:
        raise ValueError("block must be at least 1")
    if block == 1:
        draws = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    else:
        draws = values[block_indices(values.size, rng, n_boot, block)].mean(axis=1)
    a = (1.0 - level) / 2.0
    return float(values.mean()), float(np.quantile(draws, a)), float(np.quantile(draws, 1 - a))


@dataclass(frozen=True)
class EventStudy:
    """Event rows plus the ordinary-day control sample."""

    rows: tuple[StudyRow, ...]
    control_implied: np.ndarray
    control_return: np.ndarray
    start: date
    end: date

    def by_kind(self, kind: str | Iterable[str]) -> list[StudyRow]:
        kinds = {kind} if isinstance(kind, str) else set(kind)
        return [r for r in self.rows if r.kind in kinds]

    def summarise(
        self,
        rows: list[StudyRow] | None = None,
        rng: np.random.Generator | None = None,
        n_boot: int = 10_000,
    ) -> dict[str, float | int]:
        """Counts, means and bootstrap intervals for a set of event rows."""
        rows = list(self.rows) if rows is None else rows
        rng = np.random.default_rng(0) if rng is None else rng
        imp = np.array([r.implied_var for r in rows])
        real = np.array([r.realised_var for r in rows])
        ret = np.array([r.log_return for r in rows])
        sd = np.sqrt(imp)
        out: dict[str, float | int] = {"n": len(rows)}
        for name, values in (
            ("implied_var", imp),
            ("realised_var", real),
            ("premium_var", imp - real),
            ("z_squared", (ret / sd) ** 2),
            ("realised_abs", np.abs(ret)),
            ("implied_abs", sd * np.sqrt(2 / np.pi)),
        ):
            m, lo, hi = bootstrap_mean(values, rng, n_boot)
            out[name] = m
            out[f"{name}_lo"] = lo
            out[f"{name}_hi"] = hi
        pairs = np.column_stack([imp, real])
        draws = rng.integers(0, len(rows), size=(n_boot, len(rows)))
        ratios = pairs[draws, 1].mean(axis=1) / pairs[draws, 0].mean(axis=1)
        out["ratio"] = float(real.mean() / imp.mean())
        out["ratio_lo"] = float(np.quantile(ratios, 0.025))
        out["ratio_hi"] = float(np.quantile(ratios, 0.975))
        out["implied_sd_mean"] = float(sd.mean())
        out["share_exceeding_implied_sd"] = float(np.mean(np.abs(ret) > sd))
        return out

    def summarise_control(
        self, rng: np.random.Generator | None = None, n_boot: int = 10_000
    ) -> dict[str, float | int]:
        """The same statistics for ordinary days, with a circular block bootstrap."""
        rng = np.random.default_rng(0) if rng is None else rng
        imp, ret = self.control_implied, self.control_return
        out: dict[str, float | int] = {"n": int(imp.size)}
        real = ret**2
        for name, values in (
            ("implied_var", imp),
            ("realised_var", real),
            ("premium_var", imp - real),
            ("z_squared", real / imp),
        ):
            m, lo, hi = bootstrap_mean(values, rng, n_boot, block=CONTROL_BLOCK)
            out[name] = m
            out[f"{name}_lo"] = lo
            out[f"{name}_hi"] = hi
        index = block_indices(imp.size, rng, n_boot, CONTROL_BLOCK)
        ratios = real[index].mean(axis=1) / imp[index].mean(axis=1)
        out["ratio"] = float(real.mean() / imp.mean())
        out["ratio_lo"] = float(np.quantile(ratios, 0.025))
        out["ratio_hi"] = float(np.quantile(ratios, 0.975))
        out["implied_sd_mean"] = float(np.sqrt(imp).mean())
        return out


def run_event_study(
    events: Iterable[Event],
    vix1d: Mapping[date, float],
    spx: Mapping[date, float],
    start: date = date(2022, 5, 16),
    kinds: tuple[str, ...] = ("CPI", "FOMC", "NFP"),
) -> EventStudy:
    """Match events with the previous-day VIX1D close and the S&P 500 close-to-close return.

    Events whose previous trading day has no VIX1D close, or without S&P 500 closes on both
    days, are skipped; ordinary days (no event of the studied kinds) form the control sample.
    """
    end = max(set(vix1d) & set(spx))
    event_sessions: dict[date, list[Event]] = {}
    for ev in events:
        if ev.kind in kinds:
            event_sessions.setdefault(ev.session, []).append(ev)

    rows: list[StudyRow] = []
    control_imp: list[float] = []
    control_ret: list[float] = []
    day = start
    while day <= end:
        if is_trading_day(day):
            prev = previous_trading_day(day)
            if prev in vix1d and day in spx and prev in spx:
                implied = implied_session_variance(vix1d[prev], prev, day)
                ret = float(np.log(spx[day] / spx[prev]))
                if day in event_sessions:
                    for ev in event_sessions[day]:
                        rows.append(
                            StudyRow(
                                day, ev.kind, ev.label, vix1d[prev], implied, ret, (day - prev).days
                            )
                        )
                else:
                    control_imp.append(implied)
                    control_ret.append(ret)
        day = date.fromordinal(day.toordinal() + 1)
    return EventStudy(tuple(rows), np.array(control_imp), np.array(control_ret), start, end)
