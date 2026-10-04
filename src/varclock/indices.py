"""Replication of the Cboe S&P 500 volatility indices from one option-chain snapshot.

Configurations follow the published methodologies: VIX (30 days, bracket method, standard
monthly and end-of-week SPXW series), VIX9D (9 days, nearest-term method with a two-day
exclusion), VIX3M (93 days, bracket method, standard monthly and end-of-month SPXW series)
and VIX1D (one business day, business-minute clock, PM-settled SPXW series).

The module reproduces the closing level of each index (computed at 16:15 ET); it does not
implement the intraday level and series filters of the official calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .quotes import OptionStrip, Snapshot
from .rates import YieldCurve
from .tradingcal import (
    BUSINESS_MINUTES_PER_YEAR,
    MINUTES_PER_CALENDAR_YEAR,
    SESSION_MINUTES,
    business_minutes,
    calendar_minutes,
    last_trading_day_of_month,
    last_trading_day_of_week,
    next_trading_day,
)
from .variance import TermVariance, VarianceError, interpolate_variance, single_term_variance

__all__ = [
    "INDEX_SPECS",
    "IndexResult",
    "IndexSpec",
    "IndexUnavailableError",
    "TermUsed",
    "compute_index",
    "eligible_strips",
    "replicate",
]


class IndexUnavailableError(ValueError):
    """The snapshot does not contain the series needed for an index."""


@dataclass(frozen=True)
class IndexSpec:
    """Methodology attributes of one volatility index."""

    name: str
    target: float
    clock: str
    universe: str
    method: str
    min_days: float = 0.0

    @property
    def minutes_year(self) -> int:
        return MINUTES_PER_CALENDAR_YEAR if self.clock == "calendar" else BUSINESS_MINUTES_PER_YEAR

    @property
    def minutes_target(self) -> float:
        return self.target * 1440.0 if self.clock == "calendar" else float(self.target)


INDEX_SPECS: dict[str, IndexSpec] = {
    "VIX": IndexSpec("VIX", 30, "calendar", "end_of_week", "bracket"),
    "VIX9D": IndexSpec("VIX9D", 9, "calendar", "end_of_week", "nearest", min_days=2.0),
    "VIX3M": IndexSpec("VIX3M", 93, "calendar", "end_of_month", "bracket"),
    "VIX1D": IndexSpec("VIX1D", SESSION_MINUTES, "business", "pm_all", "nearest"),
}


@dataclass(frozen=True)
class TermUsed:
    """One expiration entering an index: the strip, its clock and its variance."""

    strip: OptionStrip
    minutes: int
    T: float
    rate: float
    variance: TermVariance


@dataclass(frozen=True)
class IndexResult:
    """Replicated index level with the near and next terms behind it."""

    spec: IndexSpec
    value: float
    near: TermUsed | None
    next: TermUsed
    warnings: tuple[str, ...] = field(default_factory=tuple)


def eligible_strips(snapshot: Snapshot, universe: str) -> list[OptionStrip]:
    """Strips that may serve as near or next term for the given contract universe."""
    am = {s.expiry for s in snapshot.strips if s.root == "SPX"}

    def listed(s: OptionStrip) -> bool:
        if universe == "pm_all":
            return s.root == "SPXW"
        if s.root == "SPX":
            return True
        if s.expiry in am:
            return False
        last = last_trading_day_of_week if universe == "end_of_week" else last_trading_day_of_month
        return s.expiry == last(s.expiry)

    return sorted((s for s in snapshot.strips if listed(s)), key=lambda s: s.expiry_time)


def _minutes(spec: IndexSpec, snapshot: Snapshot, strip: OptionStrip) -> int:
    if spec.clock == "calendar":
        return calendar_minutes(snapshot.asof, strip.expiry_time)
    return business_minutes(snapshot.asof, strip.expiry_time)


def _select_terms(
    spec: IndexSpec, snapshot: Snapshot, strips: list[OptionStrip]
) -> tuple[OptionStrip | None, OptionStrip]:
    live = [s for s in strips if calendar_minutes(snapshot.asof, s.expiry_time) > 0]
    if spec.method == "bracket":
        if not live:
            raise IndexUnavailableError(f"{spec.name}: no unexpired series in the snapshot")
        days = [calendar_minutes(snapshot.asof, s.expiry_time) / 1440.0 for s in live]
        within = [i for i, d in enumerate(days) if d <= spec.target]
        i_near = within[-1] if within else 0
        if i_near + 1 >= len(live):
            raise IndexUnavailableError(f"{spec.name}: no series expiring after the near term")
        return live[i_near], live[i_near + 1]
    if spec.name == "VIX1D":
        today = snapshot.asof.date()
        near = [s for s in live if s.expiry == today]
        later = [s for s in live if s.expiry > today]
        if not later or later[0].expiry != next_trading_day(today):
            raise IndexUnavailableError("VIX1D: no SPXW series for the next trading day")
        return (near[0] if near else None), later[0]
    kept = [
        s for s in live if calendar_minutes(snapshot.asof, s.expiry_time) / 1440.0 >= spec.min_days
    ]
    if len(kept) < 2:
        raise IndexUnavailableError(f"{spec.name}: fewer than two eligible series")
    return kept[0], kept[1]


def _term(spec: IndexSpec, snapshot: Snapshot, curve: YieldCurve, strip: OptionStrip) -> TermUsed:
    minutes = _minutes(spec, snapshot, strip)
    if minutes <= 0:
        raise IndexUnavailableError(f"{spec.name}: {strip.root} {strip.expiry} has no time left")
    T = minutes / spec.minutes_year
    rate = curve.rate(strip.expiry)
    try:
        variance = single_term_variance(strip, T, rate)
    except VarianceError as exc:
        raise IndexUnavailableError(f"{spec.name}: {strip.root} {strip.expiry}: {exc}") from exc
    return TermUsed(strip, minutes, T, rate, variance)


def compute_index(snapshot: Snapshot, curve: YieldCurve, name: str) -> IndexResult:
    """Closing-level replication of ``name`` (``VIX``, ``VIX9D``, ``VIX3M`` or ``VIX1D``)."""
    try:
        spec = INDEX_SPECS[name.upper()]
    except KeyError:
        raise ValueError(f"unknown index {name!r}; choose from {sorted(INDEX_SPECS)}") from None
    strips = eligible_strips(snapshot, spec.universe)
    near_strip, next_strip = _select_terms(spec, snapshot, strips)
    nxt = _term(spec, snapshot, curve, next_strip)
    near = _term(spec, snapshot, curve, near_strip) if near_strip is not None else None
    if near is not None and spec.name == "VIX1D" and nxt.minutes < spec.minutes_target:
        near = None  # no extrapolation: the next term alone, as in the methodology
    warnings: list[str] = []
    if near is not None and spec.name == "VIX1D" and near.minutes < 60:
        warnings.append(
            "near term has under 60 business minutes left: the official index freezes its "
            "variance, this snapshot cannot"
        )
    if near is not None and not near.minutes <= spec.minutes_target <= nxt.minutes:
        warnings.append("the target maturity lies outside the two terms: extrapolation")
    value = interpolate_variance(
        None if near is None else near.variance.sigma2,
        nxt.variance.sigma2,
        None if near is None else near.minutes,
        nxt.minutes,
        spec.minutes_target,
        spec.minutes_year,
    )
    return IndexResult(spec, value, near, nxt, tuple(warnings))


def replicate(
    snapshot: Snapshot,
    curve: YieldCurve,
    names: tuple[str, ...] = ("VIX", "VIX9D", "VIX3M", "VIX1D"),
) -> dict[str, IndexResult | IndexUnavailableError]:
    """Replicate several indices from one snapshot; failures are returned, not raised."""
    out: dict[str, IndexResult | IndexUnavailableError] = {}
    for name in names:
        try:
            out[name] = compute_index(snapshot, curve, name)
        except IndexUnavailableError as exc:
            out[name] = exc
    return out
