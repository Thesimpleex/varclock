"""Variance clock: a smooth base variance rate plus event bumps and weekend weights.

The strip of daily expirations gives the cumulative variance ``w_j`` accrued between the
snapshot and each expiration ``j``. The clock explains it as a sum over trading sessions ``s``,

    w_j = sum_{s <= j} v_s ,    v_s = f_s b(s) + W weekend_s + H holidays_s + sum_e J_e x_{e,s}

where ``b`` is a smooth, non-negative base variance per full session, ``f_s`` the fraction of
the session still ahead (1 except for a snapshot taken intraday), ``W`` and ``H`` the variance
carried by a weekend and by a weekday holiday, and ``J_e`` the extra variance of event ``e``
(``x_{e,s}`` is its indicator, or the number of names for the earnings term). Every parameter
enters linearly and is constrained to be non-negative, so the fit is a weighted non-negative
least-squares problem (Lawson and Hanson, 1974). Weights come from the bid-ask half-widths of
the totals with a 2% relative floor for errors the quotes do not reveal.

Two base families are provided: ``"meanrev"``, ``b(s) = b0 e^{-k s} + b_inf (1 - e^{-k s})``
(three parameters, the form of expected variance under mean-reverting dynamics), and
``"spline"``, a piecewise-linear curve with a roughness penalty.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
from scipy.optimize import nnls

from .events import Event
from .strip import QUOTE_CORRELATION, ForwardVarianceStrip
from .tradingcal import (
    TRADING_DAYS_PER_YEAR,
    business_minutes,
    expiry_datetime,
    next_trading_day,
    previous_trading_day,
    session_close,
    trading_days,
)

__all__ = [
    "ClockBootstrap",
    "ClockData",
    "ClockFit",
    "EventTerm",
    "SessionCalendar",
    "bootstrap_clock",
    "event_terms",
    "fit_clock",
]

RELATIVE_FLOOR = 0.02
KAPPA_GRID = (0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3)


@dataclass(frozen=True)
class ClockData:
    """Cumulative total variance at each expiration and its quote-based half-width."""

    asof: datetime
    expiries: tuple[date, ...]
    cumulative: np.ndarray
    band: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.expiries)
        if n < 3:
            raise ValueError("need at least three expirations")
        for name in ("cumulative", "band"):
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.shape != (n,):
                raise ValueError(f"{name} must have one entry per expiration")
            object.__setattr__(self, name, arr)
        if any(b <= a for a, b in zip(self.expiries, self.expiries[1:], strict=False)):
            raise ValueError("expiries must be strictly increasing")

    @classmethod
    def from_strip(cls, strip: ForwardVarianceStrip) -> ClockData:
        return cls(strip.asof, tuple(strip.expiries), strip.cumulative, strip.cumulative_band)

    @property
    def weights(self) -> np.ndarray:
        """Standard errors used to weight the fit: quote band with a relative floor."""
        return np.sqrt(self.band**2 + (RELATIVE_FLOOR * self.cumulative) ** 2)


@dataclass(frozen=True)
class SessionCalendar:
    """Trading sessions between the snapshot and the last expiration, with gap flags."""

    dates: tuple[date, ...]
    fraction: np.ndarray
    weekend: np.ndarray
    holidays: np.ndarray
    expiry_index: np.ndarray

    @classmethod
    def build(cls, asof: datetime, expiries: Sequence[date]) -> SessionCalendar:
        today = asof.date()
        intraday = expiries[0] == today
        first = today if intraday else next_trading_day(today)
        days = trading_days(first, expiries[-1])
        index = {d: i for i, d in enumerate(days)}
        missing = [e for e in expiries if e not in index]
        if missing:
            raise ValueError(f"expirations on non-trading days: {missing}")
        fraction = np.ones(len(days))
        if intraday:
            full = (session_close(today) - session_close(today).replace(hour=9, minute=30)).seconds
            left = business_minutes(asof, expiry_datetime(today, "PM")) * 60
            fraction[0] = min(1.0, left / full)
        weekend = np.zeros(len(days))
        holidays = np.zeros(len(days))
        for i, d in enumerate(days):
            if i == 0 and intraday:
                continue
            gap = [
                previous_trading_day(d) + timedelta(days=k + 1)
                for k in range((d - previous_trading_day(d)).days - 1)
            ]
            weekdays = {g.weekday() for g in gap}
            weekend[i] = 1.0 if {5, 6} <= weekdays else 0.0
            holidays[i] = float(sum(1 for g in gap if g.weekday() < 5))
        return cls(tuple(days), fraction, weekend, holidays, np.array([index[e] for e in expiries]))

    @property
    def n(self) -> int:
        return len(self.dates)


@dataclass(frozen=True)
class EventTerm:
    """One event bump parameter: the sessions it acts on and the multiplicity in each."""

    key: str
    kind: str
    label: str
    loadings: np.ndarray

    @property
    def sessions(self) -> list[int]:
        return [int(i) for i in np.flatnonzero(self.loadings)]


def event_terms(
    events: Sequence[Event], calendar: SessionCalendar, asof: datetime
) -> list[EventTerm]:
    """Parameter groups: one bump per FOMC, CPI, employment and election event, and one
    per-name bump shared by all earnings reports (multiplicity = number of names per session)."""
    index = {d: i for i, d in enumerate(calendar.dates)}
    terms: list[EventTerm] = []
    earnings = np.zeros(calendar.n)
    names: list[str] = []
    for ev in events:
        i = index.get(ev.session)
        if i is None:
            continue
        if ev.session == asof.date() and ev.moment <= asof:
            continue
        if ev.kind == "EARNINGS":
            earnings[i] += 1.0
            names.append(ev.label.split(" ")[0])
            continue
        loading = np.zeros(calendar.n)
        loading[i] = 1.0
        terms.append(EventTerm(f"{ev.kind} {ev.session.isoformat()}", ev.kind, ev.label, loading))
    if earnings.any():
        terms.append(EventTerm("EARNINGS per name", "EARNINGS", ", ".join(names), earnings))
    return terms


def _base_columns(
    family: str, calendar: SessionCalendar, kappa: float, knots: int
) -> tuple[np.ndarray, list[str]]:
    s = np.arange(calendar.n, dtype=float)
    f = calendar.fraction
    if family == "meanrev":
        e = np.exp(-kappa * s)
        return np.column_stack([e * f, (1.0 - e) * f]), ["b0", "b_inf"]
    if family == "spline":
        nodes = np.linspace(0.0, calendar.n - 1.0, knots)
        cols = [np.interp(s, nodes, np.eye(knots)[k]) * f for k in range(knots)]
        return np.column_stack(cols), [f"knot{k}" for k in range(knots)]
    raise ValueError(f"unknown base family {family!r}")


def _penalty(knots: int, strength: float, scale: float, n_cols: int) -> np.ndarray:
    rows = np.zeros((max(knots - 2, 0), n_cols))
    for i in range(rows.shape[0]):
        rows[i, i], rows[i, i + 1], rows[i, i + 2] = 1.0, -2.0, 1.0
    return np.sqrt(strength) * rows / scale


@dataclass(frozen=True)
class ClockFit:
    """Fitted variance clock.

    ``base`` is the base variance per *full* session; ``session_variance`` the fitted variance of
    each session including weekend, holiday and event terms. Variances are in total-variance
    units (a vol of 16% a year is ``0.16**2 / 252`` per session).
    """

    data: ClockData
    calendar: SessionCalendar
    terms: tuple[EventTerm, ...]
    family: str
    kappa: float
    knots: int
    penalty: float
    param_names: tuple[str, ...]
    theta: np.ndarray
    base: np.ndarray
    weekend: float
    holiday: float
    bumps: np.ndarray
    fitted_cumulative: np.ndarray
    standardized_residuals: np.ndarray

    @property
    def session_variance(self) -> np.ndarray:
        cal = self.calendar
        v = self.base * cal.fraction + self.weekend * cal.weekend + self.holiday * cal.holidays
        for term, j in zip(self.terms, self.bumps, strict=True):
            v = v + j * term.loadings
        return v

    @property
    def multiple(self) -> np.ndarray:
        """Session variance divided by the base variance of a full session."""
        return self.session_variance / (self.base * self.calendar.fraction)

    @property
    def base_vol(self) -> np.ndarray:
        """Annualised base volatility (252 sessions a year)."""
        return np.sqrt(self.base * TRADING_DAYS_PER_YEAR)

    @property
    def session_vol(self) -> np.ndarray:
        """Annualised volatility of each session as a full session would show it."""
        return np.sqrt(self.session_variance / self.calendar.fraction * TRADING_DAYS_PER_YEAR)

    @property
    def weekend_ratio(self) -> float:
        """Weekend variance relative to the base of the sessions that follow weekends."""
        mask = self.calendar.weekend > 0
        return float(self.weekend / np.mean(self.base[mask])) if mask.any() else float("nan")

    def event_variance(self, i: int) -> float:
        """Variance of the first session of event ``i`` including the event.

        For the per-name earnings term this is the base variance plus the bump of *one* report,
        whatever number of names report in that session; other events add their bump to the
        session variance, including any weekend or holiday term of that session.
        """
        term = self.terms[i]
        s = term.sessions[0]
        cal = self.calendar
        base = self.base[s] * cal.fraction[s]
        if term.kind == "EARNINGS":
            return float(base + self.bumps[i])
        extra = self.weekend * cal.weekend[s] + self.holiday * cal.holidays[s]
        return float(base + extra + self.bumps[i] * term.loadings[s])

    def event_multiple(self, i: int) -> float:
        """``event_variance(i)`` over that session's base variance."""
        s = self.terms[i].sessions[0]
        return self.event_variance(i) / float(self.base[s] * self.calendar.fraction[s])

    @property
    def chi2_dof(self) -> float:
        dof = max(self.data.cumulative.size - np.count_nonzero(self.theta), 1)
        return float(np.sum(self.standardized_residuals**2) / dof)


def _solve(
    data: ClockData,
    calendar: SessionCalendar,
    terms: Sequence[EventTerm],
    family: str,
    kappa: float,
    knots: int,
    penalty: float,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    base, base_names = _base_columns(family, calendar, kappa, knots)
    extra = [calendar.weekend]
    names = [*base_names, "weekend"]
    if calendar.holidays.any():
        extra.append(calendar.holidays)
        names.append("holiday")
    for t in terms:
        extra.append(t.loadings)
        names.append(t.key)
    design = np.column_stack([base, *extra])
    mask = (np.arange(calendar.n)[None, :] <= calendar.expiry_index[:, None]).astype(float)
    a = mask @ design
    h = data.weights
    aw, yw = a / h[:, None], y / h
    if family == "spline" and penalty > 0 and knots > 2:
        scale = RELATIVE_FLOOR * data.cumulative[-1] / calendar.n
        pen = _penalty(knots, penalty, scale, design.shape[1])
        aw = np.vstack([aw, pen])
        yw = np.concatenate([yw, np.zeros(pen.shape[0])])
    theta, rnorm = nnls(aw, yw, maxiter=50 * aw.shape[1])
    return theta, a @ theta, names, float(rnorm)


def _fit_arrays(
    data: ClockData,
    calendar: SessionCalendar,
    terms: Sequence[EventTerm],
    family: str,
    kappa: float | None,
    knots: int,
    penalty: float,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str], float, float]:
    grid: Sequence[float] = (kappa,) if kappa is not None else KAPPA_GRID
    if family == "spline":
        grid = (0.0,)
    best = None
    for k in grid:
        theta, fitted, names, rnorm = _solve(data, calendar, terms, family, k, knots, penalty, y)
        if best is None or rnorm < best[3]:
            best = (theta, fitted, names, rnorm, float(k))
    assert best is not None
    return best


def fit_clock(
    data: ClockData,
    events: Sequence[Event] = (),
    *,
    family: str = "meanrev",
    kappa: float | None = None,
    knots: int = 8,
    penalty: float = 1.0,
) -> ClockFit:
    """Fit the variance clock to cumulative variances by weighted non-negative least squares.

    Parameters
    ----------
    family
        ``"meanrev"`` (base ``b0 e^{-k s} + b_inf (1 - e^{-k s})`` with ``k`` profiled over a grid
        unless ``kappa`` is given) or ``"spline"`` (``knots`` piecewise-linear nodes with a
        second-difference penalty of strength ``penalty``).
    """
    calendar = SessionCalendar.build(data.asof, data.expiries)
    terms = event_terms(events, calendar, data.asof)
    theta, fitted, names, _, k = _fit_arrays(
        data, calendar, terms, family, kappa, knots, penalty, data.cumulative
    )
    return _assemble(data, calendar, terms, family, k, knots, penalty, theta, fitted, names)


def _assemble(
    data: ClockData,
    calendar: SessionCalendar,
    terms: Sequence[EventTerm],
    family: str,
    kappa: float,
    knots: int,
    penalty: float,
    theta: np.ndarray,
    fitted: np.ndarray,
    names: Sequence[str],
) -> ClockFit:
    base_cols, base_names = _base_columns(family, calendar, kappa, knots)
    nb = len(base_names)
    f = np.where(calendar.fraction > 0, calendar.fraction, 1.0)
    base = (base_cols @ theta[:nb]) / f
    pos = nb
    weekend = float(theta[pos])
    pos += 1
    holiday = 0.0
    if "holiday" in names:
        holiday = float(theta[pos])
        pos += 1
    bumps = np.asarray(theta[pos : pos + len(terms)])
    resid = (data.cumulative - fitted) / data.weights
    return ClockFit(
        data=data,
        calendar=calendar,
        terms=tuple(terms),
        family=family,
        kappa=kappa,
        knots=knots,
        penalty=penalty,
        param_names=tuple(names),
        theta=theta,
        base=base,
        weekend=weekend,
        holiday=holiday,
        bumps=bumps,
        fitted_cumulative=fitted,
        standardized_residuals=resid,
    )


@dataclass(frozen=True)
class ClockBootstrap:
    """Bootstrap replicates of the clock parameters (percentile intervals)."""

    level: float
    base: np.ndarray
    session_variance: np.ndarray
    multiple: np.ndarray
    weekend_ratio: np.ndarray
    bumps: np.ndarray
    event_multiple: np.ndarray

    def interval(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper percentile bounds of a replicate array along its first axis."""
        a = (1.0 - self.level) / 2.0
        draws = getattr(self, name)
        return np.nanquantile(draws, a, axis=0), np.nanquantile(draws, 1.0 - a, axis=0)

    def vol_interval(self, name: str = "session_variance") -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.interval(name)
        return np.sqrt(lo * TRADING_DAYS_PER_YEAR), np.sqrt(hi * TRADING_DAYS_PER_YEAR)


def bootstrap_clock(
    fit: ClockFit,
    strip: ForwardVarianceStrip | None = None,
    n_boot: int = 400,
    rng: np.random.Generator | None = None,
    level: float = 0.9,
    progress: Callable[[int], None] | None = None,
    *,
    resample_quotes: bool = True,
    resample_residuals: bool = True,
    quote_correlation: float = QUOTE_CORRELATION,
) -> ClockBootstrap:
    """Parametric-plus-residual bootstrap of the fit.

    Each replicate perturbs the data around the fitted curve by (i) redrawing every option
    price inside its bid-ask band (if ``strip`` is given), with the positions of one expiration
    correlated by ``quote_correlation`` (see ``ForwardVarianceStrip.resampled_cumulative``), and
    (ii) adding residuals resampled with replacement from the standardized residuals, then
    refits the same model (re-profiling ``kappa`` for the mean-reverting base). Intervals are
    percentiles. ``resample_quotes`` and ``resample_residuals`` switch the two components off
    to see which one drives the interval widths.
    """
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    if n_boot < 20:
        raise ValueError("n_boot must be at least 20")
    rng = np.random.default_rng() if rng is None else rng
    data = fit.data
    n_terms = len(fit.terms)
    base = np.empty((n_boot, fit.calendar.n))
    sess = np.empty_like(base)
    mult = np.empty_like(base)
    wk = np.empty(n_boot)
    bumps = np.empty((n_boot, n_terms))
    emult = np.empty((n_boot, n_terms))
    h = data.weights
    use_quotes = strip is not None and resample_quotes
    if not use_quotes and not resample_residuals:
        raise ValueError("at least one noise component must be switched on")
    resid = fit.standardized_residuals
    kappa = None if fit.family == "meanrev" else fit.kappa
    for b in range(n_boot):
        y = fit.fitted_cumulative.copy()
        if use_quotes:
            y += strip.resampled_cumulative(rng, quote_correlation) - strip.cumulative
        if resample_residuals:
            y += h * rng.choice(resid, size=resid.size, replace=True)
        theta, fitted, names, _, k = _fit_arrays(
            data, fit.calendar, fit.terms, fit.family, kappa, fit.knots, fit.penalty, y
        )
        rep = _assemble(
            data,
            fit.calendar,
            fit.terms,
            fit.family,
            k,
            fit.knots,
            fit.penalty,
            theta,
            fitted,
            names,
        )
        base[b], sess[b], mult[b] = rep.base, rep.session_variance, rep.multiple
        wk[b] = rep.weekend_ratio
        bumps[b] = rep.bumps
        emult[b] = [rep.event_multiple(i) for i in range(n_terms)]
        if progress is not None:
            progress(b)
    return ClockBootstrap(level, base, sess, mult, wk, bumps, emult)
