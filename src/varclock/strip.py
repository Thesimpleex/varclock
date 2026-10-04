"""Forward-variance strip over consecutive daily SPXW expirations.

With total variance ``w_j = sigma_j^2 T_j`` for the expiration ``j`` the forward variance
between two adjacent expirations is ``v_j = (w_{j+1} - w_j) / (T_{j+1} - T_j)``. A negative
value is a calendar-spread arbitrage in the mid quotes.

Cboe's truncation rule stops each strip at its own pair of zero bids, so adjacent expirations
cover different strike ranges. Two treatments of coverage are available (``coverage``):

* ``"own"`` (default): every total is computed on the expiration's own selected strikes and the
  increment is the difference of two totals. The wing variance that the truncation leaves out is
  a small, slowly varying amount.
* ``"common"``: each increment is computed on the strike window common to the two expirations.
  This removes the later expiration's wing variance from the increment, and the losses add up
  along the strip; on synthetic chains with a known truth it biases cumulative variance down and
  the weekend weight up (see ``tests/test_strip.py``), so it is kept as a sensitivity check.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
from scipy.special import ndtr

from .quotes import OptionStrip, Snapshot
from .rates import YieldCurve
from .tradingcal import MINUTES_PER_CALENDAR_YEAR, calendar_minutes, next_trading_day
from .variance import TermVariance, VarianceError, single_term_variance, strike_spacing

__all__ = [
    "QUOTE_CORRELATION",
    "ForwardVarianceStrip",
    "StripPoint",
    "build_strip",
    "window_total_variance",
]

#: Correlation of the bid-ask positions of the prices of one expiration in the bootstrap.
QUOTE_CORRELATION = 0.5


@dataclass(frozen=True)
class StripPoint:
    """One daily expiration in the strip."""

    expiry: date
    minutes: int
    T: float
    rate: float
    term: TermVariance

    @property
    def total_variance(self) -> float:
        return self.term.total_variance

    @property
    def band_half_width(self) -> float:
        """Half-width of the bid-ask band of the total variance (own strike range)."""
        return _band(self.term, np.ones(self.term.strikes.size, dtype=bool))[1]


def _window_weights(term: TermVariance, mask: np.ndarray) -> np.ndarray:
    """``2 dK / K^2 exp(RT)`` recomputed on the strikes selected by ``mask``."""
    k = term.strikes[mask]
    return 2.0 * strike_spacing(k) / k**2 * np.exp(term.rate * term.T)


def _window_mask(term: TermVariance, lo: float, hi: float) -> np.ndarray:
    mask = (term.strikes >= lo) & (term.strikes <= hi)
    if mask.sum() < 3 or not (lo <= term.k0 <= hi):
        return np.ones(term.strikes.size, dtype=bool)
    return mask


def _band(term: TermVariance, mask: np.ndarray) -> tuple[float, float]:
    """Total variance on ``mask`` and the half-width of its bid-ask band."""
    w = _window_weights(term, mask)
    total = float(w @ term.prices[mask] - (term.forward / term.k0 - 1.0) ** 2)
    half = float(w @ (term.price_hi[mask] - term.price_lo[mask]) / 2.0)
    growth = np.exp(term.rate * term.T)
    d_forward = growth * (
        (term.atm_call_band[1] - term.atm_call_band[0]) / 2.0
        + (term.atm_put_band[1] - term.atm_put_band[0]) / 2.0
    )
    half += abs(2.0 * (term.forward / term.k0 - 1.0) / term.k0) * d_forward
    return total, half


def window_total_variance(term: TermVariance, lo: float, hi: float) -> float:
    """Total variance of ``term`` restricted to strikes in ``[lo, hi]`` (edges use one-sided dK)."""
    return _band(term, _window_mask(term, lo, hi))[0]


@dataclass(frozen=True)
class ForwardVarianceStrip:
    """Daily forward variances, with strike coverage treated as set by ``coverage``.

    Attributes
    ----------
    points
        One entry per expiration, in time order.
    increments
        ``Delta w_j``: variance accumulated between expiration ``j-1`` and ``j`` (the first entry
        is the total variance up to the first expiration), on the strike windows of ``coverage``.
    band
        Half-width of the bid-ask band of each increment.
    cumulative
        Running sum of ``increments``: the total variance up to each expiration.
    cumulative_band
        Half-width of the bid-ask band of each own-range total variance (weights of the fit).
    delta_T
        Calendar-time gaps in years (the first is ``T_1``).
    skipped
        Expirations on the daily run whose quotes could not be used, with the reason.
    """

    asof: datetime
    points: tuple[StripPoint, ...]
    increments: np.ndarray
    band: np.ndarray
    cumulative: np.ndarray
    cumulative_band: np.ndarray
    delta_T: np.ndarray
    skipped: tuple[tuple[date, str], ...]
    windows: tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]
    coverage: str = "own"

    @property
    def expiries(self) -> list[date]:
        return [p.expiry for p in self.points]

    @property
    def forward_variance(self) -> np.ndarray:
        """``v_j``: annualised (calendar-time) forward variance of each interval."""
        return self.increments / self.delta_T

    @property
    def violations(self) -> np.ndarray:
        """Boolean mask of intervals with a negative forward variance (calendar arbitrage)."""
        return self.increments < 0

    @property
    def significant_violations(self) -> np.ndarray:
        """Negative even after adding the bid-ask band: not explainable by quote spreads."""
        return self.increments + self.band < 0

    def resampled_cumulative(
        self, rng: np.random.Generator, quote_correlation: float = QUOTE_CORRELATION
    ) -> np.ndarray:
        """Cumulative totals with every option price redrawn inside its bid-ask band.

        The position of a price inside its band is uniform. Positions within one expiration share
        a common factor through a Gaussian copula with correlation ``quote_correlation``, so quote
        errors that move a whole chain together (stale or wide markets) are represented and do
        not average out over hundreds of strikes. The strike selection and ``K0`` are held fixed;
        the forward moves with the redrawn call and put at the forward strike. Used by the
        bootstrap of the clock fit.
        """
        if not 0.0 <= quote_correlation < 1.0:
            raise ValueError("quote_correlation must be in [0, 1)")
        n = len(self.points)
        a, b = np.sqrt(quote_correlation), np.sqrt(1.0 - quote_correlation)
        own = []
        for p in self.points:
            t = p.term
            common = rng.standard_normal()
            q = t.price_lo + ndtr(a * common + b * rng.standard_normal(t.prices.size)) * (
                t.price_hi - t.price_lo
            )
            c = t.atm_call_band[0] + ndtr(a * common + b * rng.standard_normal()) * (
                t.atm_call_band[1] - t.atm_call_band[0]
            )
            pu = t.atm_put_band[0] + ndtr(a * common + b * rng.standard_normal()) * (
                t.atm_put_band[1] - t.atm_put_band[0]
            )
            fwd = t.atm_strike + np.exp(t.rate * t.T) * (c - pu)
            own.append((q, fwd))
        inc = np.empty(n)
        first = self.points[0].term
        q0, f0 = own[0]
        inc[0] = first.weights @ q0 - (f0 / first.k0 - 1.0) ** 2
        for j in range(n - 1):
            m_a, m_b, w_a, w_b = self.windows[j]
            ta, tb = self.points[j].term, self.points[j + 1].term
            qa, fa = own[j]
            qb, fb = own[j + 1]
            wa = w_a @ qa[m_a] - (fa / ta.k0 - 1.0) ** 2
            wb = w_b @ qb[m_b] - (fb / tb.k0 - 1.0) ** 2
            inc[j + 1] = wb - wa
        return np.cumsum(inc)


def _daily_run(snapshot: Snapshot) -> list[OptionStrip]:
    """Unexpired SPXW strips for every consecutive trading day, starting at the snapshot day."""
    asof = snapshot.asof
    by_date = {
        s.expiry: s
        for s in snapshot.strips
        if s.root == "SPXW" and calendar_minutes(asof, s.expiry_time) > 0
    }
    run: list[OptionStrip] = []
    d = asof.date()
    if d in by_date:
        run.append(by_date[d])
    while True:
        d = next_trading_day(d)
        if d not in by_date:
            return run
        run.append(by_date[d])


def build_strip(
    snapshot: Snapshot,
    curve: YieldCurve,
    max_expiries: int | None = None,
    coverage: str = "own",
) -> ForwardVarianceStrip:
    """Build the forward-variance strip from the daily SPXW expirations.

    The strip is the longest run of expirations on consecutive trading days; during a session
    it starts with the same-day expiration while that still has time left.

    Parameters
    ----------
    coverage
        ``"own"`` takes each expiration's total on its own strike selection, ``"common"``
        computes every increment on the strikes the two expirations share (module docstring).
    """
    if coverage not in {"own", "common"}:
        raise ValueError(f"coverage must be 'own' or 'common', got {coverage!r}")
    asof = snapshot.asof
    run = _daily_run(snapshot)
    if max_expiries is not None:
        run = run[:max_expiries]
    if len(run) < 3:
        raise ValueError("need at least three consecutive daily SPXW expirations")

    points: list[StripPoint] = []
    skipped: list[tuple[date, str]] = []
    for s in run:
        minutes = calendar_minutes(asof, s.expiry_time)
        T = minutes / MINUTES_PER_CALENDAR_YEAR
        rate = curve.rate(s.expiry)
        try:
            term = single_term_variance(s, T, rate)
        except VarianceError as exc:
            skipped.append((s.expiry, str(exc)))
            continue
        points.append(StripPoint(s.expiry, minutes, T, rate, term))
    if len(points) < 3:
        raise ValueError("fewer than three usable daily SPXW expirations")

    n = len(points)
    windows = []
    inc = np.empty(n)
    band = np.empty(n)
    cum_band = np.array([p.band_half_width for p in points])
    first = points[0].term
    inc[0], band[0] = _band(first, np.ones(first.strikes.size, dtype=bool))
    for j in range(n - 1):
        a, b = points[j].term, points[j + 1].term
        if coverage == "common":
            lo = max(a.strikes[0], b.strikes[0])
            hi = min(a.strikes[-1], b.strikes[-1])
            m_a, m_b = _window_mask(a, lo, hi), _window_mask(b, lo, hi)
        else:
            m_a = np.ones(a.strikes.size, dtype=bool)
            m_b = np.ones(b.strikes.size, dtype=bool)
        wa, ba = _band(a, m_a)
        wb, bb = _band(b, m_b)
        inc[j + 1] = wb - wa
        band[j + 1] = ba + bb
        windows.append((m_a, m_b, _window_weights(a, m_a), _window_weights(b, m_b)))
    delta_T = np.diff([0.0, *[p.T for p in points]])
    return ForwardVarianceStrip(
        asof=asof,
        points=tuple(points),
        increments=inc,
        band=band,
        cumulative=np.cumsum(inc),
        cumulative_band=cum_band,
        delta_T=delta_T,
        skipped=tuple(skipped),
        windows=tuple(windows),
        coverage=coverage,
    )
