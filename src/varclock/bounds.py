"""Arbitrage-free bid/ask bands for model-free variance (linear programming).

The published index uses mid quotes. Given the bid-ask spreads, the set of call-price curves
that are consistent with the quotes *and* free of static arbitrage (decreasing, convex, slope
at least minus one, above intrinsic value) is a polytope. Because the variance estimate is
linear in the option prices and the forward, its smallest and largest values over that
polytope are two linear programs (solved with the HiGHS solver of SciPy).

Variables are the undiscounted call values ``u_i`` at the strikes selected by the Cboe rule and
the forward ``F``; put values follow from parity, ``p_i = u_i - (F - K_i)``. The discount factor
is fixed at its Treasury value, and the small term ``(F/K0 - 1)^2`` is bounded from the safe
side by a chord (for the minimum) and a tangent (for the maximum) over ``K0 <= F < K0 + gap``,
so both bounds are valid outer bounds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .quotes import OptionStrip, Snapshot
from .strip import ForwardVarianceStrip
from .variance import TermVariance, strike_spacing

__all__ = ["VarianceBand", "increment_band", "strip_bands", "variance_band"]


@dataclass(frozen=True)
class VarianceBand:
    """Total-variance band for one expiration.

    ``lo`` and ``hi`` are the arbitrage-free LP bounds; ``naive_lo`` and ``naive_hi`` come from
    pricing every option at its bid or at its ask (no no-arbitrage constraints). ``feasible`` is
    false if the quotes admit no arbitrage-free curve at all.
    """

    mid: float
    lo: float
    hi: float
    naive_lo: float
    naive_hi: float
    widen: float

    @property
    def feasible(self) -> bool:
        return bool(np.isfinite(self.lo))

    @property
    def width(self) -> float:
        return self.hi - self.lo

    @property
    def naive_width(self) -> float:
        return self.naive_hi - self.naive_lo


def _problem(
    term: TermVariance,
    strip: OptionStrip,
    discount: float,
    mask: np.ndarray | None,
    widen: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, list[tuple[float, float | None]], float]:
    """Objective ``c @ z + const`` and constraints ``A z <= b`` with variable bounds."""
    k = term.strikes
    n = k.size
    mask = np.ones(n, dtype=bool) if mask is None else mask
    idx0 = int(np.searchsorted(strip.strikes, term.k0))
    nxt = strip.strikes[idx0 + 1] if idx0 + 1 < strip.strikes.size else term.k0 * 1.001
    gap = float(nxt - term.k0)

    kk = k[mask]
    coef = np.zeros(n)
    coef[mask] = 2.0 * strike_spacing(kk) / kk**2

    c = np.zeros(n + 1)
    const = 0.0
    for i in range(n):
        if term.kinds[i] == "P":
            c[i] += coef[i]
            c[n] -= coef[i]
            const += coef[i] * k[i]
        elif term.kinds[i] == "C":
            c[i] += coef[i]
        else:
            c[i] += coef[i]
            c[n] -= coef[i] / 2.0
            const += coef[i] * k[i] / 2.0

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    bounds: list[tuple[float, float | None]] = [(0.0, None)] * n + [(term.k0, term.k0 + gap)]

    def add(row: np.ndarray, upper: float) -> None:
        rows.append(row)
        rhs.append(upper)

    def put_row(i: int, sign: float) -> np.ndarray:
        r = np.zeros(n + 1)
        r[i], r[n] = sign, -sign
        return r

    for i in range(n):
        j = int(np.searchsorted(strip.strikes, k[i]))
        cb = max(strip.call_bid[j] - widen, 0.0) / discount
        ca = (strip.call_ask[j] + widen) / discount
        pb = max(strip.put_bid[j] - widen, 0.0) / discount
        pa = (strip.put_ask[j] + widen) / discount
        if term.kinds[i] in ("C", "B"):
            bounds[i] = (cb, ca)
        if term.kinds[i] in ("P", "B"):
            # p = u - F + K in [pb, pa]
            add(put_row(i, 1.0), pa - k[i])
            add(put_row(i, -1.0), -(pb - k[i]))
    for i in range(n - 1):
        r = np.zeros(n + 1)
        r[i + 1], r[i] = 1.0, -1.0
        add(r, 0.0)  # u decreasing
        r = np.zeros(n + 1)
        r[i], r[i + 1] = 1.0, -1.0
        add(r, k[i + 1] - k[i])  # slope at least -1
    for i in range(1, n - 1):
        h1, h2 = k[i] - k[i - 1], k[i + 1] - k[i]
        r = np.zeros(n + 1)
        r[i - 1], r[i], r[i + 1] = -1.0 / h1, 1.0 / h1 + 1.0 / h2, -1.0 / h2
        add(r, 0.0)  # convex
    for i in range(n):
        r = np.zeros(n + 1)
        r[n], r[i] = 1.0, -1.0
        add(r, k[i])  # u >= F - K
    return c, const, np.array(rows), np.array(rhs), bounds, gap


def _feasible(
    term: TermVariance, strip: OptionStrip, discount: float, mask: np.ndarray | None, widen: float
) -> bool:
    c, _, a, b, bounds, _ = _problem(term, strip, discount, mask, widen)
    res = linprog(np.zeros(c.size), A_ub=a, b_ub=b, bounds=bounds, method="highs")
    return res.status == 0


def _minimal_widening(
    term: TermVariance, strip: OptionStrip, discount: float, mask: np.ndarray | None
) -> float | None:
    """Smallest uniform extra half-spread (index points) that makes the quotes feasible."""
    if _feasible(term, strip, discount, mask, 0.0):
        return 0.0
    hi = 0.05
    while not _feasible(term, strip, discount, mask, hi):
        hi *= 2.0
        if hi > 64.0:
            return None
    lo = 0.0
    for _ in range(24):
        mid = (lo + hi) / 2.0
        if _feasible(term, strip, discount, mask, mid):
            hi = mid
        else:
            lo = mid
    return hi


def variance_band(
    term: TermVariance,
    strip: OptionStrip,
    discount: float,
    mask: np.ndarray | None = None,
) -> VarianceBand:
    """LP bounds on the total variance of ``term`` (optionally on the strikes in ``mask``).

    If the quotes admit no arbitrage-free curve, every bid and ask is widened by the smallest
    common amount that does, recorded in ``VarianceBand.widen``.
    """
    sel = np.ones(term.strikes.size, dtype=bool) if mask is None else mask
    kk = term.strikes[sel]
    w = np.zeros(term.strikes.size)
    w[sel] = 2.0 * strike_spacing(kk) / kk**2 * np.exp(term.rate * term.T)
    pen = (term.forward / term.k0 - 1.0) ** 2
    mid = float(w @ term.prices - pen)
    naive_lo = float(w @ term.price_lo - pen)
    naive_hi = float(w @ term.price_hi - pen)

    widen = _minimal_widening(term, strip, discount, mask)
    if widen is None:
        return VarianceBand(mid, np.nan, np.nan, naive_lo, naive_hi, float("inf"))
    widen = widen * 1.0001 + (1e-9 if widen > 0 else 0.0)
    c, const, a, b, bounds, gap = _problem(term, strip, discount, mask, widen)
    n = c.size - 1
    k0 = term.k0
    f_mid = min(max(term.forward, k0), k0 + gap)

    def penalty(f: float) -> float:
        return (f / k0 - 1.0) ** 2

    slope = 2.0 * (f_mid / k0 - 1.0) / k0
    chord = (penalty(k0 + gap) - penalty(k0)) / gap
    # maximum: subtract the tangent at f_mid (<= penalty); minimum: subtract the chord (>= penalty)
    obj_hi = c.copy()
    obj_hi[n] -= slope
    const_hi = const - (penalty(f_mid) - slope * f_mid)
    obj_lo = c.copy()
    obj_lo[n] -= chord
    const_lo = const - (penalty(k0) - chord * k0)
    lo = linprog(obj_lo, A_ub=a, b_ub=b, bounds=bounds, method="highs")
    hi = linprog(-obj_hi, A_ub=a, b_ub=b, bounds=bounds, method="highs")
    if lo.status != 0 or hi.status != 0:
        return VarianceBand(mid, np.nan, np.nan, naive_lo, naive_hi, widen)
    return VarianceBand(
        mid=mid,
        lo=float(lo.fun + const_lo),
        hi=float(-hi.fun + const_hi),
        naive_lo=naive_lo,
        naive_hi=naive_hi,
        widen=widen,
    )


def increment_band(near: VarianceBand, far: VarianceBand) -> tuple[float, float]:
    """Bounds on ``w_far - w_near`` when the two expirations are bounded independently."""
    return far.lo - near.hi, far.hi - near.lo


def strip_bands(snapshot: Snapshot, strip: ForwardVarianceStrip) -> list[VarianceBand]:
    """LP bands for every expiration of a strip (own strike range, Treasury discount factor)."""
    return [
        variance_band(p.term, snapshot.strip("SPXW", p.expiry), float(np.exp(-p.rate * p.T)))
        for p in strip.points
    ]
