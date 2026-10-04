"""Synthetic option chains with known variance, for tests and method checks.

* Black-76 prices with a given total variance (flat smile): replicated variance must equal it.
* Heston (1993) prices by Lewis' (2000) Fourier integral: expected integrated variance is
  ``theta T + (v0 - theta)(1 - exp(-kappa T)) / kappa``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime

import numpy as np
from scipy.stats import norm

from .quotes import OptionStrip, Snapshot

__all__ = [
    "black76_prices",
    "heston_expected_variance",
    "heston_prices",
    "synthetic_daily_snapshot",
    "synthetic_strip",
]


def black76_prices(
    forward: float, strikes: np.ndarray, total_variance: float, discount: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Call and put prices under lognormal terminal prices with ``Var(ln S_T) = total_variance``."""
    k = np.asarray(strikes, dtype=float)
    sd = np.sqrt(total_variance)
    d1 = (np.log(forward / k) + 0.5 * total_variance) / sd
    d2 = d1 - sd
    call = discount * (forward * norm.cdf(d1) - k * norm.cdf(d2))
    put = discount * (k * norm.cdf(-d2) - forward * norm.cdf(-d1))
    return call, put


def _heston_cf(
    u: np.ndarray, T: float, v0: float, kappa: float, theta: float, xi: float, rho: float
) -> np.ndarray:
    """Characteristic function of ``ln(S_T / F)`` under Heston (stable "little trap" form)."""
    iu = 1j * u
    d = np.sqrt((rho * xi * iu - kappa) ** 2 + xi**2 * (iu + u**2))
    g = (kappa - rho * xi * iu - d) / (kappa - rho * xi * iu + d)
    e = np.exp(-d * T)
    c = (
        kappa
        * theta
        / xi**2
        * ((kappa - rho * xi * iu - d) * T - 2.0 * np.log((1 - g * e) / (1 - g)))
    )
    dterm = (kappa - rho * xi * iu - d) / xi**2 * (1 - e) / (1 - g * e)
    return np.exp(c + dterm * v0)


def heston_prices(
    forward: float,
    strikes: np.ndarray,
    T: float,
    *,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    discount: float = 1.0,
    n_nodes: int = 6000,
    u_max: float = 400.0,
) -> tuple[np.ndarray, np.ndarray]:
    """European call and put prices under Heston by Lewis' formula.

    ``C = D [F - sqrt(F K) / pi * integral_0^inf Re(exp(i u k) phi(u - i/2)) / (u^2 + 1/4) du]``
    with ``k = ln(F / K)``. The integral uses the midpoint rule on ``[0, u_max]``.
    """
    k = np.log(forward / np.asarray(strikes, dtype=float))
    du = u_max / n_nodes
    u = (np.arange(n_nodes) + 0.5) * du
    phi = _heston_cf(u - 0.5j, T, v0, kappa, theta, xi, rho)
    kernel = phi / (u**2 + 0.25)
    integral = np.real(np.exp(1j * np.outer(k, u)) @ kernel) * du
    call = discount * (forward - np.sqrt(forward * np.asarray(strikes)) / np.pi * integral)
    put = call - discount * (forward - np.asarray(strikes))
    return np.maximum(call, 0.0), np.maximum(put, 0.0)


def heston_expected_variance(T: float, v0: float, kappa: float, theta: float) -> float:
    """``E[integral_0^T v_t dt]`` for the CIR variance process."""
    return theta * T + (v0 - theta) * (1.0 - np.exp(-kappa * T)) / kappa


def synthetic_strip(
    root: str,
    expiry: date,
    strikes: np.ndarray,
    call: np.ndarray,
    put: np.ndarray,
    half_spread: float = 0.0,
    tick: float = 0.0,
) -> OptionStrip:
    """Quotes ``price +- half_spread`` for each option; prices below ``tick`` get a zero bid."""

    def quote(price: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bid = np.maximum(price - half_spread, 0.0)
        ask = price + half_spread
        bid = np.where(price < tick, 0.0, bid)
        return bid, ask

    cb, ca = quote(np.asarray(call))
    pb, pa = quote(np.asarray(put))
    return OptionStrip(root, expiry, np.asarray(strikes, dtype=float), cb, ca, pb, pa)


def synthetic_daily_snapshot(
    asof: datetime,
    sessions: Sequence[date],
    session_variance: Sequence[float],
    forward: float = 1000.0,
    half_spread: float = 0.0,
    spacing: float = 1.0,
    tick: float = 0.0,
) -> Snapshot:
    """Snapshot of daily PM-settled strips with total variances that are known running sums.

    The expiry on ``sessions[j]`` has total variance ``sum(session_variance[: j + 1])`` and a flat
    smile; rates are zero, so the forward equals ``forward`` at every expiry. With ``tick > 0``
    options priced below the tick have a zero bid, so the Cboe truncation rule cuts the wings of
    each strip at a different strike, as on a real chain.
    """
    totals = np.cumsum(np.asarray(session_variance, dtype=float))
    strips = []
    for day, w in zip(sessions, totals, strict=True):
        sd = np.sqrt(w)
        lo = max(spacing, np.floor(forward * np.exp(-9 * sd) / spacing) * spacing)
        hi = np.ceil(forward * np.exp(9 * sd) / spacing) * spacing
        k = np.arange(lo, hi + spacing, spacing)
        call, put = black76_prices(forward, k, float(w))
        strips.append(
            synthetic_strip("SPXW", day, k, call, put, half_spread=half_spread, tick=tick)
        )
    return Snapshot("SPX", "synthetic", asof, asof.astimezone(UTC), forward, tuple(strips))
