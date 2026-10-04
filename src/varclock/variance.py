"""Model-free variance of one expiration, following the Cboe volatility-index methodology.

For an expiration at time ``T`` (years) with forward ``F`` and first strike ``K0`` at or below
the forward, the annualised variance is

    sigma^2 = (2 / T) * sum_i (dK_i / K_i^2) * exp(R T) * Q(K_i) - (1 / T) * (F / K0 - 1)^2

where ``Q`` is the mid quote of the out-of-the-money option at ``K_i`` (the average of call and
put at ``K0``). This is the discrete form of the variance-swap replication of Demeterfi, Derman,
Kamal and Zou (1999); see also Carr and Madan (1998) and Neuberger (1994).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .quotes import OptionStrip

__all__ = [
    "TermVariance",
    "VarianceError",
    "interpolate_variance",
    "single_term_variance",
    "strike_spacing",
]

_TIE_TOLERANCE = 1e-9


class VarianceError(ValueError):
    """Raised where the Cboe methodology says the index cannot be calculated."""


@dataclass(frozen=True)
class TermVariance:
    """Result of the single-term variance calculation, with everything needed to audit it.

    Attributes
    ----------
    sigma2
        Annualised variance ``sigma^2``.
    total_variance
        ``sigma^2 * T``: variance accumulated up to expiration.
    forward, atm_strike, k0
        Implied forward, the strike with the smallest call-put mid difference, and ``K0``.
    strikes, prices, kinds, delta_k
        Selected strikes (ascending), their mid prices, ``"P"``/``"C"``/``"B"`` (both, at
        ``K0``) and the ``Delta K`` of each.
    weights
        ``2 * delta_k / strikes^2 * exp(rate * T)``: total variance is
        ``weights @ prices - (forward / k0 - 1)^2``.
    price_lo, price_hi
        Bid and ask of each selected option (averaged at ``K0``).
    atm_call_band, atm_put_band
        ``(bid, ask)`` of the call and put at the forward strike.
    """

    sigma2: float
    forward: float
    atm_strike: float
    k0: float
    T: float
    rate: float
    strikes: np.ndarray
    prices: np.ndarray
    kinds: np.ndarray
    delta_k: np.ndarray
    weights: np.ndarray
    price_lo: np.ndarray
    price_hi: np.ndarray
    atm_call_band: tuple[float, float]
    atm_put_band: tuple[float, float]

    @property
    def total_variance(self) -> float:
        return self.sigma2 * self.T

    @property
    def contributions(self) -> np.ndarray:
        """Per-strike term ``(dK / K^2) exp(R T) Q`` of the sum in the variance formula."""
        return self.weights * self.prices / 2.0

    @property
    def volatility(self) -> float:
        """``sqrt(sigma2)``: the single-term index level divided by 100."""
        return float(np.sqrt(max(self.sigma2, 0.0)))

    @property
    def n_options(self) -> int:
        return int(self.strikes.size + np.count_nonzero(self.kinds == "B"))

    def total_variance_with(
        self, prices: np.ndarray, forward: float, strikes_mask: np.ndarray | None = None
    ) -> float:
        """Total variance for alternative prices and forward on the same strike selection."""
        w = self.weights if strikes_mask is None else self.weights * strikes_mask
        return float(w @ prices - (forward / self.k0 - 1.0) ** 2)


def strike_spacing(strikes: np.ndarray) -> np.ndarray:
    """Cboe's ``Delta K``: half the distance between neighbours, one-sided at both ends."""
    dk = np.empty_like(strikes)
    dk[0] = strikes[1] - strikes[0]
    dk[-1] = strikes[-1] - strikes[-2]
    dk[1:-1] = (strikes[2:] - strikes[:-2]) / 2.0
    return dk


def _valid(bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    """Quotes that are present and not crossed."""
    with np.errstate(invalid="ignore"):
        return ~np.isnan(bid) & ~np.isnan(ask) & (bid <= ask)


def _scan_wing(order: np.ndarray, bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    """Indices (in ``order``) kept when walking away from ``K0``.

    Options with a zero bid or zero ask are skipped; two consecutive such strikes end the walk.
    """
    zero = (bid[order] <= 0) | (ask[order] <= 0)
    stops = np.flatnonzero(zero[:-1] & zero[1:])
    cut = int(stops[0]) if stops.size else order.size
    keep = order[:cut]
    return keep[~zero[:cut]]


def single_term_variance(strip: OptionStrip, T: float, rate: float) -> TermVariance:
    """Cboe single-term variance of ``strip`` at time to expiry ``T`` (years) and rate ``rate``.

    Raises
    ------
    VarianceError
        If no two-sided call-put pair exists, if the ``K0`` options have no valid quotes, or if
        one wing has no usable out-of-the-money option.
    """
    if not np.isfinite(T) or T <= 0:
        raise ValueError("T must be a positive finite number of years")
    if not np.isfinite(rate):
        raise ValueError("rate must be finite")
    k = strip.strikes
    c_ok = _valid(strip.call_bid, strip.call_ask)
    p_ok = _valid(strip.put_bid, strip.put_ask)
    c_mid = (strip.call_bid + strip.call_ask) / 2.0
    p_mid = (strip.put_bid + strip.put_ask) / 2.0

    # Forward from the strike with the smallest |call - put| among two-sided quote pairs.
    with np.errstate(invalid="ignore"):
        two_sided = (
            c_ok
            & p_ok
            & (strip.call_bid > 0)
            & (strip.call_ask > 0)
            & (strip.put_bid > 0)
            & (strip.put_ask > 0)
        )
    if not np.any(two_sided):
        raise VarianceError("no strike with two-sided call and put quotes")
    diff = np.where(two_sided, np.abs(c_mid - p_mid), np.inf)
    i_atm = int(np.flatnonzero(diff <= diff.min() + _TIE_TOLERANCE)[0])
    growth = float(np.exp(rate * T))
    forward = float(k[i_atm] + growth * (c_mid[i_atm] - p_mid[i_atm]))

    below = np.flatnonzero(k <= forward)
    if below.size == 0:
        raise VarianceError("forward lies below the lowest listed strike")
    i0 = int(below[-1])
    if not (c_ok[i0] and p_ok[i0]):
        raise VarianceError(f"K0 = {k[i0]:g} has no valid call and put quote")
    k0 = float(k[i0])

    put_order = np.flatnonzero((np.arange(k.size) < i0) & p_ok)[::-1]
    call_order = np.flatnonzero((np.arange(k.size) > i0) & c_ok)
    puts = _scan_wing(put_order, strip.put_bid, strip.put_ask)
    calls = _scan_wing(call_order, strip.call_bid, strip.call_ask)
    if puts.size == 0:
        raise VarianceError("no out-of-the-money put with a usable quote")
    if calls.size == 0:
        raise VarianceError("no out-of-the-money call with a usable quote")

    puts = puts[::-1]
    strikes = np.concatenate([k[puts], [k0], k[calls]])
    kinds = np.array(["P"] * puts.size + ["B"] + ["C"] * calls.size)
    prices = np.concatenate([p_mid[puts], [(c_mid[i0] + p_mid[i0]) / 2.0], c_mid[calls]])
    lo = np.concatenate(
        [
            strip.put_bid[puts],
            [(strip.call_bid[i0] + strip.put_bid[i0]) / 2.0],
            strip.call_bid[calls],
        ]
    )
    hi = np.concatenate(
        [
            strip.put_ask[puts],
            [(strip.call_ask[i0] + strip.put_ask[i0]) / 2.0],
            strip.call_ask[calls],
        ]
    )

    delta_k = strike_spacing(strikes)
    weights = 2.0 * delta_k / strikes**2 * growth
    total = float(weights @ prices - (forward / k0 - 1.0) ** 2)
    return TermVariance(
        sigma2=total / T,
        forward=forward,
        atm_strike=float(k[i_atm]),
        k0=k0,
        T=float(T),
        rate=float(rate),
        strikes=strikes,
        prices=prices,
        kinds=kinds,
        delta_k=delta_k,
        weights=weights,
        price_lo=lo,
        price_hi=hi,
        atm_call_band=(float(strip.call_bid[i_atm]), float(strip.call_ask[i_atm])),
        atm_put_band=(float(strip.put_bid[i_atm]), float(strip.put_ask[i_atm])),
    )


def interpolate_variance(
    sigma2_near: float | None,
    sigma2_next: float,
    minutes_near: float | None,
    minutes_next: float,
    minutes_target: float,
    minutes_year: float,
) -> float:
    """Constant-maturity index level, ``100 * sqrt(variance)``.

    Implements section 3(b) of the Cboe mathematics methodology,

        100 * sqrt( [T1 s1 (M2 - M) / (M2 - M1) + T2 s2 (M - M1) / (M2 - M1)] * M_year / M )

    with ``T_i = M_i / M_year``. Without a near term the next term is used on its own.
    """
    if not minutes_target > 0 or not minutes_year > 0:
        raise ValueError("minutes_target and minutes_year must be positive")
    if sigma2_next < 0 or (sigma2_near is not None and sigma2_near < 0):
        raise ValueError("term variances must be non-negative")
    if sigma2_near is None or minutes_near is None:
        return 100.0 * float(np.sqrt(sigma2_next))
    if minutes_next <= minutes_near:
        raise ValueError("next term must expire after the near term")
    t1 = minutes_near / minutes_year
    t2 = minutes_next / minutes_year
    span = minutes_next - minutes_near
    w1 = (minutes_next - minutes_target) / span
    w2 = (minutes_target - minutes_near) / span
    total = t1 * sigma2_near * w1 + t2 * sigma2_next * w2
    return 100.0 * float(np.sqrt(total * minutes_year / minutes_target))
