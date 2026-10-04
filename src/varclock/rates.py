"""Risk-free rates from Treasury par yields: bounded natural cubic spline, BEY to continuous.

Follows section 2 of the Cboe Volatility Index Mathematics Methodology: a natural cubic
spline through the constant-maturity Treasury (CMT) yields, clipped between the two adjacent
yields (or between two extrapolation lines below the shortest maturity), then converted from a
bond-equivalent yield to an annual percentage yield and to a continuously compounded rate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import numpy as np
from scipy.interpolate import CubicSpline

__all__ = ["CMT_DAYS", "YieldCurve", "bounded_spline_bey", "continuous_rate"]

#: Maturity label to number of days used by Cboe for the spline nodes.
CMT_DAYS: dict[str, int] = {
    "1 Mo": 30,
    "2 Mo": 60,
    "3 Mo": 91,
    "6 Mo": 182,
    "1 Yr": 365,
    "2 Yr": 730,
    "3 Yr": 1095,
    "5 Yr": 1825,
    "7 Yr": 2555,
    "10 Yr": 3650,
    "20 Yr": 7300,
    "30 Yr": 10950,
}


def bounded_spline_bey(days: np.ndarray, cmt: np.ndarray, t: float) -> float:
    """Bond-equivalent yield (decimal) at ``t`` days from CMT nodes ``(days, cmt)`` in decimals."""
    days = np.asarray(days, dtype=float)
    cmt = np.asarray(cmt, dtype=float)
    if days.size < 2 or days.size != cmt.size or np.any(np.diff(days) <= 0):
        raise ValueError("need at least two increasing maturities with matching yields")
    if t <= 0:
        raise ValueError("t must be positive")
    y = float(CubicSpline(days, cmt, bc_type="natural")(t))
    if t < days[0]:
        later = np.arange(1, days.size)
        up = later[cmt[1:] >= cmt[0]]
        down = later[cmt[1:] <= cmt[0]]
        slope_low = (cmt[up[0]] - cmt[0]) / (days[up[0]] - days[0]) if up.size else 0.0
        slope_high = (cmt[down[0]] - cmt[0]) / (days[down[0]] - days[0]) if down.size else 0.0
        lo = cmt[0] + slope_low * (t - days[0])
        hi = cmt[0] + slope_high * (t - days[0])
        return float(min(max(y, lo), hi))
    i = min(int(np.searchsorted(days, t, side="right")) - 1, days.size - 2)
    lo, hi = sorted((cmt[i], cmt[i + 1]))
    return float(min(max(y, lo), hi))


def continuous_rate(bey: float) -> float:
    """Continuously compounded rate from a bond-equivalent yield: ``ln(1 + APY)``."""
    apy = (1.0 + bey / 2.0) ** 2 - 1.0
    return float(np.log1p(apy))


@dataclass(frozen=True)
class YieldCurve:
    """Treasury par yield curve of one business day, yields in percent."""

    curve_date: date
    yields_pct: Mapping[str, float]

    def __post_init__(self) -> None:
        nodes = [(CMT_DAYS[k], v) for k, v in self.yields_pct.items() if k in CMT_DAYS]
        nodes = [(d, v) for d, v in nodes if v is not None and np.isfinite(v)]
        if len(nodes) < 2:
            raise ValueError("yield curve needs at least two valid CMT maturities")

    def _nodes(self) -> tuple[np.ndarray, np.ndarray]:
        pairs = sorted(
            (CMT_DAYS[k], float(v))
            for k, v in self.yields_pct.items()
            if k in CMT_DAYS and v is not None and np.isfinite(v)
        )
        return np.array([p[0] for p in pairs], float), np.array([p[1] for p in pairs], float) / 100

    def rate(self, expiry: date) -> float:
        """Continuously compounded rate for an option expiring on ``expiry``.

        The day count is the number of calendar days from the curve date to the expiration
        date, which reproduces the rates printed in Cboe's worked examples.
        """
        days = (expiry - self.curve_date).days
        if days <= 0:
            raise ValueError("expiry must be after the curve date")
        d, y = self._nodes()
        return continuous_rate(bounded_spline_bey(d, y, float(days)))
