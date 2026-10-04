"""Shared parsing of OCC-style SPX option symbols into option strips."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from datetime import date

from ..quotes import OptionStrip

__all__ = ["parse_symbol", "strips_from_rows"]

_SYMBOL = re.compile(r"^(SPXW|SPX)(\d{6})([CP])(\d{8})$")


def parse_symbol(symbol: str) -> tuple[str, date, str, float] | None:
    """``SPXW261005C03200000`` -> ``("SPXW", date(2026, 10, 5), "C", 3200.0)``; else ``None``."""
    m = _SYMBOL.match(symbol.strip())
    if m is None:
        return None
    root, ymd, kind, strike = m.groups()
    expiry = date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]))
    return root, expiry, kind, int(strike) / 1000.0


def strips_from_rows(rows: Iterable[tuple[str, float | None, float | None]]) -> list[OptionStrip]:
    """Group ``(symbol, bid, ask)`` rows into one strip per root and expiration.

    Rows with a missing bid or ask are dropped (a missing quote is not a zero quote).
    """
    calls: dict[tuple[str, date], dict[float, tuple[float, float]]] = defaultdict(dict)
    puts: dict[tuple[str, date], dict[float, tuple[float, float]]] = defaultdict(dict)
    for symbol, bid, ask in rows:
        parsed = parse_symbol(symbol)
        if parsed is None or bid is None or ask is None:
            continue
        if bid != bid or ask != ask:
            continue
        root, expiry, kind, strike = parsed
        (calls if kind == "C" else puts)[(root, expiry)][strike] = (float(bid), float(ask))
    keys = sorted(set(calls) | set(puts), key=lambda k: (k[1], k[0]))
    return [
        OptionStrip.from_quotes(
            root, expiry, calls.get((root, expiry), {}), puts.get((root, expiry), {})
        )
        for root, expiry in keys
    ]
