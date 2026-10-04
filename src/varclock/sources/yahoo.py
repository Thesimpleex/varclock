"""Yahoo Finance option-chain adapter via the optional ``yfinance`` package (default source).

Yahoo chains are delayed and often truncated in the far wings; the replication errors that
result are reported by ``varclock replicate``. Downloaded quotes stay in the local cache.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

from ..quotes import Snapshot
from ..tradingcal import resolve_asof, session_close, session_end
from .chainparse import strips_from_rows

__all__ = ["fetch_yahoo_snapshot"]

_DELAY = timedelta(minutes=15)


def fetch_yahoo_snapshot(
    symbol: str = "SPX", max_expiries: int | None = None, pause: float = 0.4
) -> Snapshot:
    """Download all listed expirations for ``^SPX`` (one request per expiration, sequential)."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("the Yahoo source needs `pip install varclock[yahoo]`") from exc

    sym = symbol.strip("^").upper()
    ticker = yf.Ticker(f"^{sym}")
    expirations = list(ticker.options)
    if max_expiries is not None:
        expirations = expirations[:max_expiries]
    rows: list[tuple[str, float | None, float | None]] = []
    for exp in expirations:
        chain = ticker.option_chain(exp)
        for frame in (chain.calls, chain.puts):
            rows.extend(zip(frame["contractSymbol"], frame["bid"], frame["ask"], strict=True))
        time.sleep(pause)
    fetched_at = datetime.now(UTC)
    try:
        spot = float(ticker.fast_info["last_price"])
    except Exception:
        spot = None
    asof = resolve_asof(fetched_at - _DELAY)
    if asof == session_end(asof.date()):
        # Yahoo's SPX option quotes stop at the 16:00 equity close, 15 minutes before the
        # 16:15 close of index options (see the README section on adapter errors).
        asof = session_close(asof.date())
    return Snapshot(
        symbol=sym,
        source="yahoo",
        asof=asof,
        fetched_at=fetched_at,
        spot=spot,
        strips=tuple(strips_from_rows(rows)),
        meta={"yfinance_version": getattr(yf, "__version__", "unknown")},
    )
