"""Cboe delayed-quotes JSON adapter (opt-in).

The public JSON files behind Cboe's website are offered for personal, non-commercial use only;
read and respect the Cboe terms of use before enabling this source. Requests are sequential and
spaced by at least 3.3 seconds (about 0.3 requests per second). Raw quotes stay in the local
cache and are never redistributed by this project.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from ..quotes import Snapshot
from ..tradingcal import resolve_asof
from .chainparse import strips_from_rows
from .http import RateLimiter, fetch_bytes

__all__ = ["CBOE_TERMS_NOTE", "fetch_cboe_snapshot"]

CBOE_TERMS_NOTE = (
    "Cboe delayed quotes: for personal, non-commercial use only; see Cboe's terms of use."
)
_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/_{symbol}.json"
_LIMITER = RateLimiter(3.3)
_DELAY = timedelta(minutes=15)


def fetch_cboe_snapshot(symbol: str = "SPX") -> Snapshot:
    """Download the full SPX/SPXW chain from the Cboe delayed-quotes feed (one request)."""
    sym = symbol.strip("^").upper()
    raw = fetch_bytes(_URL.format(symbol=sym), limiter=_LIMITER)
    fetched_at = datetime.now(UTC)
    payload = json.loads(raw)
    data = payload["data"]
    rows = [(o["option"], o.get("bid"), o.get("ask")) for o in data["options"]]
    return Snapshot(
        symbol=sym,
        source="cboe",
        asof=resolve_asof(fetched_at - _DELAY),
        fetched_at=fetched_at,
        spot=float(data["current_price"]) if data.get("current_price") is not None else None,
        strips=tuple(strips_from_rows(rows)),
        meta={
            "provider_timestamp": payload.get("timestamp"),
            "index_last_trade_time": data.get("last_trade_time"),
            "terms": CBOE_TERMS_NOTE,
        },
    )
