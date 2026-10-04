"""Resolve a snapshot and its Treasury curve from the cache or the network."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, time
from pathlib import Path

from .quotes import Snapshot
from .rates import YieldCurve
from .sources import (
    fetch_cboe_snapshot,
    fetch_yahoo_snapshot,
    latest_snapshot,
    load_yield_curve,
    save_snapshot,
)
from .sources.cboe import CBOE_TERMS_NOTE
from .tradingcal import resolve_asof

__all__ = ["SOURCES", "close_is_due", "obtain_curve", "obtain_snapshot", "official_closes"]

SOURCES = ("yahoo", "cboe")


def obtain_snapshot(
    source: str = "yahoo",
    symbol: str = "SPX",
    path: str | Path | None = None,
    refresh: bool = False,
    notify: Callable[[str], None] | None = None,
) -> Snapshot:
    """A cached snapshot (newest first), a given file, or a fresh download.

    A download happens when ``refresh`` is set or nothing is cached for the source. A cached
    snapshot from before the last completed session is still returned, with a note through
    ``notify``.
    """
    say = notify or (lambda _: None)
    if path is not None:
        return Snapshot.load(path)
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; choose from {SOURCES}")
    if symbol.strip("^").upper() != "SPX":
        raise ValueError("only SPX is supported")
    if not refresh:
        cached = latest_snapshot(source, symbol)
        if cached is not None:
            last = resolve_asof(datetime.now(UTC))
            if cached.asof.date() < last.date():
                say(
                    f"The cached {source} snapshot is from {cached.asof:%a %Y-%m-%d %H:%M} ET, "
                    f"before the last completed session ({last:%a %Y-%m-%d}); "
                    "use --refresh to download a new one."
                )
            return cached
    if source == "cboe":
        say(CBOE_TERMS_NOTE)
        say("Downloading the Cboe delayed-quotes chain (one request)...")
        snap = fetch_cboe_snapshot(symbol)
    else:
        say("Downloading the Yahoo Finance chain, one request per expiry (about 30 seconds)...")
        snap = fetch_yahoo_snapshot(symbol)
    save_snapshot(snap)
    return snap


def obtain_curve(snapshot: Snapshot, refresh: bool = False) -> YieldCurve:
    """Treasury par yield curve of the last business day before the snapshot date."""
    return load_yield_curve(snapshot.asof.date(), refresh=refresh)


def close_is_due(snapshot: Snapshot) -> bool:
    """Whether the snapshot was valued at or after the close, so official closes exist for it."""
    return snapshot.asof.time() >= time(16, 0) or snapshot.asof.time() == time(13, 0)


def official_closes(snapshot: Snapshot, names: tuple[str, ...]) -> dict[str, float]:
    """Official closes on the snapshot date; empty if the snapshot is before the close.

    A stale cached history is downloaded again once if it ends before the snapshot date. An
    empty result for a snapshot after the close (``close_is_due``) means Cboe has not published
    that close yet.
    """
    from .sources import load_index_history

    if not close_is_due(snapshot):
        return {}
    day = snapshot.asof.date()
    out: dict[str, float] = {}
    for name in names:
        history = load_index_history(name, through=day)
        if day in history:
            out[name] = history[day]
    return out
