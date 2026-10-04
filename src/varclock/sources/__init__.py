"""Data adapters: option chains, Treasury yields and official index histories."""

from __future__ import annotations

from .cboe import fetch_cboe_snapshot
from .history import load_index_history
from .store import data_dir, latest_snapshot, save_snapshot
from .treasury import load_yield_curve
from .yahoo import fetch_yahoo_snapshot

__all__ = [
    "data_dir",
    "fetch_cboe_snapshot",
    "fetch_yahoo_snapshot",
    "latest_snapshot",
    "load_index_history",
    "load_yield_curve",
    "save_snapshot",
]
