"""Official index history files published by Cboe (daily closes)."""

from __future__ import annotations

import csv
import io
import time
from datetime import date, datetime
from pathlib import Path

from .http import fetch_bytes
from .store import data_dir

__all__ = ["load_index_history", "parse_history_csv"]

_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"


def parse_history_csv(text: str) -> dict[date, float]:
    """``DATE,...,CLOSE`` (or ``DATE,<name>``) CSV into ``{date: close}``."""
    reader = csv.DictReader(io.StringIO(text))
    out: dict[date, float] = {}
    for row in reader:
        key = "CLOSE" if "CLOSE" in row else next(k for k in row if k != "DATE")
        if row[key] in ("", None):
            continue
        out[datetime.strptime(row["DATE"], "%m/%d/%Y").date()] = float(row[key])
    return out


#: A cached history that lacks a requested date is downloaded again at most this often.
_REFETCH_AFTER_SECONDS = 6 * 3600


def load_index_history(
    name: str, refresh: bool = False, through: date | None = None
) -> dict[date, float]:
    """Daily closes of ``VIX``, ``VIX9D``, ``VIX3M``, ``VIX1D`` or ``SPX``, cached locally.

    The file is downloaded when it is missing or ``refresh`` is set. With ``through``, a cached
    file whose last row is older than that date and that was downloaded more than six hours ago
    is downloaded once more, so that a stale cache does not hide a close that has been published
    since. The returned table can still end before ``through`` if Cboe has not published it.
    """
    path = data_dir() / f"cboe_{name}_History.csv"
    if refresh or not path.exists():
        _download(name, path)
    history = parse_history_csv(path.read_text(encoding="utf-8"))
    stale = through is not None and (not history or max(history) < through)
    if stale and time.time() - path.stat().st_mtime > _REFETCH_AFTER_SECONDS:
        _download(name, path)
        history = parse_history_csv(path.read_text(encoding="utf-8"))
    return history


def _download(name: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(fetch_bytes(_URL.format(name=name)))
