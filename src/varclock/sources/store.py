"""Local, gitignored cache for downloaded data and timestamped option snapshots."""

from __future__ import annotations

import os
from pathlib import Path

from ..quotes import Snapshot

__all__ = ["data_dir", "latest_snapshot", "save_snapshot", "snapshot_files"]


def data_dir() -> Path:
    """Cache root: ``$VARCLOCK_DATA_DIR``, else ``varclock`` in ``$XDG_CACHE_HOME`` or ``~/.cache``.

    The default does not depend on the working directory, so a command run from any folder
    finds the same cache. Never committed.
    """
    explicit = os.environ.get("VARCLOCK_DATA_DIR")
    if explicit:
        return Path(explicit)
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "varclock"


def _snapshot_dir() -> Path:
    return data_dir() / "snapshots"


def save_snapshot(snapshot: Snapshot) -> Path:
    """Store a snapshot as ``<source>_<symbol>_<fetch time UTC>.json.gz``."""
    stamp = snapshot.fetched_at.strftime("%Y%m%dT%H%M%SZ")
    name = f"{snapshot.source}_{snapshot.symbol.strip('^')}_{stamp}.json.gz"
    return snapshot.save(_snapshot_dir() / name)


def snapshot_files(source: str | None = None, symbol: str = "SPX") -> list[Path]:
    """Cached snapshot files, oldest first, optionally restricted to one source."""
    files = sorted(_snapshot_dir().glob(f"*_{symbol.strip('^')}_*.json.gz"))
    return [f for f in files if source is None or f.name.startswith(f"{source}_")]


def latest_snapshot(source: str | None = None, symbol: str = "SPX") -> Snapshot | None:
    """Most recent cached snapshot, or ``None``."""
    files = snapshot_files(source, symbol)
    return Snapshot.load(files[-1]) if files else None
