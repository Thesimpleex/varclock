"""Shared paths and helpers for the experiment scripts."""

from __future__ import annotations

import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"
FIGURES = ROOT / "docs" / "figures"
SCREENSHOTS = ROOT / "docs" / "screenshots"

os.environ.setdefault("VARCLOCK_DATA_DIR", str(ROOT / "data" / "raw"))
os.environ.setdefault("VARCLOCK_USER_AGENT", "varclock research research@example.org")
sys.path.insert(0, str(Path(__file__).resolve().parent))


def write_json(name: str, payload: dict[str, Any]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def read_json(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "scipy", "matplotlib", "pandas", "yfinance", "rich", "varclock"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
    return out


def snapshot_meta(snap) -> dict[str, Any]:
    return {
        "source": snap.source,
        "symbol": snap.symbol,
        "valuation_time_et": snap.asof.isoformat(),
        "downloaded_utc": snap.fetched_at.isoformat(),
        "spot": snap.spot,
        "n_expiries": len(snap.strips),
        "provider_timestamp": snap.meta.get("provider_timestamp"),
    }


def notify(message: str) -> None:
    print(message, file=sys.stderr)
