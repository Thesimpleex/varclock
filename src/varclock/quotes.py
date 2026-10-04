"""Option-chain containers: one expiry strip and a timestamped snapshot of many strips."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .tradingcal import ET, expiry_datetime

__all__ = ["OptionStrip", "Snapshot"]

_ROOT_SETTLEMENT = {"SPX": "AM", "SPXW": "PM"}


def _as_float_array(name: str, values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return arr


@dataclass(frozen=True)
class OptionStrip:
    """Bid and ask quotes of all calls and puts of one root and one expiration date.

    Missing quotes are ``nan``; a quote of zero means "no bid" (or "no offer") and is meaningful.
    ``root`` is ``"SPX"`` for AM-settled standard options and ``"SPXW"`` for PM-settled weeklys.
    """

    root: str
    expiry: date
    strikes: np.ndarray
    call_bid: np.ndarray
    call_ask: np.ndarray
    put_bid: np.ndarray
    put_ask: np.ndarray

    def __post_init__(self) -> None:
        if self.root not in _ROOT_SETTLEMENT:
            raise ValueError(
                f"unknown root {self.root!r}; expected one of {sorted(_ROOT_SETTLEMENT)}"
            )
        for name in ("strikes", "call_bid", "call_ask", "put_bid", "put_ask"):
            object.__setattr__(self, name, _as_float_array(name, getattr(self, name)))
        n = self.strikes.size
        for name in ("call_bid", "call_ask", "put_bid", "put_ask"):
            if getattr(self, name).size != n:
                raise ValueError(f"{name} must have the same length as strikes")
        if n and (not np.all(np.isfinite(self.strikes)) or np.any(np.diff(self.strikes) <= 0)):
            raise ValueError("strikes must be finite and strictly increasing")
        for name in ("call_bid", "call_ask", "put_bid", "put_ask"):
            q = getattr(self, name)
            if np.any(np.isinf(q)):
                raise ValueError(f"{name} contains infinite quotes")
            if np.any(q[~np.isnan(q)] < 0):
                raise ValueError(f"{name} contains negative quotes")

    @property
    def settlement(self) -> str:
        return _ROOT_SETTLEMENT[self.root]

    @property
    def expiry_time(self) -> datetime:
        return expiry_datetime(self.expiry, self.settlement)

    @classmethod
    def from_quotes(
        cls,
        root: str,
        expiry: date,
        calls: Mapping[float, tuple[float, float]],
        puts: Mapping[float, tuple[float, float]],
    ) -> OptionStrip:
        """Merge ``{strike: (bid, ask)}`` mappings for calls and puts into one strip."""
        strikes = np.array(sorted(set(calls) | set(puts)), dtype=float)
        nan = (np.nan, np.nan)
        c = np.array([calls.get(k, nan) for k in strikes], dtype=float).reshape(-1, 2)
        p = np.array([puts.get(k, nan) for k in strikes], dtype=float).reshape(-1, 2)
        return cls(root, expiry, strikes, c[:, 0], c[:, 1], p[:, 0], p[:, 1])

    def with_quotes(
        self,
        call_mid_shift: np.ndarray | None = None,
        put_mid_shift: np.ndarray | None = None,
    ) -> OptionStrip:
        """Copy with mid prices shifted by the given amounts (bid and ask move together)."""
        cs = np.zeros_like(self.strikes) if call_mid_shift is None else call_mid_shift
        ps = np.zeros_like(self.strikes) if put_mid_shift is None else put_mid_shift
        return OptionStrip(
            self.root,
            self.expiry,
            self.strikes,
            self.call_bid + cs,
            self.call_ask + cs,
            self.put_bid + ps,
            self.put_ask + ps,
        )

    def to_dict(self) -> dict[str, Any]:
        def clean(a: np.ndarray) -> list[float | None]:
            return [None if np.isnan(x) else float(x) for x in a]

        return {
            "root": self.root,
            "expiry": self.expiry.isoformat(),
            "strikes": [float(k) for k in self.strikes],
            "call_bid": clean(self.call_bid),
            "call_ask": clean(self.call_ask),
            "put_bid": clean(self.put_bid),
            "put_ask": clean(self.put_ask),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> OptionStrip:
        def arr(key: str) -> np.ndarray:
            return np.array([np.nan if x is None else x for x in d[key]], dtype=float)

        return cls(
            d["root"],
            date.fromisoformat(d["expiry"]),
            arr("strikes"),
            arr("call_bid"),
            arr("call_ask"),
            arr("put_bid"),
            arr("put_ask"),
        )


@dataclass(frozen=True)
class Snapshot:
    """All strips of one underlying at one valuation time.

    ``asof`` is the instant (America/New_York) at which times to expiry are measured;
    ``fetched_at`` is the download time (UTC) and may be later, for example on a weekend.
    """

    symbol: str
    source: str
    asof: datetime
    fetched_at: datetime
    spot: float | None
    strips: tuple[OptionStrip, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.asof.tzinfo is None or self.fetched_at.tzinfo is None:
            raise ValueError("asof and fetched_at must be timezone-aware")
        object.__setattr__(self, "asof", self.asof.astimezone(ET))
        object.__setattr__(self, "strips", tuple(self.strips))

    def strip(self, root: str, expiry: date) -> OptionStrip:
        for s in self.strips:
            if s.root == root and s.expiry == expiry:
                return s
        raise KeyError(f"no {root} strip expiring {expiry}")

    def by_root(self, root: str) -> list[OptionStrip]:
        return sorted((s for s in self.strips if s.root == root), key=lambda s: s.expiry)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "source": self.source,
            "asof": self.asof.isoformat(),
            "fetched_at": self.fetched_at.isoformat(),
            "spot": self.spot,
            "meta": self.meta,
            "strips": [s.to_dict() for s in self.strips],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Snapshot:
        return cls(
            d["symbol"],
            d["source"],
            datetime.fromisoformat(d["asof"]),
            datetime.fromisoformat(d["fetched_at"]),
            d["spot"],
            tuple(OptionStrip.from_dict(s) for s in d["strips"]),
            dict(d.get("meta", {})),
        )

    def save(self, path: str | Path) -> Path:
        """Write a gzip-compressed JSON cache file (local only; raw quotes are never committed)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, separators=(",", ":"))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Snapshot:
        """Read a file written by ``save``; unreadable or malformed files raise ``ValueError``."""
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                return cls.from_dict(json.load(fh))
        except (OSError, EOFError, KeyError, TypeError, ValueError) as exc:
            reason = exc.strerror if isinstance(exc, OSError) and exc.strerror else str(exc)
            raise ValueError(f"cannot read snapshot file {path}: {reason}") from exc
