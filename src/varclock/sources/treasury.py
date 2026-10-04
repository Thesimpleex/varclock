"""U.S. Treasury daily par yield curve rates (the CMT yields used by Cboe)."""

from __future__ import annotations

import csv
import io
from datetime import UTC, date, datetime

from ..rates import CMT_DAYS, YieldCurve
from ..tradingcal import previous_trading_day
from .http import fetch_bytes
from .store import data_dir

__all__ = ["load_yield_curve", "parse_par_yield_csv"]

_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
    "&field_tdr_date_value={year}&page&_format=csv"
)


def parse_par_yield_csv(text: str) -> dict[date, dict[str, float]]:
    """Parse the Treasury CSV into ``{date: {label: yield in percent}}`` for the Cboe maturities."""
    out: dict[date, dict[str, float]] = {}
    for row in csv.DictReader(io.StringIO(text)):
        d = datetime.strptime(row["Date"], "%m/%d/%Y").date()
        out[d] = {k: float(row[k]) for k in CMT_DAYS if row.get(k) not in (None, "", "N/A")}
    return out


def _year_table(year: int, refresh: bool) -> dict[date, dict[str, float]]:
    path = data_dir() / f"treasury_par_yields_{year}.csv"
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fetch_bytes(_URL.format(year=year)))
    return parse_par_yield_csv(path.read_text(encoding="utf-8"))


def _downloaded_before_end_of(year: int, day: date) -> bool:
    """Whether the cached file of ``year`` was downloaded before ``day`` was over (UTC)."""
    path = data_dir() / f"treasury_par_yields_{year}.csv"
    if not path.exists():
        return True
    written = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).date()
    return written <= day


def load_yield_curve(before: date, refresh: bool = False) -> YieldCurve:
    """Par yield curve of the last business day strictly before ``before``.

    This mirrors the Cboe rule that a calculation on day ``d`` uses the CMT yields published
    for the preceding business day. If the newest cached curve is older than the previous
    trading day and the cache was written before that day ended, the current year's file is
    downloaded once more, so a cache never serves an older curve than the Treasury has published.
    """
    expected = previous_trading_day(before)
    for attempt in range(2):
        merged: dict[date, dict[str, float]] = {}
        for year in (before.year, before.year - 1):
            merged.update(_year_table(year, refresh or (attempt == 1 and year == before.year)))
        earlier = [d for d in merged if d < before]
        if earlier:
            best = max(earlier)
            fresh = best >= expected or not _downloaded_before_end_of(before.year, expected)
            if fresh or attempt == 1:
                return YieldCurve(best, merged[best])
        elif attempt == 1:
            break
    raise RuntimeError(f"no Treasury yield curve found before {before}")
