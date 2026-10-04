"""Rebuild the committed event calendars in ``src/varclock/calendars/``.

Sources (retrieved at run time and recorded in each file):

* FOMC meeting calendar: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
* CPI and Employment Situation release dates: https://www.bls.gov/schedule/<year>/home.htm
  (BLS requires a declared User-Agent: set ``VARCLOCK_USER_AGENT`` to "Name email")
* U.S. general elections: the Tuesday after the first Monday of November (2 U.S.C. section 7)
* Mega-cap earnings dates: next announced or projected dates from Yahoo Finance (optional)

Usage: ``python scripts/build_calendars.py [--earnings | --earnings-only]``
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from varclock.sources.http import fetch_bytes
from varclock.tradingcal import is_trading_day

OUT = Path(__file__).resolve().parents[1] / "src" / "varclock" / "calendars"
FIRST_YEAR = 2022
FOMC_LAST_YEAR = 2027  # the Federal Reserve publishes the next year's meetings in advance
BLS_LAST_YEAR = 2027  # skipped silently while BLS has not published the schedule yet
ELECTION_LAST_YEAR = 2028  # fixed by statute, so known well ahead
MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    )
}
FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
BLS_URL = "https://www.bls.gov/schedule/{year}/home.htm"


def _text(html: str) -> str:
    html = re.sub(r"<script.*?</script>|<style.*?</style>", "", html, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "|", html))


def fomc_rows() -> list[dict[str, str]]:
    text = _text(fetch_bytes(FOMC_URL).decode("utf-8", "replace"))
    rows = []
    for year in range(FIRST_YEAR, FOMC_LAST_YEAR + 1):
        start = text.find(f"{year} FOMC Meetings")
        end = text.find(f"{year - 1} FOMC Meetings", start + 10)
        panel = text[start : end if end > 0 else None]
        pattern = re.compile(r"\|\|([A-Z][a-z]+(?:/[A-Z][a-z]+)?)\|\| \|(\d{1,2})-(\d{1,2})(\*?)\|")
        for month, _, d2, sep in pattern.findall(panel):
            parts = month.lower().split("/")
            m_end = MONTHS[parts[-1][:3]]
            statement = date(year, m_end, int(d2))
            label = "FOMC statement" + (" and projections" if sep else "")
            rows.append(
                {
                    "date": statement.isoformat(),
                    "time_et": "14:00",
                    "kind": "FOMC",
                    "label": label,
                    "source": FOMC_URL,
                }
            )
    return rows


def bls_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    cpi, emp = [], []
    for year in range(FIRST_YEAR, BLS_LAST_YEAR + 1):
        url = BLS_URL.format(year=year)
        try:
            html = fetch_bytes(url).decode("utf-8", "replace")
        except RuntimeError:
            if year < BLS_LAST_YEAR:
                raise
            sys.stderr.write(f"BLS {year}: schedule not published yet, skipped\n")
            continue
        html = re.sub(r"<script.*?</script>|<style.*?</style>", "", html, flags=re.S)
        for row in re.findall(r"<tr.*?</tr>", html, flags=re.S):
            cells = [
                re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", c)).replace("&nbsp;", "").strip()
                for c in re.findall(r"<t[hd].*?</t[hd]>", row, flags=re.S)
            ]
            if len(cells) < 3:
                continue
            kind = None
            if cells[2].startswith("Consumer Price Index for"):
                kind, target = "CPI", cpi
            elif cells[2].startswith("Employment Situation for"):
                kind, target = "NFP", emp
            if kind is None:
                continue
            day = datetime.strptime(cells[0].split(", ", 1)[1], "%B %d, %Y").date()
            time_et = datetime.strptime(cells[1], "%I:%M %p").strftime("%H:%M")
            target.append(
                {
                    "date": day.isoformat(),
                    "time_et": time_et,
                    "kind": kind,
                    "label": cells[2],
                    "source": url,
                }
            )
        sys.stderr.write(f"BLS {year}: {len(cpi)} CPI, {len(emp)} employment so far\n")
    return cpi, emp


def election_rows() -> list[dict[str, str]]:
    rows = []
    for year in range(FIRST_YEAR, ELECTION_LAST_YEAR + 1):
        if year % 2:
            continue
        first = date(year, 11, 1)
        monday = first + timedelta(days=(0 - first.weekday()) % 7)
        day = monday + timedelta(days=1)
        rows.append(
            {
                "date": day.isoformat(),
                "time_et": "20:00",
                "kind": "ELECTION",
                "label": f"U.S. general election {year} (reaction session follows)",
                "source": "2 U.S.C. section 7 (Tuesday after the first Monday in November)",
            }
        )
    return rows


def earnings_rows(top: int = 10) -> list[dict[str, str]]:
    import time

    import yfinance as yf

    universe = [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "BRK-B", "JPM",
        "LLY", "V", "XOM", "WMT", "ORCL", "MA", "NFLX", "COST",
    ]  # fmt: skip
    caps = {}
    for sym in universe:
        caps[sym] = float(yf.Ticker(sym).fast_info["market_cap"])
        time.sleep(0.4)
    largest = sorted(caps, key=caps.get, reverse=True)[:top]  # type: ignore[arg-type]
    rows = []
    for sym in largest:
        next_date = yf.Ticker(sym).calendar.get("Earnings Date")
        time.sleep(0.4)
        if not next_date:
            continue
        d = next_date[0]
        after_close = sym not in {"JPM", "BRK-B", "XOM", "WMT", "LLY", "MA"}
        rows.append(
            {
                "date": d.isoformat(),
                "time_et": "16:05" if after_close else "07:00",
                "kind": "EARNINGS",
                "label": f"{sym} earnings (announced or projected)",
                "source": "Yahoo Finance next earnings date; top-ten market cap as of "
                f"{date.today().isoformat()}",
            }
        )
    return sorted(rows, key=lambda r: r["date"])


def write(name: str, rows: list[dict[str, str]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["date"], r["label"]))
    for r in rows:
        day = date.fromisoformat(r["date"])
        if r["kind"] in {"FOMC", "CPI", "NFP"} and not is_trading_day(day):
            sys.stderr.write(f"warning: {r['kind']} on non-trading day {day}\n")
    with (OUT / name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["date", "time_et", "kind", "label", "source"])
        w.writeheader()
        w.writerows(rows)
    sys.stderr.write(f"wrote {name}: {len(rows)} rows\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--earnings", action="store_true", help="also refresh the earnings file")
    parser.add_argument("--earnings-only", action="store_true", help="refresh only earnings.csv")
    args = parser.parse_args()
    if not args.earnings_only:
        write("fomc.csv", fomc_rows())
        cpi, emp = bls_rows()
        write("cpi.csv", cpi)
        write("employment.csv", emp)
        write("elections.csv", election_rows())
    if args.earnings or args.earnings_only:
        write("earnings.csv", earnings_rows())


if __name__ == "__main__":
    main()
