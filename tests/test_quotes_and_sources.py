from __future__ import annotations

import gzip
import json
import sys
import types
import urllib.error
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from varclock import tradingcal as tc
from varclock.quotes import OptionStrip, Snapshot
from varclock.sources import cboe, chainparse, history, store, treasury, yahoo
from varclock.sources.http import RateLimiter, fetch_bytes, user_agent


def make_strip():
    return OptionStrip.from_quotes(
        "SPXW",
        date(2026, 10, 5),
        {7700.0: (30.0, 31.0), 7725.0: (18.0, 19.0)},
        {7700.0: (14.0, 15.0), 7750.0: (30.0, 31.0)},
    )


def test_strip_merges_calls_and_puts_with_nan_for_missing_quotes():
    s = make_strip()
    assert list(s.strikes) == [7700.0, 7725.0, 7750.0]
    assert np.isnan(s.put_bid[1])
    assert np.isnan(s.call_ask[2])
    assert s.settlement == "PM"
    assert s.expiry_time.hour == 16


def test_strip_validation():
    k = np.array([1.0, 2.0])
    ok = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="unknown root"):
        OptionStrip("ABC", date(2026, 1, 1), k, ok, ok, ok, ok)
    with pytest.raises(ValueError, match="same length"):
        OptionStrip("SPX", date(2026, 1, 1), k, ok[:1], ok, ok, ok)
    with pytest.raises(ValueError, match="increasing"):
        OptionStrip("SPX", date(2026, 1, 1), k[::-1], ok, ok, ok, ok)
    with pytest.raises(ValueError, match="negative"):
        OptionStrip("SPX", date(2026, 1, 1), k, -ok, ok, ok, ok)
    with pytest.raises(ValueError, match="infinite"):
        OptionStrip("SPX", date(2026, 1, 1), k, np.array([1.0, np.inf]), ok, ok, ok)
    with pytest.raises(ValueError, match="increasing"):
        OptionStrip("SPX", date(2026, 1, 1), np.array([1.0, 1.0]), ok, ok, ok, ok)
    with pytest.raises(ValueError, match="finite"):
        OptionStrip("SPX", date(2026, 1, 1), np.array([1.0, np.nan]), ok, ok, ok, ok)
    with pytest.raises(ValueError, match="one-dimensional"):
        OptionStrip("SPX", date(2026, 1, 1), k, ok.reshape(2, 1), ok, ok, ok)
    assert OptionStrip("SPX", date(2026, 1, 1), k, ok, ok, ok, ok).settlement == "AM"


def test_with_quotes_shifts_mids():
    s = make_strip().with_quotes(call_mid_shift=np.array([1.0, 1.0, 1.0]))
    assert s.call_bid[0] == 31.0
    assert s.put_bid[0] == 14.0


def test_snapshot_round_trip_through_json_and_gzip(tmp_path):
    asof = datetime(2026, 10, 2, 16, 15, tzinfo=tc.ET)
    snap = Snapshot(
        "SPX", "test", asof, datetime(2026, 10, 4, tzinfo=UTC), 7722.7, (make_strip(),), {"k": 1}
    )
    path = snap.save(tmp_path / "nested" / "s.json.gz")
    back = Snapshot.load(path)
    assert back.asof == asof
    assert back.meta == {"k": 1}
    assert back.spot == 7722.7
    np.testing.assert_array_equal(back.strips[0].strikes, snap.strips[0].strikes)
    assert np.isnan(back.strips[0].put_bid[1])
    assert back.strip("SPXW", date(2026, 10, 5)).root == "SPXW"
    assert [s.expiry for s in back.by_root("SPXW")] == [date(2026, 10, 5)]
    with pytest.raises(KeyError):
        back.strip("SPX", date(2026, 10, 5))
    with pytest.raises(ValueError, match="timezone-aware"):
        Snapshot("SPX", "t", datetime(2026, 1, 1), datetime(2026, 1, 1, tzinfo=UTC), None, ())


def test_symbol_parsing_and_row_grouping():
    assert chainparse.parse_symbol("SPXW261005C03200000") == (
        "SPXW",
        date(2026, 10, 5),
        "C",
        3200.0,
    )
    assert chainparse.parse_symbol("SPX261016P07722500") == ("SPX", date(2026, 10, 16), "P", 7722.5)
    assert chainparse.parse_symbol("AAPL261016C00200000") is None
    rows = [
        ("SPXW261005C07700000", 30.0, 31.0),
        ("SPXW261005P07700000", 14.0, 15.0),
        ("SPXW261005C07725000", None, 19.0),
        ("SPXW261005P07725000", float("nan"), 3.0),
        ("SPX261016C07700000", 40.0, 41.0),
        ("SPY261016C00700000", 1.0, 2.0),
    ]
    strips = chainparse.strips_from_rows(rows)
    assert [(s.root, s.expiry) for s in strips] == [
        ("SPXW", date(2026, 10, 5)),
        ("SPX", date(2026, 10, 16)),
    ]
    assert list(strips[0].strikes) == [7700.0]


def test_treasury_csv_parsing():
    text = (
        'Date,"1 Mo","1.5 Month","2 Mo","3 Mo","4 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr",'
        '"7 Yr","10 Yr","20 Yr","30 Yr"\n'
        "10/02/2026,4.04,4.09,4.11,4.19,4.26,4.27,4.46,4.83,4.96,5.06,5.17,5.28,5.67,5.63\n"
        "10/01/2026,4.06,4.10,4.13,4.17,4.26,4.27,4.44,4.78,4.91,5.01,5.12,5.24,5.64,N/A\n"
    )
    table = treasury.parse_par_yield_csv(text)
    assert table[date(2026, 10, 2)]["1 Mo"] == 4.04
    assert "1.5 Month" not in table[date(2026, 10, 2)]
    assert "30 Yr" not in table[date(2026, 10, 1)]


def test_history_csv_parsing():
    ohlc = "DATE,OPEN,HIGH,LOW,CLOSE\n10/01/2026,1,2,0.5,15.5\n10/02/2026,1,2,0.5,15.31\n"
    assert history.parse_history_csv(ohlc)[date(2026, 10, 2)] == 15.31
    close_only = "DATE,SPX\n10/02/2026,7722.72\n10/03/2026,\n"
    assert history.parse_history_csv(close_only) == {date(2026, 10, 2): 7722.72}


def test_store_saves_and_finds_latest_snapshots(tmp_path, monkeypatch):
    monkeypatch.setenv("VARCLOCK_DATA_DIR", str(tmp_path))
    assert store.latest_snapshot("yahoo") is None
    base = Snapshot(
        "SPX",
        "yahoo",
        datetime(2026, 10, 2, 16, 0, tzinfo=tc.ET),
        datetime(2026, 10, 4, 1, tzinfo=UTC),
        None,
        (),
    )
    later = Snapshot("SPX", "yahoo", base.asof, base.fetched_at + timedelta(days=1), None, ())
    cboe_snap = Snapshot("SPX", "cboe", base.asof, base.fetched_at, None, ())
    for s in (later, base, cboe_snap):
        store.save_snapshot(s)
    assert store.latest_snapshot("yahoo").fetched_at == later.fetched_at
    assert store.latest_snapshot("cboe").source == "cboe"
    assert len(store.snapshot_files()) == 3
    assert len(store.snapshot_files("yahoo")) == 2
    assert store.snapshot_files(symbol="NDX") == []


def test_cboe_adapter_parses_the_feed(monkeypatch):
    payload = {
        "timestamp": "2026-10-03 22:26:08",
        "data": {
            "current_price": 7722.72,
            "last_trade_time": "2026-10-02T16:14:59",
            "options": [
                {"option": "SPXW261005C07700000", "bid": 30.0, "ask": 31.0},
                {"option": "SPXW261005P07700000", "bid": 14.0, "ask": 15.0},
            ],
        },
    }
    body = json.dumps(payload).encode()
    monkeypatch.setattr(cboe, "fetch_bytes", lambda url, limiter=None: body)
    snap = cboe.fetch_cboe_snapshot("^SPX")
    assert snap.source == "cboe"
    assert snap.spot == 7722.72
    assert snap.meta["provider_timestamp"] == "2026-10-03 22:26:08"
    assert "non-commercial" in snap.meta["terms"]
    assert snap.asof.hour in (9, 10, 11, 12, 13, 14, 15, 16)
    assert len(snap.strips) == 1


def test_yahoo_adapter_with_a_fake_yfinance(monkeypatch):
    class FakeTicker:
        options = ("2026-10-05", "2026-10-06")
        fast_info = {"last_price": 7722.7}  # noqa: RUF012

        def __init__(self, symbol):
            assert symbol == "^SPX"

        def option_chain(self, expiry):
            ymd = expiry[2:].replace("-", "")
            calls = {"contractSymbol": [f"SPXW{ymd}C07700000"], "bid": [30.0], "ask": [31.0]}
            puts = {"contractSymbol": [f"SPXW{ymd}P07700000"], "bid": [14.0], "ask": [15.0]}
            return types.SimpleNamespace(calls=calls, puts=puts)

    fake = types.SimpleNamespace(Ticker=FakeTicker, __version__="0.0-test")
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    snap = yahoo.fetch_yahoo_snapshot("SPX", pause=0.0)
    assert snap.source == "yahoo"
    assert [s.expiry for s in snap.strips] == [date(2026, 10, 5), date(2026, 10, 6)]
    assert snap.meta["yfinance_version"] == "0.0-test"
    assert yahoo.fetch_yahoo_snapshot("SPX", max_expiries=1, pause=0.0).strips[0].expiry == date(
        2026, 10, 5
    )


def test_http_helpers(monkeypatch):
    monkeypatch.setenv("VARCLOCK_USER_AGENT", "Test test@example.org")
    assert user_agent() == "Test test@example.org"
    monkeypatch.delenv("VARCLOCK_USER_AGENT")
    assert "varclock" in user_agent()
    limiter = RateLimiter(0.0)
    limiter.wait()
    limiter.wait()

    def fail(*args, **kwargs):
        raise urllib.error.URLError("no network in tests")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="could not download"):
        fetch_bytes("https://example.invalid", retries=1)


def test_gzip_cache_file_is_plain_json(tmp_path):
    snap = Snapshot(
        "SPX",
        "t",
        datetime(2026, 10, 2, 16, 15, tzinfo=tc.ET),
        datetime(2026, 10, 4, tzinfo=UTC),
        None,
        (make_strip(),),
    )
    path = snap.save(tmp_path / "x.json.gz")
    with gzip.open(path, "rt") as fh:
        assert json.load(fh)["strips"][0]["root"] == "SPXW"


def test_parser_assigns_calls_and_puts_by_the_symbol_letter():
    rows = [
        ("SPXW261005C07700000", 30.0, 31.0),
        ("SPXW261005P07700000", 14.0, 15.0),
        ("SPXW261005C07750000", 5.0, 6.0),
        ("SPXW261005P07750000", 50.0, 52.0),
        ("SPX261016C07700000", 40.0, 41.0),
    ]
    by_key = {(s.root, s.expiry): s for s in chainparse.strips_from_rows(rows)}
    weekly = by_key[("SPXW", date(2026, 10, 5))]
    assert list(weekly.strikes) == [7700.0, 7750.0]
    assert list(weekly.call_bid) == [30.0, 5.0]
    assert list(weekly.call_ask) == [31.0, 6.0]
    assert list(weekly.put_bid) == [14.0, 50.0]
    assert list(weekly.put_ask) == [15.0, 52.0]
    monthly = by_key[("SPX", date(2026, 10, 16))]
    assert list(monthly.call_bid) == [40.0]
    assert np.isnan(monthly.put_bid[0])


def test_parser_rejects_malformed_symbols_and_missing_quotes():
    assert chainparse.parse_symbol("AAPL261005C00150000") is None
    assert chainparse.parse_symbol("SPXW261005X07700000") is None
    assert chainparse.parse_symbol(" SPXW261005P07700000 ") == (
        "SPXW",
        date(2026, 10, 5),
        "P",
        7700.0,
    )
    rows = [
        ("SPXW261005C07700000", None, 31.0),
        ("SPXW261005P07700000", 14.0, float("nan")),
        ("SPXW261005C07750000", 0.0, 6.0),
    ]
    (strip,) = chainparse.strips_from_rows(rows)
    assert list(strip.strikes) == [7750.0]
    assert strip.call_bid[0] == 0.0


CSV_HEADER = (
    'Date,"1 Mo","2 Mo","3 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr","10 Yr","20 Yr","30 Yr"\n'
)


def _csv(*rows):
    return CSV_HEADER + "".join(f"{d},{','.join(['4.0'] * 12)}\n" for d in rows)


def _age(path, days):
    import os
    import time

    stamp = time.time() - days * 86400
    os.utime(path, (stamp, stamp))


def _written(path, day):
    import os

    stamp = datetime(day.year, day.month, day.day, 12, tzinfo=UTC).timestamp()
    os.utime(path, (stamp, stamp))


def test_treasury_cache_is_refreshed_once_when_older_than_the_previous_trading_day(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("VARCLOCK_DATA_DIR", str(tmp_path))
    year_file = tmp_path / "treasury_par_yields_2026.csv"
    year_file.write_text(_csv("10/05/2026"))
    (tmp_path / "treasury_par_yields_2025.csv").write_text(_csv("12/31/2025"))
    _written(year_file, date(2026, 10, 6))
    calls = []

    def fake_fetch(url, **kwargs):
        calls.append(url)
        return _csv("10/05/2026", "10/06/2026", "10/07/2026").encode()

    monkeypatch.setattr(treasury, "fetch_bytes", fake_fetch)
    curve = treasury.load_yield_curve(date(2026, 10, 8))  # Thursday: Wednesday's curve expected
    assert curve.curve_date == date(2026, 10, 7)
    assert len(calls) == 1


def test_treasury_fresh_or_current_cache_is_not_downloaded_again(tmp_path, monkeypatch):
    monkeypatch.setenv("VARCLOCK_DATA_DIR", str(tmp_path))
    year_file = tmp_path / "treasury_par_yields_2026.csv"
    year_file.write_text(_csv("10/06/2026", "10/07/2026"))
    (tmp_path / "treasury_par_yields_2025.csv").write_text(_csv("12/31/2025"))
    _written(year_file, date(2026, 10, 6))

    def fail(url, **kwargs):
        raise AssertionError("no download expected")

    monkeypatch.setattr(treasury, "fetch_bytes", fail)
    assert treasury.load_yield_curve(date(2026, 10, 8)).curve_date == date(2026, 10, 7)
    # a cache written after the expected day (Wednesday) ended already holds the newest curve
    year_file.write_text(_csv("10/05/2026"))
    _written(year_file, date(2026, 10, 9))
    assert treasury.load_yield_curve(date(2026, 10, 8)).curve_date == date(2026, 10, 5)


def test_default_cache_directory_does_not_depend_on_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("VARCLOCK_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert store.data_dir() == tmp_path / "xdg" / "varclock"
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setattr(store.Path, "home", classmethod(lambda cls: tmp_path))
    assert store.data_dir() == tmp_path / ".cache" / "varclock"
    monkeypatch.setenv("VARCLOCK_DATA_DIR", str(tmp_path / "explicit"))
    assert store.data_dir() == tmp_path / "explicit"


def test_index_history_is_downloaded_again_when_it_lacks_the_requested_date(tmp_path, monkeypatch):
    monkeypatch.setenv("VARCLOCK_DATA_DIR", str(tmp_path))
    old = "DATE,OPEN,HIGH,LOW,CLOSE\n10/01/2026,1,2,0.5,15.5\n"
    new = old + "10/02/2026,1,2,0.5,15.31\n"
    path = tmp_path / "cboe_VIX_History.csv"
    path.write_text(old)
    calls = []

    def fake_fetch(url, **kwargs):
        calls.append(url)
        return new.encode()

    monkeypatch.setattr(history, "fetch_bytes", fake_fetch)
    # a recent file is trusted: the close may simply not be published yet
    assert date(2026, 10, 2) not in history.load_index_history("VIX", through=date(2026, 10, 2))
    assert calls == []
    _age(path, 2)
    assert history.load_index_history("VIX", through=date(2026, 10, 2))[date(2026, 10, 2)] == 15.31
    assert len(calls) == 1
    assert history.load_index_history("VIX", through=date(2026, 10, 2))  # now complete
    assert len(calls) == 1
    history.load_index_history("VIX")  # no date requested: never downloaded again
    assert len(calls) == 1
