from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from helpers import ASOF, CURVE, N_SESSIONS
from varclock import tradingcal as tc
from varclock.analysis import build_calendar
from varclock.cli import main
from varclock.events import load_events
from varclock.sources.store import save_snapshot


def test_calendar_pipeline_finds_the_events_and_labels_sessions(result, truth):
    days, _, _ = truth
    assert len(result.rows) == N_SESSIONS
    by_day = {r.session: r for r in result.rows}
    assert "CPI" in by_day[date(2026, 10, 14)].events
    assert "FOMC" in by_day[date(2026, 10, 28)].events
    assert by_day[date(2026, 10, 5)].weekend
    assert not by_day[date(2026, 10, 6)].weekend
    assert any("earnings" in e for e in by_day[date(2026, 10, 29)].events)
    kinds = {e.kind for e in result.event_rows}
    assert {"CPI", "FOMC"} <= kinds
    cpi = next(e for e in result.event_rows if e.key == "CPI 2026-10-14")
    assert cpi.multiple == pytest.approx(
        1 + 0.5 / (0.6 + 0.03 * days.index(cpi.session)) / 1.0, rel=0.05
    )
    assert cpi.multiple_lo <= cpi.multiple <= cpi.multiple_hi
    assert cpi.implied_sd > 0


def test_forward_vols_and_multiples_have_consistent_intervals(result):
    for r in result.rows:
        assert r.fwd_vol_lo <= r.fwd_vol <= r.fwd_vol_hi + 1e-12
        assert r.multiple_lo - 1e-12 <= r.multiple <= r.multiple_hi + 1e-12
        assert r.multiple >= 1.0 - 1e-12
    non_event = [r for r in result.rows if not r.events and not r.weekend]
    assert all(r.multiple == pytest.approx(1.0) for r in non_event)
    assert result.rows[1].observed_vol is not None
    assert result.asof == ASOF


def test_calendar_is_reproducible_for_a_given_seed(snapshot):
    a = build_calendar(snapshot, CURVE, n_boot=30, seed=7)
    b = build_calendar(snapshot, CURVE, n_boot=30, seed=7)
    assert [r.fwd_vol_lo for r in a.rows] == [r.fwd_vol_lo for r in b.rows]
    c = build_calendar(snapshot, CURVE, n_boot=30, seed=8)
    assert [r.fwd_vol_lo for r in a.rows] != [r.fwd_vol_lo for r in c.rows]


@pytest.fixture
def data_dir(tmp_path, monkeypatch, snapshot):
    """A cache with a synthetic snapshot, zero Treasury yields and small index histories."""
    monkeypatch.setenv("VARCLOCK_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("COLUMNS", "160")
    save_snapshot(
        type(snapshot)(
            snapshot.symbol,
            "yahoo",
            snapshot.asof,
            snapshot.fetched_at,
            snapshot.spot,
            snapshot.strips,
        )
    )
    labels = ["1 Mo", "2 Mo", "3 Mo", "6 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr"]
    labels += ["10 Yr", "20 Yr", "30 Yr"]
    header = "Date," + ",".join(f'"{x}"' for x in labels) + "\n"
    zero = ",".join(["0.0"] * 12)
    (tmp_path / "treasury_par_yields_2026.csv").write_text(header + f"10/01/2026,{zero}\n")
    (tmp_path / "treasury_par_yields_2025.csv").write_text(header + f"12/31/2025,{zero}\n")
    for name, value in (("VIX", 20.0), ("VIX9D", 20.0), ("VIX3M", 20.0), ("VIX1D", 20.0)):
        (tmp_path / f"cboe_{name}_History.csv").write_text(
            f"DATE,OPEN,HIGH,LOW,CLOSE\n10/01/2026,1,1,1,1\n10/02/2026,{value},{value},{value},{value}\n"
        )
    days = tc.trading_days(date(2022, 5, 12), date(2026, 10, 2))
    rng = np.random.default_rng(0)
    level, rows, vix1d = 4000.0, ["DATE,SPX"], ["DATE,OPEN,HIGH,LOW,CLOSE"]
    for d in days:
        level *= float(np.exp(0.008 * rng.standard_normal()))
        rows.append(f"{d:%m/%d/%Y},{level:.2f}")
        vix1d.append(f"{d:%m/%d/%Y},14,14,14,14")
    (tmp_path / "cboe_SPX_History.csv").write_text("\n".join(rows) + "\n")
    (tmp_path / "cboe_VIX1D_History.csv").write_text("\n".join(vix1d) + "\n")
    return tmp_path


def test_cli_calendar_prints_the_table(data_dir, capsys):
    assert main(["calendar", "SPX", "--boot", "30", "--sessions", "12"]) == 0
    out = capsys.readouterr().out
    assert "SPX variance calendar" in out
    assert "x base" in out
    assert "CPI" in out
    assert "weekend" in out
    assert "Extra variance" in out


def test_cli_calendar_accepts_a_snapshot_file_and_spline_family(data_dir, capsys):
    from varclock.sources.store import snapshot_files

    path = snapshot_files("yahoo")[0]
    assert main(["calendar", "--snapshot", str(path), "--boot", "30", "--family", "spline"]) == 0
    assert "spline" in capsys.readouterr().out


def test_cli_replicate_compares_with_official_closes(data_dir, capsys):
    assert main(["replicate", "SPX", "--source", "yahoo"]) == 0
    out = capsys.readouterr().out
    assert "Official" in out
    assert "VIX1D" in out
    assert "n/a" in out  # the synthetic chain has no three-month expiries


def test_cli_bounds_prints_a_band_table(data_dir, capsys):
    assert main(["bounds", "SPX", "--expiries", "6"]) == 0
    out = capsys.readouterr().out
    assert "LP band" in out
    assert "Bid/ask band" in out


def test_cli_events_lists_upcoming_and_history(data_dir, capsys):
    assert main(["events", "--asof", "2026-10-03", "--days", "40"]) == 0
    out = capsys.readouterr().out
    assert "FOMC" in out
    assert "CPI" in out
    assert main(["events", "--history"]) == 0
    out = capsys.readouterr().out
    assert "Ordinary days" in out
    assert "Implied 1-sd" in out


def test_cli_report_writes_html(data_dir, tmp_path, capsys):
    out_file = tmp_path / "r.html"
    assert main(["report", "SPX", "--out", str(out_file), "--boot", "30"]) == 0
    html = out_file.read_text()
    assert html.startswith("<!doctype html>")
    assert "data:image/png;base64," in html
    assert "not affiliated" in html
    assert "Session by session" in html
    assert "Wrote" in capsys.readouterr().out


def test_cli_errors_are_reported_not_raised(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VARCLOCK_DATA_DIR", str(tmp_path))
    assert main(["calendar", "NDX", "--source", "yahoo"]) == 2
    assert "only SPX" in capsys.readouterr().out


def test_events_filter_and_calendar_horizon_use_the_committed_files(result):
    kinds = {e.kind for e in result.events}
    assert kinds == {"CPI", "FOMC", "ELECTION", "NFP", "EARNINGS"} or kinds <= {
        "CPI",
        "FOMC",
        "ELECTION",
        "NFP",
        "EARNINGS",
    }
    assert load_events(["FOMC"], ASOF.date(), ASOF.date() + timedelta(days=60))


def test_implied_sd_is_the_square_root_of_the_session_variance(result):
    cal = result.fit.calendar
    for e in result.event_rows:
        if e.kind == "EARNINGS":
            continue
        s = cal.dates.index(e.session)
        assert e.implied_sd == pytest.approx(np.sqrt(result.fit.session_variance[s]), rel=1e-12)
    earnings = next(e for e in result.event_rows if e.kind == "EARNINGS")
    i = [t.kind for t in result.fit.terms].index("EARNINGS")
    assert earnings.implied_sd == pytest.approx(np.sqrt(result.fit.event_variance(i)), rel=1e-12)


def test_calendar_notes_warn_when_an_event_file_ends_before_the_last_expiry(snapshot, monkeypatch):
    import varclock.analysis as analysis

    assert build_calendar(snapshot, CURVE, n_boot=20, seed=1).notes == ()
    monkeypatch.setattr(analysis, "calendar_end", lambda kind: date(2026, 10, 20))
    notes = build_calendar(snapshot, CURVE, n_boot=20, seed=1).notes
    assert len(notes) == 3
    assert all("ends 2026-10-20" in n for n in notes)


def test_cli_rejects_non_positive_numeric_options(data_dir, capsys):
    for argv in (
        ["calendar", "--sessions", "0"],
        ["calendar", "--sessions", "-3"],
        ["calendar", "--boot", "0"],
        ["events", "--days", "-5"],
        ["bounds", "--expiries", "1"],
        ["bounds", "--expiries", "0"],
    ):
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert exc.value.code == 2
    assert "at least 3" in capsys.readouterr().err


def test_cli_reports_unreadable_snapshot_files_without_a_traceback(data_dir, tmp_path, capsys):
    assert main(["calendar", "--snapshot", str(tmp_path / "missing.json.gz")]) == 2
    assert "cannot read snapshot file" in capsys.readouterr().out
    junk = tmp_path / "junk.json.gz"
    junk.write_bytes(b"not a gzip file")
    assert main(["calendar", "--snapshot", str(junk)]) == 2
    assert "cannot read snapshot file" in capsys.readouterr().out
    import gzip

    wrong = tmp_path / "wrong.json.gz"
    with gzip.open(wrong, "wt") as fh:
        fh.write('{"symbol": "SPX"}')
    assert main(["calendar", "--snapshot", str(wrong)]) == 2
    assert "cannot read snapshot file" in capsys.readouterr().out
    unreadable_history = data_dir / "cboe_VIX_History.csv"
    unreadable_history.unlink()
    unreadable_history.mkdir()
    assert main(["replicate", "SPX", "--source", "yahoo"]) == 2
    assert "error: cannot read" in capsys.readouterr().out


def test_cli_calendar_json_is_machine_readable(data_dir, capsys):
    import json

    assert main(["calendar", "SPX", "--boot", "30", "--sessions", "5", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["sessions"]) == 5
    assert payload["snapshot"]["strike_coverage"] == "own"
    cpi = next(e for e in payload["events"] if e["kind"] == "CPI")
    assert cpi["extra_variance_pct_sq"] == pytest.approx(cpi["extra_variance"] * 1e4)
    assert cpi["multiple_90pct"][0] <= cpi["multiple_of_base"] <= cpi["multiple_90pct"][1]


def test_cli_events_json_lists_events_and_history(data_dir, capsys):
    import json

    assert main(["events", "--asof", "2026-10-03", "--days", "40", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert {"FOMC", "CPI"} <= {e["kind"] for e in listed}
    assert main(["events", "--history", "--json"]) == 0
    study = json.loads(capsys.readouterr().out)
    assert study["ordinary_days"]["n"] > 500
    assert study["CPI"]["ratio_lo"] < study["CPI"]["ratio_hi"]


def test_cli_calendar_layout_fits_an_80_column_terminal(data_dir, monkeypatch, capsys):
    monkeypatch.setenv("COLUMNS", "80")
    assert main(["calendar", "SPX", "--boot", "30", "--sessions", "12"]) == 0
    out = capsys.readouterr().out
    assert "Strip" not in out
    assert "…" not in out
    assert max(len(line) for line in out.splitlines()) <= 80
    monkeypatch.setenv("COLUMNS", "130")
    assert main(["calendar", "SPX", "--boot", "30", "--sessions", "12"]) == 0
    assert "Strip" in capsys.readouterr().out


def test_cli_replicate_distinguishes_an_unpublished_close_from_an_early_snapshot(data_dir, capsys):
    (data_dir / "cboe_VIX_History.csv").write_text("DATE,OPEN,HIGH,LOW,CLOSE\n10/01/2026,1,1,1,1\n")
    for name in ("VIX9D", "VIX3M", "VIX1D"):
        (data_dir / f"cboe_{name}_History.csv").write_text(
            "DATE,OPEN,HIGH,LOW,CLOSE\n10/01/2026,1,1,1,1\n"
        )
    assert main(["replicate", "SPX", "--source", "yahoo"]) == 1
    assert "not in the Cboe index history yet" in capsys.readouterr().out


def test_obtain_snapshot_notes_a_cached_snapshot_from_before_the_last_session(
    data_dir, monkeypatch
):
    from datetime import datetime

    import varclock.loaders as loaders

    notes = []
    monkeypatch.setattr(loaders, "resolve_asof", lambda now: ASOF)
    loaders.obtain_snapshot("yahoo", notify=notes.append)
    assert notes == []
    later = datetime(2026, 10, 9, 16, 15, tzinfo=tc.ET)
    monkeypatch.setattr(loaders, "resolve_asof", lambda now: later)
    loaders.obtain_snapshot("yahoo", notify=notes.append)
    assert len(notes) == 1
    assert "before the last completed session" in notes[0]
    assert "--refresh" in notes[0]
