from __future__ import annotations

from datetime import date, time

import pytest

from varclock.events import EVENT_FILES, event_session, load_events


def test_calendar_files_have_the_expected_coverage():
    counts = {k: len(load_events([k])) for k in EVENT_FILES}
    assert counts["FOMC"] == 48  # eight scheduled meetings a year, 2022 to 2027
    assert counts["CPI"] >= 58
    assert counts["NFP"] >= 58
    assert counts["ELECTION"] == 4
    assert 5 <= counts["EARNINGS"] <= 12


def test_known_dates():
    fomc = {e.day for e in load_events(["FOMC"])}
    assert date(2022, 6, 15) in fomc
    assert date(2024, 9, 18) in fomc
    assert date(2026, 10, 28) in fomc
    assert date(2027, 10, 27) in fomc
    cpi = {e.day for e in load_events(["CPI"])}
    assert date(2022, 6, 10) in cpi
    assert date(2026, 10, 14) in cpi
    elections = [e.day for e in load_events(["ELECTION"])]
    assert elections == [
        date(2022, 11, 8),
        date(2024, 11, 5),
        date(2026, 11, 3),
        date(2028, 11, 7),
    ]


def test_every_event_has_a_source_and_valid_session():
    for ev in load_events():
        assert ev.source
        assert ev.session >= ev.day


def test_event_session_rules():
    assert event_session(date(2026, 10, 14), time(8, 30)) == date(2026, 10, 14)
    assert event_session(date(2026, 10, 28), time(14, 0)) == date(2026, 10, 28)
    assert event_session(date(2026, 10, 28), time(16, 5)) == date(2026, 10, 29)
    assert event_session(date(2026, 11, 3), time(20, 0)) == date(2026, 11, 4)
    # Good Friday 2026 is a market holiday; a release that morning moves the Monday session
    assert event_session(date(2026, 4, 3), time(8, 30)) == date(2026, 4, 6)
    # a Saturday report belongs to the next trading day
    assert event_session(date(2026, 11, 7), time(7, 0)) == date(2026, 11, 9)
    # on an early-close day a 13:30 release is after the close
    assert event_session(date(2025, 11, 28), time(13, 30)) == date(2025, 12, 1)


def test_filtering_by_kind_and_window():
    events = load_events(["CPI", "FOMC"], date(2026, 10, 1), date(2026, 11, 30))
    assert [(e.kind, e.session) for e in events] == [
        ("CPI", date(2026, 10, 14)),
        ("FOMC", date(2026, 10, 28)),
        ("CPI", date(2026, 11, 10)),
    ]
    assert load_events(["fomc"], date(2026, 1, 1), date(2026, 1, 31))[0].day == date(2026, 1, 28)


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown event kinds"):
        load_events(["WEATHER"])


def test_calendar_end_reports_the_last_listed_event_and_validates_the_kind():
    from varclock.events import calendar_end

    assert calendar_end("fomc") == date(2027, 12, 8)
    assert calendar_end("CPI") == max(e.day for e in load_events(["CPI"]))
    with pytest.raises(ValueError, match="unknown event kind"):
        calendar_end("GDP")
