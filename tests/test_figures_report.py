from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from helpers import ASOF  # noqa: E402
from varclock.figures import (  # noqa: E402
    _calendar_title,
    build_calendar_figure,
    build_replication_figure,
    build_robustness_figure,
    build_study_figure,
)
from varclock.report import render_report, write_report  # noqa: E402
from varclock.viz import picture_html, save_figure, style  # noqa: E402


def test_calendar_figure_is_rendered_in_both_themes(result, tmp_path):
    paths = save_figure(build_calendar_figure(result), tmp_path / "cal")
    assert [p.name for p in paths] == ["cal-light.png", "cal-dark.png"]
    assert all(p.stat().st_size > 20_000 for p in paths)
    assert "times a base session" in _calendar_title(result)


def test_replication_and_robustness_figures(tmp_path):
    official = {"VIX": 15.31, "VIX1D": 10.76}
    errors = {"cboe": {"VIX": 0.008, "VIX1D": 0.004}, "yahoo": {"VIX": 0.104, "VIX1D": 0.026}}
    paths = save_figure(build_replication_figure(official, errors, "label"), tmp_path / "rep")
    assert all(p.exists() for p in paths)
    table = {
        "A": {"CPI 10-14": (1.7, 1.6, 1.8), "FOMC 10-28": (1.5, 1.3, 1.7)},
        "B": {"CPI 10-14": (1.8, 1.7, 1.9), "FOMC 10-28": (1.4, 1.2, 1.6)},
    }
    paths = save_figure(build_robustness_figure(table), tmp_path / "rob")
    assert all(p.exists() for p in paths)


def test_study_figure(tmp_path):
    from helpers import make_events, synthetic_history
    from varclock.eventstudy import run_event_study

    vix1d, spx, days = synthetic_history(n_days=1500, seed=2)
    events = make_events(days, "CPI", 9) + make_events(days, "FOMC", 23)
    study = run_event_study(events, vix1d, spx, start=days[5], kinds=("CPI", "FOMC"))
    paths = save_figure(build_study_figure(study, ("CPI", "FOMC")), tmp_path / "study")
    assert all(p.stat().st_size > 10_000 for p in paths)


def test_picture_html_serves_the_dark_variant():
    html = picture_html("docs/figures/calendar", "alt text", width=800)
    assert "prefers-color-scheme: dark" in html
    assert "calendar-dark.png" in html
    assert 'src="docs/figures/calendar-light.png"' in html
    assert 'width="800"' in html


def test_style_palettes_cover_eight_series_and_both_themes():
    for theme in ("light", "dark"):
        with style(theme) as p:
            assert len(p["series"]) == 8
            assert len(set(p["series"])) == 8


def test_report_html_contains_the_story_and_the_disclaimers(result, tmp_path):
    replication = {"VIX": (15.31, 15.318), "VIX1D": (10.76, 10.764)}
    html = render_report(result, replication)
    assert html.startswith("<!doctype html>")
    assert "SPX variance calendar" in html
    assert "Session by session" in html
    assert "What each event adds" in html
    assert "Check against the official closes" in html
    assert "not investment advice" in html
    assert "not affiliated" in html
    assert html.count("<tr") > 30
    assert 'class="event"' in html
    path = write_report(result, tmp_path / "r.html")
    assert path.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert "Check against" not in render_report(result)
    assert ASOF.strftime("%d %B %Y") in html
    assert np.isfinite(result.fit.weekend_ratio)


def test_report_tiles_show_the_nearest_event_of_each_kind(result):
    from varclock.report import render_report

    html = render_report(result)
    first_cpi = next(e for e in result.event_rows if e.kind == "CPI")
    assert f"CPI, {first_cpi.session:%d %b}" in html
