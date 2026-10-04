"""Regenerate the product screenshots and animations from the real code.

Terminal shots and the animated terminal cast run the installed command line on the newest cached
snapshots; the report shot captures the generated HTML report; the method animation replays the
calendar of ``docs/results/calendar.json``. Run ``scripts/reproduce_all.py`` first if the cache is
empty or the results are stale.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from animate import save_animation, terminal_cast_svg
from common import ROOT, SCREENSHOTS, notify, read_json
from screenshots import report_screenshot, terminal_screenshot

from varclock.viz import style

PY = sys.executable
ANIMATIONS = ROOT / "docs" / "animations"
CAST_COLUMNS = 112

TERMINAL_SHOTS = (
    (
        "replicate",
        [PY, "-m", "varclock.cli", "replicate", "SPX", "--source", "yahoo", "--source", "cboe"],
        "varclock replicate SPX --source yahoo --source cboe",
        "varclock",
        104,
    ),
    (
        "events-history",
        [PY, "-m", "varclock.cli", "events", "--history"],
        "varclock events --history",
        "varclock",
        122,
    ),
    (
        "bounds",
        [PY, "-m", "varclock.cli", "bounds", "SPX", "--source", "cboe", "--expiries", "10"],
        "varclock bounds SPX --source cboe --expiries 10",
        "varclock",
        118,
    ),
)


def _calendar_frames(source: str = "cboe"):
    """Per-session series for the method animation, read from the committed results."""
    data = read_json("calendar")[source]
    sessions = data["sessions"]
    fit_events = {e["session"]: e for e in data["events"]}
    n = len(sessions)
    vol = np.array([r["fwd_vol"] for r in sessions]) * 100
    base = np.array([r["base_vol"] for r in sessions]) * 100
    strip = np.array([np.nan if r["strip_vol"] is None else r["strip_vol"] for r in sessions]) * 100
    kind = []
    labels = {}
    for i, r in enumerate(sessions):
        named = [e for e in r["events"] if not e.startswith("earnings")]
        ev = fit_events.get(r["session"])
        if named and ev is not None and ev["extra_variance"] > 0:
            distinct = ev["multiple_90pct"][0] > 1.05
            kind.append("event" if distinct else "weak")
            name = ", ".join(named)
            if distinct:
                labels[i] = f"{name} {r['multiple']:.1f}x"
        else:
            kind.append("weekend" if r["weekend"] else "plain")
    first = sessions[0]["session"][5:]
    return n, vol, base, strip, kind, labels, first


def _draw_clock(ax, p, i: int, frames) -> None:
    n, vol, base, strip, kind, labels, first = frames
    k = min(i, n - 1) + 1
    colour = {
        "event": p["series"][1],
        "weak": p["series"][1],
        "weekend": p["muted"],
        "plain": p["series"][0],
    }
    x = np.arange(n)
    for j in range(k):
        ax.bar(
            j,
            vol[j],
            width=0.72,
            color=colour[kind[j]],
            alpha=0.55 if kind[j] == "weak" else 0.9,
            linewidth=0,
        )
        if j in labels:
            ax.annotate(
                labels[j],
                (j, vol[j]),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=p["ink"] if kind[j] == "event" else p["ink_secondary"],
            )
    ax.plot(x[:k], base[:k], color=p["ink"], lw=1.4)
    ax.scatter(x[:k], strip[:k], s=12, color=p["ink"], zorder=4, linewidths=0)
    ax.text(k - 1 + 0.5, base[k - 1], "base", fontsize=8, va="center", color=p["ink"])
    ax.set_xlim(-0.7, n + 0.8)
    ax.set_ylim(0, np.nanmax(vol) * 1.28)
    ax.grid(axis="x", visible=False)
    ax.set_xticks([])
    ax.set_ylabel("Annualised volatility of the session, percent")
    ax.set_title("The variance calendar fills in session by session", pad=10)
    ax.text(
        0.0,
        -0.10,
        f"Cboe feed, 2026-10-02 close: fitted forward volatility per session from {first}, "
        "dots: model-free strip.\nOrange: event sessions (pale: not distinct from the base); "
        "grey: after a weekend.",
        transform=ax.transAxes,
        fontsize=7.5,
        color=p["muted"],
    )


def make_animations() -> None:
    ANIMATIONS.mkdir(parents=True, exist_ok=True)
    command = [PY, "-m", "varclock.cli", "calendar", "SPX", "--source", "cboe"]
    for theme in ("light", "dark"):
        out = ANIMATIONS / f"calendar-{theme}.svg"
        notify(f"{out.relative_to(ROOT)}")
        terminal_cast_svg(
            command,
            out,
            title="varclock",
            display_command="varclock calendar SPX --source cboe",
            columns=CAST_COLUMNS,
            cwd=ROOT,
            theme=theme,
        )
    frames = _calendar_frames()
    notify("docs/animations/variance-clock-{light,dark}.gif")
    save_animation(
        lambda ax, p, i: _draw_clock(ax, p, i, frames),
        frames[0],
        ANIMATIONS / "variance-clock",
        style=style,
        figsize=(8.0, 4.8),
        fps=14,
        hold_last=40,
    )


def main() -> None:
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    make_animations()
    for name, command, shown, title, columns in TERMINAL_SHOTS:
        for theme in ("light", "dark"):
            out = SCREENSHOTS / f"{name}-{theme}.png"
            notify(f"{out.relative_to(ROOT)}")
            terminal_screenshot(
                command,
                out,
                title=title,
                display_command=shown,
                theme=theme,
                columns=columns,
                cwd=ROOT,
            )
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "variance-calendar.html"
        import subprocess

        subprocess.run(
            [PY, "-m", "varclock.cli", "report", "SPX", "--source", "cboe", "--out", str(report)],
            check=True,
            cwd=ROOT,
            capture_output=True,
        )
        for theme in ("light", "dark"):
            out = SCREENSHOTS / f"report-{theme}.png"
            notify(f"{out.relative_to(ROOT)}")
            report_screenshot(report, out, width=1100, color_scheme=theme)


if __name__ == "__main__":
    main()
