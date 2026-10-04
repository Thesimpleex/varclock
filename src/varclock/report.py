"""Self-contained HTML variance-calendar report (inline CSS, embedded figure)."""

from __future__ import annotations

import base64
import html
import io
from collections.abc import Mapping
from pathlib import Path

from .analysis import CalendarResult, EventRow

__all__ = ["render_report", "write_report"]

_CSS = """
:root { --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --surface:#fcfcfb;
  --blue:#2a78d6; --orange:#eb6834; --tint:#f0efec; }
* { box-sizing: border-box; }
body { margin:0; background:var(--surface); color:var(--ink);
  font:15px/1.5 "Helvetica Neue", Helvetica, Arial, sans-serif; }
main { max-width:1040px; margin:0 auto; padding:56px 40px 72px; }
.eyebrow { font-size:12px; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); }
h1 { font-size:34px; line-height:1.15; margin:8px 0 6px; font-weight:600; letter-spacing:-.01em; }
.sub { color:var(--ink2); margin:0 0 36px; }
h2 { font-size:19px; margin:44px 0 12px; font-weight:600; }
.kpis { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin:0 0 8px; }
.kpi { border-top:2px solid var(--ink); padding-top:10px; }
.kpi .v { font-size:28px; font-weight:600; letter-spacing:-.01em; }
.kpi .l { font-size:12.5px; color:var(--ink2); }
.kpi .i { font-size:12px; color:var(--muted); }
img.fig { width:100%; height:auto; display:block; margin:8px 0 4px; }
table { border-collapse:collapse; width:100%; font-size:13.5px; font-variant-numeric:tabular-nums; }
th { text-align:right; font-weight:600; color:var(--ink2); border-bottom:1px solid var(--ink);
  padding:6px 10px; }
td { text-align:right; padding:5px 10px; border-bottom:1px solid var(--grid); }
th:first-child, td:first-child, th.l, td.l { text-align:left; }
tr.event td { background:#fdf3ee; }
tr.weekend td { color:var(--muted); }
.bar { display:inline-block; height:9px; background:var(--blue); border-radius:2px;
  vertical-align:middle; }
tr.event .bar { background:var(--orange); }
.note { color:var(--ink2); font-size:13.5px; max-width:780px; }
.fine { color:var(--muted); font-size:12px; margin-top:40px; border-top:1px solid var(--grid);
  padding-top:14px; }
.ok { color:#006300; } .warn { color:#9a6a00; } .bad { color:#c93a39; }
@media (max-width:760px) {
  .kpis { grid-template-columns:repeat(2,1fr); } main { padding:32px 20px; } }
@media (prefers-color-scheme: dark) {
  :root { --ink:#f2f1ed; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --surface:#1a1a19;
    --blue:#3987e5; --orange:#d95926; --tint:#242422; }
  tr.event td { background:#2a2119; }
  .ok { color:#3cc993; } .warn { color:#f5b83a; } .bad { color:#f08c8b; }
}
"""


def _figure_png_base64(result: CalendarResult, sessions: int) -> tuple[str, str]:
    """Light and dark PNG of the calendar figure as base64 strings."""
    try:
        import matplotlib.pyplot as plt

        from .figures import build_calendar_figure
        from .viz import style
    except ImportError as exc:  # pragma: no cover - matplotlib is an optional extra
        raise ImportError("the HTML report needs `pip install varclock[plot]`") from exc
    out = []
    for theme in ("light", "dark"):
        with style(theme) as palette:
            fig = build_calendar_figure(result, sessions)(palette)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=170)
            plt.close(fig)
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out[0], out[1]


def _e(text: object) -> str:
    return html.escape(str(text))


def render_report(
    result: CalendarResult,
    replication: Mapping[str, tuple[float, float]] | None = None,
    sessions: int = 25,
) -> str:
    """HTML for the variance calendar of ``result``.

    ``replication`` maps an index name to ``(official close, rebuilt value)`` for the same
    snapshot and is shown as a validation block when given.
    """
    snap, fit = result.snapshot, result.fit
    rows = result.rows[:sessions]
    top = max(r.fwd_vol_hi for r in rows)
    events: dict[str, EventRow] = {}
    for e in result.event_rows:
        if e.kind != "EARNINGS":
            events.setdefault(e.kind, e)  # the first (nearest) event of each kind

    def kpi(value: str, label: str, info: str) -> str:
        return (
            f'<div class="kpi"><div class="v">{_e(value)}</div>'
            f'<div class="l">{_e(label)}</div><div class="i">{_e(info)}</div></div>'
        )

    tiles = [
        kpi(
            f"{fit.base_vol[0] * 100:.1f}% to {fit.base_vol[len(rows) - 1] * 100:.1f}%",
            "Base volatility, first to last session",
            "ordinary session, no event, no weekend",
        )
    ]
    for kind, name in (
        ("CPI", "CPI"),
        ("FOMC", "FOMC decision"),
        ("ELECTION", "Election, day after"),
    ):
        e = events.get(kind)
        if e is not None:
            tiles.append(
                kpi(
                    f"{e.multiple:.2f}x",
                    f"{name}, {e.session:%d %b}",
                    f"90% interval {e.multiple_lo:.2f} to {e.multiple_hi:.2f}",
                )
            )
    tiles.append(
        kpi(
            f"{fit.weekend_ratio:.2f}",
            "A weekend, in base sessions",
            "variance of Saturday and Sunday",
        )
    )
    tiles = tiles[:4] if len(tiles) > 4 else tiles

    body_rows = []
    for r in rows:
        cls = "event" if r.events else ("weekend" if r.weekend else "")
        label = ", ".join(r.events)
        if r.weekend:
            label = f"weekend, {label}" if label else "weekend"
        interval = (
            "-"
            if abs(r.multiple_hi - r.multiple_lo) < 1e-9
            else f"{r.multiple_lo:.2f} to {r.multiple_hi:.2f}"
        )
        strip = "-" if r.observed_vol is None else f"{r.observed_vol * 100:.1f}%"
        width = round(150 * r.fwd_vol / top)
        body_rows.append(
            f'<tr class="{cls}"><td>{r.session:%a %d %b}</td>'
            f"<td>{r.fwd_vol * 100:.1f}%</td>"
            f"<td>{r.fwd_vol_lo * 100:.1f} to {r.fwd_vol_hi * 100:.1f}</td>"
            f"<td>{r.multiple:.2f}</td><td>{interval}</td><td>{strip}</td>"
            f'<td class="l"><span class="bar" style="width:{width}px"></span></td>'
            f'<td class="l">{_e(label)}</td></tr>'
        )

    event_rows = []
    for e in result.event_rows:
        name = {"EARNINGS": "Mega-cap earnings, per report"}.get(e.kind, e.label)
        event_rows.append(
            f'<tr><td class="l">{_e(name)}</td><td>{e.session:%a %d %b}</td>'
            f"<td>{e.bump_var * 1e4:.2f} %&sup2;</td><td>{e.multiple:.2f}</td>"
            f"<td>{e.multiple_lo:.2f} to {e.multiple_hi:.2f}</td>"
            f"<td>{e.implied_sd * 100:.2f}%</td></tr>"
        )

    validation = ""
    if replication:
        vrows = []
        for name, (official, rebuilt) in replication.items():
            err = rebuilt - official
            cls = "ok" if abs(err) < 0.05 else ("warn" if abs(err) < 0.15 else "bad")
            vrows.append(
                f"<tr><td>{_e(name)}</td><td>{official:.2f}</td><td>{rebuilt:.3f}</td>"
                f'<td class="{cls}">{err:+.3f}</td></tr>'
            )
        validation = (
            "<h2>Check against the official closes</h2>"
            "<table><tr><th>Index</th><th>Official close</th><th>Rebuilt here</th>"
            f"<th>Error</th></tr>{''.join(vrows)}</table>"
            '<p class="note">The same code rebuilds the Cboe volatility indices from this '
            "snapshot; the calendar above uses the identical variance engine.</p>"
        )

    note_skipped = ""
    if result.strip.skipped:
        days = ", ".join(f"{d:%d %b}" for d, _ in result.strip.skipped)
        note_skipped = f'<p class="note warn">Expiries without usable quotes, skipped: {days}.</p>'

    fig_light, fig_dark = _figure_png_base64(result, sessions)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(snap.symbol)} variance calendar</title><style>{_CSS}</style></head>
<body><main>
<div class="eyebrow">varclock &middot; model-free variance from option chains</div>
<h1>{_e(snap.symbol)} variance calendar</h1>
<p class="sub">Snapshot {snap.asof:%A %d %B %Y, %H:%M} ET &middot; source {_e(snap.source)}
&middot; Treasury curve of {result.curve.curve_date:%d %b %Y}
&middot; {len(result.strip.points)} daily SPXW expiries
&middot; {result.n_boot} bootstrap draws (seed {result.seed})</p>
<div class="kpis">{"".join(tiles)}</div>
<picture>
<source media="(prefers-color-scheme: dark)" srcset="data:image/png;base64,{fig_dark}">
<img class="fig" alt="Forward volatility per session with events"
 src="data:image/png;base64,{fig_light}">
</picture>
{note_skipped}
<h2>Session by session</h2>
<table><tr><th>Session</th><th>Forward vol</th><th>90% interval</th><th>x base</th>
<th>90% interval</th><th>Strip</th><th class="l"></th><th class="l">Event</th></tr>
{"".join(body_rows)}</table>
<p class="note">Forward vol is the fitted volatility of the session, annualised over 252
sessions; x base is that session's variance over the base variance of an ordinary session;
Strip is the model-free forward volatility read directly from adjacent daily expiries.</p>
<h2>What each event adds</h2>
<table><tr><th class="l">Event</th><th>Session</th><th>Extra variance</th><th>x base</th>
<th>90% interval</th><th>1-sd move</th></tr>{"".join(event_rows)}</table>
{validation}
<p class="fine">Illustrative research, not investment advice. Quotes: {_e(snap.source)}
snapshot downloaded {snap.fetched_at:%Y-%m-%d %H:%M} UTC. varclock is not affiliated with,
endorsed by or sponsored by Cboe Global Markets or S&amp;P Dow Jones Indices; VIX and SPX are
their trademarks. Option quotes from the Cboe delayed-quotes feed are for personal,
non-commercial use. Intervals are bootstrap intervals conditional on the fitted model family
and do not cover model misspecification.</p>
</main></body></html>
"""


def write_report(
    result: CalendarResult,
    path: str | Path,
    replication: Mapping[str, tuple[float, float]] | None = None,
    sessions: int = 25,
) -> Path:
    """Write the report to ``path`` and return the path."""
    path = Path(path)
    path.write_text(render_report(result, replication, sessions), encoding="utf-8")
    return path
