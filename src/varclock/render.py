"""Rich renderings of the analysis results (the command-line output)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

import numpy as np
from rich import box
from rich.console import Group
from rich.table import Table
from rich.text import Text

from .analysis import CalendarResult
from .bounds import VarianceBand
from .eventstudy import EventStudy
from .indices import IndexResult, IndexUnavailableError

__all__ = [
    "bounds_view",
    "calendar_view",
    "events_view",
    "header",
    "replicate_view",
    "study_view",
    "upcoming_events_view",
]

_BAR = 22
#: Terminal width from which the calendar shows the strip and the bar columns.
FULL_WIDTH = 112


def _day(d: date) -> str:
    return d.strftime("%a %m-%d")


def _interval(lo: float, hi: float, digits: int = 1, scale: float = 1.0) -> str:
    if abs(hi - lo) < 1e-9:
        return "-"
    return f"[{lo * scale:.{digits}f}, {hi * scale:.{digits}f}]"


def header(result: CalendarResult) -> Group:
    snap = result.snapshot
    first = Text()
    first.append(f"{snap.symbol} variance calendar", style="bold")
    first.append(f"  snapshot {snap.asof:%a %Y-%m-%d %H:%M} ET from {snap.source}", style="dim")
    second = Text(
        f"downloaded {snap.fetched_at:%Y-%m-%d %H:%M} UTC | Treasury curve "
        f"{result.curve.curve_date} | {len(result.strip.points)} SPXW expiries | "
        f"{result.n_boot} draws, seed {result.seed}",
        style="dim",
    )
    return Group(first, second)


def calendar_view(result: CalendarResult, sessions: int = 25, width: int | None = None) -> Group:
    """Table of forward volatility per session with intervals, multiples and event labels.

    Below ``FULL_WIDTH`` columns the strip and bar columns are left out so that no column wraps
    or is cut; ``width`` defaults to the full layout.
    """
    rows = result.rows[:sessions]
    top = max(r.fwd_vol_hi for r in rows)
    full = width is None or width >= FULL_WIDTH
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, header_style="bold")
    table.add_column("Session", no_wrap=True)
    table.add_column("Fwd vol", justify="right", no_wrap=True)
    table.add_column("90% interval", justify="right", no_wrap=True)
    table.add_column("x base", justify="right", no_wrap=True)
    table.add_column("90% interval", justify="right", no_wrap=True)
    if full:
        table.add_column("Strip", justify="right", no_wrap=True)
        table.add_column("", no_wrap=True)
    table.add_column("Event", no_wrap=True)
    for r in rows:
        label = ", ".join(r.events)
        if r.weekend:
            label = ("weekend, " + label) if label else "weekend"
        hot = bool(r.events)
        style = "yellow" if hot else ("dim" if r.weekend else None)
        bar = "\u2588" * round(_BAR * r.fwd_vol / top)
        bar_style = "yellow" if hot else ("grey50" if r.weekend else "blue")
        strip = "-" if r.observed_vol is None else f"{r.observed_vol * 100:.1f}"
        cells: list[str | Text] = [
            _day(r.session) + ("*" if r.fraction < 1 else ""),
            f"{r.fwd_vol * 100:.1f}%",
            _interval(r.fwd_vol_lo, r.fwd_vol_hi, scale=100),
            f"{r.multiple:.2f}",
            _interval(r.multiple_lo, r.multiple_hi, digits=2),
        ]
        if full:
            cells += [strip, Text(bar, style=bar_style)]
        table.add_row(*cells, label, style=style)
    fit = result.fit
    foot = Text()
    foot.append(
        f"Base vol {fit.base_vol[0] * 100:.1f}% on the first session, "
        f"{fit.base_vol[len(rows) - 1] * 100:.1f}% on the last shown; "
        f"a weekend carries {fit.weekend_ratio:.2f} of a base session "
        f"(base family {fit.family}, chi2/dof {fit.chi2_dof:.2f}).",
        style="dim",
    )
    notes = []
    if result.strip.skipped:
        skipped = ", ".join(d.strftime("%m-%d") for d, _ in result.strip.skipped)
        notes.append(Text(f"Expiries without usable quotes, skipped: {skipped}.", style="yellow"))
    if result.strip.significant_violations.any():
        bad = [
            e
            for e, v in zip(result.strip.expiries, result.strip.significant_violations, strict=True)
            if v
        ]
        notes.append(
            Text(f"Calendar arbitrage in mid quotes at: {', '.join(map(str, bad))}", style="red")
        )
    notes += [Text(note, style="yellow") for note in result.notes]
    return Group(table, foot, *notes)


def events_view(result: CalendarResult) -> Group:
    """Fitted event bumps with bootstrap intervals."""
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, header_style="bold")
    table.add_column("Event")
    table.add_column("Session")
    table.add_column("Extra variance", justify="right")
    table.add_column("x base", justify="right")
    table.add_column("90% interval", justify="right")
    table.add_column("1-sd move", justify="right")
    for e in result.event_rows:
        name = {"EARNINGS": "Mega-cap earnings, per report"}.get(e.kind, e.label)
        if e.kind in {"CPI", "NFP"}:
            period = e.label.split(" for ", 1)[-1]
            name = f"{'CPI' if e.kind == 'CPI' else 'Jobs report'}, {period}"
        if e.kind == "ELECTION":
            name = "U.S. election, day after"
        if e.kind == "FOMC":
            name = "FOMC decision"
        table.add_row(
            name,
            _day(e.session),
            f"{e.bump_var * 1e4:.2f} %\u00b2",
            f"{e.multiple:.2f}",
            _interval(e.multiple_lo, e.multiple_hi, digits=2),
            f"{e.implied_sd * 100:.2f}%",
            style="yellow" if e.multiple_lo > 1.05 else None,
        )
    return Group(table)


def replicate_view(
    snapshot_label: str,
    official: Mapping[str, float],
    results: Mapping[str, Mapping[str, IndexResult | IndexUnavailableError]],
) -> Group:
    """Rebuilt versus official index closes, one column group per adapter."""
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, header_style="bold")
    table.add_column("Index")
    table.add_column("Official", justify="right")
    for source in results:
        table.add_column(f"{source} rebuilt", justify="right")
        table.add_column("error", justify="right")
    table.add_column("Terms used")
    names = list(official)
    for name in names:
        row = [name, f"{official[name]:.2f}"]
        terms = ""
        for per_index in results.values():
            r = per_index.get(name)
            if isinstance(r, IndexResult):
                err = r.value - official[name]
                colour = "green" if abs(err) < 0.05 else ("yellow" if abs(err) < 0.15 else "red")
                row += [f"{r.value:.3f}", Text(f"{err:+.3f}", style=colour)]
                if not terms:
                    near = "" if r.near is None else f"{r.near.strip.root} {r.near.strip.expiry} + "
                    terms = f"{near}{r.next.strip.root} {r.next.strip.expiry}"
            else:
                row += ["n/a", ""]
        row.append(terms)
        table.add_row(*row)
    return Group(Text(snapshot_label, style="dim"), table)


def _pm(lo: float, hi: float, scale: float = 1.0, digits: int = 2) -> str:
    return f"[{lo * scale:.{digits}f}, {hi * scale:.{digits}f}]"


def study_view(
    study: EventStudy, kinds: Sequence[str] = ("CPI", "FOMC", "NFP"), seed: int = 0
) -> Group:
    """Implied versus realised event-day variance with 95% bootstrap intervals."""
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, header_style="bold")
    table.add_column("Sample")
    table.add_column("n", justify="right")
    table.add_column("Implied 1-sd", justify="right")
    table.add_column("Mean |move|", justify="right")
    table.add_column("Realised / implied var", justify="right")
    table.add_column("95% interval", justify="right")
    table.add_column("Premium (%\u00b2)", justify="right")
    table.add_column("95% interval", justify="right")
    samples = [
        (k, study.summarise(study.by_kind(k), np.random.default_rng(seed)))
        for k in kinds
        if study.by_kind(k)
    ]
    samples.append(("Ordinary days", study.summarise_control(np.random.default_rng(seed))))
    for name, s in samples:
        control = name == "Ordinary days"
        table.add_row(
            name,
            str(s["n"]),
            f"{s['implied_sd_mean'] * 100:.2f}%",
            "-" if control else f"{s['realised_abs'] * 100:.2f}%",
            f"{s['ratio']:.2f}",
            _pm(s["ratio_lo"], s["ratio_hi"]),
            f"{s['premium_var'] * 1e4:+.2f}",
            _pm(s["premium_var_lo"], s["premium_var_hi"], 1e4),
            style="dim" if control else None,
        )
    foot = Text(
        f"{study.start} to {study.end}. Implied = (VIX1D/100)^2 x 390/102,060 from the previous "
        "close; realised = squared S&P 500 close-to-close log return. Premium = implied minus "
        "realised, in squared percent. Ordinary days: 20-session block bootstrap.",
        style="dim",
    )
    return Group(table, foot)


def upcoming_events_view(events: Sequence, asof: date, horizon: int = 45) -> Group:
    """Calendar of the events whose session falls within ``horizon`` days after ``asof``."""
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, header_style="bold")
    table.add_column("Session")
    table.add_column("Kind")
    table.add_column("Event")
    shown = [e for e in events if 0 <= (e.session - asof).days <= horizon]
    for e in shown:
        table.add_row(_day(e.session), e.kind, e.label)
    if not shown:
        table.add_row("-", "-", "no events in the calendar files for this window")
    foot = Text(
        f"Sources: Federal Reserve, BLS, election law, Yahoo Finance; {len(shown)} events "
        f"within {horizon} days of {asof}. Earnings dates are company-announced or projected.",
        style="dim",
    )
    return Group(table, foot)


def _vol(total_variance: float, t: float) -> float:
    return float(np.sqrt(max(total_variance, 0.0) / t)) * 100


def bounds_view(
    expiries: Sequence[date], bands: Sequence[VarianceBand], minutes: Sequence[int]
) -> Group:
    """Mid-quote variance with its arbitrage-free LP band and the naive bid/ask band."""
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, header_style="bold")
    table.add_column("Expiry")
    table.add_column("Vol (mid)", justify="right")
    table.add_column("LP band", justify="right")
    table.add_column("Bid/ask band", justify="right")
    table.add_column("LP width / naive", justify="right")
    table.add_column("Fwd var step (%\u00b2)", justify="right")
    table.add_column("LP band (%\u00b2)", justify="right")
    year = 525_600.0
    prev = None
    for d, b, m in zip(expiries, bands, minutes, strict=True):
        t = m / year

        step = "-" if prev is None else f"{(b.mid - prev.mid) * 1e4:.2f}"
        band = "-"
        if prev is not None and prev.feasible and b.feasible:
            lo, hi = b.lo - prev.hi, b.hi - prev.lo
            band = f"[{lo * 1e4:.2f}, {hi * 1e4:.2f}]"
        if b.feasible:
            table.add_row(
                _day(d),
                f"{_vol(b.mid, t):.2f}",
                f"[{_vol(b.lo, t):.2f}, {_vol(b.hi, t):.2f}]",
                f"[{_vol(b.naive_lo, t):.2f}, {_vol(b.naive_hi, t):.2f}]",
                "-" if b.naive_width <= 0 else f"{b.width / b.naive_width:.2f}",
                step,
                band,
                style="yellow" if b.widen > 0 else None,
            )
        else:
            table.add_row(
                _day(d), f"{_vol(b.mid, t):.2f}", "infeasible", "", "", step, band, style="red"
            )
        prev = b
    foot = Text(
        "Volatilities in percent, annualised over calendar time. LP band: smallest and largest "
        "variance among arbitrage-free call curves inside the quotes; rows in yellow needed the "
        "quotes widened by under a tick to admit any such curve.",
        style="dim",
    )
    return Group(table, foot)
