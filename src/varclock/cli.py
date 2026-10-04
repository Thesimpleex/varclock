"""Command-line interface: ``varclock calendar | replicate | events | report | snapshot``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from rich.console import Console

from . import __version__
from .analysis import build_calendar
from .events import EVENT_FILES, load_events
from .export import calendar_records, study_records, upcoming_records
from .indices import replicate
from .loaders import SOURCES, close_is_due, obtain_curve, obtain_snapshot, official_closes
from .render import (
    bounds_view,
    calendar_view,
    events_view,
    header,
    replicate_view,
    study_view,
    upcoming_events_view,
)
from .tradingcal import resolve_asof

INDEX_NAMES = ("VIX", "VIX9D", "VIX3M", "VIX1D")


def _console(stderr: bool = False) -> Console:
    return Console(highlight=False, stderr=stderr)


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text}")
    return value


def _min_expiries(text: str) -> int:
    value = int(text)
    if value < 3:
        raise argparse.ArgumentTypeError(f"needs at least 3 consecutive expiries, got {text}")
    return value


def _add_snapshot_args(p: argparse.ArgumentParser, multi_source: bool = False) -> None:
    p.add_argument("symbol", nargs="?", default="SPX", help="underlying (only SPX is supported)")
    if multi_source:
        p.add_argument(
            "--source",
            action="append",
            choices=[*SOURCES, "all"],
            help="data source, repeatable (default: yahoo)",
        )
    else:
        p.add_argument("--source", choices=SOURCES, default="yahoo", help="option-chain source")
    p.add_argument("--snapshot", type=Path, help="use this cached snapshot file")
    p.add_argument("--refresh", action="store_true", help="download a fresh snapshot")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="varclock",
        description="The market's variance calendar, read model-free from one option-chain "
        "snapshot. Not affiliated with Cboe; set VARCLOCK_USER_AGENT to 'Name email'.",
    )
    parser.add_argument("--version", action="version", version=f"varclock {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    cal = sub.add_parser("calendar", help="forward volatility for the next sessions")
    _add_snapshot_args(cal)
    cal.add_argument(
        "--sessions", type=_positive_int, default=25, help="sessions to show (default 25)"
    )
    cal.add_argument(
        "--boot", type=_positive_int, default=400, help="bootstrap draws (default 400)"
    )
    cal.add_argument("--seed", type=int, default=0)
    cal.add_argument("--family", choices=("meanrev", "spline"), default="meanrev")
    cal.add_argument(
        "--coverage",
        choices=("own", "common"),
        default="own",
        help="strike windows of the strip: each expiry's own range or the common window",
    )
    cal.add_argument("--json", action="store_true", help="print JSON instead of tables")

    rep = sub.add_parser("replicate", help="rebuild VIX, VIX9D, VIX3M, VIX1D and compare")
    _add_snapshot_args(rep, multi_source=True)

    ev = sub.add_parser("events", help="upcoming events, or the event history study")
    ev.add_argument("--history", action="store_true", help="implied vs realised on past events")
    ev.add_argument(
        "--days", type=_positive_int, default=45, help="horizon of the upcoming list (days)"
    )
    ev.add_argument("--json", action="store_true", help="print JSON instead of tables")
    ev.add_argument("--kinds", nargs="+", default=["CPI", "FOMC", "NFP"], choices=list(EVENT_FILES))
    ev.add_argument("--asof", type=date.fromisoformat, help="list events after this date")
    ev.add_argument("--seed", type=int, default=0)

    rpt = sub.add_parser("report", help="write the self-contained HTML variance calendar")
    _add_snapshot_args(rpt)
    rpt.add_argument("--out", type=Path, default=Path("variance-calendar.html"))
    rpt.add_argument("--boot", type=_positive_int, default=400)
    rpt.add_argument("--seed", type=int, default=0)

    bnd = sub.add_parser("bounds", help="arbitrage-free LP bands for each daily expiry")
    _add_snapshot_args(bnd)
    bnd.add_argument(
        "--expiries",
        type=_min_expiries,
        default=12,
        help="expiries to show (minimum 3, default 12)",
    )

    snap = sub.add_parser("snapshot", help="download and cache an option-chain snapshot")
    snap.add_argument("symbol", nargs="?", default="SPX")
    snap.add_argument("--source", choices=SOURCES, default="yahoo")
    return parser


def _notify(console: Console):
    def say(message: str) -> None:
        console.print(message, style="dim", markup=False)

    return say


def _cmd_calendar(args: argparse.Namespace, console: Console) -> int:
    log = _console(stderr=True) if args.json else console
    snap = obtain_snapshot(args.source, args.symbol, args.snapshot, args.refresh, _notify(log))
    curve = obtain_curve(snap)
    result = build_calendar(
        snap, curve, family=args.family, n_boot=args.boot, seed=args.seed, coverage=args.coverage
    )
    if args.json:
        records = calendar_records(result)
        records["sessions"] = records["sessions"][: args.sessions]
        console.print_json(json.dumps(records))
        return 0
    console.print(header(result))
    console.print(calendar_view(result, args.sessions, console.width))
    console.print(events_view(result))
    return 0


def _cmd_replicate(args: argparse.Namespace, console: Console) -> int:
    sources = args.source or ["yahoo"]
    if "all" in sources:
        sources = list(SOURCES)
    results = {}
    official: dict[str, float] = {}
    stamps: list[str] = []
    curve_date = None
    for source in dict.fromkeys(sources):
        snap = obtain_snapshot(source, args.symbol, args.snapshot, args.refresh, _notify(console))
        curve = obtain_curve(snap)
        results[source] = replicate(snap, curve, INDEX_NAMES)
        official = official_closes(snap, INDEX_NAMES)
        stamps.append(f"{snap.source} valued {snap.asof:%a %Y-%m-%d %H:%M} ET")
        curve_date = curve.curve_date
    label = (
        f"Snapshots: {', '.join(stamps)}; Treasury curve {curve_date}; "
        "official closes from Cboe index history"
    )
    if not official:
        if close_is_due(snap):
            console.print(
                f"The official close of {snap.asof:%Y-%m-%d} is not in the Cboe index history yet; "
                "try again after Cboe has published it.",
                style="red",
            )
        else:
            console.print(
                "The snapshot predates the close; no official close to compare.", style="red"
            )
        return 1
    console.print(replicate_view(label, official, results))
    return 0


def _cmd_events(args: argparse.Namespace, console: Console) -> int:
    from .eventstudy import run_event_study
    from .sources import load_index_history

    if args.history:
        events = load_events(args.kinds)
        latest = resolve_asof(datetime.now(UTC)).date()
        study = run_event_study(
            events,
            load_index_history("VIX1D", through=latest),
            load_index_history("SPX", through=latest),
            kinds=tuple(args.kinds),
        )
        if args.json:
            console.print_json(json.dumps(study_records(study, args.kinds, args.seed)))
            return 0
        console.print(study_view(study, tuple(args.kinds), args.seed))
        if study.end < latest:
            console.print(
                f"The index histories end {study.end}, before the last completed session "
                f"{latest}: Cboe has not published later closes.",
                style="yellow",
            )
        return 0
    today = args.asof or date.today()
    events = load_events(None, today - timedelta(days=1), today + timedelta(days=args.days))
    if args.json:
        console.print_json(json.dumps(upcoming_records(events, today, args.days)))
        return 0
    console.print(upcoming_events_view(events, today, args.days))
    return 0


def _cmd_report(args: argparse.Namespace, console: Console) -> int:
    from .report import write_report

    snap = obtain_snapshot(args.source, args.symbol, args.snapshot, args.refresh, _notify(console))
    curve = obtain_curve(snap)
    result = build_calendar(snap, curve, n_boot=args.boot, seed=args.seed)
    replication: dict[str, tuple[float, float]] = {}
    note = None
    try:
        official = official_closes(snap, INDEX_NAMES)
        for name, rebuilt in replicate(snap, curve, INDEX_NAMES).items():
            if name in official and hasattr(rebuilt, "value"):
                replication[name] = (official[name], rebuilt.value)
        if not official and close_is_due(snap):
            note = f"the official close of {snap.asof:%Y-%m-%d} is not in the Cboe history yet"
    except RuntimeError as exc:
        note = f"official closes unavailable ({exc})"
    path = write_report(result, args.out, replication)
    console.print(f"Wrote {path}")
    if note and not replication:
        console.print(f"The report has no replication table: {note}.", style="yellow")
    return 0


def _cmd_bounds(args: argparse.Namespace, console: Console) -> int:
    from .bounds import strip_bands
    from .strip import build_strip

    snap = obtain_snapshot(args.source, args.symbol, args.snapshot, args.refresh, _notify(console))
    curve = obtain_curve(snap)
    strip = build_strip(snap, curve, max_expiries=args.expiries)
    bands = strip_bands(snap, strip)
    console.print(
        f"{snap.symbol} snapshot {snap.asof:%a %Y-%m-%d %H:%M} ET from {snap.source}", style="dim"
    )
    console.print(bounds_view(strip.expiries, bands, [p.minutes for p in strip.points]))
    return 0


def _cmd_snapshot(args: argparse.Namespace, console: Console) -> int:
    snap = obtain_snapshot(args.source, args.symbol, None, True, _notify(console))
    console.print(
        f"Cached {snap.source} snapshot of {snap.symbol}: {len(snap.strips)} expiries, "
        f"valuation time {snap.asof:%Y-%m-%d %H:%M} ET"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    console = _console()
    handlers = {
        "calendar": _cmd_calendar,
        "replicate": _cmd_replicate,
        "events": _cmd_events,
        "report": _cmd_report,
        "bounds": _cmd_bounds,
        "snapshot": _cmd_snapshot,
    }
    try:
        return handlers[args.command](args, console)
    except json.JSONDecodeError as exc:
        console.print(f"error: cannot parse a JSON file: {exc}", style="red", markup=False)
        return 2
    except OSError as exc:
        what = exc.filename or "a data file"
        console.print(
            f"error: cannot read {what}: {exc.strerror or exc}", style="red", markup=False
        )
        return 2
    except (ValueError, RuntimeError, ImportError) as exc:
        console.print(f"error: {exc}", style="red", markup=False)
        return 2


if __name__ == "__main__":
    sys.exit(main())
