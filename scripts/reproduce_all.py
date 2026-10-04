"""Run every experiment, write all figures and docs/results.json.

Usage::

    python scripts/reproduce_all.py            # use cached snapshots, download if missing
    python scripts/reproduce_all.py --refresh  # download fresh option chains and index data

The historical event study and the validation need no option snapshot. The calendar,
replication and bounds experiments use the newest cached snapshot per source, so a later run
reports the market of the later date; the snapshot vintage is recorded in the results file.
Raw quotes live in the gitignored ``data/raw`` and are never committed.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp_bounds
import exp_calendar
import exp_event_study
import exp_replicate
import exp_validation
from common import ROOT, notify, read_json, versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="download fresh data first")
    args = parser.parse_args()
    started = time.time()

    if args.refresh:
        from varclock.sources import load_index_history

        for name in ("VIX", "VIX9D", "VIX3M", "VIX1D", "SPX"):
            load_index_history(name, refresh=True)

    steps = (
        ("validation", exp_validation.run, {}),
        ("replication", exp_replicate.run, {"refresh": args.refresh}),
        ("calendar", exp_calendar.run, {"refresh": args.refresh}),
        ("bounds", exp_bounds.run, {}),
        ("event_study", exp_event_study.run, {}),
    )
    for name, fn, kwargs in steps:
        notify(f"running {name} ...")
        fn(**kwargs)

    from varclock.sources import load_index_history

    histories = {n: load_index_history(n) for n in ("VIX", "VIX9D", "VIX3M", "VIX1D", "SPX")}
    replication = read_json("replication")
    results = {
        "generated_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "versions": versions(),
        "data_vintage": {
            "option_snapshots": replication["snapshots"],
            "treasury_curve_date": replication["treasury_curve_date"],
            "official_index_history_last_observation": {
                n: max(h).isoformat() for n, h in histories.items()
            },
            "event_calendars": "src/varclock/calendars/*.csv (sources and retrieval in each file)",
        },
        "validation": read_json("validation"),
        "replication": replication,
        "calendar": read_json("calendar"),
        "bounds": read_json("bounds"),
        "event_study": read_json("event_study"),
    }
    out = ROOT / "docs" / "results.json"
    import json

    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    notify(f"wrote {out} in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
