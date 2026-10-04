"""Figure builders (matplotlib). Each ``build_*`` returns a function ``palette -> Figure`` for
``varclock.viz.save_figure``; all data is computed before drawing."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .analysis import CalendarResult
from .eventstudy import EventStudy

__all__ = [
    "build_calendar_figure",
    "build_replication_figure",
    "build_robustness_figure",
    "build_study_figure",
]

Builder = Callable[[dict[str, Any]], Figure]


def _calendar_title(result: CalendarResult) -> str:
    big = [
        e
        for e in result.event_rows
        if e.kind in {"CPI", "FOMC", "ELECTION"} and e.multiple_lo > 1.05
    ]
    if not big:
        return "Forward volatility per session"
    lo = min(e.multiple for e in big)
    hi = max(e.multiple for e in big)
    return (
        f"The market prices CPI, FOMC and election sessions at {lo:.1f} to {hi:.1f} times "
        "a base session"
    )


def build_calendar_figure(result: CalendarResult, sessions: int = 25) -> Builder:
    rows = result.rows[:sessions]
    x = np.arange(len(rows))
    vol = np.array([r.fwd_vol for r in rows]) * 100
    lo = np.array([r.fwd_vol_lo for r in rows]) * 100
    hi = np.array([r.fwd_vol_hi for r in rows]) * 100
    base = np.array([r.base_vol for r in rows]) * 100
    obs = np.array([np.nan if r.observed_vol is None else r.observed_vol for r in rows]) * 100
    band = (
        np.array([np.nan if r.observed_vol_band is None else r.observed_vol_band for r in rows])
        * 100
    )
    labels = [r.session.strftime("%m-%d") if r.session.weekday() == 0 else "" for r in rows]
    by_session = {e.session: e for e in result.event_rows if e.kind != "EARNINGS"}
    annot = []
    for i, r in enumerate(rows):
        names = [e for e in r.events if not e.startswith("earnings")]
        if names:
            ev = by_session.get(r.session)
            distinct = ev is None or ev.multiple_lo > 1.05
            text = (
                f"{', '.join(names)} {r.multiple:.1f}x" if distinct else f"{', '.join(names)} n.s."
            )
            annot.append((i, text, distinct))
        elif r.events:
            annot.append((i, r.events[0].replace("earnings ", "earn. "), False))
    title = _calendar_title(result)
    snap = result.snapshot
    sub = (
        f"{snap.symbol} options, {snap.source} snapshot of {snap.asof:%Y-%m-%d %H:%M} ET; "
        f"{result.n_boot} bootstrap draws, 90% intervals; n.s.: interval of the multiple includes 1"
    )

    def build(p: dict[str, Any]) -> Figure:
        fig, ax = plt.subplots(figsize=(9.6, 5.0))
        c_fit, c_base, c_obs = p["series"][0], p["series"][1], p["muted"]
        ax.fill_between(x, lo, hi, color=c_fit, alpha=0.22, linewidth=0)
        ax.errorbar(
            x,
            obs,
            yerr=band,
            fmt="o",
            ms=4,
            color=c_obs,
            ecolor=c_obs,
            elinewidth=0.7,
            alpha=0.4,
            capsize=0,
            label="Model-free strip, with quote band",
            zorder=2,
        )
        ax.plot(
            x, base, color=c_base, lw=1.6, label="Base volatility (no event, no weekend)", zorder=3
        )
        ax.plot(
            x, vol, color=c_fit, lw=2.0, label="Forward volatility, fitted (90% interval)", zorder=4
        )
        top = float(np.nanmax(hi))
        reach = np.fmax(hi, np.nan_to_num(obs + band, nan=-np.inf))
        levels: list[tuple[float, int]] = []
        for i, text, distinct in annot:
            level = 0
            while any(abs(i - j) < 5 and level == lv for j, lv in levels):
                level += 1
            levels.append((i, level))
            ax.annotate(
                text,
                (i, reach[i]),
                xytext=(0, 6 + 13 * level),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5 if distinct and "x" in text[-3:] else 7.5,
                color=p["ink"] if distinct and "x" in text[-3:] else p["ink_secondary"],
                arrowprops={"arrowstyle": "-", "color": p["muted"], "lw": 0.5, "shrinkB": 1}
                if level
                else None,
            )
        depth = max((lv for _, lv in levels), default=0)
        ax.set_ylim(max(0.0, float(np.nanmin(lo)) - 4.0), top * (1.2 + 0.08 * depth))
        ax.grid(axis="x", visible=False)
        ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_xlabel("Trading session (Mondays labelled)")
        ax.set_ylabel("Annualised volatility, percent (252 sessions)")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8.5)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.text(0.01, 0.01, sub, fontsize=8, color=p["muted"], ha="left", va="bottom")
        return fig

    return build


def build_study_figure(
    study: EventStudy, kinds: tuple[str, ...] = ("CPI", "FOMC", "NFP")
) -> Builder:
    samples = {}
    for k in kinds:
        rows = study.by_kind(k)
        samples[k] = (
            np.array([r.implied_sd * np.sqrt(2 / np.pi) for r in rows]) * 100,
            np.array([abs(r.log_return) for r in rows]) * 100,
            study.summarise(rows),
        )
    control = study.summarise_control()
    full = {k: v[2] for k, v in samples.items()}
    ratios = [full[k]["ratio"] for k in kinds]
    title = (
        "Event-day moves were not distinguishable from what the one-day index implied: "
        f"variance ratios {min(ratios):.2f} to {max(ratios):.2f}, intervals all contain 1"
        if all(full[k]["ratio_lo"] < 1 < full[k]["ratio_hi"] for k in kinds)
        else "Realised versus implied event-day variance"
    )
    names = {"CPI": "CPI", "FOMC": "FOMC", "NFP": "Employment report"}

    def build(p: dict[str, Any]) -> Figure:
        fig, (a, b) = plt.subplots(
            1, 2, figsize=(10.4, 4.7), gridspec_kw={"width_ratios": [1.15, 1.0]}
        )
        for i, k in enumerate(kinds):
            xs, ys, _ = samples[k]
            a.scatter(
                xs,
                ys,
                s=26,
                color=p["series"][i],
                label=f"{names[k]} (n={xs.size})",
                linewidths=0.6,
            )
        top = max(max(v[0].max(), v[1].max()) for v in samples.values()) * 1.05
        a.plot([0, top], [0, top], color=p["axis"], lw=1.2, zorder=0)
        a.text(
            top * 0.97,
            top * 0.9,
            "realised = implied",
            ha="right",
            va="top",
            fontsize=8.5,
            color=p["muted"],
            rotation=0,
        )
        a.set_xlim(0, top)
        a.set_ylim(0, top)
        a.set_xlabel("Implied expected |move| from previous close, percent")
        a.set_ylabel("Realised |move| on the day, percent")
        a.set_title("Each dot is one event day")
        a.legend(loc="upper left", fontsize=8.5)

        ys = np.arange(len(kinds) + 1)[::-1]
        items = [(names[k], full[k]) for k in kinds] + [("Ordinary days", control)]
        for j, (name, s) in zip(ys, items, strict=True):
            colour = (
                p["series"][kinds.index(next(k for k in kinds if names[k] == name))]
                if name != "Ordinary days"
                else p["muted"]
            )
            if "ratio_lo" in s:
                b.plot(
                    [s["ratio_lo"], s["ratio_hi"]],
                    [j, j],
                    color=colour,
                    lw=2.0,
                    solid_capstyle="round",
                )
            else:
                b.plot(
                    [s["z_squared_lo"], s["z_squared_hi"]],
                    [j, j],
                    color=colour,
                    lw=2.0,
                    solid_capstyle="round",
                )
            val = s["ratio"] if "ratio_lo" in s else s["z_squared"]
            b.scatter(
                [val], [j], s=46, color=colour, zorder=3, edgecolors=p["surface"], linewidths=1.5
            )
            b.text(
                -0.02,
                j,
                f"{name} (n={s['n']})",
                ha="right",
                va="center",
                transform=b.get_yaxis_transform(),
                fontsize=9,
                color=p["ink_secondary"],
            )
        b.axvline(1.0, color=p["axis"], lw=1.2, zorder=0)
        b.set_yticks([])
        b.set_ylim(-0.7, len(kinds) + 0.6)
        b.set_xlim(0, max(s.get("ratio_hi", s.get("z_squared_hi", 1)) for _, s in items) * 1.08)
        b.set_xlabel("Realised / implied variance (95% bootstrap interval); 1 = fair")
        b.set_title("Mean realised over mean implied variance")
        b.grid(axis="y", visible=False)
        b.spines["left"].set_visible(False)
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=p["ink"])
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    return build


def build_replication_figure(
    official: Mapping[str, float], errors: Mapping[str, Mapping[str, float]], asof_label: str
) -> Builder:
    names = list(official)
    sources = list(errors)
    worst = {s: max(abs(v) for v in errors[s].values()) for s in sources}
    title = (
        "Rebuilt closes versus official: "
        + ", ".join(f"{s} chain within {worst[s]:.2f}" for s in sources)
        + " volatility points"
    )

    def build(p: dict[str, Any]) -> Figure:
        fig, ax = plt.subplots(figsize=(8.4, 4.2))
        n = len(sources)
        width = 0.28
        xs = np.arange(len(names))
        for k, s in enumerate(sources):
            off = (k - (n - 1) / 2) * (width + 0.04)
            vals = [errors[s][nm] for nm in names]
            ax.bar(
                xs + off, vals, width=width, color=p["series"][k], label=f"{s} chain", linewidth=0
            )
            for xi, v in zip(xs + off, vals, strict=True):
                ax.annotate(
                    f"{v:+.3f}",
                    (xi, v),
                    xytext=(0, 3 if v >= 0 else -3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=8,
                    color=p["ink_secondary"],
                )
        ax.axhline(0, color=p["axis"], lw=1.2)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(min(-0.012, ax.get_ylim()[0]), ax.get_ylim()[1] * 1.04)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{nm}\nofficial {official[nm]:.2f}" for nm in names])
        ax.set_ylabel("Rebuilt minus official close, volatility points")
        ax.set_title(title, fontsize=10.5)
        ax.legend(loc="upper right", fontsize=8.5)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.text(0.01, 0.01, asof_label, fontsize=8, color=p["muted"], ha="left", va="bottom")
        return fig

    return build


def build_robustness_figure(
    table: Mapping[str, Mapping[str, tuple[float, float, float]]],
) -> Builder:
    """Event multiples by base family: ``table[family][event] = (estimate, lo, hi)``."""
    families = list(table)
    events = list(next(iter(table.values())))
    title = "Event multiples move with the base curve's flexibility by more than the intervals show"

    def build(p: dict[str, Any]) -> Figure:
        fig, ax = plt.subplots(figsize=(8.4, 4.2))
        ys = np.arange(len(events))[::-1]
        n = len(families)
        for k, fam in enumerate(families):
            off = (k - (n - 1) / 2) * 0.24
            for y, ev in zip(ys, events, strict=True):
                est, lo, hi = table[fam][ev]
                ax.plot(
                    [lo, hi],
                    [y + off, y + off],
                    color=p["series"][k],
                    lw=2.0,
                    solid_capstyle="round",
                )
                ax.scatter(
                    [est],
                    [y + off],
                    s=40,
                    color=p["series"][k],
                    zorder=3,
                    edgecolors=p["surface"],
                    linewidths=1.5,
                    label=fam if y == ys[0] else None,
                )
        ax.axvline(1.0, color=p["axis"], lw=1.2, zorder=0)
        ax.set_yticks(ys)
        ax.set_yticklabels(events)
        ax.set_xlabel("Session variance over base variance (90% bootstrap interval)")
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right", fontsize=8.5)
        fig.tight_layout()
        return fig

    return build
