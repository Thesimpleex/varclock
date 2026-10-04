"""Figure style shared by all scripts.

Every figure is rendered twice, on a light and a dark surface, so that the README
can serve the variant matching the reader's GitHub theme via ``<picture>``.
Categorical colours are assigned in a fixed order (never cycled past eight);
sequential maps use a single blue ramp; diverging maps use blue/red arms around a
neutral grey midpoint.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

THEMES: dict[str, dict[str, Any]] = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "neutral": "#f0efec",
        "series": (
            "#2a78d6",
            "#eb6834",
            "#1baf7a",
            "#eda100",
            "#e87ba4",
            "#008300",
            "#4a3aa7",
            "#e34948",
        ),
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "neutral": "#383835",
        "series": (
            "#3987e5",
            "#d95926",
            "#199e70",
            "#c98500",
            "#d55181",
            "#008300",
            "#9085e9",
            "#e66767",
        ),
    },
}

_BLUE_RAMP = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)

_SANS = ["Helvetica Neue", "Helvetica", "DejaVu Sans"]


def sequential_cmap(theme: str = "light") -> LinearSegmentedColormap:
    """Single-hue magnitude map; values near zero recede toward the surface."""
    ramp = _BLUE_RAMP if theme == "light" else _BLUE_RAMP[::-1]
    return LinearSegmentedColormap.from_list(f"seq-{theme}", ramp)


def diverging_cmap(theme: str = "light") -> LinearSegmentedColormap:
    """Blue (negative) to red (positive) with a neutral grey midpoint."""
    mid = THEMES[theme]["neutral"]
    arms = ("#184f95", "#5598e7", mid, "#e66767", "#a8322f")
    return LinearSegmentedColormap.from_list(f"div-{theme}", arms)


@contextmanager
def style(theme: str = "light") -> Iterator[dict[str, Any]]:
    """Apply the house style for one theme and yield its palette."""
    p = THEMES[theme]
    rc = {
        "figure.facecolor": p["surface"],
        "axes.facecolor": p["surface"],
        "savefig.facecolor": p["surface"],
        "axes.edgecolor": p["axis"],
        "axes.labelcolor": p["ink_secondary"],
        "axes.titlecolor": p["ink"],
        "text.color": p["ink"],
        "xtick.color": p["axis"],
        "ytick.color": p["axis"],
        "xtick.labelcolor": p["ink_secondary"],
        "ytick.labelcolor": p["ink_secondary"],
        "axes.grid": True,
        "grid.color": p["grid"],
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=list(p["series"])),
        "lines.linewidth": 1.8,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.markersize": 6,
        "patch.edgecolor": p["surface"],
        "legend.frameon": False,
        "legend.labelcolor": p["ink_secondary"],
        "font.family": "sans-serif",
        "font.sans-serif": _SANS,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
    with mpl.rc_context(rc):
        yield p


def save_figure(
    build: Callable[[dict[str, Any]], Figure],
    stem: str | Path,
    themes: Sequence[str] = ("light", "dark"),
) -> list[Path]:
    """Render ``build(palette)`` once per theme and save ``<stem>-<theme>.png``.

    ``build`` must only draw: compute all data beforehand so that both variants
    show identical content.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for theme in themes:
        with style(theme) as p:
            fig = build(p)
            path = stem.with_name(f"{stem.name}-{theme}.png")
            fig.savefig(path)
            plt.close(fig)
            paths.append(path)
    return paths


def picture_html(stem: str, alt: str, width: int | None = None) -> str:
    """README snippet that serves the dark variant to dark-theme readers."""
    size = f' width="{width}"' if width else ""
    return (
        "<picture>\n"
        f'  <source media="(prefers-color-scheme: dark)" srcset="{stem}-dark.png">\n'
        f'  <img alt="{alt}" src="{stem}-light.png"{size}>\n'
        "</picture>"
    )
