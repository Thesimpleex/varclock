"""Animated README visuals rendered from real program output.

``terminal_cast_svg`` runs a real command and writes a looping animated SVG of a
terminal session: the command is typed with a blinking cursor, then the actual
output appears line by line. The SVG uses only CSS animations, so it plays inside
GitHub READMEs, stays sharp at any zoom and is a few kilobytes in size.

``save_animation`` renders a sequence of matplotlib frames to light and dark GIFs
for explanatory animations of a method.

Requires ``rich`` (and ``matplotlib`` plus Pillow for GIFs).
"""

from __future__ import annotations

import html
import io
import os
import shlex
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.terminal_theme import TerminalTheme
from rich.text import Text


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


_THEMES = {
    "dark": {
        "window": "#1a1a19",
        "bar": "#242422",
        "ink": "#e9e8e3",
        "muted": "#898781",
        "prompt": "#3987e5",
        "cursor": "#e9e8e3",
        "ring": "#2c2c2a",
        "dot": "#4a4a47",
        "shadow": 0.45,
        "ansi": (
            [
                "#383835",
                "#e66767",
                "#1baf7a",
                "#eda100",
                "#3987e5",
                "#d55181",
                "#6da7ec",
                "#c3c2b7",
            ],
            [
                "#898781",
                "#f08c8b",
                "#3cc993",
                "#f5b83a",
                "#6da7ec",
                "#e87ba4",
                "#9ec5f4",
                "#ffffff",
            ],
        ),
    },
    "light": {
        "window": "#fcfcfb",
        "bar": "#f0efec",
        "ink": "#0b0b0b",
        "muted": "#898781",
        "prompt": "#2a78d6",
        "cursor": "#0b0b0b",
        "ring": "#d9d8d2",
        "dot": "#d6d5cf",
        "shadow": 0.10,
        "ansi": (
            [
                "#0b0b0b",
                "#c93a39",
                "#007a00",
                "#9a6a00",
                "#2a78d6",
                "#c2457a",
                "#1c5cab",
                "#52514e",
            ],
            [
                "#52514e",
                "#e34948",
                "#008300",
                "#b07c00",
                "#256abf",
                "#d55181",
                "#184f95",
                "#0b0b0b",
            ],
        ),
    },
}


def _ansi_theme(t: dict) -> TerminalTheme:
    normal, bright = t["ansi"]
    return TerminalTheme(
        _rgb(t["window"]), _rgb(t["ink"]), [_rgb(c) for c in normal], [_rgb(c) for c in bright]
    )


def capture_command(
    command: Sequence[str] | str, columns: int = 100, cwd: str | Path | None = None
) -> str:
    """Run ``command`` with colour forced on and return its ANSI output."""
    argv = shlex.split(command) if isinstance(command, str) else list(command)
    env = {**os.environ, "FORCE_COLOR": "1", "TERM": "xterm-256color", "COLUMNS": str(columns)}
    env.pop("NO_COLOR", None)
    result = subprocess.run(argv, env=env, cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout


def _lines_from_ansi(ansi: str, columns: int, t: dict) -> list[list[tuple[str, str, bool, bool]]]:
    """Split ANSI output into lines of (text, hex colour, bold, dim) runs."""
    console = Console(
        width=columns, file=io.StringIO(), force_terminal=True, color_system="truecolor"
    )
    lines = []
    for line in Text.from_ansi(ansi.rstrip("\n")).split("\n", allow_blank=True):
        runs = []
        for seg in line.render(console):
            if not seg.text or seg.control:
                continue
            style = seg.style
            colour = t["ink"]
            bold = dim = False
            if style is not None:
                if style.color is not None:
                    colour = style.color.get_truecolor(_ansi_theme(t), foreground=True).hex
                bold = bool(style.bold)
                dim = bool(style.dim)
            runs.append((seg.text, colour, bold, dim))
        lines.append(runs)
    return lines


def terminal_cast_svg(
    command: Sequence[str] | str,
    out: str | Path,
    title: str = "",
    display_command: str | None = None,
    columns: int = 100,
    cwd: str | Path | None = None,
    type_seconds_per_char: float = 0.045,
    line_seconds: float = 0.07,
    hold_seconds: float = 5.0,
    font_size: float = 13.0,
    theme: str = "dark",
) -> Path:
    """Run a real command and save a looping animated SVG of the terminal session.

    Write both themes and serve them with ``<picture>`` so the window matches the
    reader's GitHub theme.
    """
    ansi = capture_command(command, columns=columns, cwd=cwd)
    shown = display_command or (command if isinstance(command, str) else shlex.join(command))
    return Path(
        _write_cast(
            ansi,
            shown,
            Path(out),
            title,
            columns,
            type_seconds_per_char,
            line_seconds,
            hold_seconds,
            font_size,
            theme,
        )
    )


def _write_cast(
    ansi: str,
    shown: str,
    out: Path,
    title: str,
    columns: int,
    type_seconds_per_char: float,
    line_seconds: float,
    hold_seconds: float,
    font_size: float,
    theme: str = "dark",
) -> Path:
    th = _THEMES[theme]
    lines = _lines_from_ansi(ansi, columns, th)
    char_w = 0.6 * font_size
    line_h = 1.45 * font_size
    pad_x, pad_top, pad_bottom, bar_h, margin = 22.0, 18.0, 22.0, 36.0, 32.0
    widest = max([len(shown) + 2] + [sum(len(r[0]) for r in ln) for ln in lines])
    win_w = max(560.0, 2 * pad_x + widest * char_w)
    n_rows = len(lines) + 2
    win_h = bar_h + pad_top + n_rows * line_h + pad_bottom
    total_w, total_h = win_w + 2 * margin, win_h + 2 * margin

    t_type0 = 0.6
    t_type1 = t_type0 + type_seconds_per_char * len(shown)
    t_out0 = t_type1 + 0.35
    t_out1 = t_out0 + line_seconds * len(lines)
    cycle = t_out1 + hold_seconds + 0.8

    def pct(t: float) -> str:
        return f"{100.0 * t / cycle:.3f}%"

    css = [
        "text{font-family:ui-monospace,'SF Mono',Menlo,Consolas,'DejaVu Sans Mono',monospace;"
        f"font-size:{font_size}px;white-space:pre}}",
        ".t{font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;font-size:12.5px}",
        f".cast{{animation:fade {cycle:.3f}s infinite}}",
        f"@keyframes fade{{0%{{opacity:1}}{pct(cycle - 0.6)}{{opacity:1}}100%{{opacity:0}}}}",
        f".cmd{{animation:type {cycle:.3f}s steps(1,end) infinite}}",
        f".cur{{animation:blink 1s steps(1,end) infinite,move {cycle:.3f}s linear infinite}}",
        "@keyframes blink{0%{opacity:1}50%{opacity:0}}",
        f".cw{{animation:cw {cycle:.3f}s steps(1,end) infinite}}",
        f"@keyframes cw{{0%{{opacity:1}}{pct(t_out0)}{{opacity:0}}100%{{opacity:0}}}}",
        ".cb{animation:blink 1s steps(1,end) infinite}",
    ]
    type_frames = [
        f"0%{{clip-path:inset(0 100% 0 0)}}{pct(t_type0)}{{clip-path:inset(0 100% 0 0)}}"
    ]
    move_frames = [f"0%{{transform:translateX(0px)}}{pct(t_type0)}{{transform:translateX(0px)}}"]
    n = max(1, len(shown))
    for i in range(1, n + 1):
        t = t_type0 + type_seconds_per_char * i
        frac = 100.0 * (1 - i / n)
        type_frames.append(f"{pct(t)}{{clip-path:inset(0 {frac:.3f}% 0 0)}}")
        move_frames.append(f"{pct(t)}{{transform:translateX({i * char_w:.2f}px)}}")
    type_frames.append("100%{clip-path:inset(0 0 0 0)}")
    move_frames.append(f"100%{{transform:translateX({n * char_w:.2f}px)}}")
    css.append("@keyframes type{" + "".join(type_frames) + "}")
    css.append("@keyframes move{" + "".join(move_frames) + "}")
    for k in range(len(lines) + 1):
        t = t_out0 + line_seconds * k + (0.25 if k == len(lines) else 0.0)
        css.append(
            f".l{k}{{opacity:0;animation:l{k} {cycle:.3f}s steps(1,end) infinite}}"
            f"@keyframes l{k}{{0%{{opacity:0}}{pct(t)}{{opacity:1}}100%{{opacity:1}}}}"
        )

    x0 = margin + pad_x
    y_cmd = margin + bar_h + pad_top + font_size
    prompt_w = 2 * char_w
    body = []
    for k, runs in enumerate(lines):
        y = y_cmd + (k + 1) * line_h
        spans = []
        for text, colour, bold, dim in runs:
            attrs = f'fill="{colour}"'
            if bold:
                attrs += ' font-weight="700"'
            if dim:
                attrs += ' fill-opacity="0.62"'
            spans.append(f"<tspan {attrs}>{html.escape(text)}</tspan>")
        body.append(f'<text class="l{k}" x="{x0:.2f}" y="{y:.2f}">{"".join(spans)}</text>')

    y_end = y_cmd + (len(lines) + 1) * line_h
    d = th
    cy = margin + bar_h / 2
    cursor = (
        f'width="{char_w:.2f}" height="{font_size + 2:.2f}" '
        f'fill="{d["cursor"]}" fill-opacity="0.85"'
    )
    win = f'x="{margin}" y="{margin}" width="{win_w:.2f}" height="{win_h:.2f}" rx="11"'
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w:.0f}" height="{total_h:.0f}" '
        f'viewBox="0 0 {total_w:.2f} {total_h:.2f}">',
        f"<style>{''.join(css)}</style>",
        '<defs><filter id="sh" x="-10%" y="-10%" width="120%" height="130%">'
        f'<feDropShadow dx="0" dy="6" stdDeviation="9" flood-color="#000" '
        f'flood-opacity="{d["shadow"]}"/></filter>',
        f'<clipPath id="win"><rect {win}/></clipPath></defs>',
        f'<rect {win} fill="{d["window"]}" filter="url(#sh)"/>',
        '<g clip-path="url(#win)">',
        f'<rect x="{margin}" y="{margin}" width="{win_w:.2f}" height="{bar_h}" fill="{d["bar"]}"/>',
        f'<circle cx="{margin + 20}" cy="{cy}" r="6" fill="{d["dot"]}"/>',
        f'<circle cx="{margin + 40}" cy="{cy}" r="6" fill="{d["dot"]}"/>',
        f'<circle cx="{margin + 60}" cy="{cy}" r="6" fill="{d["dot"]}"/>',
        f'<text class="t" x="{margin + win_w / 2:.2f}" y="{cy + 4.5:.2f}" text-anchor="middle" '
        f'fill="{d["muted"]}">{html.escape(title)}</text>',
        '<g class="cast">',
        f'<text x="{x0:.2f}" y="{y_cmd:.2f}" fill="{d["prompt"]}">$</text>',
        f'<text class="cmd" x="{x0 + prompt_w:.2f}" y="{y_cmd:.2f}" fill="{d["ink"]}" '
        f'font-weight="700">{html.escape(shown)}</text>',
        f'<g class="cw"><rect class="cur" x="{x0 + prompt_w:.2f}" '
        f'y="{y_cmd - font_size + 2:.2f}" {cursor}/></g>',
        *body,
        f'<g class="l{len(lines)}"><text x="{x0:.2f}" y="{y_end:.2f}" '
        f'fill="{d["prompt"]}">$</text><rect class="cb" x="{x0 + prompt_w:.2f}" '
        f'y="{y_end - font_size + 2:.2f}" {cursor}/></g>',
        "</g></g>",
        f'<rect {win} fill="none" stroke="{d["ring"]}"/>',
        "</svg>",
    ]
    svg = "\n".join(parts) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    return out


def save_animation(
    draw_frame: Callable[[Any, dict[str, Any], int], None],
    n_frames: int,
    stem: str | Path,
    style: Callable[[str], Any],
    figsize: tuple[float, float] = (8.0, 4.5),
    fps: int = 20,
    dpi: int = 110,
    hold_last: int = 30,
    themes: Sequence[str] = ("light", "dark"),
) -> list[Path]:
    """Render ``draw_frame(ax, palette, i)`` for ``i < n_frames`` into looping GIFs.

    Data must be computed beforehand; ``draw_frame`` only draws frame ``i`` on a
    cleared axes. The last frame is held for ``hold_last`` frames. ``style`` is the
    package's ``viz.style`` context manager.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for theme in themes:
        with style(theme) as p:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            frames = []
            for i in range(n_frames):
                ax.clear()
                draw_frame(ax, p, i)
                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                frames.append(
                    Image.frombuffer("RGBA", fig.canvas.get_width_height(), buf).convert("RGB")
                )
            plt.close(fig)
        frames += [frames[-1]] * hold_last
        palette_frames = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
        path = stem.with_name(f"{stem.name}-{theme}.gif")
        palette_frames[0].save(
            path,
            save_all=True,
            append_images=palette_frames[1:],
            duration=int(1000 / fps),
            loop=0,
            optimize=True,
            disposal=2,
        )
        paths.append(path)
    return paths
