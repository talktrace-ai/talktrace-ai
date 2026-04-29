"""Theme-bewusstes Styling für matplotlib-Plots im Results-Tab.

Source of truth für die Farben ist talktrace_ai/static/theme.css. Die
Hex-Werte werden hier gespiegelt, weil CSS-Custom-Properties aus matplotlib
heraus nicht erreichbar sind. Werte ändern sich selten — bei Theme-Updates
in theme.css auch hier nachziehen.
"""
from __future__ import annotations

import colorsys
from typing import Literal

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


Mode = Literal["dark", "light"]


PALETTES: dict[str, dict] = {
    "light": {
        "bg":      "#EEF1F1",
        "surface": "#F6F8F8",
        "text":    "#23292B",
        "muted":   "#6F7779",
        "border":  "#D2D8D8",
        "primary": "#5E8784",
        "accent":  "#5E8784",
        "secondary": "#B57A72",  # tt-danger als 2. Serie für Two-Series-Plots
    },
    "dark": {
        "bg":      "#161B19",
        "surface": "#1E2421",
        "text":    "#D5DDD8",
        "muted":   "#85918B",
        "border":  "#2F3833",
        "primary": "#6F9382",
        "accent":  "#8FB29A",
        "secondary": "#B0786A",
    },
}


def _shade(hex_color: str, lightness_delta: float) -> str:
    """Hellt/dunkelt eine Farbe im HLS-Raum auf. delta in [-1, 1]."""
    r, g, b = mcolors.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.05, min(0.95, l + lightness_delta))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return mcolors.to_hex((r2, g2, b2))


def _build_stack(primary: str) -> list[str]:
    """7 Schattierungen der Primärfarbe von dunkel bis hell."""
    return [
        _shade(primary, -0.20),
        _shade(primary, -0.10),
        _shade(primary, 0.00),
        _shade(primary, 0.10),
        _shade(primary, 0.20),
        _shade(primary, 0.30),
        _shade(primary, 0.40),
    ]


for _mode, _pal in PALETTES.items():
    _pal["stack"] = _build_stack(_pal["primary"])


def resolve_mode(input_obj, force_mode: Mode | None = None) -> Mode:
    """Liest input.dark_mode() defensiv. Default: 'light'."""
    if force_mode is not None:
        return force_mode
    try:
        val = input_obj.dark_mode()
    except Exception:
        return "light"
    return "dark" if str(val).lower() == "dark" else "light"


def palette(mode: Mode) -> dict:
    return PALETTES[mode]


def primary_color(mode: Mode) -> str:
    return PALETTES[mode]["primary"]


def secondary_color(mode: Mode) -> str:
    return PALETTES[mode]["secondary"]


def stack_colors(mode: Mode, n: int) -> list[str]:
    stack = PALETTES[mode]["stack"]
    if n <= 0:
        return []
    if n <= len(stack):
        # Bei kleiner Anzahl gleichmässig über die Palette verteilen,
        # damit die Stapel klar unterscheidbar bleiben.
        step = max(1, len(stack) // n)
        return [stack[(i * step) % len(stack)] for i in range(n)]
    # Wrap-around bei sehr grossen Codebüchern.
    return [stack[i % len(stack)] for i in range(n)]


def apply_axes_style(ax, mode: Mode) -> None:
    """Setzt Hintergrund, Spines, Ticks, Grid und Labels passend zum Theme."""
    pal = PALETTES[mode]
    fig = ax.figure
    fig.patch.set_facecolor(pal["surface"])
    ax.set_facecolor(pal["surface"])

    for spine in ax.spines.values():
        spine.set_color(pal["border"])
        spine.set_linewidth(0.8)
    # Top/Right Spines verstecken — clean look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(colors=pal["muted"], which="both")
    ax.xaxis.label.set_color(pal["text"])
    ax.yaxis.label.set_color(pal["text"])
    ax.title.set_color(pal["text"])

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(pal["text"])

    ax.grid(color=pal["border"], axis="y", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    # Bestehende Bar-Labels (Zahlen über/in Balken) ebenfalls einfärben.
    for txt in ax.texts:
        txt.set_color(pal["text"])

    legend = ax.get_legend()
    if legend is not None:
        frame = legend.get_frame()
        frame.set_facecolor(pal["surface"])
        frame.set_edgecolor(pal["border"])
        for txt in legend.get_texts():
            txt.set_color(pal["text"])
        if legend.get_title() is not None:
            legend.get_title().set_color(pal["text"])


def round_bar_corners(ax, radius: float = 0.06) -> None:
    """Ersetzt jede Rectangle-Bar durch eine FancyBboxPatch mit
    abgerundeten Ecken. Die `radius` ist in Daten-x-Einheiten — für
    übliche Bar-Plots (kategoriale x, Bar-Breite ~0.5) ist 0.06 dezent.
    Für Stacked-Bars NICHT aufrufen — die Übergänge sehen sonst kaputt aus.
    """
    rectangles = []
    for container in list(ax.containers):
        for patch in container:
            if isinstance(patch, mpatches.Rectangle):
                rectangles.append(patch)

    for patch in rectangles:
        x, y = patch.get_xy()
        w = patch.get_width()
        h = patch.get_height()
        if w <= 0 or h == 0:
            continue
        color = patch.get_facecolor()
        r = min(abs(radius), abs(w) / 2)
        # BoxStyle.Round mit pad=0, rounding_size=r dehnt sich um r aus —
        # daher nach innen verschieben, damit der Aussen-Bbox stimmt.
        if h > 0:
            new_xy = (x + r, y + r)
            new_h = h - 2 * r
        else:
            new_xy = (x + r, y - r)
            new_h = h + 2 * r
        if new_h * (1 if h > 0 else -1) <= 0:
            # Bar zu kurz für gewünschten Radius — ohne Rundung übernehmen.
            continue
        bbp = mpatches.FancyBboxPatch(
            new_xy,
            w - 2 * r,
            new_h,
            boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=r),
            facecolor=color,
            edgecolor="none",
            linewidth=0,
        )
        patch.set_visible(False)
        ax.add_patch(bbp)


def style_no_data_axes(ax, mode: Mode) -> None:
    """Hintergrund + Textfarbe für die 'no data'-Fallback-Plots setzen."""
    pal = PALETTES[mode]
    ax.figure.patch.set_facecolor(pal["surface"])
    ax.set_facecolor(pal["surface"])
    for txt in ax.texts:
        txt.set_color(pal["muted"])


def _snapshot_style(ax) -> dict:
    """Sichert die für apply_axes_style relevanten Farben einer Axes,
    damit sie nach einem temporären Re-Style wiederhergestellt werden können."""
    fig = ax.figure
    legend = ax.get_legend()
    snap = {
        "fig_face": fig.get_facecolor(),
        "ax_face": ax.get_facecolor(),
        "spines": {
            name: (s.get_edgecolor(), s.get_linewidth(), s.get_visible())
            for name, s in ax.spines.items()
        },
        "x_label": ax.xaxis.label.get_color(),
        "y_label": ax.yaxis.label.get_color(),
        "title": ax.title.get_color(),
        "x_tick_labels": [l.get_color() for l in ax.get_xticklabels()],
        "y_tick_labels": [l.get_color() for l in ax.get_yticklabels()],
        "texts": [t.get_color() for t in ax.texts],
        "legend": None,
    }
    if legend is not None:
        frame = legend.get_frame()
        snap["legend"] = {
            "frame_face": frame.get_facecolor(),
            "frame_edge": frame.get_edgecolor(),
            "texts": [t.get_color() for t in legend.get_texts()],
            "title": legend.get_title().get_color() if legend.get_title() is not None else None,
        }
    return snap


def _restore_style(ax, snap: dict) -> None:
    fig = ax.figure
    fig.patch.set_facecolor(snap["fig_face"])
    ax.set_facecolor(snap["ax_face"])
    for name, (color, lw, visible) in snap["spines"].items():
        ax.spines[name].set_edgecolor(color)
        ax.spines[name].set_linewidth(lw)
        ax.spines[name].set_visible(visible)
    ax.xaxis.label.set_color(snap["x_label"])
    ax.yaxis.label.set_color(snap["y_label"])
    ax.title.set_color(snap["title"])
    for label, c in zip(ax.get_xticklabels(), snap["x_tick_labels"]):
        label.set_color(c)
    for label, c in zip(ax.get_yticklabels(), snap["y_tick_labels"]):
        label.set_color(c)
    for txt, c in zip(ax.texts, snap["texts"]):
        txt.set_color(c)
    legend = ax.get_legend()
    if legend is not None and snap["legend"] is not None:
        frame = legend.get_frame()
        frame.set_facecolor(snap["legend"]["frame_face"])
        frame.set_edgecolor(snap["legend"]["frame_edge"])
        for txt, c in zip(legend.get_texts(), snap["legend"]["texts"]):
            txt.set_color(c)
        if legend.get_title() is not None and snap["legend"]["title"] is not None:
            legend.get_title().set_color(snap["legend"]["title"])


class light_export_style:
    """Context-Manager: setzt Light-Stil auf einer Axes (für PNG-Export in
    Reports), stellt den vorherigen Zustand beim Verlassen wieder her — damit
    das gemeinsam mit der UI gecachte Axes-Objekt nicht "verfärbt" bleibt."""

    def __init__(self, ax):
        self.ax = ax
        self._snap = None

    def __enter__(self):
        self._snap = _snapshot_style(self.ax)
        apply_axes_style(self.ax, "light")
        return self.ax

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._snap is not None:
            _restore_style(self.ax, self._snap)
        return False
