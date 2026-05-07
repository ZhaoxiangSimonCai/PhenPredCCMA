"""Shared plotting style for publication-quality matplotlib figures.

Drop this file into ``<project>/scripts/_plot_style.py``. Every plot script
should ``from _plot_style import configure_nature_style, PALETTE,
SCATTER_ALPHA, panel_label, save_figure`` rather than defining its own
colour or font constants. This keeps every figure visually consistent.

Design principles
-----------------
1. **Focus set in saturated colour, reference set in neutral grey.**
   Avoids the contrast issue when two scatter clouds overlap; matches the
   Nature/Cell convention.
2. **Okabe-Ito-derived palette** — colourblind-safe and widely recognised.
3. **Sans-serif fonts** (Arial / Helvetica fallback) at sizes that stay
   legible at 300 dpi PNG output.
4. **No top/right spines, thin axis rules, no-frame legends** — clean
   typographic style.
5. **Type-42 (TrueType) embedding in PDF** so text remains selectable and
   editable in Illustrator / Inkscape.

Usage
-----
    from _plot_style import (
        PALETTE, SCATTER_ALPHA,
        configure_nature_style, panel_label, save_figure,
    )

    configure_nature_style("composite")    # or "column" or "full"
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.scatter(..., color=PALETTE["new"], alpha=SCATTER_ALPHA["new"],
               rasterized=True)
    panel_label(ax, "a")
    save_figure(fig, out_dir / "fig_X_name")   # writes PNG + PDF
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


PALETTE = {
    # Two-way contrasts (most common case).
    "new":       "#0173B2",   # deep blue — focus / highlighted set
    "common":    "#888888",   # neutral grey — reference background
    "lost":      "#DE8F05",   # orange — small contrastive set

    # Aliases for "old vs new" framings.
    "astral":    "#0173B2",   # alias for "new"
    "old":       "#888888",   # alias for "common"

    # Highlights & accents.
    "highlight": "#CC3311",   # brick red — significance, threshold lines
    "accent":    "#029E73",   # teal — third-category accent

    # Soft grey for desaturated panels (e.g. unselected scatter points).
    "neutral":   "#BCBCBC",
}


# Recommended scatter-plot alpha values — lighter for "common" so the
# saturated "new" points pop on top.
SCATTER_ALPHA = {
    "new":    0.55,
    "common": 0.30,
    "lost":   0.55,
}


# Type-size presets keyed by intended layout context.
NATURE_TYPE_SCALES = {
    # Compact panels inside a multi-panel figure (Nature 2-column figure
    # divided into ~6 sub-panels). Print at ~3.5 cm wide each.
    "composite": dict(font=8.5, title=10, label=9, tick=8, legend=8),
    # Standalone single-column figure (~89 mm / 3.5 in wide).
    "column":    dict(font=9.5, title=11, label=10, tick=8.5, legend=9),
    # Full-page or 1.5-column figure (~120-180 mm). Slightly larger than
    # column for headline plots.
    "full":      dict(font=10.5, title=12, label=10.5, tick=9.5, legend=9.5),
}


def configure_nature_style(scale: str = "composite") -> None:
    """Apply publication-grade matplotlib rcParams.

    Sets:
    - 300 dpi PNG, white facecolor
    - Type 42 fonts in PDF + PS (TrueType embedded as Type0/CIDFontType2)
      so text remains editable in Illustrator / Inkscape
    - Tighter type sizes driven by *scale*
    - Thin axis lines (0.7 pt) suitable for print
    - No top/right spines, frameless legends

    *scale* selects from ``NATURE_TYPE_SCALES``: "composite" (default),
    "column", or "full".
    """
    if scale not in NATURE_TYPE_SCALES:
        raise ValueError(
            f"unknown scale {scale!r}; expected one of {list(NATURE_TYPE_SCALES)}"
        )
    sizes = NATURE_TYPE_SCALES[scale]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": sizes["font"],
        "axes.titlesize": sizes["title"],
        "axes.titleweight": "bold",
        "axes.labelsize": sizes["label"],
        "axes.labelweight": "regular",
        "xtick.labelsize": sizes["tick"],
        "ytick.labelsize": sizes["tick"],
        "legend.fontsize": sizes["legend"],
        "legend.frameon": False,
        "legend.title_fontsize": sizes["legend"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "figure.dpi": 110,
        "figure.facecolor": "white",
        "axes.labelpad": 4,
        "axes.titlepad": 7,
        "axes.titlelocation": "left",
        # Editable text in vector outputs.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        # Tighter line + marker defaults for print.
        "lines.linewidth": 1.2,
        "lines.markersize": 4.5,
        "patch.linewidth": 0.5,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
    })


def panel_label(ax, letter: str, *, offset=(-0.13, 1.04),
                fontsize: float | None = None, weight: str = "bold") -> None:
    """Draw a Nature-style panel label (a, b, c, ...) at axis-relative top-left.

    Place this *after* axis titles/labels so it sits above the panel.
    """
    if fontsize is None:
        fontsize = plt.rcParams["axes.titlesize"] + 1
    ax.text(
        offset[0], offset[1], letter,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=fontsize, fontweight=weight,
    )


def save_figure(fig, path_no_ext, *, formats=("png", "pdf"),
                close: bool = True) -> None:
    """Write the same figure to PNG + PDF (default) under *path_no_ext*.

    PNG goes out at 300 dpi (set via rcParams). PDF embeds Type 42 fonts so
    text stays editable in Illustrator / Inkscape. Parent directory is
    created if missing.
    """
    p = Path(path_no_ext)
    p.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(p.with_suffix(f".{ext}"))
    if close:
        plt.close(fig)
