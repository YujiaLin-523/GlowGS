"""
Trade-off bubble chart: PSNR vs FPS vs Size (MB)
-------------------------------------------------
Edit DATA to update values; edit CONFIG/STYLE for appearance.
Uses adjustText for automatic non-overlapping labels.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

# =====================
# CONFIG & DATA (edit here to update figure)
# =====================
CONFIG = {
    "figsize": (8.0, 7.0),              # squarer canvas to condense horizontal gap
    "bubble_scale": 4.5,                # Increased scale to magnify size differences
    "bubble_min": 10,                   # Lower floor to let small models show their tiny size
    "bubble_max": 3000,                 # Raised ceiling for large models
    "save_pdf": "paper_experiments/04_trade-off/tradeoff_psnr_fps_size.pdf",
    "save_png": "paper_experiments/04_trade-off/tradeoff_psnr_fps_size.png",
    "png_dpi": 300,
    "x_label": "Rendering speed (FPS)",
    "y_label": "PSNR (dB)",
    "x_margin_ratio": 0.15,             # 15% padding for tighter, centered layout
    "y_margin_ratio": 0.15,             # 15% padding for tighter, centered layout
    
    # -------------------------------------------------------------
    # LABEL POSITIONING
    # -------------------------------------------------------------
    "manual_layout_mode": True,        # Set True to DISABLE auto-adjustment and use strict manual offsets below
    
    # Offsets are in DATA COORDINATES (FPS, dB). 
    # (0, 0) means center of text is at center of bubble.
    # Adjust (dx, dy) to place text next to bubbles.
    "label_offsets": {
        "3DGS":            (10.0, 0.4),    # Large bubble, move up significantly
        "Scaffold-GS":     (-10, 0.3),     # Medium-large, move up
        "Mip-Splatting":   (33, 0),   # Large bubble, move down significantly
        "Compact3DGS":     (25, 0),      # Small bubble, move right
        "LightGaussian":   (25, 0),    # Medium-small, move up
        "HAC":             (0, -0.2),   # Small bubble, move down
        "GlowGS (Ours)":   (30, 0),   # Small bubble, move right
    },
}

STYLE = {
    # Per-method colors (fill color; edge is black for all)
    "method_colors": {
        "3DGS":            "#E64B35",   # Red
        "Scaffold-GS":     "#F39B7F",   # Orange
        "Mip-Splatting":   "#8491B4",   # Purple
        "Compact3DGS":     "#4DBBD5",   # Teal
        "LightGaussian":   "#ECB02E",   # Yellow
        "HAC":             "#00A087",   # Green
        "GlowGS (Ours)":   "#3C5488",   # Royal Blue (highlighted)
    },
    # Fonts & lines
    "font_family": "serif",
    "font_options": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext": "stix",
    "axes_linewidth": 1.5,
    "spine_color": "black",
    "grid_color": "gray",
    "grid_alpha": 0.3,
    "grid_linewidth": 0.8,
    "grid_linestyle": "--",
    "tick_labelsize": 14,
    "axis_labelsize": 18,
    "axis_label_weight": "bold",
    # Bubble styling
    "alpha_fill": 0.92,
    "edge_color": "black",
    "edge_width": 0.6,
    # Label styling
    "label_fontsize": 12,
    "label_fontsize_ours": 13,
    "label_fontweight_ours": "bold",
    # Legend bubble sizes (MB)
    "legend_sizes": [10, 50, 200],
    "legend_fontsize": 11,
    "legend_title_fontsize": 12,
}

# Method data: add/remove/edit rows here
DATA = [
    {"name": "3DGS",            "psnr": 27.45, "fps": 134, "size_mb": 825.3},
    {"name": "Scaffold-GS",    "psnr": 27.65, "fps": 124, "size_mb": 189.2},
    {"name": "Mip-Splatting",  "psnr": 27.39, "fps": 135, "size_mb": 734.4},
    {"name": "Compact3DGS",    "psnr": 26.97, "fps": 148, "size_mb": 26.34},
    {"name": "LightGaussian",  "psnr": 26.91, "fps": 235, "size_mb": 53.93},
    {"name": "HAC",            "psnr": 27.47, "fps": 113, "size_mb": 17.01},
    {"name": "GlowGS (Ours)",  "psnr": 28.92, "fps": 196, "size_mb": 10.53},
]


def apply_style():
    """Apply global matplotlib rcParams for academic-paper style."""
    plt.rcParams.update({
        "font.family": STYLE["font_family"],
        "font.serif": STYLE["font_options"],
        "mathtext.fontset": STYLE["mathtext"],
        "axes.linewidth": STYLE["axes_linewidth"],
        "xtick.labelsize": STYLE["tick_labelsize"],
        "ytick.labelsize": STYLE["tick_labelsize"],
        "axes.edgecolor": STYLE["spine_color"],
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    })


def size_to_area(size_mb: float) -> float:
    """
    Map storage size (MB) to scatter area (points^2).
    Uses linear mapping with min/max clamping for visibility.
    """
    area = size_mb * CONFIG["bubble_scale"]
    area = max(area, CONFIG["bubble_min"])
    area = min(area, CONFIG["bubble_max"])
    return area


def get_text_color(fill_color: str) -> str:
    """Return text color: use the fill color itself (or black for light fills)."""
    # For yellow, use darker variant for readability
    if fill_color == "#F1C40F":
        return "#B7950B"
    return fill_color


def plot():
    apply_style()
    fig, ax = plt.subplots(figsize=CONFIG["figsize"])

    # Compute axis limits (User requested explicit limits: 60-300 FPS, 25.5-31 dB)
    # fps_vals = [d["fps"] for d in DATA]
    # psnr_vals = [d["psnr"] for d in DATA]
    # x_min, x_max = min(fps_vals), max(fps_vals)
    # y_min, y_max = min(psnr_vals), max(psnr_vals)
    # padding_x = (x_max - x_min) * CONFIG["x_margin_ratio"]
    # padding_y = (y_max - y_min) * CONFIG["y_margin_ratio"]
    # ax.set_xlim(x_min - padding_x, x_max + padding_x)
    # ax.set_ylim(y_min - padding_y, y_max + padding_y)
    
    # Set fixed limits as requested
    ax.set_xlim(60, 300)
    ax.set_ylim(26, 30)

    # Grid behind bubbles
    ax.grid(
        True,
        color=STYLE["grid_color"],
        alpha=STYLE["grid_alpha"],
        linewidth=STYLE["grid_linewidth"],
        linestyle=STYLE["grid_linestyle"],
        zorder=0,
    )
    ax.set_axisbelow(True)

    # Clean axis ticks (integer FPS, one decimal PSNR)
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Axis labels
    ax.set_xlabel(
        CONFIG["x_label"],
        fontsize=STYLE["axis_labelsize"],
        fontweight=STYLE["axis_label_weight"],
        labelpad=8,
    )
    ax.set_ylabel(
        CONFIG["y_label"],
        fontsize=STYLE["axis_labelsize"],
        fontweight=STYLE["axis_label_weight"],
        labelpad=8,
    )

    # Draw bubbles
    texts = []
    label_offsets = CONFIG.get("label_offsets", {})
    for item in DATA:
        name = item["name"]
        fps = item["fps"]
        psnr = item["psnr"]
        size_mb = item["size_mb"]
        s = size_to_area(size_mb)

        fill = STYLE["method_colors"].get(name, "#888888")
        txt_color = get_text_color(fill)

        ax.scatter(
            fps,
            psnr,
            s=s,
            color=fill,
            edgecolors=STYLE["edge_color"],
            linewidth=STYLE["edge_width"],
            alpha=STYLE["alpha_fill"],
            zorder=3,
        )

        # Prepare label with optional manual offset for crowded areas
        is_ours = name == "GlowGS (Ours)"
        fontsize = STYLE["label_fontsize_ours"] if is_ours else STYLE["label_fontsize"]
        fontweight = STYLE["label_fontweight_ours"] if is_ours else "normal"
        label_text = f"{name}\n{size_mb:.1f} MB"

        # Apply manual offset if defined (dx, dy in data units)
        dx, dy = label_offsets.get(name, (0, 0))
        txt = ax.text(
            fps + dx,
            psnr + dy,
            label_text,
            fontsize=fontsize,
            fontweight=fontweight,
            color=txt_color,
            ha="center",
            va="center",
            zorder=5,
        )
        texts.append(txt)

    # Apply auto-adjustment layout (adjustText) ONLY if manual mode is False
    if not CONFIG.get("manual_layout_mode", False):
        try:
            adjust_text(
                texts,
                x=[d["fps"] for d in DATA],
                y=[d["psnr"] for d in DATA],
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5), # Restore lines if auto-mode
                expand_points=(1.6, 1.6),
                expand_text=(1.3, 1.3),
                force_text=(0.5, 1.0),
                lim=800,
            )
        except Exception as e:
            print(f"Warning: adjustText failed ({e}), falling back to manual offsets.")
    else:
        print("Manual layout mode enabled: Skipping adjustText. Using 'label_offsets' from CONFIG.")
    
    # Add bubble size legend (upper left to avoid crowded bottom-right)
    add_size_legend(ax)

    fig.tight_layout()
    fig.savefig(CONFIG["save_pdf"], bbox_inches="tight", pad_inches=0.05)
    fig.savefig(CONFIG["save_png"], dpi=CONFIG["png_dpi"], bbox_inches="tight", pad_inches=0.05)
    plt.show()


def add_size_legend(ax):
    """Draw a minimal bubble-size legend in the upper-left corner (avoids crowded bottom-right)."""
    handles = []
    labels = []
    for mb in STYLE["legend_sizes"]:
        s = size_to_area(mb)
        sc = ax.scatter(
            [],
            [],
            s=s,
            color="white",
            edgecolors=STYLE["edge_color"],
            linewidth=STYLE["edge_width"],
            alpha=1.0,
        )
        handles.append(sc)
        labels.append(f"{mb} MB")

    legend = ax.legend(
        handles,
        labels,
        title="Size (MB)",
        scatterpoints=1,
        fontsize=STYLE["legend_fontsize"],
        title_fontsize=STYLE["legend_title_fontsize"],
        frameon=True,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        borderpad=1,       # Increased padding inside the box
        handletextpad=1.0,   # More space between bubble and text
        labelspacing=1.5,    # Much larger vertical spacing between bubbles
        framealpha=0.9,
        edgecolor="gray",
    )
    legend.get_frame().set_linewidth(0.5)


def main():
	plot()


if __name__ == "__main__":
	main()
