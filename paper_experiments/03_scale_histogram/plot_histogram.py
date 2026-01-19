"""
Publication-Quality Opacity Histogram (Refined)
Clean layout, adjusted legend, specific unit labels.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator
from plyfile import PlyData
import os

# ==============================================================================
# 🎨 Nature/Science Style Configuration
# ==============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.linewidth': 1.0,       # 边框稍微加粗一点点，更有质感
    'axes.edgecolor': '#333333', # 深灰色边框，比纯黑更柔和
    'grid.color': '#DDDDDD',     # 极淡的网格线
    'grid.linestyle': '-',
    'grid.linewidth': 0.8,
    'figure.dpi': 300,
})

# ==============================================================================
# 🔴 Data Paths (保持不变)
# ==============================================================================
PATH_BASELINE = "/home/ubuntu/lyj/Project/gaussian-splatting/output/170780ab-c/point_cloud/iteration_30000/point_cloud.ply"
PATH_OURS     = "/home/ubuntu/lyj/Project/GlowGS/output/bicycle/point_cloud/iteration_30000/point_cloud.ply"
OUTPUT_PATH   = "./paper_experiments/03_scale_histogram/figure_3_scale_histogram.png"

# 🎨 Refined Palette (更为沉稳的配色)
COLOR_BASE_FILL = '#E6B0AA'  # 浅红 (填充)
COLOR_BASE_LINE = '#C0392B'  # 深红 (线条)
COLOR_OURS_FILL = '#AED6F1'  # 浅蓝 (填充)
COLOR_OURS_LINE = '#1F618D'  # 深蓝 (线条)

def get_data(path):
    """Load opacity values from PLY file"""
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return np.array([]) 
    
    ply = PlyData.read(path)
    v = ply['vertex']
    op = np.asarray(v['opacity'])
    opacity = 1.0 / (1.0 + np.exp(-op))
    return opacity

def format_count(count):
    """Format count to M (Millions) or K (Thousands)"""
    if count >= 1e6:
        return f"{count/1e6:.2f}M"
    elif count >= 1e3:
        return f"{count/1e3:.1f}K"
    return str(count)

def main():
    print("Loading Data...")
    op_base = get_data(PATH_BASELINE)
    op_ours = get_data(PATH_OURS)
    
    if len(op_base) == 0 or len(op_ours) == 0:
        print("Error: Could not load data. Check paths.")
        return

    count_base = len(op_base)
    count_ours = len(op_ours)
    
    print(f"Baseline: {count_base}, Ours: {count_ours}")

    # =========================================================================
    # 🎨 Plotting
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 5.5)) 
    
    # Bins setup
    bins = np.linspace(0, 1.0, 60)

    # --- Plot 1: 3DGS (Baseline) ---
    # Layer 1: Fill (Background)
    ax.hist(op_base, bins=bins, color=COLOR_BASE_FILL, 
            alpha=0.6, log=True, zorder=1, histtype='stepfilled')
    # Layer 2: Edge (Foreground, crisp line)
    ax.hist(op_base, bins=bins, color=COLOR_BASE_LINE, 
            lw=1.5, log=True, zorder=2, histtype='step', 
            label=f'3DGS ({format_count(count_base)} Primitives)')

    # --- Plot 2: Ours (GlowGS) ---
    # Layer 1: Fill
    ax.hist(op_ours, bins=bins, color=COLOR_OURS_FILL, 
            alpha=0.7, log=True, zorder=3, histtype='stepfilled')
    # Layer 2: Edge
    ax.hist(op_ours, bins=bins, color=COLOR_OURS_LINE, 
            lw=2.0, log=True, zorder=4, histtype='step', 
            label=f'Ours ({format_count(count_ours)} Primitives)')

    # --- Styling ---
    
    # 1. Remove Top and Right Spines (Despine)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 2. Axis Limits & Grid
    ax.set_xlim(0, 1.02)
    ax.set_ylim(bottom=10) # Log scale shouldn't start at 0
    
    # Only show horizontal grid (y-axis) for cleanliness
    ax.grid(True, which='major', axis='y', alpha=0.5, zorder=0)
    ax.grid(False, axis='x') 

    # 3. Labels
    ax.set_xlabel(r'Gaussian Opacity ($\alpha$)', fontweight='bold')
    ax.set_ylabel('Number of Primitives (Log Scale)', fontweight='bold')

    # 4. Legend Optimization
    # loc='upper right' but shifted left using bbox_to_anchor
    # (x, y): (0.85, 1.0) means the legend's top-right corner is at x=0.85 of the axes
    legend = ax.legend(
        loc='upper right', 
        bbox_to_anchor=(0.82, 1.0), # 向左移动，避开右侧 x=1.0 处的蓝色波峰
        frameon=False,              # 无边框，更简洁
        fontsize=12
    )

    # --- Save ---
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()