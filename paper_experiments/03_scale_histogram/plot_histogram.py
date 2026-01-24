import os
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData
import matplotlib.font_manager as fm

# === 1. 强制顶刊样式 ===
COLOR_BASE = "#D6404E"  # 稍微加深一点的红色 (Nature Red)
COLOR_OURS = "#4A7EBB"  # 稍微加深一点的蓝色 (Nature Blue)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"], # 强制 Arial
    "font.size": 14,              # 字号加大，防止留白过多显得字小
    "axes.linewidth": 1.2,
    "axes.edgecolor": "#333333",
    "xtick.major.pad": 6,         # 增加刻度文字距离
    "ytick.major.pad": 6,
    "pdf.fonttype": 42,
    "figure.dpi": 300,
})

PATH_BASELINE = "/home/ubuntu/lyj/Project/gaussian-splatting/output/bicycle/point_cloud/iteration_30000/point_cloud.ply"
PATH_OURS = "/home/ubuntu/lyj/Project/GlowGS/output/bicycle/point_cloud/iteration_30000/point_cloud.ply"
OUTPUT_DIR = "./paper_experiments/03_scale_histogram"

def load_data(path):
    ply = PlyData.read(path)
    op = np.asarray(ply["vertex"]["opacity"], dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-op))

def plot_fig5_layout_fix(op_base, op_ours):
    print("Plotting Fig 5 with fixed layout...")
    
    # === 关键修改：手动切分画布 ===
    # 宽度比例 3:2，让左边的直方图宽一点，右边的柱状图窄一点
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [1.6, 1]})
    
    # === 关键修改：像素级控制边距 ===
    # left/right/top/bottom 控制画布的有效区域
    # wspace 控制两个子图中间的缝隙
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.18, wspace=0.25)

    # --- 左图：直方图 ---
    bins = np.linspace(0, 1.0, 50)
    # 线条加粗，颜色加深
    ax1.hist(op_base, bins=bins, color=COLOR_BASE, histtype="step", lw=2.5, zorder=2, label="3DGS")
    ax1.hist(op_ours, bins=bins, color=COLOR_OURS, histtype="step", lw=3.0, zorder=3, label="GlowGS")
    # 填充极淡的颜色增加层次
    ax1.hist(op_base, bins=bins, color=COLOR_BASE, alpha=0.1, histtype="stepfilled", zorder=1)
    ax1.hist(op_ours, bins=bins, color=COLOR_OURS, alpha=0.1, histtype="stepfilled", zorder=1)

    ax1.set_yscale('log')
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(bottom=10) # Log scale 不能从0开始
    
    # 美化坐标轴
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xlabel("Gaussian Opacity", fontweight='bold')
    ax1.set_ylabel("Count (Log)", fontweight='bold')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
    
    # 图例放在图内上方，无边框
    ax1.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=12)

    # --- 右图：Tail Mass ---
    thresholds = [0.1, 0.05]
    vals_b = [np.mean(op_base < t) for t in thresholds]
    vals_o = [np.mean(op_ours < t) for t in thresholds]
    
    x = np.arange(2)
    width = 0.35
    
    bars_b = ax2.bar(x - width/2, vals_b, width, color=COLOR_BASE, alpha=0.9, label='3DGS')
    bars_o = ax2.bar(x + width/2, vals_o, width, color=COLOR_OURS, alpha=0.9, label='GlowGS')

    # 标数值
    for rect in bars_b + bars_o:
        height = rect.get_height()
        if height > 0.001:
            ax2.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                     f'{height*100:.1f}%', ha='center', va='bottom', fontsize=11, color='black')

    ax2.set_xticks(x)
    # 使用 LaTeX 格式让 α 看起来更专业
    ax2.set_xticklabels([r"$\alpha < 0.1$", r"$\alpha < 0.05$"])
    ax2.set_ylabel("Primitive Fraction", fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    # 让 Y 轴上限稍微高一点，留出标数字的空间
    ax2.set_ylim(0, max(max(vals_b), max(vals_o)) * 1.2)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)

    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "opacity_stats.pdf")
    plt.savefig(save_path) # 不需要 bbox_inches='tight'，因为我们手动 adjust 了
    plt.savefig(save_path.replace(".pdf", ".png"))
    print(f"Done! Check layout at: {save_path}")

if __name__ == "__main__":
    print("Loading data...")
    op_base = load_data(PATH_BASELINE)
    op_ours = load_data(PATH_OURS)
    plot_fig5_layout_fix(op_base, op_ours)