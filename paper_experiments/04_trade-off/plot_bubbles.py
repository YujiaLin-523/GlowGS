import matplotlib.pyplot as plt
import numpy as np

# =====================
# 1. 基础配置
# =====================
CONFIG = {
    "figsize": (9, 7),
    "save_path": "paper_experiments/04_trade-off/tradeoff.pdf",
    "dpi": 300,
    "bubble_scale": 6.5,
    "bubble_min": 60,
    "bubble_max": 2800,
}

# Muted 统一色板
COLORS = {
    "Ours":     "#F09496",   # Muted Brick Red
    "3DGS":     "#7F7F7F",   # Neutral Gray
    "Scaffold": "#5F97C6",   # Muted Blue
    "Mip":      "#9C9BE9",   # Muted Purple
    "Compact":  "#7BC8C8",   # Muted Teal
    "Light":    "#AFC778",   # Muted Green
    "HAC":      "#D4A76A",   # Muted Amber
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 28,
    "axes.linewidth": 1.5,
    "axes.edgecolor": "#333333",
    "pdf.fonttype": 42,
})

# =====================
# 2. 数据录入
# =====================
# [FPS, PSNR, Size(MB), Color, Name]
DATA = [
    # 核心组
    [134, 27.45, 825.3, COLORS["3DGS"],     "3DGS"],
    [124, 27.65, 189.2, COLORS["Scaffold"], "Scaffold-GS"],
    [135, 27.39, 734.4, COLORS["Mip"],      "Mip-Splatting"],
    
    # 轻量组
    [148, 26.97, 26.3,  COLORS["Compact"],  "Compact3DGS"],
    [113, 27.47, 17.0,  COLORS["HAC"],      "HAC"],
    
    # 极速组
    [235, 26.91, 53.9,  COLORS["Light"],    "LightGaussian"],
    
    # Ours
    [196, 28.92, 10.5,  COLORS["Ours"],   "Ours"],
]

# =====================
# 3. 标签微调 (距离拉近版)
# =====================

LABEL_POS = {
    "3DGS":           (24, 0, 'center', 'bottom'),
    "Scaffold-GS":    (30, 0.25, 'right', 'center'),
    "Mip-Splatting":  (10, 0, 'left', 'top'),
    "Compact3DGS":    (-30, -0.1, 'left', 'top'),
    "HAC":            (8, -0.1, 'right', 'top'),
    "LightGaussian":  (0, 0.15, 'center', 'bottom'),
    "Ours":  (0, -0.1, 'center', 'top'),
}

def size_to_area(size_mb):
    return np.clip(size_mb * CONFIG["bubble_scale"], CONFIG["bubble_min"], CONFIG["bubble_max"])

def main():
    fig, ax = plt.subplots(figsize=CONFIG["figsize"])

    # 按大小排序，防止遮挡
    DATA.sort(key=lambda x: x[2], reverse=True)

    for fps, psnr, size, color, name in DATA:
        is_ours = "Ours" in name
        
        # 样式逻辑
        marker = 'o'
        alpha = 1.0 if is_ours else 0.7  # 其他半透明
        edge_color = 'white'
        lw = 1.5 if is_ours else 1.0
        zorder = 100 if is_ours else 50
        
        area = size_to_area(size)

        ax.scatter(fps, psnr, s=area, c=color, marker=marker, 
                   edgecolors=edge_color, linewidth=lw, alpha=alpha, zorder=zorder)

        # 绘制标签
        dx, dy, ha, va = LABEL_POS.get(name, (0, 0.2, 'center', 'bottom'))
        
        # 字体加粗 & 颜色匹配
        weight = 'bold' if is_ours else 'medium'
        fontsize = 28 if is_ours else 26
        
        # === 关键修改：文字颜色直接使用气泡颜色 ===
        text_color = color 
        
        ax.text(fps + dx, psnr + dy, f"{name}", 
                fontsize=fontsize, fontweight=weight, color=text_color,
                ha=ha, va=va, zorder=200)

    # --- 装饰 ---
    ax.set_xlabel("Rendering Speed (FPS)", fontweight='bold', fontsize=24, labelpad=10)
    ax.set_ylabel("PSNR (dB)", fontweight='bold', fontsize=24, labelpad=10)
    
    ax.set_xlim(100, 250)
    ax.set_ylim(26.5, 29.5)
    
    ax.grid(True, which='major', linestyle='--', alpha=0.3, color='#bbbbbb', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- 图例 (右移版) ---
    legend_x, legend_y = 130, 29.3 
    
    # ax.text(legend_x, legend_y + 0.3, "Model Size", fontweight='bold', fontsize=24, ha='center', color='#333333')
    
    sizes_demo = [10, 100, 500]
    spacing = 0.35
    for i, s in enumerate(sizes_demo):
        y_pos = legend_y - i * spacing
        # 画空心圆
        ax.scatter(legend_x - 15, y_pos, s=size_to_area(s), 
                   facecolors='none', edgecolors='#666666', linewidth=1)
        # 写文字（左对齐，保持等距）
        ax.text(legend_x, y_pos, f"{s} MB", va='center', ha='left', fontsize=24, color='#666666')

    plt.tight_layout()
    plt.savefig(CONFIG["save_path"], bbox_inches='tight')
    plt.savefig(CONFIG["save_path"].replace(".pdf", ".png"), dpi=300)
    print("Done.")

if __name__ == "__main__":
    main()