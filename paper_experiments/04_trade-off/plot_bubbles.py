import matplotlib.pyplot as plt
import numpy as np

# =====================
# 1. 基础配置
# =====================
CONFIG = {
    "figsize": (9, 7),
    "save_path": "paper_experiments/04_trade-off/tradeoff.pdf",
    "dpi": 300,
    "bubble_scale": 6.5,   # 略微增大全局气泡
    "bubble_min": 60,      # 提高最小值，防止小气泡太难看
    "bubble_max": 2800,
}

# 顶刊配色 (保持不变)
COLORS = {
    "GlowGS":   "#2F5C8F", # Deep Blue
    "3DGS":     "#C84449", # Red
    "Scaffold": "#E6A01B", # Golden
    "Mip":      "#8E7CC3", # Purple
    "Compact":  "#17BECF", # Cyan
    "Light":    "#BCBD22", # Olive
    "HAC":      "#2CA02C", # Green
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 14,
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
    [196, 28.92, 10.5,  COLORS["GlowGS"],   "GlowGS (Ours)"],
]

# =====================
# 3. 标签微调 (距离拉近版)
# =====================
# 格式: Name: (x_offset, y_offset, ha, va)
# 这里的 offset 是数据坐标偏移，我把数值调小了，让字靠近气泡
LABEL_POS = {
    "3DGS":           (0, 0.35, 'center', 'bottom'), # 之前是 0.45
    "Scaffold-GS":    (-9, 0.0, 'right', 'center'),  # 之前是 -12
    "Mip-Splatting":  (12, -0.05, 'left', 'top'),    # 之前是 15
    "Compact3DGS":    (6, -0.25, 'left', 'top'),     # 之前是 8
    "HAC":            (-6, -0.25, 'right', 'top'),   # 之前是 -8
    "LightGaussian":  (0, 0.25, 'center', 'bottom'), # 之前是 0.35
    "GlowGS (Ours)":  (0, -0.3, 'center', 'top'),    # 之前是 -0.4
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
        marker = '*' if is_ours else 'o'
        alpha = 1.0 if is_ours else 0.7  # 其他半透明
        edge_color = 'white'
        lw = 1.5 if is_ours else 1.0
        zorder = 100 if is_ours else 50
        
        # === 关键修改：大幅增加星星的显示面积 ===
        area = size_to_area(size)
        if is_ours: 
            area *= 2.5 # 之前是1.5，现在改成2.5，保证星星够大

        ax.scatter(fps, psnr, s=area, c=color, marker=marker, 
                   edgecolors=edge_color, linewidth=lw, alpha=alpha, zorder=zorder)

        # 绘制标签
        dx, dy, ha, va = LABEL_POS.get(name, (0, 0.2, 'center', 'bottom'))
        
        # 字体加粗 & 颜色匹配
        weight = 'bold' if is_ours else 'medium'
        fontsize = 14 if is_ours else 11
        
        # === 关键修改：文字颜色直接使用气泡颜色 ===
        text_color = color 
        
        ax.text(fps + dx, psnr + dy, f"{name}\n{size:.1f}MB", 
                fontsize=fontsize, fontweight=weight, color=text_color,
                ha=ha, va=va, zorder=200)

    # --- 装饰 ---
    ax.set_xlabel("Rendering Speed (FPS)", fontweight='bold', fontsize=14, labelpad=10)
    ax.set_ylabel("PSNR (dB)", fontweight='bold', fontsize=14, labelpad=10)
    
    ax.set_xlim(50, 280)
    ax.set_ylim(26.0, 30.0)
    
    ax.grid(True, which='major', linestyle='--', alpha=0.3, color='#bbbbbb', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- 图例 (右移版) ---
    # legend_x 从 70 改为 75
    legend_x, legend_y = 75, 29.5 
    
    ax.text(legend_x, legend_y + 0.3, "Model Size", fontweight='bold', fontsize=12, ha='center', color='#333333')
    
    sizes_demo = [10, 100, 500]
    for i, s in enumerate(sizes_demo):
        y_pos = legend_y - i * 0.25
        # 画空心圆
        ax.scatter(legend_x - 10, y_pos, s=size_to_area(s), 
                   facecolors='none', edgecolors='#666666', linewidth=1)
        # 写文字
        ax.text(legend_x, y_pos, f"{s} MB", va='center', fontsize=10, color='#666666')

    plt.tight_layout()
    plt.savefig(CONFIG["save_path"], bbox_inches='tight')
    plt.savefig(CONFIG["save_path"].replace(".pdf", ".png"), dpi=300)
    print("Done.")

if __name__ == "__main__":
    main()