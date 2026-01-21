import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "#333333",
    "grid.color": "#dddddd",
    "grid.linestyle": "-",
    "grid.linewidth": 0.8,
    "figure.dpi": 300,
})

PATH_BASELINE = "/home/ubuntu/lyj/Project/gaussian-splatting/output/170780ab-c/point_cloud/iteration_30000/point_cloud.ply"
PATH_OURS = "/home/ubuntu/lyj/Project/GlowGS/output/bicycle/point_cloud/iteration_30000/point_cloud.ply"
OUTPUT_DIR = "./paper_experiments/03_scale_histogram"

N_RAYS = 3
N_RAYS_DIST = 256
RAY_RADIUS = 0.03
RANDOM_SEED = 2024

COLOR_BASE_FILL = "#e6b0aa"
COLOR_BASE_LINE = "#c0392b"
COLOR_OURS_FILL = "#aed6f1"
COLOR_OURS_LINE = "#1f618d"

def load_xyz_opacity(path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        print(f"Warning: file not found: {path}")
        return np.empty((0, 3)), np.empty((0,))
    ply = PlyData.read(path)
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    op = np.asarray(v["opacity"], dtype=np.float32)
    alpha = 1.0 / (1.0 + np.exp(-op))
    return xyz, alpha

def format_count(count: int) -> str:
    if count >= 1_000_000:
        return f"{count / 1e6:.2f}M"
    if count >= 1_000:
        return f"{count / 1e3:.1f}K"
    return str(count)

def tail_mass(alpha: np.ndarray, thresholds: List[float] = None) -> dict:
    if thresholds is None:
        thresholds = [0.1, 0.05]
    res = {}
    total = float(len(alpha)) if len(alpha) > 0 else 1.0
    for t in thresholds:
        res[t] = float(np.sum(alpha < t)) / total
    return res

def choose_rays_with_hits(
    xyz_base: np.ndarray,
    op_base: np.ndarray,
    xyz_ours: np.ndarray,
    op_ours: np.ndarray,
    n_rays: int,
    seed: int,
    radius: float,
    alpha_range: Tuple[float, float] = (0.02, 0.3),
) -> List[Tuple[Optional[Tuple[np.ndarray, np.ndarray]], Optional[Tuple[np.ndarray, np.ndarray]]]]:
    rng = np.random.default_rng(seed)
    lo, hi = alpha_range
    mask_base = (op_base > lo) & (op_base < hi)
    mask_ours = (op_ours > lo) & (op_ours < hi)

    candidates = []
    if np.any(mask_ours):
        candidates.append(xyz_ours[mask_ours])
    if np.any(mask_base):
        candidates.append(xyz_base[mask_base])
    if not candidates:
        return [(None, None)] * n_rays

    all_candidates = np.concatenate(candidates, axis=0)
    rng.shuffle(all_candidates)

    rays: List[Tuple[Optional[Tuple[np.ndarray, np.ndarray]], Optional[Tuple[np.ndarray, np.ndarray]]]] = []
    for origin in all_candidates:
        direction = rng.normal(size=3)
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        direction = direction / norm

        prof_base = ray_profile(xyz_base, op_base, origin, direction, radius)
        prof_ours = ray_profile(xyz_ours, op_ours, origin, direction, radius)
        if prof_base is None and prof_ours is None:
            continue
        rays.append((prof_base, prof_ours))
        if len(rays) >= n_rays:
            break

    while len(rays) < n_rays:
        rays.append((None, None))
    return rays

def ray_profile(
    xyz: np.ndarray,
    alpha: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    radius: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    rel = xyz - origin[None, :]
    t = np.dot(rel, direction)
    mask_front = t > 0
    if not np.any(mask_front):
        return None
    rel = rel[mask_front]
    alpha_f = alpha[mask_front]
    t = t[mask_front]

    closest = rel - np.outer(t, direction)
    dist = np.linalg.norm(closest, axis=1)
    mask_ray = dist < radius
    if not np.any(mask_ray):
        return None

    t = t[mask_ray]
    alpha_f = alpha_f[mask_ray]
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    alpha_f = alpha_f[sort_idx]

    t = t - t.min()
    t_cut = np.percentile(t, 99.5)
    keep = t <= t_cut
    t = t[keep]
    alpha_f = alpha_f[keep]

    trans = np.cumprod(1.0 - alpha_f)
    accum = 1.0 - trans
    return t, accum

def compute_thickness(profile: Tuple[np.ndarray, np.ndarray]) -> Optional[Tuple[float, float, float]]:
    t, accum = profile
    if accum.size < 2:
        return None
    z10 = float(np.interp(0.1, accum, t, left=np.nan, right=np.nan))
    z90 = float(np.interp(0.9, accum, t, left=np.nan, right=np.nan))
    if np.isnan(z10) or np.isnan(z90):
        return None
    return z10, z90, z90 - z10

def compute_zoom_limit(
    profiles: List[Tuple[Optional[Tuple[np.ndarray, np.ndarray]], Optional[Tuple[np.ndarray, np.ndarray]]]],
    default: float = 0.05,
) -> float:
    candidates: List[float] = []
    for pb, po in profiles:
        for prof in (pb, po):
            if prof is None:
                continue
            t, acc = prof
            mask = acc < 0.995
            if np.any(mask):
                candidates.append(float(np.max(t[mask])))
            else:
                candidates.append(float(np.max(t)))
    if not candidates:
        return default
    zoom = np.percentile(candidates, 90) * 1.05
    return min(default, zoom)

def plot_histogram(op_base: np.ndarray, op_ours: np.ndarray, count_base: int, count_ours: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bins = np.linspace(0, 1.0, 60)
    ax.hist(op_base, bins=bins, color=COLOR_BASE_FILL, alpha=0.6, log=True, zorder=1, histtype="stepfilled")
    ax.hist(op_base, bins=bins, color=COLOR_BASE_LINE, lw=1.3, log=True, zorder=2, histtype="step", label=f"3DGS ({format_count(count_base)} prim)")
    ax.hist(op_ours, bins=bins, color=COLOR_OURS_FILL, alpha=0.65, log=True, zorder=3, histtype="stepfilled")
    ax.hist(op_ours, bins=bins, color=COLOR_OURS_LINE, lw=1.8, log=True, zorder=4, histtype="step", label=f"GlowGS ({format_count(count_ours)} prim)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(bottom=10)
    ax.grid(True, which="major", axis="y", alpha=0.5, zorder=0)
    ax.grid(False, axis="x")
    ax.set_xlabel(r"Gaussian Opacity", fontweight="bold")
    ax.set_ylabel("Number of Primitives (Log)", fontweight="bold")
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "opacity_hist.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_tail_mass(tail_base: dict, tail_ours: dict) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    thresholds = [0.1, 0.05]
    x_pos = np.arange(len(thresholds))
    width = 0.32
    vals_base = [tail_base[t] for t in thresholds]
    vals_ours = [tail_ours[t] for t in thresholds]
    ax.bar(x_pos - width / 2, vals_base, width=width, color=COLOR_BASE_LINE, alpha=0.8, label="3DGS")
    ax.bar(x_pos + width / 2, vals_ours, width=width, color=COLOR_OURS_LINE, alpha=0.8, label="GlowGS")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["α<0.1", "α<0.05"])
    ax.set_ylabel("Primitive Fraction")
    ymax = max(vals_base + vals_ours + [0.05]) * 1.15
    ax.set_ylim(0, ymax)
    for x, v in zip(x_pos - width / 2, vals_base):
        ax.text(x, v + 0.01, f"{v * 100:.1f}%", ha="center", va="bottom", fontsize=9, color=COLOR_BASE_LINE)
    for x, v in zip(x_pos + width / 2, vals_ours):
        ax.text(x, v + 0.01, f"{v * 100:.1f}%", ha="center", va="bottom", fontsize=9, color=COLOR_OURS_LINE)
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "tail_mass.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_rays(ray_profiles: List[Tuple[Optional[Tuple[np.ndarray, np.ndarray]], Optional[Tuple[np.ndarray, np.ndarray]]]]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    max_t = 0.0
    colors = ["#d62728", "#e377c2", "#ff9896", "#1f77b4", "#17becf", "#aec7e8"]
    t_placeholder = np.array([0.0, 0.05], dtype=np.float32)
    for i, (prof_base, prof_ours) in enumerate(ray_profiles):
        if prof_base is not None:
            t_b, a_b = prof_base
            ax.plot(t_b, a_b, color=colors[i], lw=1.5, alpha=0.9)
            max_t = max(max_t, float(t_b.max()))
        else:
            ax.plot(t_placeholder, np.zeros_like(t_placeholder), color=colors[i], lw=1.5, alpha=0.3)
        if prof_ours is not None:
            t_o, a_o = prof_ours
            ax.plot(t_o, a_o, color=colors[i + N_RAYS], lw=1.5, alpha=0.9)
            max_t = max(max_t, float(t_o.max()))
        else:
            ax.plot(t_placeholder, np.zeros_like(t_placeholder), color=colors[i + N_RAYS], lw=1.5, alpha=0.3)
    ax.set_xlabel("Depth along ray (m)")
    ax.set_ylabel("Accumulated Opacity")
    ax.set_ylim(0, 1.05)
    if max_t > 0:
        ax.set_xlim(0, min(0.05, max_t))
    else:
        ax.set_xlim(0, 0.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="both", alpha=0.3)
    legend_lines = [plt.Line2D([0], [0], color=colors[i], lw=1.5) for i in range(4)]
    legend_labels = ["Ray1-3DGS", "Ray2-3DGS", "Ray1-GlowGS", "Ray2-GlowGS"]
    ax.legend(legend_lines, legend_labels, frameon=False, fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "ray_spread.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_box(
    ray_profiles_dist: List[Tuple[Optional[Tuple[np.ndarray, np.ndarray]], Optional[Tuple[np.ndarray, np.ndarray]]]],
    ray_profiles_tail: List[Tuple[Optional[Tuple[np.ndarray, np.ndarray]], Optional[Tuple[np.ndarray, np.ndarray]]]],
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.2))

    def collect_delta(profiles: List[Tuple[Optional[Tuple[np.ndarray, np.ndarray]], Optional[Tuple[np.ndarray, np.ndarray]]]]) -> Tuple[List[float], List[float]]:
        delta_b: List[float] = []
        delta_o: List[float] = []
        for prof_base, prof_ours in profiles:
            if prof_base is not None:
                thick = compute_thickness(prof_base)
                if thick is not None:
                    delta_b.append(thick[2])
            if prof_ours is not None:
                thick = compute_thickness(prof_ours)
                if thick is not None:
                    delta_o.append(thick[2])
        return delta_b, delta_o

    delta_base_dist, delta_ours_dist = collect_delta(ray_profiles_dist)
    delta_base_tail, delta_ours_tail = collect_delta(ray_profiles_tail)

    data_box = [delta_base_dist, delta_ours_dist, delta_base_tail, delta_ours_tail]
    colors_box = [COLOR_BASE_FILL, COLOR_OURS_FILL, COLOR_BASE_FILL, COLOR_OURS_FILL]
    edge_colors = [COLOR_BASE_LINE, COLOR_OURS_LINE, COLOR_BASE_LINE, COLOR_OURS_LINE]
    labels = ["3DGS", "GlowGS", "3DGS α<0.1", "GlowGS α<0.1"]

    bp = ax.boxplot(
        data_box,
        vert=True,
        patch_artist=True,
        labels=labels,
        showfliers=False,
        medianprops=dict(color="#222222"),
        whiskerprops=dict(color="#666666"),
        capprops=dict(color="#666666"),
    )
    for patch, face, edge in zip(bp["boxes"], colors_box, edge_colors):
        patch.set_facecolor(face)
        patch.set_edgecolor(edge)
    ax.set_ylabel("Δz (m)")
    ax.tick_params(axis="x", labelsize=9, rotation=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    mean_base = np.mean(delta_base_dist) if delta_base_dist else float("nan")
    mean_ours = np.mean(delta_ours_dist) if delta_ours_dist else float("nan")
    mean_base_tail = np.mean(delta_base_tail) if delta_base_tail else float("nan")
    mean_ours_tail = np.mean(delta_ours_tail) if delta_ours_tail else float("nan")
    ax.text(
        0.97,
        0.97,
        f"mean Δz (main/tail): 3DGS={mean_base*1000:.1f}/{mean_base_tail*1000:.1f}mm, "
        f"GlowGS={mean_ours*1000:.1f}/{mean_ours_tail*1000:.1f}mm",
        transform=ax.transAxes,
        fontsize=8,
        color="#333333",
        ha="right",
        va="top",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "dz_box.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    print("Loading data...")
    xyz_base, op_base = load_xyz_opacity(PATH_BASELINE)
    xyz_ours, op_ours = load_xyz_opacity(PATH_OURS)
    if op_base.size == 0 or op_ours.size == 0:
        print("Error: missing data; check paths.")
        return

    count_base = len(op_base)
    count_ours = len(op_ours)
    print(f"Baseline: {count_base}, Ours: {count_ours}")

    tail_base = tail_mass(op_base)
    tail_ours = tail_mass(op_ours)

    ray_profiles = choose_rays_with_hits(
        xyz_base, op_base, xyz_ours, op_ours, n_rays=N_RAYS, seed=RANDOM_SEED, radius=RAY_RADIUS
    )
    ray_profiles_dist = choose_rays_with_hits(
        xyz_base, op_base, xyz_ours, op_ours, n_rays=N_RAYS_DIST, seed=RANDOM_SEED + 1, radius=RAY_RADIUS
    )
    ray_profiles_tail = choose_rays_with_hits(
        xyz_base, op_base, xyz_ours, op_ours, n_rays=N_RAYS_DIST, seed=RANDOM_SEED + 2, radius=RAY_RADIUS, alpha_range=(0.0, 0.1)
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_histogram(op_base, op_ours, count_base, count_ours)
    plot_tail_mass(tail_base, tail_ours)
    plot_rays(ray_profiles)
    plot_box(ray_profiles_dist, ray_profiles_tail)
    print(f"Saved figures to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
