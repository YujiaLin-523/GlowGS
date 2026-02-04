import json
import os
from pathlib import Path
import yaml
from typing import Optional, Tuple, Dict, Any

# TODO(stage2-task0): extend parsing to validate view_id/roi completeness and types

def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_paths(cfg: dict, base_dir: Path) -> dict:
    """Resolve root-relative paths and attach derived locations.
    Returns a shallow copy with pathlib Paths.
    """
    out = dict(cfg)
    out["legacy_root"] = (base_dir / cfg["legacy_root"]).resolve()
    out["hash_only_root"] = (base_dir / cfg["hash_only_root"]).resolve()
    out["outputs_dir"] = (base_dir / cfg.get("outputs_dir", "outputs")).resolve()
    out["cache_dir"] = (base_dir / cfg.get("cache_dir", "cache")).resolve()
    out["manifest_path"] = (base_dir / cfg.get("manifest_path", "outputs/manifest.json")).resolve()
    return out


def select_scenes(cfg: dict) -> list:
    scenes = list(cfg.get("scenes_main", []))
    if cfg.get("use_supp", False):
        scenes.extend(cfg.get("scenes_supp", []))
    return scenes


def ensure_dirs(paths: dict):
    for key in ["outputs_dir", "cache_dir", "manifest_path"]:
        path = paths[key]
        # manifest_path is a file path; ensure parent
        target_dir = path if path.is_dir() else path.parent
        target_dir.mkdir(parents=True, exist_ok=True)


def dry_run_report(cfg: dict, paths: dict, scenes: list) -> dict:
    """Prepare a human-readable dry run plan without executing anything."""
    plan = {
        "legacy_root": str(paths["legacy_root"]),
        "hash_only_root": str(paths["hash_only_root"]),
        "outputs_dir": str(paths["outputs_dir"]),
        "cache_dir": str(paths["cache_dir"]),
        "manifest_path": str(paths["manifest_path"]),
        "scenes": [],
    }
    for scene in scenes:
        entry = {
            "scene": scene,
            "hybrid": {
                "model_dir": str(paths["legacy_root"] / scene),
                "iteration": cfg.get("iteration_hybrid", -1),
            },
            "hash_only": {
                "model_dir": str(paths["hash_only_root"] / scene),
                "iteration": cfg.get("iteration_hash_only", -1),
            },
            "view_id": cfg.get("view_id_map", {}).get(scene, None),
            "roi": cfg.get("roi_map", {}).get(scene, []),
            "depth_range": cfg.get("colormap_depth_range", {}).get(scene, {}),
        }
        plan["scenes"].append(entry)
    return plan


def save_manifest(manifest_path: Path, manifest: dict):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def find_iteration_dir(model_dir: Path, iteration: int) -> Tuple[Optional[int], Optional[Path]]:
    """Resolve the compression iteration directory for a model.

    Returns (iter_number, path) or (None, None) if not found.
    """
    compression_root = model_dir / "compression"
    if not compression_root.is_dir():
        return None, None
    if iteration is not None and iteration >= 0:
        target = compression_root / f"iteration_{iteration}"
        return iteration, target if target.is_dir() else None
    # iteration == -1: find latest
    max_iter = None
    max_dir = None
    for child in compression_root.iterdir():
        if child.is_dir() and child.name.startswith("iteration_"):
            try:
                it = int(child.name.split("iteration_")[-1])
            except ValueError:
                continue
            if (max_iter is None) or (it > max_iter):
                max_iter, max_dir = it, child
    return max_iter, max_dir


def ensure_symlink(src: Path, dst: Path) -> str:
    """Create dst symlink pointing to src if not present. Returns status string."""
    if dst.exists() or dst.is_symlink():
        return "reuse"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src, target_is_directory=src.is_dir())
    return "linked"


def build_artifact_plan(scene: str, method: str, model_dir: Path, iteration: Optional[int], cache_dir: Path, view_id: Optional[int]) -> Dict[str, Any]:
    """Construct required artifact paths and existence flags."""
    plan: Dict[str, Any] = {
        "scene": scene,
        "method": method,
        "model_dir": str(model_dir),
        "iteration": iteration,
        "artifacts": {},
    }
    iter_num, comp_dir = find_iteration_dir(model_dir, iteration if iteration is not None else -1)
    plan["resolved_iteration"] = iter_num
    plan["artifacts"]["compression"] = {
        "path": str(comp_dir) if comp_dir else None,
        "exists": bool(comp_dir and comp_dir.is_dir()),
    }
    render_dir = None
    if iter_num is not None:
        render_dir = model_dir / "test" / f"ours_{iter_num}" / "renders"
    plan["artifacts"]["renders"] = {
        "path": str(render_dir) if render_dir else None,
        "exists": bool(render_dir and render_dir.is_dir()),
    }
    ply_path = None
    if iter_num is not None:
        ply_path = model_dir / "point_cloud" / f"iteration_{iter_num}" / "point_cloud.ply"
    plan["artifacts"]["ply"] = {
        "path": str(ply_path) if ply_path else None,
        "exists": bool(ply_path and ply_path.is_file()),
    }

    proxy_dir = cache_dir / scene / method / "proxies"
    proxy_depth = proxy_dir / f"depth_view{view_id}.npz" if view_id is not None else None
    proxy_normal = proxy_dir / f"normal_view{view_id}.npz" if view_id is not None else None
    plan["artifacts"]["proxy_depth"] = {
        "path": str(proxy_depth) if proxy_depth else None,
        "exists": bool(proxy_depth and proxy_depth.is_file()),
    }
    plan["artifacts"]["proxy_normal"] = {
        "path": str(proxy_normal) if proxy_normal else None,
        "exists": bool(proxy_normal and proxy_normal.is_file()),
    }
    plan["cache_root"] = str(cache_dir / scene / method)
    return plan

