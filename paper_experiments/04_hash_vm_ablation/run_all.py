import argparse
import json
from pathlib import Path
from typing import Dict, Any
import subprocess
import shutil
import sys

from utils_io import load_config, resolve_paths, select_scenes, ensure_dirs, dry_run_report, save_manifest, build_artifact_plan, ensure_symlink
from utils_plot import placeholder_figure, assemble_rgb_depth_normal


# TODO(stage2-task3): wire plotting pipelines (hist/tomo/render_view) and assembly
# TODO(stage2-task4): emit full manifest with per-figure provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hash vs Hybrid ablation runner")
    parser.add_argument("--config", type=Path, required=True, help="Path to config.yaml")
    parser.add_argument("--dry_run", action="store_true", help="Print planned actions without executing")
    parser.add_argument("--manifest_only", action="store_true", help="Only emit manifest from dry run plan")
    parser.add_argument("--execute", action="store_true", help="Actually run missing artifact commands (render/convert)")
    parser.add_argument("--fig_rgb_depth_normal", action="store_true", help="Assemble RGB/Depth/Normal ROI figures")
    return parser.parse_args()


def run_cmd(cmd: list, cwd: Path | None = None, dry_run: bool = True) -> Dict[str, Any]:
    desc = " ".join(cmd)
    if dry_run:
        return {"command": cmd, "status": "planned", "cwd": str(cwd) if cwd else None}
    try:
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        return {"command": cmd, "status": "ok", "stdout": result.stdout, "stderr": result.stderr, "cwd": str(cwd) if cwd else None}
    except subprocess.CalledProcessError as e:
        return {"command": cmd, "status": "error", "stdout": e.stdout, "stderr": e.stderr, "returncode": e.returncode, "cwd": str(cwd) if cwd else None}


def prepare_artifacts(scene: str, method: str, plan_entry: Dict[str, Any], cfg: dict, paths: dict, execute: bool) -> Dict[str, Any]:
    """Ensure artifacts exist; for hybrid only reuse, for hash_only generate missing."""
    model_dir = Path(plan_entry[method]["model_dir"])
    iteration = plan_entry[method]["iteration"]
    view_id = plan_entry.get("view_id", None)
    cache_root = Path(paths["cache_dir"]) / scene / method
    artifact_plan = build_artifact_plan(scene, method, model_dir, iteration, Path(paths["cache_dir"]), view_id)

    actions = {"scene": scene, "method": method, "status": "ok", "commands": [], "symlinks": [], "warnings": [], "artifact_plan": artifact_plan}

    if method == "hybrid":
        # Only reuse; link compression and renders into cache
        comp = artifact_plan["artifacts"].get("compression", {})
        rnd = artifact_plan["artifacts"].get("renders", {})
        if not comp.get("exists"):
            actions["status"] = "missing"
            actions["warnings"].append("Hybrid compression missing; cannot proceed.")
            return actions
        comp_dst = cache_root / "compression"
        actions["symlinks"].append({"src": comp["path"], "dst": str(comp_dst), "result": ensure_symlink(Path(comp["path"]), comp_dst)})
        if rnd.get("exists"):
            rnd_dst = cache_root / "renders"
            actions["symlinks"].append({"src": rnd["path"], "dst": str(rnd_dst), "result": ensure_symlink(Path(rnd["path"]), rnd_dst)})
        else:
            actions["warnings"].append("Hybrid renders missing; leave to downstream if needed.")
        return actions

    # hash_only: generate if missing
    comp = artifact_plan["artifacts"].get("compression", {})
    rnd = artifact_plan["artifacts"].get("renders", {})
    ply = artifact_plan["artifacts"].get("ply", {})
    proxy_d = artifact_plan["artifacts"].get("proxy_depth", {})
    proxy_n = artifact_plan["artifacts"].get("proxy_normal", {})

    iter_num = artifact_plan.get("resolved_iteration", iteration)
    if iter_num is None:
        actions["status"] = "missing"
        actions["warnings"].append("No compression iteration found for hash_only.")
        return actions

    # render
    if not rnd.get("exists"):
        cmd = [sys.executable, "render.py", "-m", str(model_dir), "--iteration", str(iter_num), "--skip_train", "--encoder_variant", "hash_only", "--quiet"]
        actions["commands"].append(run_cmd(cmd, cwd=Path(paths["legacy_root"]).parent, dry_run=not execute))
    else:
        rnd_dst = cache_root / "renders"
        actions["symlinks"].append({"src": rnd["path"], "dst": str(rnd_dst), "result": ensure_symlink(Path(rnd["path"]), rnd_dst)})

    # convert2ply
    if not ply.get("exists"):
        cmd = [sys.executable, "convert2ply.py", "-m", str(model_dir), "--iteration", str(iter_num), "--encoder_variant", "hash_only", "--quiet"]
        actions["commands"].append(run_cmd(cmd, cwd=Path(paths["legacy_root"]).parent, dry_run=not execute))

    # proxies (depth/normal)
    if view_id is not None and (not proxy_d.get("exists") or not proxy_n.get("exists")):
        proxy_dir = Path(artifact_plan["cache_root"]) / "proxies"
        proxy_dir.mkdir(parents=True, exist_ok=True)
        render_view = Path("paper_experiments/02_geo_comparison/render_view.py")
        cmd = [
            sys.executable,
            str(render_view),
            "--model_path", str(model_dir),
            "--iteration", str(iter_num),
            "--view_ids", str(view_id),
            "--output", str(proxy_dir),
            "--save_depth",
            "--save_normal",
            "--encoder_variant", "hash_only",
            "--quiet",
        ]
        actions["commands"].append(run_cmd(cmd, cwd=Path(paths["legacy_root"]).parent, dry_run=not execute))

    return actions


def main():
    args = parse_args()
    cfg = load_config(args.config)
    base_dir = args.config.parent
    paths = resolve_paths(cfg, base_dir)
    ensure_dirs(paths)
    scenes = select_scenes(cfg)

    plan = dry_run_report(cfg, paths, scenes)

    if args.dry_run:
        print(json.dumps(plan, indent=2))
        if args.manifest_only:
            save_manifest(paths["manifest_path"], plan)
        return

    manifest: Dict[str, Any] = {"plan": plan, "figures": [], "artifacts": []}

    # Prepare artifacts per scene/method
    for entry in plan["scenes"]:
        scene = entry["scene"]
        for method in ["hybrid", "hash_only"]:
            act = prepare_artifacts(scene, method, entry, cfg, paths, execute=args.execute)
            manifest["artifacts"].append(act)

    # Assemble RGB/Depth/Normal ROI grids if requested
    if args.fig_rgb_depth_normal:
        for entry in plan["scenes"]:
            scene = entry["scene"]
            roi_list = entry.get("roi", [])
            depth_range_cfg = entry.get("depth_range", {}) or {}
            view_id = entry.get("view_id", None)
            if view_id is None:
                print(f"[WARN] skip rgb-depth-normal for {scene}: view_id missing")
                continue
            for method in ["hybrid", "hash_only"]:
                cache_root = Path(paths["cache_dir"]) / scene / method
                # Prefer cached renders; fallback to model_dir render path
                iter_num = None
                for act in manifest["artifacts"]:
                    if act.get("scene") == scene and act.get("method") == method:
                        iter_num = act.get("artifact_plan", {}).get("resolved_iteration")
                        break
                if iter_num is None:
                    iter_num = entry[method]["iteration"]
                render_dir_cache = cache_root / "renders"
                if render_dir_cache.is_dir():
                    rgb_path = render_dir_cache / f"{view_id:05d}.png"
                else:
                    render_dir_model = Path(entry[method]["model_dir"]) / "test" / f"ours_{iter_num}" / "renders"
                    rgb_path = render_dir_model / f"{view_id:05d}.png"

                proxy_dir = cache_root / "proxies"
                depth_npz = proxy_dir / f"depth_view{view_id}.npz"
                normal_npz = proxy_dir / f"normal_view{view_id}.npz"

                if not (rgb_path.is_file() and depth_npz.is_file() and normal_npz.is_file()):
                    print(f"[WARN] missing inputs for rgb-depth-normal: scene={scene} method={method}")
                    continue

                fig_path = Path(paths["outputs_dir"]) / "figures" / f"fig_rgb_depth_normal_roi_{scene}_{method}.png"
                res = assemble_rgb_depth_normal(
                    scene=scene,
                    method=method,
                    rgb_path=rgb_path,
                    depth_npz=depth_npz,
                    normal_npz=normal_npz,
                    roi_list=roi_list,
                    depth_range=depth_range_cfg,
                    save_path=fig_path,
                )
                manifest["figures"].append({"type": "rgb_depth_normal", **res})

    # Minimal placeholder: generate empty figures to validate paths
    for scene in scenes:
        fig_path = paths["outputs_dir"] / "figures" / f"placeholder_{scene}.png"
        placeholder_figure(f"placeholder-{scene}", fig_path)
        manifest["figures"].append({"scene": scene, "path": str(fig_path), "type": "placeholder"})

    save_manifest(paths["manifest_path"], manifest)
    print(f"[DONE] Wrote manifest to {paths['manifest_path']}")


if __name__ == "__main__":
    main()
