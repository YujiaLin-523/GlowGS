#!/usr/bin/env bash
# Batch convert checkpoints to PLY for bicycle (full + ablations)
# Uses convert2ply.py; skips existing PLY unless --force is given.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ITERATION=30000
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --iteration)
            ITERATION=$2
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--iteration N] [--force]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--iteration N] [--force]"
            exit 1
            ;;
    esac
done

# config_name|model_dir|encoder_variant|feature_mod_type|densification_mode|use_edge_loss
declare -a CONFIGS=(
    "ours|output/legacy_output/bicycle|hybrid|film|mass_aware|True"
    "wo_vm|output/bicycle_wo_vm|hash_only|film|mass_aware|True"
    "wo_mass|output/bicycle_wo_mass|hybrid|film|standard|True"
    "wo_edge|output/bicycle_wo_edge|hybrid|film|mass_aware|False"
)

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r name model_dir encoder mod densify edge <<< "$entry"
    ply_dir="$model_dir/point_cloud/iteration_${ITERATION}"
    ply_path="$ply_dir/point_cloud.ply"
    vm_root="$model_dir/compression/vm_planes_fp16.npz"
    vm_iter="$model_dir/compression/iteration_${ITERATION}/vm_planes_fp16.npz"
    vm_grid="$model_dir/compression/iteration_${ITERATION}/grid_all.npz"

    echo "------------------------------------------------------------"
    echo "Config: $name"
    echo "Model: $model_dir"
    echo "Iter:  iteration_${ITERATION}"

    if [[ -f "$ply_path" && "$FORCE" == false ]]; then
        echo "PLY exists, skip (use --force to overwrite): $ply_path"
        continue
    fi

    # For hybrid runs, convert2ply expects vm_planes_fp16.npz under compression/.
    # If missing, try to stage from iteration directory; if still missing, and grid_all exists,
    # warn user to rebuild via grid_all (manual or helper script).
    if [[ "$encoder" == "hybrid" && ! -f "$vm_root" ]]; then
        if [[ -f "$vm_iter" ]]; then
            echo "Staging VM planes to compression root for validator..."
            cp "$vm_iter" "$vm_root"
        else
            echo "[WARN] VM planes missing: $vm_root and $vm_iter"
            if [[ -f "$vm_grid" ]]; then
                echo "        grid_all.npz is present; run the helper to rebuild vm_planes_fp16.npz from grid_all if needed."
            fi
        fi
    fi

    mkdir -p "$ply_dir"

    cmd=(python convert2ply.py \
        -s data/360_v2/bicycle \
        -m "$model_dir" \
        --iteration "$ITERATION" \
        --encoder_variant "$encoder" \
        --feature_mod_type "$mod" \
        --densification_mode "$densify" \
        --use_edge_loss "$edge")

    echo "Running: CONVERT_SKIP_ENCODER_VALIDATION=1 ${cmd[*]}"
    CONVERT_SKIP_ENCODER_VALIDATION=1 "${cmd[@]}"
    echo "Saved: $ply_path"

done

echo "============================================================"
echo "Done."
