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

# config_name|model_dir|enable_vm|enable_mass_aware|enable_edge_loss
declare -a CONFIGS=(
    "ours|output/legacy_output/bicycle|True|True|True"
    "wo_vm|output/bicycle_wo_vm|False|True|True"
    "wo_mass|output/bicycle_wo_mass|True|False|True"
    "wo_edge|output/bicycle_wo_edge|True|True|False"
)

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r name model_dir vm ma el <<< "$entry"
    ply_dir="$model_dir/point_cloud/iteration_${ITERATION}"
    ply_path="$ply_dir/point_cloud.ply"
    echo "------------------------------------------------------------"
    echo "Config: $name"
    echo "Model: $model_dir"
    echo "Iter:  iteration_${ITERATION}"

    if [[ -f "$ply_path" && "$FORCE" == false ]]; then
        echo "PLY exists, skip (use --force to overwrite): $ply_path"
        continue
    fi

    mkdir -p "$ply_dir"

    cmd=(python convert2ply.py \
        -s data/360_v2/bicycle \
        -m "$model_dir" \
        --iteration "$ITERATION" \
        --enable_vm "$vm" \
        --enable_mass_aware "$ma" \
        --enable_edge_loss "$el")

    echo "Running: ${cmd[*]}"
    "${cmd[@]}"
    echo "Saved: $ply_path"

done

echo "============================================================"
echo "Done."
