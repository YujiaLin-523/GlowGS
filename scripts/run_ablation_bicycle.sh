#!/usr/bin/env bash
# ==============================================================================
# GlowGS Ablation Study - Bicycle Scene (Paper Figure Generation)
# ==============================================================================
# 
# Configurations (3 ablations, baseline results already exist):
#   1. w/o VM:           Hash-Only Encoder (no VM/tri-plane geometry)
#   2. w/o Mass-aware:   Hybrid + FiLM + Standard Densification (no mass-aware)
#   3. w/o Edge Loss:    Hybrid + FiLM + Mass-Aware (no edge loss)
#
# Pre-existing results (DO NOT re-train):
#   - Ours (Full):  output/bicycle/  (or output/legacy_output/bicycle/)
#   - 3DGS:         /home/ubuntu/lyj/Project/gaussian-splatting/output/bicycle/
#
# Output Structure:
#   output/bicycle_wo_vm/              # w/o VM (enable_vm=False, hash-only bypass)
#   output/bicycle_wo_mass/            # w/o Mass-aware
#   output/bicycle_wo_edge/            # w/o Edge Loss
#
# Usage:
#   bash scripts/run_ablation_bicycle.sh
#   bash scripts/run_ablation_bicycle.sh --dry-run
#   bash scripts/run_ablation_bicycle.sh --config wo_vm        # Run single config
# ==============================================================================

set -e

# Configuration
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_PATH="data/360_v2/bicycle"
ITERATIONS=30000
SAVE_ITERATIONS="30000"
TEST_ITERATIONS="7000 30000"

# Parse arguments
DRY_RUN=false
SINGLE_CONFIG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --config)
            SINGLE_CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--config <wo_vm|wo_mass|wo_edge>]"
            exit 1
            ;;
    esac
done

# ==============================================================================
# Define ablation configurations (3 ablations only)
# ==============================================================================
# Format: config_name|output_dir|enable_vm|enable_mass_aware|enable_edge_loss
# Values use True/False for the simplified bool flag pattern
declare -A ABLATION_CONFIGS
ABLATION_CONFIGS["wo_vm"]="bicycle_wo_vm|False|True|True"
ABLATION_CONFIGS["wo_mass"]="bicycle_wo_mass|True|False|True"
ABLATION_CONFIGS["wo_edge"]="bicycle_wo_edge|True|True|False"

# Ordered list
CONFIG_ORDER=("wo_vm" "wo_mass" "wo_edge")

# Filter to single config if specified
if [[ -n "$SINGLE_CONFIG" ]]; then
    if [[ -z "${ABLATION_CONFIGS[$SINGLE_CONFIG]}" ]]; then
        echo "Error: Unknown config '$SINGLE_CONFIG'"
        echo "Available: ${CONFIG_ORDER[*]}"
        exit 1
    fi
    CONFIG_ORDER=("$SINGLE_CONFIG")
fi

# ==============================================================================
# Print experiment plan
# ==============================================================================
echo ""
echo "========================================================================"
echo "  GlowGS Ablation Study - Bicycle Scene"
echo "========================================================================"
echo ""
echo "  Configurations to run:"
for cfg in "${CONFIG_ORDER[@]}"; do
    IFS='|' read -r out_dir vm ma el <<< "${ABLATION_CONFIGS[$cfg]}"
    printf "    %-12s → output/%-20s [vm=%s, mass=%s, edge=%s]\n" \
           "$cfg" "$out_dir" "$vm" "$ma" "$el"
done
echo ""
echo "  Iterations: $ITERATIONS"
echo "  Data: $DATA_PATH"
echo ""
if [ "$DRY_RUN" = true ]; then
    echo "  🔍 DRY RUN MODE - Commands will be printed but not executed"
fi
echo "========================================================================"
echo ""

# ==============================================================================
# Run training
# ==============================================================================
run_training() {
    local config_name="$1"
    local config_str="${ABLATION_CONFIGS[$config_name]}"
    
    IFS='|' read -r out_dir vm ma el <<< "$config_str"
    
    local model_path="output/$out_dir"
    
    echo ""
    echo "========================================================================"
    echo "  Training: $config_name → $model_path"
    echo "========================================================================"
    
    local cmd="python train.py \
        -s $DATA_PATH \
        -i images_4 \
        -m $model_path \
        --enable_vm $vm \
        --enable_mass_aware $ma \
        --enable_edge_loss $el \
        --iterations $ITERATIONS \
        --save_iterations $SAVE_ITERATIONS \
        --test_iterations $TEST_ITERATIONS \
        --eval"
    
    echo "Command:"
    echo "  $cmd"
    echo ""
    
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Skipping execution."
    else
        eval "$cmd"
        echo ""
        echo "✅ Completed: $config_name"
    fi
}

# Run all configs
for cfg in "${CONFIG_ORDER[@]}"; do
    run_training "$cfg"
done

echo ""
echo "========================================================================"
echo "  All training completed!"
echo "========================================================================"
echo ""
echo "  Next steps:"
echo "    1. Render NPZ files for comparison plots:"
echo "       python paper_experiments/06_ablation_study/render_ablation_views.py"
echo ""
echo "    2. Generate ablation figures:"
echo "       python paper_experiments/06_ablation_study/01_ply_tomography/plot_slice.py"
echo "       python paper_experiments/06_ablation_study/03_scale_histogram_vm/plot_histogram.py"
echo "       python paper_experiments/06_ablation_study/03_scale_histogram_vm/plot_geo_analysis.py"
echo ""
echo "========================================================================"
