#!/usr/bin/env bash

# ==============================================================================
# GlowGS ECCV Ablation Study - Mip-NeRF 360 Dataset
# ==============================================================================
# Runs the complete additive ablation study (A→B→C→D) across 7 Mip-NeRF 360 scenes.
#
# Ablation Models (Additive - "Trinity" of contributions):
#   A (Naive):    Hybrid Encoder (Concat mode), Standard Densification, No Edge Loss
#   B (+FiLM):    Hybrid Encoder (FiLM mode),   Standard Densification, No Edge Loss
#   C (+Mass):    Hybrid Encoder (FiLM mode),   Mass-Aware Densification, No Edge Loss
#   D (Full):     Hybrid Encoder (FiLM mode),   Mass-Aware Densification, Edge Loss
#
# Output:
#   - Models: output/ablation_study/<scene>_<model>/
#   - Logs:   output/ablation_study/logs/
#   - Stats:  output/ablation_study/<scene>_<model>/stats.json
#   - CSV:    output/ablation_study/ablation_results.csv
#
# Usage:
#   bash scripts/run_ablation_360.sh
#   bash scripts/run_ablation_360.sh --dry-run    # Print commands without executing
#   bash scripts/run_ablation_360.sh --scenes "bicycle garden"  # Run specific scenes
# ==============================================================================

set -e  # Exit on error

# Configuration
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_BASE="output/ablation_study"
LOG_DIR="$OUTPUT_BASE/logs"
RESULTS_CSV="$OUTPUT_BASE/ablation_results.csv"
DATA_ROOT="data/360_v2"

# Mip-NeRF 360 scenes (7 scenes total)
ALL_SCENES=(bicycle garden stump room counter kitchen bonsai)

# Training configuration
ITERATIONS=30000
TEST_ITERATIONS="7000 30000"
SAVE_ITERATIONS="30000"

# Parse command line arguments
DRY_RUN=false
FORCE=false
SCENES=("${ALL_SCENES[@]}")

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force|-f)
            FORCE=true
            shift
            ;;
        --scenes)
            IFS=' ' read -r -a SCENES <<< "$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ==============================================================================
# Safety: Clean up old experiments (with warning)
# ==============================================================================
if [ -d "$OUTPUT_BASE" ]; then
    if [ "$FORCE" = true ]; then
        echo "[INFO] Force mode: Removing old experiments..."
        rm -rf "$OUTPUT_BASE"
    else
        echo ""
        echo "========================================================================"
        echo "  ⚠️  WARNING: Existing ablation study found at $OUTPUT_BASE"
        echo "========================================================================"
        echo ""
        read -p "Delete and start fresh? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "[INFO] Removing old experiments..."
            rm -rf "$OUTPUT_BASE"
        else
            echo "[INFO] Keeping existing experiments. New runs will overwrite matching configs."
        fi
    fi
fi

# Create directories
mkdir -p "$OUTPUT_BASE"
mkdir -p "$LOG_DIR"

# ==============================================================================
# Define ablation configurations
# ==============================================================================
# Format: MODEL_NAME:feature_mod_type:densification_mode:use_edge_loss
declare -A CONFIGS
CONFIGS["A_Concat"]="concat:standard:False"
CONFIGS["B_FiLM"]="film:standard:False"
CONFIGS["C_MassAware"]="film:mass_aware:False"
CONFIGS["D_Full"]="film:mass_aware:True"

# Ordered list for iteration
CONFIG_ORDER=("A_Concat" "B_FiLM" "C_MassAware" "D_Full")

# ==============================================================================
# Print experiment plan
# ==============================================================================
echo ""
echo "========================================================================"
echo "  GlowGS ECCV Ablation Study - Mip-NeRF 360"
echo "========================================================================"
echo ""
echo "  Scenes (${#SCENES[@]}): ${SCENES[*]}"
echo ""
echo "  Ablation Models:"
echo "    A_Concat:    Hybrid (Concat), Standard Densify, No Edge Loss"
echo "    B_FiLM:      Hybrid (FiLM),   Standard Densify, No Edge Loss"
echo "    C_MassAware: Hybrid (FiLM),   Mass-Aware Densify, No Edge Loss"
echo "    D_Full:      Hybrid (FiLM),   Mass-Aware Densify, Edge Loss"
echo ""
echo "  Training Config:"
echo "    Iterations: $ITERATIONS"
echo "    Test iters: $TEST_ITERATIONS"
echo ""
echo "  Output: $OUTPUT_BASE"
echo "  Results CSV: $RESULTS_CSV"
echo ""
echo "  Total experiments: $(( ${#SCENES[@]} * ${#CONFIG_ORDER[@]} ))"
echo ""
if [ "$DRY_RUN" = true ]; then
    echo "  🔍 DRY RUN MODE - Commands will be printed but not executed"
    echo ""
fi
echo "========================================================================"
echo ""

# ==============================================================================
# Run experiments
# ==============================================================================
run_experiment() {
    local scene=$1
    local model=$2
    local config=$3
    
    # Parse config string
    IFS=':' read -r mod_type densify_mode edge_loss <<< "$config"
    
    local scene_path="$DATA_ROOT/$scene"
    local output_dir="$OUTPUT_BASE/${scene}_${model}"
    local log_file="$LOG_DIR/${scene}_${model}.log"
    
    # Check for pre-generated point cloud
    local pcd_arg=""
    local pcd_path="output/${scene}/nerfacto/run/point_cloud.ply"
    if [ -f "$pcd_path" ]; then
        pcd_arg="--pcd_path $pcd_path"
    fi
    
    echo ""
    echo "──────────────────────────────────────────────────────────────────────"
    echo "  Running: $scene / $model"
    echo "──────────────────────────────────────────────────────────────────────"
    echo "  Config: mod_type=$mod_type, densify=$densify_mode, edge_loss=$edge_loss"
    echo "  Output: $output_dir"
    echo "  Log: $log_file"
    echo ""
    
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] python -u train.py -s $scene_path -m $output_dir --iterations $ITERATIONS --eval --feature_mod_type $mod_type --densification_mode $densify_mode --use_edge_loss $edge_loss --test_iterations $TEST_ITERATIONS --save_iterations $SAVE_ITERATIONS $pcd_arg"
        echo ""
        return 0
    fi
    
    # Run training with unbuffered output, visible in terminal AND logged to file
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║  [TRAIN] Starting: $scene / $model"                                  ║
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    if ! python -u train.py \
        -s "$scene_path" \
        -m "$output_dir" \
        --iterations $ITERATIONS \
        --eval \
        --feature_mod_type $mod_type \
        --densification_mode $densify_mode \
        --use_edge_loss $edge_loss \
        --test_iterations $TEST_ITERATIONS \
        --save_iterations $SAVE_ITERATIONS \
        $pcd_arg 2>&1 | tee "$log_file"; then
        echo ""
        echo "[ERROR] Training failed for $scene / $model. Check $log_file"
        return 1
    fi
    
    echo ""
    echo "  ✓ Training finished: $scene / $model"
    echo ""
    
    # Run render.py
    echo "[RENDER] Rendering test views for $scene / $model ..."
    python -u render.py -m "$output_dir" 2>&1 | tee -a "$log_file"
    echo "  ✓ Rendering finished"
    
    # Run metrics.py  
    echo "[METRICS] Computing final metrics for $scene / $model ..."
    python -u metrics.py -m "$output_dir" 2>&1 | tee -a "$log_file"
    echo "  ✓ Metrics computed"
    
    echo ""
    echo "══════════════════════════════════════════════════════════════════════"
    echo "  ✅ COMPLETED: $scene / $model"
    echo "══════════════════════════════════════════════════════════════════════"
    echo ""
}

# Main experiment loop
experiment_count=0
total_experiments=$(( ${#SCENES[@]} * ${#CONFIG_ORDER[@]} ))

for scene in "${SCENES[@]}"; do
    for model in "${CONFIG_ORDER[@]}"; do
        experiment_count=$((experiment_count + 1))
        echo ""
        echo "========================================================================"
        echo "  Experiment $experiment_count / $total_experiments"
        echo "========================================================================"
        
        run_experiment "$scene" "$model" "${CONFIGS[$model]}"
    done
done

# ==============================================================================
# Aggregate results into CSV
# ==============================================================================
echo ""
echo "========================================================================"
echo "  Aggregating Results to CSV"
echo "========================================================================"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would aggregate stats.json files to $RESULTS_CSV"
else
    # Create CSV header
    echo "Model,Scene,PSNR,SSIM,LPIPS,Size(MB),FPS,Gaussians" > "$RESULTS_CSV"
    
    # Python one-liner to aggregate all stats.json files
    python3 << 'PYTHON_SCRIPT'
import os
import json
import glob

output_base = "output/ablation_study"
csv_path = os.path.join(output_base, "ablation_results.csv")

# Find all stats.json files
stats_files = glob.glob(os.path.join(output_base, "**/stats.json"), recursive=True)

rows = []
for stats_file in sorted(stats_files):
    try:
        with open(stats_file) as f:
            stats = json.load(f)
        
        row = {
            "model": stats.get("method", "Unknown"),
            "scene": stats.get("scene", "Unknown"),
            "psnr": stats.get("psnr", 0),
            "ssim": stats.get("ssim", 0),
            "lpips": stats.get("lpips", 0),
            "size_mb": stats.get("size_mb", 0),
            "fps": stats.get("fps", 0),
            "gaussians": stats.get("num_gaussians", 0),
        }
        rows.append(row)
        print(f"  Loaded: {stats_file}")
    except Exception as e:
        print(f"  [WARN] Failed to load {stats_file}: {e}")

# Write CSV (append to existing header)
with open(csv_path, 'a') as f:
    for row in rows:
        line = f"{row['model']},{row['scene']},{row['psnr']},{row['ssim']},{row['lpips']},{row['size_mb']},{row['fps']},{row['gaussians']}"
        f.write(line + "\n")

print(f"\n  Total: {len(rows)} experiments aggregated")
print(f"  CSV saved to: {csv_path}")
PYTHON_SCRIPT

    echo ""
    echo "========================================================================"
    echo "  Results Summary"
    echo "========================================================================"
    if [ -f "$RESULTS_CSV" ]; then
        echo ""
        cat "$RESULTS_CSV" | column -t -s ','
        echo ""
    fi
fi

echo ""
echo "========================================================================"
echo "  ✅ Ablation Study Complete!"
echo "========================================================================"
echo ""
echo "  Output directory: $OUTPUT_BASE"
echo "  Results CSV: $RESULTS_CSV"
echo "  Logs: $LOG_DIR"
echo ""
echo "  To view results:"
echo "    cat $RESULTS_CSV | column -t -s ','"
echo ""
echo "========================================================================"
