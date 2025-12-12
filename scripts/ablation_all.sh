#!/usr/bin/env bash

# ==============================================================================
# GlowGS Ablation Experiments - All Datasets
# ==============================================================================
# Runs complete ablation study across MipNeRF360, Tanks&Temples, and Deep Blending
#
# Usage:
#   bash scripts/ablation_all_datasets.sh
#
# Datasets & Scenes:
#   - MipNeRF360: garden, stump, bicycle
#   - Tanks&Temples: train, truck
#   - Deep Blending: drjohnson, playroom
#
# Ablation Variants (4 configurations per scene):
#   V0: 3DGS Baseline (all innovations OFF)
#   V1: + Hybrid Encoder only
#   V2: + Hybrid Encoder + Edge Loss
#   V3: Full GlowGS (all innovations ON)
#
# Output:
#   - Models: output/ablation/<scene>/<variant>/
#   - Logs: logs/ablation/<scene>/<variant>.log
#   - Summary: logs/ablation/ablation_results.tsv
# ==============================================================================

set -e  # Exit on error

# Configuration
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="logs/ablation"
RESULTS_FILE="$LOG_DIR/ablation_results.tsv"

# Create log directory
mkdir -p "$LOG_DIR"

# Dataset scene lists
MIP360_SCENES=(garden stump bicycle)
TANDT_SCENES=(train truck)
DB_SCENES=(drjohnson playroom)

# Training iterations (can be adjusted for faster testing)
ITERATIONS=30000

# Initialize results file with header if it doesn't exist
if [ ! -f "$RESULTS_FILE" ]; then
    echo -e "dataset\tscene\tvariant\tuse_hybrid_encoder\tuse_edge_loss\tuse_feature_densify\tPSNR\tSSIM\tLPIPS" > "$RESULTS_FILE"
    echo "[INFO] Created results file: $RESULTS_FILE"
fi

# ==============================================================================
# Helper function to run a single training experiment
# ==============================================================================
run_experiment() {
    local dataset=$1
    local scene=$2
    local variant=$3
    local use_hybrid=$4
    local use_edge=$5
    local use_densify=$6
    local scene_path=$7
    
    local output_dir="output/${dataset}_${scene}_${variant}"
    local log_file="$LOG_DIR/${dataset}_${scene}_${variant}.log"
    
    echo ""
    echo "========================================================================"
    echo "  Running: ${dataset}/${scene} - ${variant}"
    echo "  Output: ${output_dir}"
    echo "  Log: ${log_file}"
    echo "========================================================================"
    
    # Run training
    python train.py \
        -s "$scene_path" \
        -m "$output_dir" \
        --iterations "$ITERATIONS" \
        --eval \
        --use_hybrid_encoder="$use_hybrid" \
        --use_edge_loss="$use_edge" \
        --use_feature_densify="$use_densify" \
        > "$log_file"
    
    echo "[INFO] Training completed, running render and evaluation..."
    
    # Run render.py to generate test images
    python render.py -m "$output_dir" >> "$log_file"
    
    # Run metrics.py to get final evaluation
    python metrics.py -m "$output_dir" >> "$log_file"
    
    # Extract metrics from log file
    # Look for patterns from metrics.py output: "  PSNR : XX.XXXXXXX"
    local psnr=$(grep -oP "PSNR\s*:\s*\K[0-9]+\.[0-9]+" "$log_file" | tail -1 || echo "N/A")
    local ssim=$(grep -oP "SSIM\s*:\s*\K[0-9]+\.[0-9]+" "$log_file" | tail -1 || echo "N/A")
    local lpips=$(grep -oP "LPIPS:\s*\K[0-9]+\.[0-9]+" "$log_file" | tail -1 || echo "N/A")
    
    # If metrics not found, try alternative patterns from train.py output
    if [ "$psnr" == "N/A" ]; then
        psnr=$(grep -oP "Evaluating test.*PSNR\s+\K[0-9]+\.[0-9]+" "$log_file" | tail -1 || echo "N/A")
    fi
    
    # Append results to TSV file
    echo -e "${dataset}\t${scene}\t${variant}\t${use_hybrid}\t${use_edge}\t${use_densify}\t${psnr}\t${ssim}\t${lpips}" >> "$RESULTS_FILE"
    
    echo "[DONE] ${dataset}/${scene} - ${variant}"
    echo "  PSNR: ${psnr}  SSIM: ${ssim}  LPIPS: ${lpips}"
}

# ==============================================================================
# Main experiment loop
# ==============================================================================

echo ""
echo "========================================================================"
echo "  GlowGS Ablation Experiments"
echo "========================================================================"
echo "  Datasets:"
echo "    - mipnerf360: ${MIP360_SCENES[@]}"
echo "    - Tanks&Temples: ${TANDT_SCENES[@]}"
echo "    - DB: ${DB_SCENES[@]}"
echo ""
echo "  Variants:"
echo "    - V0: 3DGS baseline (all OFF)"
echo "    - V1: +Hybrid Encoder"
echo "    - V2: +Hybrid + Edge Loss"
echo "    - V3: Full GlowGS (all ON)"
echo ""
echo "  Iterations: ${ITERATIONS}"
echo "  Results: ${RESULTS_FILE}"
echo "========================================================================"
echo ""

# Run mipnerf360 scenes
for scene in "${MIP360_SCENES[@]}"; do
    scene_path="data/360_v2/${scene}"
    
    # V0: 3DGS baseline
    run_experiment "mip360" "$scene" "V0" "False" "False" "False" "$scene_path"
    
    # V1: +Hybrid Encoder
    run_experiment "mip360" "$scene" "V1" "True" "False" "False" "$scene_path"
    
    # V2: +Edge Loss
    run_experiment "mip360" "$scene" "V2" "True" "True" "False" "$scene_path"
    
    # V3: Full GlowGS
    run_experiment "mip360" "$scene" "V3" "True" "True" "True" "$scene_path"
done

# Run Tanks&Temples scenes
for scene in "${TANDT_SCENES[@]}"; do
    scene_path="data/tandt/${scene}"
    
    run_experiment "tandt" "$scene" "V0" "False" "False" "False" "$scene_path"
    run_experiment "tandt" "$scene" "V1" "True" "False" "False" "$scene_path"
    run_experiment "tandt" "$scene" "V2" "True" "True" "False" "$scene_path"
    run_experiment "tandt" "$scene" "V3" "True" "True" "True" "$scene_path"
done

# Run DB scenes
for scene in "${DB_SCENES[@]}"; do
    scene_path="data/db/${scene}"
    
    run_experiment "db" "$scene" "V0" "False" "False" "False" "$scene_path"
    run_experiment "db" "$scene" "V1" "True" "False" "False" "$scene_path"
    run_experiment "db" "$scene" "V2" "True" "True" "False" "$scene_path"
    run_experiment "db" "$scene" "V3" "True" "True" "True" "$scene_path"
done

# ==============================================================================
# Summary
# ==============================================================================

echo ""
echo "========================================================================"
echo "  All experiments completed!"
echo "========================================================================"
echo "  Results saved to: ${RESULTS_FILE}"
echo ""
echo "  To view results:"
echo "    cat ${RESULTS_FILE}"
echo ""
echo "  To analyze results with column alignment:"
echo "    column -t -s $'\\t' ${RESULTS_FILE}"
echo "========================================================================"
echo ""
