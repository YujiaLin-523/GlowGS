#!/usr/bin/env bash

# Evaluate all trained GlowGS models across all datasets
# This script runs render.py and metrics.py for each model
# Usage: bash scripts/evaluate_all_datasets.sh

set -euo pipefail

echo "============================================"
echo "Evaluating GlowGS Models on ALL Datasets"
echo "============================================"
echo ""

# Function to evaluate a model
evaluate_model() {
    local model_dir=$1
    local scene_name=$2
    
    if [ ! -d "${model_dir}" ]; then
        echo "⚠ Skipping ${scene_name}: model directory not found"
        return
    fi
    
    echo "----------------------------------------"
    echo "Evaluating: ${scene_name}"
    echo "Model: ${model_dir}"
    echo "----------------------------------------"
    
    # Clean up old renders and tfevents
    echo "Cleaning up old renders and TensorBoard files..."
    rm -rf "${model_dir}/test"
    rm -rf "${model_dir}/train"
    find "${model_dir}" -name "events.out.tfevents.*" -type f -delete
    
    # Always render test images
    echo "Rendering test images..."
    python render.py -m "${model_dir}"
    
    # Compute metrics
    echo "Computing metrics..."
    python metrics.py -m "${model_dir}"
    
    echo "✓ Completed: ${scene_name}"
    echo ""
}

# ============================================
# MipNeRF360 (360_v2) Dataset
# ============================================
echo "=== Evaluating MipNeRF360 Dataset ==="
evaluate_model "output/bicycle" "bicycle"
evaluate_model "output/bonsai" "bonsai"
evaluate_model "output/counter" "counter"
evaluate_model "output/garden" "garden"
evaluate_model "output/kitchen" "kitchen"
evaluate_model "output/room" "room"
evaluate_model "output/stump" "stump"

# ============================================
# Tanks & Temples Dataset
# ============================================
echo "=== Evaluating Tanks & Temples Dataset ==="
evaluate_model "output/train" "train"
evaluate_model "output/truck" "truck"

# ============================================
# Deep Blending Dataset
# ============================================
echo "=== Evaluating Deep Blending Dataset ==="
evaluate_model "output/drjohnson" "drjohnson"
evaluate_model "output/playroom" "playroom"

echo "============================================"
echo "✓ All datasets evaluation completed!"
echo "============================================"
echo "Check metrics.json in each model's output directory"
echo "============================================"
