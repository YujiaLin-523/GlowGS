#!/usr/bin/env bash

# Train GlowGS on all datasets (mipnerf360, Tanks&Temples, Deep Blending)
# This script sequentially trains models on all scenes across all three datasets
# Usage: bash scripts/train_all_datasets.sh

set -euo pipefail

echo "============================================"
echo "Training GlowGS on ALL Datasets"
echo "============================================"
echo ""

# Function to train a scene
# Usage: train_scene <scene_path> <scene_name> [images_dir]
train_scene() {
    local scene_path=$1
    local scene_name=$2
    local images_dir="${3:-}"
    local output_dir="output/${scene_name}"
    
    # Build optional -i argument
    local images_arg=""
    if [[ -n "${images_dir}" ]]; then
        images_arg="-i ${images_dir}"
    fi

    echo "----------------------------------------"
    echo "Training: ${scene_name}"
    echo "Scene: ${scene_path}"
    echo "Output: ${output_dir}"
    [[ -n "${images_dir}" ]] && echo "Images: ${images_dir}"
    echo "----------------------------------------"
    
    python train.py \
        -s "${scene_path}" \
        -m "${output_dir}" \
        --eval \
        ${images_arg} \
    
    echo "✓ Completed: ${scene_name}"
    echo ""
}

# ============================================
# MipNeRF360 (360_v2) Dataset
# ============================================
echo "=== Training MipNeRF360 Dataset ==="
train_scene "data/360_v2/bicycle" "bicycle" "images_4"
train_scene "data/360_v2/bonsai" "bonsai" "images_2"
train_scene "data/360_v2/counter" "counter" "images_2"
train_scene "data/360_v2/garden" "garden" "images_4"
train_scene "data/360_v2/kitchen" "kitchen" "images_2"
train_scene "data/360_v2/room" "room" "images_2"
train_scene "data/360_v2/stump" "stump" "images_4"

# ============================================
# Tanks & Temples Dataset
# ============================================
echo "=== Training Tanks & Temples Dataset ==="
train_scene "data/tandt/train" "train"
train_scene "data/tandt/truck" "truck"

# ============================================
# Deep Blending Dataset
# ============================================
echo "=== Training Deep Blending Dataset ==="
train_scene "data/db/drjohnson" "drjohnson"
train_scene "data/db/playroom" "playroom"

echo "============================================"
echo "✓ All datasets training completed!"
echo "============================================"
echo "Total scenes trained: 11"
echo "  - MipNeRF360: 7 scenes"
echo "  - Tanks&Temples: 2 scenes"
echo "  - Deep Blending: 2 scenes"
echo ""
echo "Models saved in output/ directory"
echo "============================================"

find ./output -maxdepth 2 -type d -print0 | \
while IFS= read -r -d '' d; do t="$d/compression/iteration_30000"; if [ -d "$t" ]; then s=$(find "$t" -type f -printf '%s\n' | awk '{s+=$1}END{print s+0}'); printf "%s\t%.2f MB\n" "$t" "$(awk -v b="$s" 'BEGIN{printf "%.2f", b/1048576}')"; fi; done