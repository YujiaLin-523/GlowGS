#!/usr/bin/env bash

# ==============================================================================
# GlowGS Ablation Experiment - Single Scene Test
# ==============================================================================
# Quick test of ablation variants on a single scene
# Usage: 
#   bash scripts/ablation_single_scene.sh [scene] [variant]
# Examples:
#   bash scripts/ablation_single_scene.sh data/360_v2/bicycle V0
#   bash scripts/ablation_single_scene.sh  # Uses defaults
# ==============================================================================

set -euo pipefail

SCENE_PATH="${1:-data/360_v2/bicycle}"
VARIANT="${2:-V0}"
OUTPUT_BASE="output/ablation_test"
LOG_DIR="logs/ablation_test"
mkdir -p "$LOG_DIR" "$OUTPUT_BASE"

OUTPUT_DIR="$OUTPUT_BASE/$VARIANT"
LOG_FILE="$LOG_DIR/${VARIANT}.log"
RENDER_LOG="$LOG_DIR/${VARIANT}_render.log"
EVAL_LOG="$LOG_DIR/${VARIANT}_eval.log"
ITERATIONS=7000  # Quick test with 7k iterations

# Configure ablation switches based on variant
case "$VARIANT" in
    V0)
        USE_HYBRID_ENCODER=False
        USE_EDGE_LOSS=False
        USE_FEATURE_DENSIFY=False
        DESC="3DGS Baseline (all OFF)"
        ;;
    V1)
        USE_HYBRID_ENCODER=True
        USE_EDGE_LOSS=False
        USE_FEATURE_DENSIFY=False
        DESC="Hybrid Encoder only"
        ;;
    V2)
        USE_HYBRID_ENCODER=True
        USE_EDGE_LOSS=True
        USE_FEATURE_DENSIFY=False
        DESC="Hybrid Encoder + Edge Loss"
        ;;
    V3)
        USE_HYBRID_ENCODER=True
        USE_EDGE_LOSS=True
        USE_FEATURE_DENSIFY=True
        DESC="Full GlowGS (all ON)"
        ;;
    *)
        echo "Unknown variant: $VARIANT"
        echo "Supported: V0, V1, V2, V3"
        exit 1
        ;;
esac

echo "========================================================================"
echo "  GlowGS Ablation Test - $VARIANT"
echo "========================================================================"
echo "  Description: $DESC"
echo "  Scene: $SCENE_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Iterations: $ITERATIONS"
echo "========================================================================"

# Clean previous output
rm -rf "$OUTPUT_DIR"

# Train
echo "Training..."
python train.py \
    -s "$SCENE_PATH" \
    -m "$OUTPUT_DIR" \
    --iterations $ITERATIONS \
    --eval \
    --use_hybrid_encoder $USE_HYBRID_ENCODER \
    --use_edge_loss $USE_EDGE_LOSS \
    --use_feature_densify $USE_FEATURE_DENSIFY \
    2>&1 | tee "$LOG_FILE"

# Render
echo ""
echo "Rendering test images..."
python render.py -m "$OUTPUT_DIR" 2>&1 | tee "$RENDER_LOG"

# Evaluate
echo ""
echo "Computing metrics..."
python metrics.py -m "$OUTPUT_DIR" 2>&1 | tee "$EVAL_LOG"

echo ""
echo "========================================================================"
echo "  ✓ Test completed!"
echo "========================================================================"
echo "  Logs saved to:"
echo "    Training: $LOG_FILE"
echo "    Render:   $RENDER_LOG"
echo "    Metrics:  $EVAL_LOG"
echo "========================================================================"
