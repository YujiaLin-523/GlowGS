#!/usr/bin/env bash

# ==============================================================================
# Quick test script for ablation experiments
# ==============================================================================
# This script runs a single quick experiment to test the ablation framework
# Usage: bash scripts/test_ablation.sh
# ==============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="logs/ablation_test"
mkdir -p "$LOG_DIR"

# Test with a quick run (reduced iterations)
SCENE_PATH="data/360_v2/garden"
OUTPUT_DIR="output/test_ablation_V0"
LOG_FILE="$LOG_DIR/test_V0.log"
ITERATIONS=100  # Very short for testing

echo "========================================================================"
echo "  Testing Ablation Framework - V0 (3DGS Baseline)"
echo "========================================================================"
echo "  Scene: $SCENE_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Log: $LOG_FILE"
echo "  Iterations: $ITERATIONS"
echo "========================================================================"

# Clean previous test output
rm -rf "$OUTPUT_DIR"

#!/usr/bin/env bash

# Consolidated quick test for ablation experiments
# This script runs a short test training, render, and evaluation for a single variant
# Usage:
#   bash scripts/test_ablation.sh [scene] [variant]
# Defaults:
#   scene: data/360_v2/bicycle
#   variant: V0

set -euo pipefail

SCENE_PATH="${1:-data/360_v2/bicycle}"
VARIANT="${2:-V0}"
OUTPUT_BASE="output/ablation_test"
LOGS_DIR="logs/ablation_test"
mkdir -p "$LOGS_DIR" "$OUTPUT_BASE"

OUTPUT_DIR="$OUTPUT_BASE/$VARIANT"
LOG_FILE="$LOGS_DIR/${VARIANT}.log"
RENDER_LOG="$LOGS_DIR/${VARIANT}_render.log"
EVAL_LOG="$LOGS_DIR/${VARIANT}_eval.log"

# Default args per variant (only V0 provided as example)
if [ "$VARIANT" = "V0" ]; then
    USE_HYBRID_ENCODER=False
    USE_EDGE_LOSS=False
    USE_FEATURE_DENSIFY=False
else
    # For other variants, user should pass booleans manually via train.sh
    USE_HYBRID_ENCODER=True
    USE_EDGE_LOSS=True
    USE_FEATURE_DENSIFY=True
fi

echo "Running quick ablation test: $VARIANT on $SCENE_PATH"
rm -rf "$OUTPUT_DIR"

python train.py \
    -s "$SCENE_PATH" \
    -m "$OUTPUT_DIR" \
    --iterations 7000 \
    --eval \
    --use_hybrid_encoder $USE_HYBRID_ENCODER \
    --use_edge_loss $USE_EDGE_LOSS \
    --use_feature_densify $USE_FEATURE_DENSIFY \
    2>&1 | tee "$LOG_FILE"

# Render and evaluate
python render.py -m "$OUTPUT_DIR" 2>&1 | tee "$RENDER_LOG"
python metrics.py -m "$OUTPUT_DIR" 2>&1 | tee "$EVAL_LOG"

echo "Test finished. Logs: $LOG_FILE, $RENDER_LOG, $EVAL_LOG"
