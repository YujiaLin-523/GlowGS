#!/usr/bin/env bash
#
# Test script: Run a single ablation variant on one scene to verify the setup.
# This helps debug the full ablation pipeline before running all experiments.
#
# Usage:
#   bash scripts/test_single_ablation.sh

set -e  # Exit on error

# Configuration
SCENE_PATH="data/360_v2/bicycle"
OUTPUT_BASE="output/ablation_test"
LOGS_DIR="logs/ablation_test"

# Create directories
mkdir -p "$LOGS_DIR"
mkdir -p "$OUTPUT_BASE"

echo "=========================================="
echo "Testing GlowGS Ablation Setup"
echo "=========================================="
echo "Scene: $SCENE_PATH"
echo "Output: $OUTPUT_BASE"
echo "Logs: $LOGS_DIR"
echo ""

# Test V0: baseline 3DGS-mode (all innovations OFF)
echo "Running V0: 3DGS baseline mode..."
VARIANT="V0"
OUTPUT_DIR="${OUTPUT_BASE}/${VARIANT}"
LOG_FILE="${LOGS_DIR}/${VARIANT}.log"

python train.py \
    -s "$SCENE_PATH" \
    -m "$OUTPUT_DIR" \
    --eval \
    --iterations 7000 \
    --use_hybrid_encoder False \
    --use_edge_loss False \
    --use_feature_densify False \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Training completed. Check log at:"
echo "  $LOG_FILE"
echo ""
echo "Now running evaluation to get final metrics..."
echo "=========================================="

# Run evaluation
EVAL_LOG="${LOGS_DIR}/${VARIANT}_eval.log"
python metrics.py -m "$OUTPUT_DIR" 2>&1 | tee "$EVAL_LOG"

echo ""
echo "=========================================="
echo "Test completed!"
echo "Check results in:"
echo "  - Training log: $LOG_FILE"
echo "  - Evaluation log: $EVAL_LOG"
echo "  - Model output: $OUTPUT_DIR"
echo "=========================================="
