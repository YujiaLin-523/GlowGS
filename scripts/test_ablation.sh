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

# Run V0 configuration
python train.py \
    -s "$SCENE_PATH" \
    -m "$OUTPUT_DIR" \
    --iterations "$ITERATIONS" \
    --eval \
    --use_hybrid_encoder=False \
    --use_edge_loss=False \
    --use_feature_densify=False \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================================================"
echo "  Test completed!"
echo "========================================================================"
echo "  Check log file: $LOG_FILE"
echo "  Check output directory: $OUTPUT_DIR"
echo ""
echo "  Configuration printed in log should show:"
echo "    use_hybrid_encoder    : False"
echo "    use_edge_loss         : False"
echo "    use_feature_densify   : False"
echo "    Variant               : V0: 3DGS Baseline"
echo "========================================================================"
