#!/usr/bin/env bash

# Unified preprocessing wrapper for datasets
# Usage:
#   bash scripts/preprocess.sh <dataset>
# Supported dataset values: 360_v2, db, tandt

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <dataset>"
  echo "Supported: 360_v2, db, tandt"
  exit 1
fi

DATASET="$1"
case "$DATASET" in
  360_v2)
    bash scripts/preprocessing_360_v2.sh;;
  db)
    bash scripts/preprocessing_db.sh;;
  tandt)
    bash scripts/preprocessing_tandt.sh;;
  *)
    echo "Unknown dataset: $DATASET"; exit 1;;
esac
