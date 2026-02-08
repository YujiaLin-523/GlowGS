#!/bin/bash
# Tanks and Temples dataset training script with GeometryAppearanceEncoder
# Feature role split is now always-on (geometry/appearance disentanglement via FiLM)

# train
python train.py \
    -s data/tandt/train \
    -m output/train \
    --eval \

# truck
python train.py \
    -s data/tandt/truck \
    -m output/truck \
    --eval \