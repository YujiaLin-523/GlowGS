#!/bin/bash
# Deep Blending dataset training script with GeometryAppearanceEncoder
# Feature role split is now always-on (geometry/appearance disentanglement via FiLM)

# drjohnson
python train.py \
    -s data/db/drjohnson \
    -m output/drjohnson \
    --eval \

# playroom
python train.py \
    -s data/db/playroom \
    -m output/playroom \
    --eval \
