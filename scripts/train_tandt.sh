#!/bin/bash
# Tanks and Temples dataset training script with GeometryAppearanceEncoder
# Feature role split is now always-on (geometry/appearance disentanglement via FiLM)

# train
python train.py \
    -s data/tandt/train \
    -m output/train \
    --eval \
    --pcd_path output/train/nerfacto/run/point_cloud.ply \

# truck
python train.py \
    -s data/tandt/truck \
    -m output/truck \
    --eval \
    --pcd_path output/truck/nerfacto/run/point_cloud.ply \