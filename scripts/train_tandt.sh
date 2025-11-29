#!/bin/bash
# Tanks and Temples dataset training script with GeometryAppearanceEncoder
# Uses --feature_role_split for geometry/appearance feature disentanglement

# train
python train.py \
    -s data/tandt/train \
    -m output/train \
    --eval \
    --pcd_path output/train/nerfacto/run/point_cloud.ply \
    --lambda_mask 0.004 \
    --feature_role_split

# truck
python train.py \
    -s data/tandt/truck \
    -m output/truck \
    --eval \
    --pcd_path output/truck/nerfacto/run/point_cloud.ply \
    --lambda_mask 0.004 \
    --feature_role_split
