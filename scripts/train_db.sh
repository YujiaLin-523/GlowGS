#!/bin/bash
# Deep Blending dataset training script with GeometryAppearanceEncoder
# Uses --feature_role_split for geometry/appearance feature disentanglement

# drjohnson
python train.py \
    -s data/db/drjohnson \
    -m output/drjohnson \
    --eval \
    --pcd_path output/drjohnson/nerfacto/run/point_cloud.ply \
    --lambda_mask 0.004 \
    --feature_role_split

# playroom
python train.py \
    -s data/db/playroom \
    -m output/playroom \
    --eval \
    --pcd_path output/playroom/nerfacto/run/point_cloud.ply \
    --lambda_mask 0.004 \
    --feature_role_split
