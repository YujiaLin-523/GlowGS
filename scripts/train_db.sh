#!/bin/bash
# Deep Blending dataset training script with GeometryAppearanceEncoder
# Uses --feature_role_split for geometry/appearance feature disentanglement

# drjohnson
python train.py \
    -s data/db/drjohnson \
    -m output/drjohnson \
    --eval --qat \
    --pcd_path output/drjohnson/nerfacto/run/point_cloud.ply \

# playroom
python train.py \
    -s data/db/playroom \
    -m output/playroom \
    --eval --qat \
    --pcd_path output/playroom/nerfacto/run/point_cloud.ply \
