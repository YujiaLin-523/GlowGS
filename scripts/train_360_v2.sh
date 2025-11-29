#!/bin/bash
# 360_v2 dataset training script with GeometryAppearanceEncoder
# Uses --feature_role_split for geometry/appearance feature disentanglement

# bicycle
python train.py \
    -s data/360_v2/bicycle \
    -m output/bicycle \
    --eval \
    --pcd_path output/bicycle/nerfacto/run/point_cloud.ply \
    --lambda_mask 0.004 \
    --feature_role_split

# bonsai
python train.py \
    -s data/360_v2/bonsai \
    -m output/bonsai \
    --eval \
    --pcd_path output/bonsai/nerfacto/run/point_cloud.ply \
    --lambda_mask 0.004 \
    --feature_role_split

# counter
python train.py \
    -s data/360_v2/counter \
    -m output/counter \
    --eval \
    --pcd_path output/counter/nerfacto/run/point_cloud.ply \
    --lambda_mask 0.004 \
    --feature_role_split

# garden
python train.py \
    -s data/360_v2/garden \
    -m output/garden \
    --eval \
    --pcd_path output/garden/nerfacto/run/point_cloud.ply \
    --lambda_mask 0.004 \
    --feature_role_split

# kitchen
python train.py \
    -s data/360_v2/kitchen \
    -m output/kitchen \
    --eval \
    --pcd_path output/kitchen/nerfacto/run/point_cloud.ply \
    --lambda_mask 0.004 \
    --feature_role_split

# room
python train.py \
    -s data/360_v2/room \
    -m output/room \
    --eval \
    --pcd_path output/room/nerfacto/run/point_cloud.ply \
    --lambda_mask 0.004 \
    --feature_role_split

# stump
python train.py \
    -s data/360_v2/stump \
    -m output/stump \
    --eval \
    --pcd_path output/stump/nerfacto/run/point_cloud.ply \
    --lambda_mask 0.004 \
    --feature_role_split
