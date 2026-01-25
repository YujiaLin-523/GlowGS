#!/bin/bash
# 360_v2 dataset training script with GeometryAppearanceEncoder
# Uses --feature_role_split for geometry/appearance feature disentanglement

# bicycle
python train.py \
    -s data/360_v2/bicycle \
    -m output/bicycle \
    --eval --qat \
    --pcd_path output/bicycle/nerfacto/run/point_cloud.ply

# bonsai
python train.py \
    -s data/360_v2/bonsai \
    -m output/bonsai \
    --eval --qat \
    --pcd_path output/bonsai/nerfacto/run/point_cloud.ply

# counter
python train.py \
    -s data/360_v2/counter \
    -m output/counter \
    --eval --qat \
    --pcd_path output/counter/nerfacto/run/point_cloud.ply

# garden
python train.py \
    -s data/360_v2/garden \
    -m output/garden \
    --eval --qat \
    --pcd_path output/garden/nerfacto/run/point_cloud.ply

# kitchen
python train.py \
    -s data/360_v2/kitchen \
    -m output/kitchen \
    --eval --qat \
    --pcd_path output/kitchen/nerfacto/run/point_cloud.ply

# room
python train.py \
    -s data/360_v2/room \
    -m output/room \
    --eval --qat \
    --pcd_path output/room/nerfacto/run/point_cloud.ply

# stump
python train.py \
    -s data/360_v2/stump \
    -m output/stump \
    --eval --qat \
    --pcd_path output/stump/nerfacto/run/point_cloud.ply
