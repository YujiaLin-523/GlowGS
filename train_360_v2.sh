#!/bin/bash
# 360_v2 dataset training script with Hybrid Encoder (VM + HashGrid)

# bicycle
python train.py \
    -s data/360_v2/bicycle \
    -m output/bicycle_hybrid \
    --eval \
    --pcd_path output/bicycle/nerfacto/run/point_cloud.ply \
    --lambda_mask 0.004

# bonsai
python train.py \
    -s data/360_v2/bonsai \
    -m output/bonsai_hybrid \
    --eval \
    --pcd_path output/bonsai/nerfacto/run/point_cloud.ply \
    --lambda_mask 0.004

# counter
python train.py \
    -s data/360_v2/counter \
    -m output/counter_hybrid \
    --eval \
    --pcd_path output/counter/nerfacto/run/point_cloud.ply \
    --lambda_mask 0.004

# garden
python train.py \
    -s data/360_v2/garden \
    -m output/garden_hybrid \
    --eval \
    --pcd_path output/garden/nerfacto/run/point_cloud.ply \
    --lambda_mask 0.004

# kitchen
python train.py \
    -s data/360_v2/kitchen \
    -m output/kitchen_hybrid \
    --eval \
    --pcd_path output/kitchen/nerfacto/run/point_cloud.ply \
    --lambda_mask 0.004

# room
python train.py \
    -s data/360_v2/room \
    -m output/room_hybrid \
    --eval \
    --pcd_path output/room/nerfacto/run/point_cloud.ply \
    --lambda_mask 0.004

# stump
python train.py \
    -s data/360_v2/stump \
    -m output/stump_hybrid \
    --eval \
    --pcd_path output/stump/nerfacto/run/point_cloud.ply \
    --lambda_mask 0.004
