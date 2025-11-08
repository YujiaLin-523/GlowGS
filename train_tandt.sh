#!/bin/bash
# tandt dataset training script with Hybrid Encoder (VM + HashGrid)

# train
python train.py \
    -s data/tandt/train \
    -m output/train_hybrid \
    --eval \
    --pcd_path output/train/nerfacto/run/point_cloud.ply \
    --lambda_mask 0.004

# truck
python train.py \
    -s data/tandt/truck \
    -m output/truck_hybrid \
    --eval \
    --pcd_path output/truck/nerfacto/run/point_cloud.ply \
    --lambda_mask 0.004
