#!/bin/bash
# db dataset training script with Hybrid Encoder (VM + HashGrid)

# drjohnson
python train.py \
    -s data/db/drjohnson \
    -m output/drjohnson_hybrid \
    --eval \
    --pcd_path output/drjohnson/nerfacto/run/point_cloud.ply \
    --lambda_mask 0.004

# playroom
python train.py \
    -s data/db/playroom \
    -m output/playroom_hybrid \
    --eval \
    --pcd_path output/playroom/nerfacto/run/point_cloud.ply \
    --lambda_mask 0.004
