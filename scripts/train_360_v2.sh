#!/bin/bash
# 360_v2 dataset training script with GeometryAppearanceEncoder
# Feature role split is now always-on (geometry/appearance disentanglement via FiLM)

# bicycle
python train.py \
    -s data/360_v2/bicycle \
    -m output/bicycle \
    --eval \

# bonsai
python train.py \
    -s data/360_v2/bonsai \
    -m output/bonsai \
    --eval \

# counter
python train.py \
    -s data/360_v2/counter \
    -m output/counter \
    --eval \

# garden
python train.py \
    -s data/360_v2/garden \
    -m output/garden \
    --eval \

# kitchen
python train.py \
    -s data/360_v2/kitchen \
    -m output/kitchen \
    --eval \

# room
python train.py \
    -s data/360_v2/room \
    -m output/room \
    --eval \

# stump
python train.py \
    -s data/360_v2/stump \
    -m output/stump \
    --eval \
