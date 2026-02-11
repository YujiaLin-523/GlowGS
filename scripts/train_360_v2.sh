#!/bin/bash
# 360_v2 dataset training script with GeometryAppearanceEncoder
# Resolution follows LocoGS convention:
#   Indoor (bonsai, counter, kitchen, room): images_2
#   Outdoor (bicycle, garden, stump):        images_4

# ── Outdoor scenes (images_4) ──────────────────────────────────────

# bicycle
python train.py \
    -s data/360_v2/bicycle \
    -i images_4 \
    -m output/bicycle \
    --eval

# garden
python train.py \
    -s data/360_v2/garden \
    -i images_4 \
    -m output/garden \
    --eval

# stump
python train.py \
    -s data/360_v2/stump \
    -i images_4 \
    -m output/stump \
    --eval

# ── Indoor scenes (images_2) ──────────────────────────────────────

# bonsai
python train.py \
    -s data/360_v2/bonsai \
    -i images_2 \
    -m output/bonsai \
    --eval

# counter
python train.py \
    -s data/360_v2/counter \
    -i images_2 \
    -m output/counter \
    --eval

# kitchen
python train.py \
    -s data/360_v2/kitchen \
    -i images_2 \
    -m output/kitchen \
    --eval

# room
python train.py \
    -s data/360_v2/room \
    -i images_2 \
    -m output/room \
    --eval
