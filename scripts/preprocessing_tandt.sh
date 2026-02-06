#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/home/ubuntu/anaconda3/envs/glowgs/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1

# train
# processing data
ns-process-data images --data data/tandt/train \
                       --output-dir data/tandt/train \
                       --skip-colmap --skip-image-processing

# fix trasform.json
python fix_tandt_train.py

# train nerfacto
ns-train nerfacto --data data/tandt/train \
                  --output-dir output \
                  --timestamp run \
                  --vis tensorboard \
                  --machine.seed 0  \
                  --pipeline.model.camera-optimizer.mode off

# export pointcloud
ns-export pointcloud --load-config output/train/nerfacto/run/config.yml \
                     --output-dir output/train/nerfacto/run \
                     --remove-outliers True \
                     --num-points 1000000 \
                     --normal-method open3d \
                     --save-world-frame True

# truck
# processing data
ns-process-data images --data data/tandt/truck \
                       --output-dir data/tandt/truck \
                       --skip-colmap --skip-image-processing

# fix trasform.json
python fix_tandt_truck.py

# train nerfacto
ns-train nerfacto --data data/tandt/truck \
                  --output-dir output \
                  --timestamp run \
                  --vis tensorboard \
                  --machine.seed 0  \
                  --pipeline.model.camera-optimizer.mode off

# export pointcloud
ns-export pointcloud --load-config output/truck/nerfacto/run/config.yml \
                     --output-dir output/truck/nerfacto/run \
                     --remove-outliers True \
                     --num-points 1000000 \
                     --normal-method open3d \
                     --save-world-frame True

