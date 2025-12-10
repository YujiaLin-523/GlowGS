#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/home/ubuntu/anaconda3/envs/locogs/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1

# drjohnson
# processing data
ns-process-data images --data ../data/db/drjohnson \
                       --output-dir ../data/db/drjohnson \
                       --skip-colmap --skip-image-processing

# train nerfacto
ns-train nerfacto --data ../data/db/drjohnson \
                  --output-dir ../output \
                  --timestamp run \
                  --vis tensorboard \
                  --machine.seed 0  \
                  --pipeline.model.camera-optimizer.mode off

# export pointcloud
ns-export pointcloud --load-config ../output/drjohnson/nerfacto/run/config.yml \
                     --output-dir ../output/drjohnson/nerfacto/run \
                     --remove-outliers True \
                     --num-points 1000000 \
                     --normal-method open3d \
                     --save-world-frame True

# playroom
# processing data
ns-process-data images --data ../data/db/playroom \
                       --output-dir ../data/db/playroom \
                       --skip-colmap --skip-image-processing

# train nerfacto
ns-train nerfacto --data ../data/db/playroom \
                  --output-dir ../output \
                  --timestamp run \
                  --vis tensorboard \
                  --machine.seed 0  \
                  --pipeline.model.camera-optimizer.mode off

# export pointcloud
ns-export pointcloud --load-config ../output/playroom/nerfacto/run/config.yml \
                     --output-dir ../output/playroom/nerfacto/run \
                     --remove-outliers True \
                     --num-points 1000000 \
                     --normal-method open3d \
                     --save-world-frame True

