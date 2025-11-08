#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/home/ubuntu/anaconda3/envs/locogs/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1

# bicycle
# processing data
ns-process-data images --data data/360_v2/bicycle \
                       --output-dir data/360_v2/bicycle \
                       --skip-colmap --skip-image-processing

# train nerfacto
ns-train nerfacto --data data/360_v2/bicycle \
                  --output-dir output \
                  --timestamp run \
                  --vis tensorboard \
                  --machine.seed 0  \
                  --pipeline.model.camera-optimizer.mode off

# export pointcloud
ns-export pointcloud --load-config output/bicycle/nerfacto/run/config.yml \
                     --output-dir output/bicycle/nerfacto/run \
                     --remove-outliers True \
                     --num-points 1000000 \
                     --normal-method open3d \
                     --save-world-frame True

# bonsai
# processing data
ns-process-data images --data data/360_v2/bonsai \
                       --output-dir data/360_v2/bonsai \
                       --skip-colmap --skip-image-processing

# train nerfacto
ns-train nerfacto --data data/360_v2/bonsai \
                  --output-dir output \
                  --timestamp run \
                  --vis tensorboard \
                  --machine.seed 0  \
                  --pipeline.model.camera-optimizer.mode off

# export pointcloud
ns-export pointcloud --load-config output/bonsai/nerfacto/run/config.yml \
                     --output-dir output/bonsai/nerfacto/run \
                     --remove-outliers True \
                     --num-points 1000000 \
                     --normal-method open3d \
                     --save-world-frame True

# counter
# processing data
ns-process-data images --data data/360_v2/counter \
                       --output-dir data/360_v2/counter \
                       --skip-colmap --skip-image-processing

# train nerfacto
ns-train nerfacto --data data/360_v2/counter \
                  --output-dir output \
                  --timestamp run \
                  --vis tensorboard \
                  --machine.seed 0  \
                  --pipeline.model.camera-optimizer.mode off

# export pointcloud
ns-export pointcloud --load-config output/counter/nerfacto/run/config.yml \
                     --output-dir output/counter/nerfacto/run \
                     --remove-outliers True \
                     --num-points 1000000 \
                     --normal-method open3d \
                     --save-world-frame True
                    
# garden
# processing data
ns-process-data images --data data/360_v2/garden \
                       --output-dir data/360_v2/garden \
                       --skip-colmap --skip-image-processing
# train nerfacto
ns-train nerfacto --data data/360_v2/garden \
                  --output-dir output \
                  --timestamp run \
                  --vis tensorboard \
                  --machine.seed 0  \
                  --pipeline.model.camera-optimizer.mode off
# export pointcloud
ns-export pointcloud --load-config output/garden/nerfacto/run/config.yml \
                     --output-dir output/garden/nerfacto/run \
                     --remove-outliers True \
                     --num-points 1000000 \
                     --normal-method open3d \
                     --save-world-frame True

# kitchen
# processing data
ns-process-data images --data data/360_v2/kitchen \
                       --output-dir data/360_v2/kitchen \
                       --skip-colmap --skip-image-processing
# train nerfacto
ns-train nerfacto --data data/360_v2/kitchen \
                  --output-dir output \
                  --timestamp run \
                  --vis tensorboard \
                  --machine.seed 0  \
                  --pipeline.model.camera-optimizer.mode off
# export pointcloud
ns-export pointcloud --load-config output/kitchen/nerfacto/run/config.yml \
                     --output-dir output/kitchen/nerfacto/run \
                     --remove-outliers True \
                     --num-points 1000000 \
                     --normal-method open3d \
                     --save-world-frame True

# room
# processing data
ns-process-data images --data data/360_v2/room \
                       --output-dir data/360_v2/room \
                       --skip-colmap --skip-image-processing
# train nerfacto
ns-train nerfacto --data data/360_v2/room \
                  --output-dir output \
                  --timestamp run \
                  --vis tensorboard \
                  --machine.seed 0  \
                  --pipeline.model.camera-optimizer.mode off
# export pointcloud
ns-export pointcloud --load-config output/room/nerfacto/run/config.yml \
                     --output-dir output/room/nerfacto/run \
                     --remove-outliers True \
                     --num-points 1000000 \
                     --normal-method open3d \
                     --save-world-frame True

# stump
# processing data
ns-process-data images --data data/360_v2/stump \
                       --output-dir data/360_v2/stump \
                       --skip-colmap --skip-image-processing
# train nerfacto
ns-train nerfacto --data data/360_v2/stump \
                  --output-dir output \
                  --timestamp run \
                  --vis tensorboard \
                  --machine.seed 0  \
                  --pipeline.model.camera-optimizer.mode off
# export pointcloud
ns-export pointcloud --load-config output/stump/nerfacto/run/config.yml \
                     --output-dir output/stump/nerfacto/run \
                     --remove-outliers True \
                     --num-points 1000000 \
                     --normal-method open3d \
                     --save-world-frame True                     