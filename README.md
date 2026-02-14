<div align="center">

# GlowGS: Efficient 3D Gaussian Splatting with <br> Hybrid Encoding and Adaptive Densification

</div>

## Overview

GlowGS introduces three key innovations for efficient and high-quality 3D Gaussian Splatting:

1. **🔀 Hybrid Hash-VM Encoder**: Combines discrete hash grids with continuous tri-plane VM decomposition for efficient geometry/appearance encoding
2. **🎯 Unified Edge-Aware Loss**: Adaptive gradient-based loss that preserves high-frequency details while regularizing flat regions
3. **⚡ Feature-Weighted Densification**: Detail-aware Gaussian allocation that prioritizes high-importance regions

This codebase provides a clean, unified interface for systematic ablation studies of these innovations.

## Installation

This project is built upon the following environment:
- Python 3.10
- CUDA 11.7
- PyTorch 2.0.1

### Conda Environment
Run the following command to build the environment.
```bash
# clone this repo
git clone --recursive https://github.com/YujiaLin-523/GlowGS.git
cd GlowGS

# create a conda environment
conda create -n glowgs python=3.10
conda activate glowgs

# install pytorch
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

# install nerfstudio
cd submodules/nerfstudio
pip install --upgrade pip setuptools
pip install -e .
cd ../../

# install submodules
pip install submodules/diff-gaussian-rasterization --no-build-isolation
pip install submodules/diff-gaussian-rasterization-sh --no-build-isolation
pip install submodules/simple-knn --no-build-isolation
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-build-isolation

# install packages
pip install -r requirements.txt --no-build-isolation
```

**Note**: If you encounter issues with `tiny-cuda-nn`, please refer to their [official installation guide](https://github.com/NVlabs/tiny-cuda-nn).

### G-PCC (Geometry-based Point Cloud Compression)

GlowGS uses [TMC13](https://github.com/MPEGGroup/mpeg-pcc-tmc13) for efficient model storage and compression. Build it with:

```bash
# build mpeg-pcc-tmc13
cd submodules/mpeg-pcc-tmc13
mkdir build
cd build
cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 .. 
make -j$(nproc)
cd ../../../
```

This step is **required** for model saving/loading functionality.

## Data

We support three datasets for evaluation. Please follow [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) to set up the data.

- [MipNeRF-360](https://jonbarron.info/mipnerf360/) 
- [Tanks & Temples + Deep Blending](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip)

```shell
├── data
    ├── 360_v2
      ├── bicycle
      ├── bonsai
      ├── ...
    ├── tandt
      ├── ...
    ├── db
      ├── ...
```

## Quick Start

### Basic Training

Train GlowGS with default settings (full method):
```bash
python train.py -s data/360_v2/bicycle -m output/bicycle --eval
```

### Training with Custom Point Cloud

If you have a dense point cloud (e.g., from NeRF):
```bash
python train.py -s data/360_v2/bicycle -m output/bicycle --eval \
                --pcd_path path/to/point_cloud.ply
``` 


## Ablation Studies

GlowGS provides a unified interface for systematic ablation studies. See [ABLATION_STUDY_GUIDE.md](ABLATION_STUDY_GUIDE.md) for detailed documentation.

### Ablation Parameters

Control three key innovations via CLI switches (all default **True**):

| Parameter | Disable | Default | Description |
|-----------|---------|---------|-------------|
| `--enable_vm` | `--enable_vm False` | True | VM tri-plane geometry branch in encoder |
| `--enable_mass_aware` | `--enable_mass_aware False` | True | Mass-aware densification (prune + budget) |
| `--enable_edge_loss` | `--enable_edge_loss False` | True | Edge-aware gradient supervision |

### Examples

**Disable individual components:**
```bash
# Without VM (hash-only encoding)
python train.py -s data/360_v2/bicycle -m output/bicycle_wo_vm \
                --enable_vm False

# Without mass-aware densification
python train.py -s data/360_v2/bicycle -m output/bicycle_wo_mass \
                --enable_mass_aware False

# Without edge loss
python train.py -s data/360_v2/bicycle -m output/bicycle_wo_edge \
                --enable_edge_loss False
```

**Full ablation (all innovations disabled = baseline):**
```bash
python train.py -s data/360_v2/bicycle -m output/bicycle_baseline \
                --enable_vm False \
                --enable_mass_aware False \
                --enable_edge_loss False
```

**Full GlowGS (default, all enabled):**
```bash
python train.py -s data/360_v2/bicycle -m output/bicycle --eval
```

### Configuration Auto-Save

Every training run automatically saves `ablation_config.yaml` in the output directory for full reproducibility:

```yaml
ablation_settings:
  enable_vm: true
  enable_mass_aware: true
  enable_edge_loss: true

training_hyperparameters:
  iterations: 30000
  lambda_grad: 0.05
  max_gaussians: 6000000
  ...
```

### Training Summary

At training start, you'll see:
```
======================================================================================
  ABLATION STUDY CONFIGURATION
--------------------------------------------------------------------------------------
  Encoder Variant      │  hybrid
  Edge Loss Mode       │  sobel_weighted (λ=0.050)
  Densification Mode   │  feature_weighted
  Max Gaussians        │  6,000,000
  Iterations           │  30,000
======================================================================================
```

## Advanced Usage

### Preprocessing (Optional Dense Initialization)

If you want to use a dense point cloud from NeRF instead of COLMAP:

``` shell
# Process data
ns-process-data images --data data/${DATASET}/${SCENE} \
                       --output-dir data/${DATASET}/${SCENE} \
                       --skip-colmap --skip-image-processing

# Train nerfacto
ns-train nerfacto --data data/${DATASET}/${SCENE} \
                  --output-dir output \
                  --timestamp run \
                  --vis tensorboard \
                  --machine.seed 0  \
                  --pipeline.model.camera-optimizer.mode off

# Export pointcloud
ns-export pointcloud --load-config output/${SCENE}/nerfacto/run/config.yml \
                     --output-dir output/${SCENE}/nerfacto/run \
                     --remove-outliers True \
                     --num-points ${NUM_PTS} \
                     --normal-method open3d \
                     --save-world-frame True
```

The point cloud will be saved as `output/${SCENE}/nerfacto/run/point_cloud.ply`.

### Training Parameters

Key hyperparameters for GlowGS:

```bash
python train.py -s <scene_path> -m <output_path> \
                --eval \                           # Enable test set evaluation
                --pcd_path <path_to_pcd> \        # Optional: custom point cloud
                --lambda_grad 0.05 \               # Edge loss weight
                --lambda_mask 0.01 \               # Mask regularization weight
                --hash_size 19 \                   # Hash grid size
                --max_gaussians 6000000 \          # Maximum Gaussian count
                --iterations 30000                 # Training iterations
```

**Parameter Details:**
- `pcd_path`: Path to initial point cloud (uses COLMAP point cloud by default)
- `lambda_grad`: Weight for edge-aware loss (default: `0.05`)
- `lambda_mask`: Weight for mask regularization (default: `0.01`)
- `hash_size`: Hash grid resolution, controls memory usage (default: `19`)
- `max_gaussians`: Hard capacity limit for Gaussians (default: `6,000,000`)
- `geo_resolution`: VM tri-plane resolution (default: `48`)
- `geo_rank`: VM rank for tensor decomposition (default: `6`)

### Evaluation

Render and evaluate a trained model:
```bash
# Render test views
python render.py -m output/bicycle

# Compute metrics (PSNR, SSIM, LPIPS)
python metrics.py -m output/bicycle
```

Results will be saved in `output/bicycle/test/ours_30000/`.


## Batch Processing Scripts

For convenience, we provide scripts for batch evaluation:

```bash
# Train all scenes in Mip-NeRF 360
bash scripts/train_360_v2.sh

# Train Tanks & Temples scenes
bash scripts/train_tandt.sh

# Train Deep Blending scenes
bash scripts/train_db.sh

# Evaluate all trained models
bash scripts/evaluate_360_v2.sh
bash scripts/evaluate_tandt.sh
bash scripts/evaluate_db.sh
```

## Project Structure

```
GlowGS/
├── encoders/
│   ├── factory.py              # Unified encoder creation (ablation support)
│   ├── vm_encoder.py           # VM tri-plane encoder (TensoRF-style)
│   └── hybrid_encoder.py       # Hybrid hash-VM encoder + FiLM modulation
├── scene/
│   └── gaussian_model.py       # Core Gaussian model with densification
├── utils/
│   ├── loss_utils.py           # Loss functions (edge-aware loss)
│   └── ...
├── train.py                    # Main training script
├── render.py                   # Rendering script
├── metrics.py                  # Evaluation metrics
└── ABLATION_STUDY_GUIDE.md    # Detailed ablation guide
```

## Key Features

✅ **Unified Ablation Interface**: Clean CLI parameters for systematic experiments  
✅ **Auto-Config Saving**: Every run saves `ablation_config.yaml` for reproducibility  
✅ **Modular Design**: Factory pattern for easy variant switching  
✅ **Backward Compatible**: Default parameters match paper baseline  
✅ **GPU Memory Efficient**: Supports both 48GB and 24GB GPUs  
✅ **Training Monitoring**: TensorBoard integration + detailed console logs  

## Tips

- **Memory**: If OOM on 24GB GPU, reduce `--max_gaussians` to `3000000` or `--hash_size` to `18`
- **Speed**: Enable AMP with `--use_amp` for faster training (experimental)
- **Quality**: For better quality, increase `--geo_resolution` to `64` (more memory)
- **Debugging**: Use `--debug_from 0` to enable detailed logging from iteration 0

## Troubleshooting

**Issue**: "CUDA out of memory"  
**Solution**: Reduce `--max_gaussians` or `--hash_size`, or enable `--use_amp`

**Issue**: "Module 'yaml' not found"  
**Solution**: `pip install pyyaml`

**Issue**: "tiny-cuda-nn installation failed"  
**Solution**: Follow [tiny-cuda-nn installation guide](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension)

**Issue**: Different results with same config  
**Solution**: Set random seed with environment variable: `PYTHONHASHSEED=0 python train.py ...`

## Acknowledgement

This codebase is built upon:
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) - Original 3DGS implementation
- [LocoGS](https://github.com/seungjooshin/LocoGS) - Locality-aware compression baseline
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) - Efficient hash grid encoding
- [TensoRF](https://github.com/apchenstu/TensoRF) - VM decomposition inspiration

Thanks to the authors for their excellent work!
