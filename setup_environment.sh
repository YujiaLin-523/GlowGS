#!/bin/bash

# LocoGS One-Click Environment Setup Script
# For CUDA 11.7 and RTX A6000

set -e  # Exit immediately if a command exits with a non-zero status

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if in conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo_error "Please activate conda environment first: conda activate locogs"
    exit 1
fi

echo_info "Current conda environment: $CONDA_DEFAULT_ENV"

# Set CUDA environment variables
echo_info "Setting CUDA environment variables..."

# Try to find CUDA installation
if [ -d "/usr/local/cuda-11.7" ]; then
    export CUDA_HOME=/usr/local/cuda-11.7
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
else
    echo_error "CUDA installation not found in /usr/local/cuda or /usr/local/cuda-11.7"
    exit 1
fi

echo_info "Using CUDA_HOME: $CUDA_HOME"

export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TCNN_CUDA_ARCHITECTURES=86  # Compute capability for RTX A6000

# Verify CUDA installation
echo_info "Verifying CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    echo_error "nvcc not found, please ensure CUDA 11.7 is installed correctly"
    exit 1
fi

echo_info "CUDA verification:"

nvcc --version
nvidia-smi

# Step 1: Install PyTorch (CUDA 11.7)
echo_info "Step 1/7: Installing PyTorch with CUDA 11.7..."

# Uninstall any existing PyTorch installation
echo_info "Removing any existing PyTorch installation..."
pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

# Install PyTorch with CUDA 11.7 support
echo_info "Installing PyTorch 2.0.1 with CUDA 11.7..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

# Verify PyTorch CUDA support
echo_info "Verifying PyTorch CUDA support..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU count: {torch.cuda.device_count()}')
else:
    print('WARNING: CUDA is not available!')
    print('Installed PyTorch package info:')
    import sys
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'show', 'torch'])
    exit(1)
"

if [ $? -ne 0 ]; then
    echo_error "PyTorch CUDA support verification failed"
    echo_error "Possible solutions:"
    echo_error "  1. Check if CUDA 11.7 is properly installed"
    echo_error "  2. Verify LD_LIBRARY_PATH includes CUDA libraries"
    echo_error "  3. Try: export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:\$LD_LIBRARY_PATH"
    exit 1
fi

# Step 2: Install basic Python dependencies
echo_info "Step 2/7: Installing basic Python dependencies..."
pip install numpy plyfile==0.8.1 tqdm imageio opencv-python imageio-ffmpeg scipy dearpygui lpips

# Step 3: Install diff-gaussian-rasterization
echo_info "Step 3/7: Installing diff-gaussian-rasterization..."
if [ -d "submodules/diff-gaussian-rasterization" ]; then
    pip install --no-build-isolation submodules/diff-gaussian-rasterization
else
    echo_error "submodules/diff-gaussian-rasterization directory not found"
    exit 1
fi

# Step 4: Install diff-gaussian-rasterization-sh
echo_info "Step 4/7: Installing diff-gaussian-rasterization-sh..."
if [ -d "submodules/diff-gaussian-rasterization-sh" ]; then
    pip install --no-build-isolation submodules/diff-gaussian-rasterization-sh
else
    echo_error "submodules/diff-gaussian-rasterization-sh directory not found"
    exit 1
fi

# Step 5: Install simple-knn
echo_info "Step 5/7: Installing simple-knn..."
if [ -d "submodules/simple-knn" ]; then
    pip install --no-build-isolation submodules/simple-knn
else
    echo_error "submodules/simple-knn directory not found"
    exit 1
fi

# Step 6: Install tiny-cuda-nn (without build isolation)
echo_info "Step 6/7: Installing tiny-cuda-nn..."
echo_info "This may take several minutes, please be patient..."
pip install --no-build-isolation git+https://github.com/NVlabs/tiny-cuda-nn.git/#subdirectory=bindings/torch

# Verify tiny-cuda-nn installation
echo_info "Verifying tiny-cuda-nn installation..."
python -c "
import tinycudann as tcnn
print(f'tiny-cuda-nn installed successfully!')
print(f'tcnn version: {tcnn.__version__}')
"

if [ $? -ne 0 ]; then
    echo_error "tiny-cuda-nn installation verification failed"
    exit 1
fi

# Step 7: Install pymeshlab (for nerfstudio point cloud export)
echo_info "Step 7/7: Installing pymeshlab (via conda-forge)..."
conda install -c conda-forge pymeshlab -y

# Verify pymeshlab installation
echo_info "Verifying pymeshlab installation..."
python -c "
import pymeshlab
print('pymeshlab installed successfully!')
"

if [ $? -ne 0 ]; then
    echo_warn "pymeshlab verification failed, trying pip installation..."
    pip install pymeshlab
fi

# Final verification
echo_info "========================================="
echo_info "Running final verification..."
echo_info "========================================="

python -c "
import sys
import torch
import numpy
import cv2
import tinycudann as tcnn
import pymeshlab

print('✓ Python version:', sys.version)
print('✓ PyTorch version:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
print('✓ NumPy version:', numpy.__version__)
print('✓ OpenCV version:', cv2.__version__)
print('✓ tiny-cuda-nn version:', tcnn.__version__)
print('✓ pymeshlab imported successfully')

# Check all required packages
required_packages = [
    'plyfile', 'tqdm', 'imageio', 'scipy', 'dearpygui', 'lpips'
]

for pkg in required_packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg} imported successfully')
    except ImportError as e:
        print(f'✗ {pkg} import failed: {e}')
        sys.exit(1)

print('\\n' + '='*50)
print('🎉 All dependencies installed successfully! Environment setup complete!')
print('='*50)
"

if [ $? -eq 0 ]; then
    echo_info ""
    echo_info "========================================="
    echo_info "🎉 Environment setup completed successfully!"
    echo_info "========================================="
    echo_info ""
    echo_info "Next steps:"
    echo_info "  1. Run training script: ./train_360_v2.sh"
    echo_info "  2. Run preprocessing script: ./preprocessing_360_v2.sh"
    echo_info "  3. Run evaluation script: ./evaluate_360_v2.sh"
    echo_info ""
else
    echo_error "Environment setup failed, please check the error messages above"
    exit 1
fi
