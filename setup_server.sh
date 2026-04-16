#!/bin/bash
# =============================================================================
# Environment setup for diffusion_policy training on Ubuntu + RTX 4090
# Driver: 535.288.01  |  CUDA: 12.2
# Scope: training + remote evaluation (no simulation, no real-robot stack)
# =============================================================================
set -e

# ---------------------------------------------------------------------------
# 0. Install Miniconda (skip if already installed)
# ---------------------------------------------------------------------------
if ! command -v conda &>/dev/null; then
    echo "[1/6] Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    conda init bash
    echo "Miniconda installed. Re-run this script or open a new shell."
else
    echo "[1/6] Conda already installed, skipping."
    eval "$(conda shell.bash hook)"
fi

# ---------------------------------------------------------------------------
# 1. Clone repo via SSH (requires SSH key added to GitHub)
# ---------------------------------------------------------------------------
REPO_DIR="$HOME/DP-LW"
if [ ! -d "$REPO_DIR" ]; then
    echo "[2/6] Cloning repo..."
    git clone git@github.com:xiaozhisky1/DP-LW.git "$REPO_DIR"
else
    echo "[2/6] Repo already exists at $REPO_DIR, skipping clone."
fi
cd "$REPO_DIR"

# ---------------------------------------------------------------------------
# 2. Create conda environment
# ---------------------------------------------------------------------------
ENV_NAME="robodiff"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[3/6] Conda env '${ENV_NAME}' already exists, skipping creation."
else
    echo "[3/6] Creating conda env '${ENV_NAME}'..."
    conda create -y -n "$ENV_NAME" python=3.9
fi

conda activate "$ENV_NAME"

# ---------------------------------------------------------------------------
# 3. Install PyTorch with CUDA 12.1 (closest stable to CUDA 12.2)
# ---------------------------------------------------------------------------
echo "[4/6] Installing PyTorch 2.1 + CUDA 12.1..."
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# ---------------------------------------------------------------------------
# 4. Install core conda packages
# ---------------------------------------------------------------------------
echo "[5/6] Installing conda packages..."
conda install -y -c conda-forge \
    numpy=1.23.3 \
    scipy=1.9.1 \
    py-opencv=4.6.0 \
    zarr=2.12.0 \
    numcodecs=0.10.2 \
    h5py=3.7.0 \
    hydra-core=1.2.0 \
    einops=0.4.1 \
    tqdm=4.64.1 \
    dill=0.3.5.1 \
    scikit-image=0.19.3 \
    imageio=2.22.0 \
    imageio-ffmpeg=0.4.7 \
    termcolor=2.0.1 \
    tensorboard=2.10.1 \
    tensorboardx=2.5.1 \
    psutil=5.9.2 \
    click=8.0.4 \
    threadpoolctl=3.1.0 \
    shapely=1.8.4 \
    matplotlib=3.6.1 \
    av=10.0.0 \
    cmake=3.24.3

# ---------------------------------------------------------------------------
# 5. Install pip packages
# ---------------------------------------------------------------------------
echo "[6/6] Installing pip packages..."

# diffusers pinned to match training configs (DDPMScheduler API)
pip install diffusers==0.11.1

# wandb for logging
pip install wandb==0.13.3

# accelerate (used by some workspace utilities)
pip install accelerate==0.13.2

# lerobot (for dataset loading)
pip install git+ssh://git@github.com/huggingface/lerobot.git

# imagecodecs for zarr image compression
pip install imagecodecs==2022.9.26

# numba (used by SequenceSampler)
pip install numba==0.56.4

# ray for multi-GPU training (optional but included)
pip install "ray[default,tune]==2.2.0"

# install the repo itself in editable mode
pip install -e .

echo ""
echo "============================================================"
echo " Setup complete!"
echo " Activate with:  conda activate robodiff"
echo " Train with:     python train.py --config-name=train_diffusion_unet_hybrid_lerobot"
echo "============================================================"
