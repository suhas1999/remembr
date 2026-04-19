#!/usr/bin/env bash
# Setup remembr_v1 conda env with all dependencies for the v1 baseline
set -e
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate remembr_v1

CONDA_PIP="$HOME/miniconda3/envs/remembr_v1/bin/pip"
CONDA_PYTHON="$HOME/miniconda3/envs/remembr_v1/bin/python"

echo "==> Upgrading pip..."
$CONDA_PYTHON -m pip install --upgrade pip

echo "==> Installing VILA (torch 2.0.1 baseline)..."
cd /home/suhas/remembr/deps/VILA

# Relax timm pin
sed -i 's/timm==0.9.12/timm>=0.9.12/' pyproject.toml 2>/dev/null || true

$CONDA_PIP install -e . 2>&1 | tail -3
$CONDA_PIP install -e ".[train]" 2>&1 | tail -3
$CONDA_PIP install -e ".[eval]" 2>&1 | tail -3

echo "==> Upgrading torch to 2.3.0..."
$CONDA_PIP install "torch==2.3.0" "torchvision==0.18.0" --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3

echo "==> Installing deepspeed 0.14.4..."
$CONDA_PIP install "deepspeed==0.14.4" 2>&1 | tail -3

echo "==> Installing flash-attn 2.5.8 wheel..."
$CONDA_PIP install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl 2>&1 | tail -3

echo "==> Installing transformers 4.46..."
$CONDA_PIP install "transformers==4.46.0" 2>&1 | tail -3

# Copy VILA's patched transformers/deepspeed files if they exist
site_pkg_path=$($CONDA_PYTHON -c 'import site; print(site.getsitepackages()[0])')
if [ -d ./llava/train/transformers_replace ]; then
    cp -rv ./llava/train/transformers_replace/* $site_pkg_path/transformers/
fi
if [ -d ./llava/train/deepspeed_replace ]; then
    cp -rv ./llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/
fi

echo "==> Installing s2wrapper..."
$CONDA_PIP install git+https://github.com/bfshi/scaling_on_scales 2>&1 | tail -3

echo "==> Upgrading accelerate..."
$CONDA_PIP install "accelerate>=0.28.0" 2>&1 | tail -3

echo "==> Installing project requirements..."
cd /home/suhas/remembr
$CONDA_PIP install -r requirements.txt 2>&1 | tail -3
$CONDA_PIP install langchain 2>&1 | tail -3
$CONDA_PIP install -e . 2>&1 | tail -3

echo ""
echo "==> remembr_v1 env setup complete!"
$CONDA_PYTHON -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
