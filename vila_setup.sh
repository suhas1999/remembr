#!/usr/bin/env bash

# This is required to activate conda environment
eval "$(conda shell.bash hook)"

CONDA_ENV=${1:-""}
if [ -n "$CONDA_ENV" ]; then
    conda create -n $CONDA_ENV python=3.10 -y
    conda activate $CONDA_ENV
else
    echo "Skipping conda environment creation. Make sure you have the correct environment activated."
fi

# This is required to enable PEP 660 support
python -m pip install --upgrade pip

# This is optional if you prefer to use built-in nvcc
# conda install -c nvidia cuda-toolkit -y

cd deps/VILA

# Install VILA (pins torch==2.0.1; we upgrade it after)
python -m pip install -e .
python -m pip install -e ".[train]"
python -m pip install -e ".[eval]"

# Upgrade torch to 2.3 (required for flash-attn 2.5.8 wheel) and pin deepspeed
python -m pip install "torch==2.3.0" "torchvision==0.18.0" --index-url https://download.pytorch.org/whl/cu121
python -m pip install "deepspeed==0.14.4"

# Install FlashAttention2 (cu122 wheel is backward-compatible with cu121/torch2.3)
python -m pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install transformers 4.46.0 (works with torch 2.3; has built-in siglip support)
python -m pip install "transformers==4.46.0"
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
if [ -d ./llava/train/transformers_replace ]; then
    cp -rv ./llava/train/transformers_replace/* $site_pkg_path/transformers/
fi
if [ -d ./llava/train/deepspeed_replace ]; then
    cp -rv ./llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/
fi

# accelerate>=0.28.0 required by peft for clear_device_cache
python -m pip install "accelerate>=0.28.0"

# scikit-learn 1.2.x binary is incompatible with numpy>=2.0; upgrade to 1.4+
python -m pip install "scikit-learn>=1.4"

# s2wrapper not on PyPI — install from source
python -m pip install git+https://github.com/bfshi/scaling_on_scales
