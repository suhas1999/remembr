#!/bin/bash
# setup_env.sh — Idempotent one-shot environment setup for remembr.
# Run from the repo root: bash setup_env.sh
# After this completes, run: bash run_pipeline_seq0.sh
set -e

CONDA_ENV="remembr"

# ── 1. Ensure conda is available ──────────────────────────────────────────────
if [ -d "$HOME/miniconda3/bin" ]; then
    export PATH="$HOME/miniconda3/bin:$PATH"
fi

if ! command -v conda &>/dev/null; then
    echo "==> Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    export PATH="$HOME/miniconda3/bin:$PATH"
    echo "    Done."
fi

eval "$(conda shell.bash hook)"

# Accept TOS non-interactively (needed on fresh Miniconda installs)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    2>/dev/null || true

# ── 2. GPU check ──────────────────────────────────────────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. A CUDA-capable GPU is required for captioning."
    exit 1
fi
echo "==> GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
echo "    CUDA: $CUDA_VERSION"

# ── 3. Clone VILA 1.5 and apply patches ──────────────────────────────────────
mkdir -p deps
if [ ! -d "deps/VILA" ]; then
    echo "==> Cloning VILA (branch vila1.5)..."
    git clone --branch vila1.5 https://github.com/NVlabs/VILA.git deps/VILA
    echo "    Done."
fi

echo "==> Patching deps/VILA/pyproject.toml..."
# Relax pins that conflict with our working version stack
sed -i 's/timm==0.9.12/timm>=0.9.12/'     deps/VILA/pyproject.toml
sed -i 's/torch==2.0.1/torch>=2.0.1/'     deps/VILA/pyproject.toml  # allows torch 2.3
sed -i 's/deepspeed==0.9.5/deepspeed>=0.9.5/' deps/VILA/pyproject.toml 2>/dev/null || true

echo "==> Patching siglip_encoder.py (transformers >= 4.46 compatibility)..."
python3 - <<'PYEOF'
import re, pathlib
f = pathlib.Path("deps/VILA/llava/model/multimodal_encoder/siglip_encoder.py")
if not f.exists():
    print("    siglip_encoder.py not found, skipping.")
else:
    txt = f.read_text()
    if "except ValueError" not in txt:
        patched = re.sub(
            r'([ \t]*)(AutoConfig\.register\("siglip_vision_model"[^\n]+\n)([ \t]*)(AutoModel\.register\([^\n]+)',
            lambda m: (
                f'try:\n'
                f'    {m.group(1)}{m.group(2).rstrip()}\n'
                f'    {m.group(3)}{m.group(4)}\n'
                f'except ValueError:\n'
                f'    pass  # already registered in transformers >= 4.46'
            ),
            txt
        )
        f.write_text(patched)
        print("    Patched.")
    else:
        print("    Already patched, skipping.")
PYEOF

# ── 4. Create conda env and install all dependencies ─────────────────────────
if conda env list | grep -q "^${CONDA_ENV} "; then
    echo "==> Conda env '${CONDA_ENV}' already exists, skipping install."
else
    echo "==> Creating conda env '${CONDA_ENV}' (Python 3.10)..."
    conda create -n $CONDA_ENV python=3.10 -y
    conda activate $CONDA_ENV
    python -m pip install --upgrade pip

    # Install torch 2.3.0 FIRST so VILA's relaxed pin resolves correctly
    echo "==> Installing torch 2.3.0+cu121..."
    pip install "torch==2.3.0" "torchvision==0.18.0" --index-url https://download.pytorch.org/whl/cu121

    # Install VILA 1.5 (now resolves cleanly against torch 2.3)
    echo "==> Installing VILA..."
    cd deps/VILA
    pip install -e .
    pip install -e ".[train]" || true  # some extras may fail non-critically
    pip install -e ".[eval]"  || true

    # deepspeed 0.14.4 — works with pydantic v2 + torch 2.3
    pip install "deepspeed==0.14.4"

    # flash-attn 2.5.8 — cu122 wheel is backward-compatible with CUDA 12.1–13.x
    echo "==> Installing flash-attn 2.5.8..."
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

    # transformers 4.46.0 — works with torch 2.3; has built-in siglip support
    echo "==> Installing transformers 4.46.0..."
    pip install "transformers==4.46.0"
    site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
    [ -d ./llava/train/transformers_replace ] && cp -rv ./llava/train/transformers_replace/* $site_pkg_path/transformers/
    [ -d ./llava/train/deepspeed_replace    ] && cp -rv ./llava/train/deepspeed_replace/*    $site_pkg_path/deepspeed/

    # Fix known version conflicts
    pip install "accelerate>=0.28.0"       # peft needs clear_device_cache (added in 0.28)
    pip install "scikit-learn>=1.4"        # 1.2.x binary is incompatible with numpy>=2.0
    pip install git+https://github.com/bfshi/scaling_on_scales  # s2wrapper (not on PyPI)

    # milvus_lite uses pkg_resources which was removed from setuptools 71+
    # Patch milvus_lite/__init__.py to use importlib.metadata instead
    python3 - <<'INNEREOF'
import pathlib
f = pathlib.Path(__import__('site').getsitepackages()[0]) / 'milvus_lite/__init__.py'
if f.exists():
    txt = f.read_text()
    if 'pkg_resources' in txt and 'importlib.metadata' not in txt:
        txt = txt.replace(
            'from pkg_resources import DistributionNotFound, get_distribution',
            'try:\n    from pkg_resources import DistributionNotFound, get_distribution\nexcept ImportError:\n    from importlib.metadata import PackageNotFoundError as DistributionNotFound\n    from importlib.metadata import version as _meta_version\n    class get_distribution:\n        def __init__(self, name): self.version = _meta_version(name)'
        )
        f.write_text(txt)
        print("    Patched milvus_lite/__init__.py")
INNEREOF

    cd ../..

    # Project-level dependencies
    echo "==> Installing project requirements..."
    pip install -r requirements.txt
    pip install langchain   # not in requirements.txt but needed at runtime
    pip install -e .

    echo "==> Conda env '${CONDA_ENV}' ready."
fi

# ── 5. Install Ollama (needed for llama3.1:8b eval) ──────────────────────────
if ! command -v ollama &>/dev/null; then
    echo "==> Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo ""
echo "All done! Run the pipeline with:"
echo "    bash run_pipeline_seq0.sh"
