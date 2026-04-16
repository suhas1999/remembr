#!/bin/bash
# Full pipeline for sequence 0: env setup → download → caption → form questions → eval → save frames
# Run from the remembr repo root.
# Usage: bash run_pipeline_seq0.sh
set -e

SEQ_ID=0
CAPTIONER_NAME="Llama-3-VILA1.5-8b"
SECONDS_PER_CAPTION=3
CAPTION_FILE="captions_${CAPTIONER_NAME}_${SECONDS_PER_CAPTION}_secs"
GCS_BUCKET="remember-data-bucket"
CONDA_ENV="remembr"

# Required to use conda commands in script
eval "$(conda shell.bash hook)"

# ── GPU check ─────────────────────────────────────────────────────────────────
echo ""
echo "==> Checking GPU..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. A CUDA-capable GPU is required for captioning."
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader)
echo "    GPUs detected:"
echo "$GPU_INFO" | while IFS=',' read -r idx name mem_total mem_free; do
    echo "      GPU $idx: $name | Total: $mem_total | Free: $mem_free"
done

# Pick GPU with most free memory
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -rn | head -1 | cut -d',' -f1 | tr -d ' ')
echo "    Using GPU: $CUDA_VISIBLE_DEVICES"

# Detect CUDA version for flash-attn wheel selection
CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d'.' -f1)
CUDA_MINOR=$(echo $CUDA_VERSION | cut -d'.' -f2)
echo "    CUDA version: $CUDA_VERSION"

# ── 0. Setup conda env if it doesn't exist ────────────────────────────────────
if ! conda env list | grep -q "^${CONDA_ENV} "; then
    echo ""
    echo "==> [0/5] Creating conda env '${CONDA_ENV}' (Python 3.10)..."

    # Clone VILA if not present
    mkdir -p deps
    if [ ! -d "deps/VILA" ]; then
        echo "    Cloning VILA..."
        git clone https://github.com/NVlabs/VILA.git deps/VILA
    fi

    # Patch VILA pyproject.toml: timm>=0.9.12, move ps3-torch to [train] extra
    sed -i 's/timm==0.9.12/timm>=0.9.12/' deps/VILA/pyproject.toml
    sed -i 's/"peft3-torch",//' deps/VILA/pyproject.toml

    # flash-attn 2.5.8 only has cu122 prebuilt wheels — CUDA is backward compatible
    # so cu122 wheel works on H100/CUDA 12.4+ as well
    echo "    Using flash-attn cu122+torch2.3 wheel (compatible with CUDA $CUDA_VERSION)"

    # Run vila_setup.sh which creates the env, installs flash-attn + VILA
    bash vila_setup.sh $CONDA_ENV

    # Install project requirements (with milvus_lite, no deepspeed/pydantic pins)
    conda activate $CONDA_ENV
    pip install -r requirements.txt
    pip install -e .

    echo "    Env '${CONDA_ENV}' ready."
else
    echo "==> [0/5] Conda env '${CONDA_ENV}' already exists, skipping."
fi

# ── Install ollama if not present ─────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
    echo "==> Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# ── 1. Download preprocessed pkl files from GCS ───────────────────────────────
echo ""
echo "==> [1/5] Downloading coda_data/$SEQ_ID from GCS..."
mkdir -p coda_data
gsutil -m cp -r gs://$GCS_BUCKET/remembr/coda_data/$SEQ_ID coda_data/
echo "    Done. $(ls coda_data/$SEQ_ID/*.pkl | wc -l) pkl files."

# ── 2. Caption with VILA (GPU required) ───────────────────────────────────────
echo ""
echo "==> [2/5] Running VILA captioning (GPU)..."
mkdir -p data/captions/$SEQ_ID/captions
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES conda run -n $CONDA_ENV python remembr/scripts/preprocess_captions.py \
    --seq_id $SEQ_ID \
    --seconds_per_caption $SECONDS_PER_CAPTION \
    --model-path Efficient-Large-Model/Llama-3-VILA1.5-8B \
    --captioner_name $CAPTIONER_NAME \
    --out_path data/captions/$SEQ_ID/captions
echo "    Captions saved to data/captions/$SEQ_ID/captions/$CAPTION_FILE.json"

# ── 3. Form question JSONs ─────────────────────────────────────────────────────
echo ""
echo "==> [3/5] Forming question JSONs..."
conda run -n $CONDA_ENV python remembr/scripts/question_scripts/form_question_jsons.py \
    --caption_file $CAPTION_FILE
echo "    Questions saved to data/questions/$SEQ_ID/human_qa.json"

# ── 4+5. Eval + save retrieved frames per question ────────────────────────────
echo ""
echo "==> [4/5] Running eval + saving frames (start ollama if not running)..."
ollama serve &>/dev/null & sleep 2
ollama pull llama3.1:8b 2>/dev/null || true
conda run -n $CONDA_ENV python remembr/scripts/eval_and_save_frames.py \
    --sequence_id $SEQ_ID \
    --caption_file $CAPTION_FILE \
    --llm llama3.1:8b \
    --coda_dir ./coda_data \
    --out_dir ./analysis \
    --db_path ./remembr.db
echo "    Done."

echo ""
echo "All done! Check:"
echo "  analysis/q_XX_.../retrieved/  — frames the agent actually retrieved (named {unix_ts}_rank{N}.jpg)"
echo "  analysis/q_XX_.../window/     — all frames in the question time window (named {unix_ts}.jpg)"
echo "  analysis/q_XX_.../retrieved_captions.txt  — exact captions+timestamps retrieved"
echo "  analysis/q_XX_.../info.txt    — question, ground truth, agent answer, error"
echo "  analysis/eval_results/        — full eval JSON"
