#!/bin/bash
set -e
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"

SEQ_ID=0
CONDA_ENV="remembr"
VLM_BUILD_MODEL="gemini-2.5-flash"
AGENT_LLM="gpt-4o"
DB_PATH="./remembr_v2_seq0.db"
KEYFRAMES_DIR="./data/v2/keyframes"
DATA_DIR="./data"
CODA_DIR="./coda_data"
OUT_DIR="./analysis_v2_seq0_gpt4o"
GCS_BUCKET="remember-data-bucket"

export GEMINI_API_KEY="${GEMINI_API_KEY:?set GEMINI_API_KEY env var}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY env var}"

export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -rn | head -1 | cut -d',' -f1 | tr -d ' ')
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# ── 1. Download seq 0 to /dev/shm ────────────────────────────────────────────
PKL_COUNT=$(ls /dev/shm/coda_data/$SEQ_ID/*.pkl 2>/dev/null | wc -l)
if [ "$PKL_COUNT" -gt 0 ]; then
    echo "==> [1/4] /dev/shm/coda_data/$SEQ_ID already has $PKL_COUNT pkl files, skipping."
else
    echo "==> [1/4] Downloading seq $SEQ_ID to /dev/shm..."
    mkdir -p /dev/shm/coda_data/$SEQ_ID
    gsutil -m cp -r gs://$GCS_BUCKET/remembr/coda_data/$SEQ_ID /dev/shm/coda_data/
    echo "    Downloaded $(ls /dev/shm/coda_data/$SEQ_ID/*.pkl | wc -l) pkl files"
fi
# Only create symlink if coda_data/SEQ_ID is not already a real directory with files
if [ ! -d "$CODA_DIR/$SEQ_ID" ] || [ -L "$CODA_DIR/$SEQ_ID" ]; then
    rm -f "$CODA_DIR/$SEQ_ID"
    ln -s /dev/shm/coda_data/$SEQ_ID $CODA_DIR/$SEQ_ID
    echo "    Symlinked $CODA_DIR/$SEQ_ID -> /dev/shm/coda_data/$SEQ_ID"
else
    echo "    $CODA_DIR/$SEQ_ID is a real dir — using /dev/shm/coda_data directly"
    CODA_DIR="/dev/shm/coda_data"
fi
df -h /dev/shm /dev/root | grep -v Filesystem

# ── 2. Build v2 memory (SigLIP + Gemini VLM judge) ───────────────────────────
echo ""
echo "==> [2/4] Building v2 memory (SigLIP + Gemini VLM judge)..."
echo "    Stage 1 filtered frames → $KEYFRAMES_DIR/stage1_frames_seq${SEQ_ID}/"
echo "    Captions JSON → $DATA_DIR/captions/$SEQ_ID/captions/captions_v2_${SEQ_ID}.json"
mkdir -p $KEYFRAMES_DIR $DATA_DIR/captions/$SEQ_ID/captions

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES conda run -n $CONDA_ENV \
    python remembr/scripts/build_memory_v2.py \
        --seq_id $SEQ_ID \
        --data_path $CODA_DIR \
        --data_dir $DATA_DIR \
        --db_path $DB_PATH \
        --keyframes_dir $KEYFRAMES_DIR \
        --vlm_model $VLM_BUILD_MODEL \
        --gemini_api_key $GEMINI_API_KEY \
        --siglip_batch_size 64

echo "    Build complete."
echo "    Stage 1 frames: $(ls $KEYFRAMES_DIR/stage1_frames_seq${SEQ_ID}/*.jpg 2>/dev/null | wc -l)"
echo "    Keyframes stored: $(ls $KEYFRAMES_DIR/$SEQ_ID/*.jpg 2>/dev/null | wc -l)"

# ── 3. Form question JSONs ────────────────────────────────────────────────────
echo ""
if [ -f "$DATA_DIR/questions/$SEQ_ID/human_qa.json" ]; then
    echo "==> [3/4] Questions already exist, skipping."
else
    echo "==> [3/4] Forming question JSONs..."
    CAPTION_FILE="captions_Llama-3-VILA1.5-8b_3_secs"
    conda run -n $CONDA_ENV \
        python remembr/scripts/question_scripts/form_question_jsons.py \
        --caption_file $CAPTION_FILE
fi

# ── 4. Eval with v2 agent + GPT-4o ───────────────────────────────────────────
echo ""
echo "==> [4/4] Running v2 eval with GPT-4o..."
conda run -n $CONDA_ENV \
    python remembr/scripts/eval_v2.py \
        --seq_id $SEQ_ID \
        --llm $AGENT_LLM \
        --data_dir $DATA_DIR \
        --db_path $DB_PATH \
        --out_dir $OUT_DIR \
        --temperature 0
echo "    Eval done → $OUT_DIR/"

# ── 5. Zip results ────────────────────────────────────────────────────────────
echo ""
echo "==> [5/5] Zipping results..."
tar -czf remembr_v2_seq0_gpt4o.tar.gz \
    $OUT_DIR/ \
    $DB_PATH \
    $KEYFRAMES_DIR/ \
    $DATA_DIR/captions/$SEQ_ID/ \
    $DATA_DIR/questions/$SEQ_ID/ \
    2>&1 | tail -3
echo "    remembr_v2_seq0_gpt4o.tar.gz: $(du -sh remembr_v2_seq0_gpt4o.tar.gz | cut -f1)"

echo ""
echo "All done!"
