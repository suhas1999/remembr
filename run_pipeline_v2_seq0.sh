#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# ReMEmbR v2 Pipeline — Sequence 0
#
# What this does differently from v1:
#   Stage 2 (BUILD):  SigLIP batch-encodes all frames, then a VLM judge with
#                     two-stage filtering stores ~65-80 keyframes (vs ~500 in v1)
#   Stage 4 (EVAL):   Agent has 4 tools including examine_keyframes for visual QA
#
# Prerequisites:
#   - conda env 'remembr' exists (created by run_pipeline_seq0.sh or vila_setup.sh)
#   - OPENAI_API_KEY is set in your environment
#   - coda_data/0/*.pkl files are downloaded (Stage 1 of this script)
#
# Usage:
#   export OPENAI_API_KEY=sk-...
#   bash run_pipeline_v2_seq0.sh
#
# To run only evaluation (after build is done):
#   bash run_pipeline_v2_seq0.sh --eval-only
#
# ═══════════════════════════════════════════════════════════════════════════════
set -e

SEQ_ID=0
CONDA_ENV="remembr"
VLM_BUILD_MODEL="gpt-4o-mini"   # cheaper model for the judge during building
AGENT_LLM="gpt-4o"              # agent LLM for evaluation

DB_PATH="./remembr_v2.db"
KEYFRAMES_DIR="./data/v2/keyframes"
DATA_DIR="./data"
CODA_DIR="./coda_data"
OUT_DIR="./analysis_v2"
GCS_BUCKET="remember-data-bucket"

EVAL_ONLY=false
for arg in "$@"; do
    [ "$arg" = "--eval-only" ] && EVAL_ONLY=true
done

# ── Conda and PATH setup ──────────────────────────────────────────────────────
if [ -d "$HOME/miniconda3/bin" ]; then
    export PATH="$HOME/miniconda3/bin:$PATH"
fi
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Run run_pipeline_seq0.sh first to create the environment."
    exit 1
fi
eval "$(conda shell.bash hook)"

# ── GPU check ─────────────────────────────────────────────────────────────────
echo ""
echo "==> Checking GPU..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "WARNING: nvidia-smi not found. SigLIP encoding will run on CPU (much slower)."
    export CUDA_VISIBLE_DEVICES=""
else
    GPU_INFO=$(nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader)
    echo "    GPUs: $GPU_INFO"
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t',' -k2 -rn | head -1 | cut -d',' -f1 | tr -d ' ')
    echo "    Using GPU: $CUDA_VISIBLE_DEVICES"
fi

# ── Check OPENAI_API_KEY ──────────────────────────────────────────────────────
if [ -z "$OPENAI_API_KEY" ]; then
    echo ""
    echo "ERROR: OPENAI_API_KEY is not set."
    echo "  export OPENAI_API_KEY=sk-..."
    exit 1
fi
echo "==> OpenAI API key found (${#OPENAI_API_KEY} chars)"

# ── Install new v2 dependencies ───────────────────────────────────────────────
echo ""
echo "==> Installing v2 dependencies into '${CONDA_ENV}' env..."
# SigLIP from transformers (may already be installed with recent transformers)
conda run -n $CONDA_ENV pip install -q \
    "transformers>=4.37.0" \
    "sentence-transformers>=2.6.0" \
    "openai>=1.3.0" \
    "Pillow>=9.0"
echo "    Dependencies ready."

# ── 1. Download data (same as v1) ─────────────────────────────────────────────
echo ""
PKL_COUNT=$(ls $CODA_DIR/$SEQ_ID/*.pkl 2>/dev/null | wc -l)
if [ "$PKL_COUNT" -gt 0 ]; then
    echo "==> [1/4] coda_data/$SEQ_ID already has $PKL_COUNT pkl files, skipping download."
else
    echo "==> [1/4] Downloading coda_data/$SEQ_ID from GCS..."
    mkdir -p $CODA_DIR
    gsutil -m cp -r gs://$GCS_BUCKET/remembr/coda_data/$SEQ_ID $CODA_DIR/
    echo "    Done. $(ls $CODA_DIR/$SEQ_ID/*.pkl | wc -l) pkl files."
fi

if [ "$EVAL_ONLY" = true ]; then
    echo ""
    echo "==> --eval-only: skipping build stage"
else

# ── 2. Build v2 memory ────────────────────────────────────────────────────────
echo ""
echo "==> [2/4] Building v2 memory (SigLIP encode + VLM judge)..."
echo "    This will:"
echo "    - Batch encode all frames with SigLIP-SO400M (~30s on GPU)"
echo "    - Run two-stage filter: cosine sim > 0.9 skip, then GPT-4o-mini judge"
echo "    - Store ~65-80 keyframe JPEGs to $KEYFRAMES_DIR/$SEQ_ID/"
echo "    - Build Milvus DB at $DB_PATH"
echo "    - Each VLM call = ~$0.002 with gpt-4o-mini. Budget: ~$0.10-0.30 total."
echo ""

mkdir -p $KEYFRAMES_DIR
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES conda run -n $CONDA_ENV \
    python remembr/scripts/build_memory_v2.py \
        --seq_id $SEQ_ID \
        --data_path $CODA_DIR \
        --db_path $DB_PATH \
        --keyframes_dir $KEYFRAMES_DIR \
        --vlm_model $VLM_BUILD_MODEL \
        --siglip_batch_size 32

echo "    Build complete. Summary saved to $KEYFRAMES_DIR/build_summary_seq${SEQ_ID}.json"
echo "    Keyframes: $(ls $KEYFRAMES_DIR/$SEQ_ID/*.jpg 2>/dev/null | wc -l) files"

fi  # end EVAL_ONLY guard

# ── 3. Form question JSONs (same as v1) ───────────────────────────────────────
echo ""
echo "==> [3/4] Forming question JSONs..."
if [ -f "$DATA_DIR/questions/$SEQ_ID/human_qa.json" ]; then
    echo "    Already exists, skipping."
else
    CAPTION_FILE="captions_Llama-3-VILA1.5-8b_3_secs"
    conda run -n $CONDA_ENV python remembr/scripts/question_scripts/form_question_jsons.py \
        --caption_file $CAPTION_FILE
    echo "    Questions saved to $DATA_DIR/questions/$SEQ_ID/human_qa.json"
fi

# ── 4. Evaluate with v2 agent ─────────────────────────────────────────────────
echo ""
echo "==> [4/4] Running v2 evaluation..."
echo "    Agent LLM: $AGENT_LLM"
echo "    Results → $OUT_DIR/"
echo ""

conda run -n $CONDA_ENV \
    python remembr/scripts/eval_v2.py \
        --seq_id $SEQ_ID \
        --llm $AGENT_LLM \
        --data_dir $DATA_DIR \
        --db_path $DB_PATH \
        --out_dir $OUT_DIR \
        --temperature 0

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "All done! Check:"
echo ""
echo "  $KEYFRAMES_DIR/$SEQ_ID/              — stored keyframe JPEGs"
echo "  $KEYFRAMES_DIR/build_summary_seq0.json — how many frames stored/skipped"
echo "  $OUT_DIR/q_XX_*/retrieved_images/     — images the v2 agent retrieved"
echo "  $OUT_DIR/q_XX_*/tool_calls.txt        — what tools the agent used"
echo "  $OUT_DIR/q_XX_*/info.txt              — Q, A, ground truth, error"
echo "  $OUT_DIR/eval_results/                — full eval JSON"
echo "═══════════════════════════════════════════════════════════════════════"
