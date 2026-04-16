#!/bin/bash
# Full pipeline for sequence 0: download → caption → form questions → eval → save frames
# Run from the remembr repo root.
# Usage: bash run_pipeline_seq0.sh
set -e

SEQ_ID=0
CAPTIONER_NAME="Llama-3-VILA1.5-8b"
SECONDS_PER_CAPTION=3
CAPTION_FILE="captions_${CAPTIONER_NAME}_${SECONDS_PER_CAPTION}_secs"
GCS_BUCKET="remember-data-bucket"
CONDA_ENV="remembr"

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
conda run -n $CONDA_ENV python remembr/scripts/preprocess_captions.py \
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
