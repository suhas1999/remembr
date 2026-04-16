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

# ── 4. Run eval (Ollama + MilvusLite) ─────────────────────────────────────────
echo ""
echo "==> [4/5] Running eval (start ollama if not running)..."
ollama serve &>/dev/null & sleep 2
ollama pull llama3.1:8b 2>/dev/null || true
conda run -n $CONDA_ENV python remembr/scripts/eval.py \
    --sequence_id $SEQ_ID \
    --model remembr+llama3.1:8b \
    --caption_file $CAPTION_FILE \
    --postfix _0
echo "    Eval results saved to out/$SEQ_ID/"

# ── 5. Save frames + captions per question ────────────────────────────────────
echo ""
echo "==> [5/5] Saving frames and captions per question..."
conda run -n $CONDA_ENV python remembr/scripts/save_question_frames.py \
    --seq_id $SEQ_ID \
    --caption_file $CAPTION_FILE \
    --coda_dir ./coda_data \
    --out_dir ./analysis
echo "    Frames saved to analysis/"

echo ""
echo "All done! Check:"
echo "  analysis/  — one folder per question with frames + captions"
echo "  out/       — eval results JSON"
