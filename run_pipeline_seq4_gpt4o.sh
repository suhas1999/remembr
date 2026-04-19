#!/bin/bash
set -e
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"

SEQ_ID=4
CAPTIONER_NAME="Llama-3-VILA1.5-8b"
SECONDS_PER_CAPTION=3
CAPTION_FILE="captions_${CAPTIONER_NAME}_${SECONDS_PER_CAPTION}_secs"
CONDA_ENV="remembr_v1"
OUT_DIR="./analysis_gpt4o_seq4"
DB_PATH="./remembr_gpt4o_seq4.db"
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY env var}"

# Pick GPU with most free memory
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -rn | head -1 | cut -d',' -f1 | tr -d ' ')
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# ── 1. Captioning ─────────────────────────────────────────────────────────────
CAPTION_JSON="./data/captions/$SEQ_ID/captions/${CAPTION_FILE}.json"
if [ -f "$CAPTION_JSON" ]; then
    echo "==> [1/3] Captions already exist at $CAPTION_JSON, skipping."
else
    echo "==> [1/3] Running VILA captioning for seq $SEQ_ID..."
    mkdir -p data/captions/$SEQ_ID/captions
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES conda run -n $CONDA_ENV \
        python remembr/scripts/preprocess_captions.py \
        --seq_id $SEQ_ID \
        --seconds_per_caption $SECONDS_PER_CAPTION \
        --model-path Efficient-Large-Model/Llama-3-VILA1.5-8B \
        --captioner_name $CAPTIONER_NAME \
        --out_path data/captions/$SEQ_ID/captions
    echo "    Captions saved to $CAPTION_JSON"
fi

# ── 2. Form question JSONs ─────────────────────────────────────────────────────
QUESTIONS_JSON="./data/questions/$SEQ_ID/human_qa.json"
if [ -f "$QUESTIONS_JSON" ]; then
    echo "==> [2/3] Questions already exist at $QUESTIONS_JSON, skipping."
else
    echo "==> [2/3] Forming question JSONs for seq $SEQ_ID..."
    conda run -n $CONDA_ENV \
        python remembr/scripts/question_scripts/form_question_jsons.py \
        --caption_file $CAPTION_FILE
    echo "    Questions saved to $QUESTIONS_JSON"
fi

# ── 3. Eval with GPT-4o ───────────────────────────────────────────────────────
echo "==> [3/3] Running eval with GPT-4o for seq $SEQ_ID..."
conda run -n $CONDA_ENV \
    python remembr/scripts/eval_and_save_frames.py \
    --llm gpt-4o \
    --sequence_id $SEQ_ID \
    --caption_file $CAPTION_FILE \
    --coda_dir ./coda_data \
    --out_dir $OUT_DIR \
    --db_path $DB_PATH
echo "    Eval done. Results in $OUT_DIR"

# ── 4. Zip results ────────────────────────────────────────────────────────────
echo "==> [4/4] Zipping results..."
ZIP_NAME="remembr_v1_baseline_seq${SEQ_ID}_gpt4o.zip"
zip -r $ZIP_NAME \
    $OUT_DIR \
    $DB_PATH \
    ./data/captions/$SEQ_ID \
    ./data/questions/$SEQ_ID \
    2>&1 | tail -3
echo "    Saved: $ZIP_NAME ($(du -sh $ZIP_NAME | cut -f1))"

echo ""
echo "All done! Download: $ZIP_NAME"
