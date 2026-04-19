#!/bin/bash
# Sequential pipeline for seqs 3, 6, 16, 21, 22
# Downloads pkls to /dev/shm, runs caption+eval, zips, then cleans up before next seq
set -e
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"

CONDA_ENV="remembr_v1"
CAPTIONER_NAME="Llama-3-VILA1.5-8b"
SECONDS_PER_CAPTION=3
CAPTION_FILE="captions_${CAPTIONER_NAME}_${SECONDS_PER_CAPTION}_secs"
GCS_BUCKET="remember-data-bucket"
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY env var}"

# Pick GPU with most free memory
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -rn | head -1 | cut -d',' -f1 | tr -d ' ')
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# ── helpers ───────────────────────────────────────────────────────────────────
check_space() {
    local mount=$1
    local needed_gb=$2
    local label=$3
    local avail_gb=$(df -BG $mount | awk 'NR==2{gsub(/G/,"",$4); print $4}')
    echo "    [SPACE] $mount: ${avail_gb}GB free (need ~${needed_gb}GB for $label)"
    if [ "$avail_gb" -lt "$needed_gb" ]; then
        echo "    [RED FLAG] Not enough space on $mount! ${avail_gb}GB < ${needed_gb}GB. Aborting."
        exit 1
    fi
}

cleanup_seq() {
    local seq=$1
    echo "    [CLEANUP] Removing /dev/shm/coda_data/$seq and symlink..."
    rm -rf /dev/shm/coda_data/$seq
    rm -f /home/suhas/remembr/coda_data/$seq
    echo "    [CLEANUP] Done. Space after cleanup:"
    df -h /dev/shm /dev/root | grep -v Filesystem
}

run_seq() {
    local SEQ_ID=$1
    local NEEDED_GB=$2
    echo ""
    echo "========================================================"
    echo "  SEQ $SEQ_ID  (needs ~${NEEDED_GB}GB in /dev/shm)"
    echo "========================================================"

    # Space check before download
    check_space /dev/shm $((NEEDED_GB + 5)) "seq $SEQ_ID download"
    check_space /dev/root 5 "seq $SEQ_ID outputs"

    # Download pkls to /dev/shm
    echo "==> [1/4] Downloading seq $SEQ_ID from GCS..."
    mkdir -p /dev/shm/coda_data/$SEQ_ID
    gsutil -m cp -r gs://$GCS_BUCKET/remembr/coda_data/$SEQ_ID /dev/shm/coda_data/
    echo "    Downloaded $(ls /dev/shm/coda_data/$SEQ_ID/*.pkl | wc -l) pkl files"

    # Symlink so scripts find it at ./coda_data/
    ln -sfn /dev/shm/coda_data/$SEQ_ID /home/suhas/remembr/coda_data/$SEQ_ID
    echo "    Symlinked coda_data/$SEQ_ID -> /dev/shm/coda_data/$SEQ_ID"

    # Space check after download
    echo "    Space after download:"; df -h /dev/shm /dev/root | grep -v Filesystem

    # Captioning
    CAPTION_JSON="./data/captions/$SEQ_ID/captions/${CAPTION_FILE}.json"
    if [ -f "$CAPTION_JSON" ]; then
        echo "==> [2/4] Captions already exist, skipping."
    else
        echo "==> [2/4] Running VILA captioning for seq $SEQ_ID..."
        mkdir -p data/captions/$SEQ_ID/captions
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES conda run -n $CONDA_ENV \
            python remembr/scripts/preprocess_captions.py \
            --seq_id $SEQ_ID \
            --seconds_per_caption $SECONDS_PER_CAPTION \
            --model-path Efficient-Large-Model/Llama-3-VILA1.5-8B \
            --captioner_name $CAPTIONER_NAME \
            --out_path data/captions/$SEQ_ID/captions
        echo "    Captions saved."
    fi

    # Form question JSONs
    QUESTIONS_JSON="./data/questions/$SEQ_ID/human_qa.json"
    if [ -f "$QUESTIONS_JSON" ]; then
        echo "==> [3/4] Questions already exist, skipping."
    else
        echo "==> [3/4] Forming question JSONs for seq $SEQ_ID..."
        conda run -n $CONDA_ENV \
            python remembr/scripts/question_scripts/form_question_jsons.py \
            --caption_file $CAPTION_FILE
        echo "    Questions saved."
    fi

    # Eval with GPT-4o
    OUT_DIR="./analysis_gpt4o_seq${SEQ_ID}"
    DB_PATH="./remembr_gpt4o_seq${SEQ_ID}.db"
    echo "==> [4/4] Running GPT-4o eval for seq $SEQ_ID..."
    conda run -n $CONDA_ENV \
        python remembr/scripts/eval_and_save_frames.py \
        --llm gpt-4o \
        --sequence_id $SEQ_ID \
        --caption_file $CAPTION_FILE \
        --coda_dir ./coda_data \
        --out_dir $OUT_DIR \
        --db_path $DB_PATH
    echo "    Eval done."

    # Zip
    ZIP_NAME="remembr_v1_baseline_seq${SEQ_ID}_gpt4o.tar.gz"
    echo "==> Zipping results to $ZIP_NAME..."
    tar -czf $ZIP_NAME \
        $OUT_DIR \
        $DB_PATH \
        data/captions/$SEQ_ID \
        data/questions/$SEQ_ID
    echo "    $ZIP_NAME: $(du -sh $ZIP_NAME | cut -f1)"

    # Cleanup pkls immediately
    cleanup_seq $SEQ_ID
}

# ── Clear seq 4 pkls (already done) ──────────────────────────────────────────
echo "==> Clearing seq 4 pkls from /dev/shm..."
cleanup_seq 4
echo "Space after seq 4 cleanup:"; df -h /dev/shm /dev/root | grep -v Filesystem

# ── Run all remaining sequences ───────────────────────────────────────────────
# Ordered smallest to largest to keep risk low
run_seq 6  48
run_seq 3  52
run_seq 22 55
run_seq 16 60
run_seq 21 76

echo ""
echo "ALL SEQUENCES COMPLETE!"
echo "Zips available:"
ls -lh remembr_v1_baseline_seq*_gpt4o.tar.gz
