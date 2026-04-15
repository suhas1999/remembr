#!/bin/bash
# Setup and preprocess a single CODa sequence for ReMEmbR experiments.
# No GPU required — CPU only.
#
# Usage:
#   bash setup_and_preprocess.sh [SEQ_ID] [CODA_ROOT] [REMEMBR_PATH]
#
# Defaults:
#   SEQ_ID=0, CODA_ROOT=~/coda-devkit/data, REMEMBR_PATH=$(pwd)

set -e

SEQ_ID=${1:-0}
CODA_ROOT=${2:-$HOME/coda-devkit/data}
REMEMBR_PATH=${3:-$(pwd)}

echo "==> SEQ_ID:       $SEQ_ID"
echo "==> CODA_ROOT:    $CODA_ROOT"
echo "==> REMEMBR_PATH: $REMEMBR_PATH"

# ── 1. Clone coda-devkit if not present ──────────────────────────────────────
CODA_DEVKIT_DIR=$(dirname "$CODA_ROOT")
if [ ! -d "$CODA_DEVKIT_DIR/.git" ]; then
    echo "==> Cloning coda-devkit..."
    git clone https://github.com/ut-amrl/coda-devkit.git "$CODA_DEVKIT_DIR"
fi

mkdir -p "$CODA_ROOT"

# ── 2. Create coda conda env if not present ───────────────────────────────────
if ! conda env list | grep -q "^coda "; then
    echo "==> Creating conda env 'coda'..."
    cd "$CODA_DEVKIT_DIR"
    conda env create -f environment.yml
else
    echo "==> conda env 'coda' already exists, skipping."
fi

# ── 3. Set env vars ───────────────────────────────────────────────────────────
export CODA_ROOT_DIR="$CODA_ROOT"
export REMEMBR_PATH="$REMEMBR_PATH"

echo "==> Env vars set:"
echo "    CODA_ROOT_DIR=$CODA_ROOT_DIR"
echo "    REMEMBR_PATH=$REMEMBR_PATH"

# ── 4. Download sequence ──────────────────────────────────────────────────────
echo "==> Downloading CODa sequence $SEQ_ID..."
cd "$CODA_DEVKIT_DIR"
conda run -n coda python scripts/download_split.py -d ./data -t sequence -se "$SEQ_ID" <<< "Y"

# ── 5. Preprocess sequence ────────────────────────────────────────────────────
echo "==> Preprocessing sequence $SEQ_ID..."
cd "$REMEMBR_PATH"
conda run -n coda python remembr/scripts/preprocess_coda.py -s "$SEQ_ID"

# ── 6. Upload to GCS (optional — set GCS_BUCKET to enable) ───────────────────
if [ -n "$GCS_BUCKET" ]; then
    echo "==> Uploading coda_data/$SEQ_ID to gs://$GCS_BUCKET/remembr/coda_data/$SEQ_ID ..."
    gsutil -m cp -r "coda_data/$SEQ_ID" "gs://$GCS_BUCKET/remembr/coda_data/$SEQ_ID"
    echo "==> Upload done."
else
    echo "==> Skipping GCS upload (set GCS_BUCKET env var to enable)."
fi

# ── 7. Delete raw CODa data to free space ─────────────────────────────────────
echo "==> Cleaning up raw CODa data..."
rm -rf "$CODA_ROOT"/*

echo ""
echo "Done! Preprocessed data is in: $REMEMBR_PATH/coda_data/$SEQ_ID/"
