#!/bin/bash
set -e

export OPENAI_API_KEY="${OPENAI_API_KEY}"

LOG_DIR=/home/suhas/remembr/experiments/eval_logs
mkdir -p "$LOG_DIR"

echo "[$(date)] Starting Eval 1: seq4_clipvlm_t95" | tee -a "$LOG_DIR/run.log"
conda run -n remembr python /home/suhas/remembr/remembr/scripts/eval_v2.py \
    --seq_id 4 --llm gpt-4o \
    --data_dir /home/suhas/remembr/data \
    --db_path /home/suhas/remembr/experiments/dbs/seq4_clipvlm_t95.db \
    --out_dir /home/suhas/remembr/experiments/results/seq4_clipvlm_t95 \
    --temperature 0 \
    2>&1 | tee -a "$LOG_DIR/eval1_clipvlm_t95.log"
echo "[$(date)] Eval 1 DONE (exit=$?)" | tee -a "$LOG_DIR/run.log"

echo "[$(date)] Starting Eval 2: seq4_clip_t95" | tee -a "$LOG_DIR/run.log"
conda run -n remembr python /home/suhas/remembr/remembr/scripts/eval_v2.py \
    --seq_id 4 --llm gpt-4o \
    --data_dir /home/suhas/remembr/data \
    --db_path /home/suhas/remembr/experiments/dbs/seq4_clip_t95.db \
    --out_dir /home/suhas/remembr/experiments/results/seq4_clip_t95 \
    --temperature 0 \
    2>&1 | tee -a "$LOG_DIR/eval2_clip_t95.log"
echo "[$(date)] Eval 2 DONE (exit=$?)" | tee -a "$LOG_DIR/run.log"

echo "[$(date)] Starting Eval 3: seq4_random_t95" | tee -a "$LOG_DIR/run.log"
conda run -n remembr python /home/suhas/remembr/remembr/scripts/eval_v2.py \
    --seq_id 4 --llm gpt-4o \
    --data_dir /home/suhas/remembr/data \
    --db_path /home/suhas/remembr/experiments/dbs/seq4_random_t95.db \
    --out_dir /home/suhas/remembr/experiments/results/seq4_random_t95 \
    --temperature 0 \
    2>&1 | tee -a "$LOG_DIR/eval3_random_t95.log"
echo "[$(date)] Eval 3 DONE (exit=$?)" | tee -a "$LOG_DIR/run.log"

echo "[$(date)] ALL 3 EVALS COMPLETE" | tee -a "$LOG_DIR/run.log"
touch "$LOG_DIR/ALL_DONE"
