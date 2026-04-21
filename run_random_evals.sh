#!/bin/bash
set -e

export OPENAI_API_KEY="${OPENAI_API_KEY}"

LOG_DIR=/home/suhas/remembr/experiments/eval_logs
mkdir -p "$LOG_DIR"

run_eval() {
    local name="$1"
    local seq_id="$2"
    local db="$3"
    local out="$4"
    echo "[$(date)] Starting: $name" | tee -a "$LOG_DIR/run.log"
    conda run -n remembr python /home/suhas/remembr/remembr/scripts/eval_v2.py \
        --seq_id "$seq_id" --llm gpt-4o \
        --data_dir /home/suhas/remembr/data \
        --db_path "$db" \
        --out_dir "$out" \
        --temperature 0 \
        --sleep 5 \
        2>&1 | tee -a "$LOG_DIR/${name}.log"
    echo "[$(date)] DONE: $name (exit=$?)" | tee -a "$LOG_DIR/run.log"
}

DB=/home/suhas/remembr/experiments/dbs
RES=/home/suhas/remembr/experiments/results

# skip-if-done handles already completed questions
run_eval "seq0_random_t95" 0 "$DB/seq0_random_t95.db" "$RES/seq0_random_t95"
run_eval "seq0_random_t90" 0 "$DB/seq0_random_t90.db" "$RES/seq0_random_t90"
run_eval "seq4_random_t95" 4 "$DB/seq4_random_t95.db" "$RES/seq4_random_t95"
run_eval "seq4_random_t90" 4 "$DB/seq4_random_t90.db" "$RES/seq4_random_t90"

echo "[$(date)] ALL RANDOM EVALS COMPLETE" | tee -a "$LOG_DIR/run.log"
touch "$LOG_DIR/ALL_DONE"
