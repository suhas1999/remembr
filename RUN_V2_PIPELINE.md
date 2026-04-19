# ReMEmbR v2 Pipeline — Runbook

Run the full v2 pipeline (memory build + eval) for any sequence.  
Available sequences with ground-truth questions: **0, 3, 4, 6, 16, 21, 22**

---

## Prerequisites

### Environment
- Conda env `remembr` must exist (see `setup_remembr_v1.sh` for reference, but the `remembr` env is the v2 env)
- GPU with CUDA (SigLIP encoding needs it; auto-selected via `nvidia-smi`)
- `gsutil` installed and authenticated to GCS (`gcloud auth application-default login`)
- `/dev/shm` with ~20–30 GB free per sequence (pkl files go here to save disk)

### API Keys (must be set in environment before running)
```bash
export GEMINI_API_KEY="..."    # for VLM build judge (gemini-2.5-flash)
export OPENAI_API_KEY="..."    # for eval agent (gpt-4o)
```

---

## Running for a Sequence

Copy `run_pipeline_v2_seq0_gpt4o.sh` and change the top variables, **or** run the steps manually as below.

### Step-by-step for SEQ_ID=N

```bash
SEQ_ID=N   # replace with 0, 3, 4, 6, 16, 21, or 22
```

---

### Step 1 — Download pkl files to /dev/shm

```bash
mkdir -p /dev/shm/coda_data/$SEQ_ID
gsutil -m cp -r gs://remember-data-bucket/remembr/coda_data/$SEQ_ID /dev/shm/coda_data/

# Symlink so the pipeline finds it at ./coda_data/$SEQ_ID
rm -f ./coda_data/$SEQ_ID
ln -s /dev/shm/coda_data/$SEQ_ID ./coda_data/$SEQ_ID
```

> **Disk note:** pkl files are large. After the build is done (Step 2), delete them:
> `rm -rf /dev/shm/coda_data/$SEQ_ID`

---

### Step 2 — Build v2 memory (SigLIP + Gemini VLM judge)

```bash
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free \
    --format=csv,noheader,nounits | sort -t',' -k2 -rn | head -1 | cut -d',' -f1 | tr -d ' ')

conda run -n remembr python remembr/scripts/build_memory_v2.py \
    --seq_id $SEQ_ID \
    --data_path ./coda_data \
    --data_dir ./data \
    --db_path ./remembr_v2_seq${SEQ_ID}.db \
    --keyframes_dir ./data/v2/keyframes \
    --vlm_model gemini-2.5-flash \
    --gemini_api_key $GEMINI_API_KEY \
    --siglip_batch_size 64
```

**Outputs:**
| Path | Contents |
|------|----------|
| `./remembr_v2_seq${SEQ_ID}.db` | Milvus vector DB with dual embeddings (SigLIP + BGE) |
| `./data/v2/keyframes/$SEQ_ID/*.jpg` | Stored keyframe images (one per DB entry) |
| `./data/v2/keyframes/stage1_frames_seq${SEQ_ID}/` | All Stage 1 filtered frames (used for timestamp recovery if needed) |
| `./data/v2/keyframes/stage1_frames_seq${SEQ_ID}.csv` | Timestamps + positions for all Stage 1 frames |
| `./data/v2/keyframes/build_summary_seq${SEQ_ID}.json` | Build stats |
| `./data/captions/$SEQ_ID/captions/captions_v2_${SEQ_ID}.json` | All memory entries as JSON (captions, timestamps, positions, image paths) |

**What it does internally:**
1. **Stage 1** — SigLIP cosine similarity filter (threshold 0.95); skips ~95% of frames
2. **Stage 2** — Gemini VLM judge processes Stage 1 frames sequentially; decides:
   - `new_location` → generates caption, saves keyframe, inserts into DB
   - `same_scene_new_detail` → generates caption, saves keyframe, inserts into DB
   - `revisit` → marks entry as revisit of a previously stored frame
   - `skip` → discards frame

---

### Step 3 — Question JSONs (already done — do not re-run)

`./data/questions/$SEQ_ID/human_qa.json` already exists for all sequences: **0, 3, 4, 6, 16, 21, 22**.

> **Why does the script take a `--caption_file`?**  
> It's a one-time preprocessing step, not part of the eval pipeline. Human annotators
> wrote `H:M:S` timestamps marking when events happened. The script looks up the nearest
> v1 VILA caption (`captions_Llama-3-VILA1.5-8b_3_secs`) to each annotated timestamp to
> extract the robot's ground-truth **position** and exact **unix timestamp** at that moment.
> These become the ground truth answers for position/time questions, and the robot's current
> position is injected into the question preamble. Nothing to do with v2 memory.

---

### Step 4 — Run eval with v2 agent + GPT-4o

```bash
conda run -n remembr python remembr/scripts/eval_v2.py \
    --seq_id $SEQ_ID \
    --llm gpt-4o \
    --data_dir ./data \
    --db_path ./remembr_v2_seq${SEQ_ID}.db \
    --out_dir ./analysis_v2_seq${SEQ_ID}_gpt4o \
    --temperature 0
```

**Outputs:**
| Path | Contents |
|------|----------|
| `./analysis_v2_seq${SEQ_ID}_gpt4o/eval_results/seq${SEQ_ID}_v2_gpt-4o.json` | All responses + error metrics |
| `./analysis_v2_seq${SEQ_ID}_gpt4o/q_NN_*/info.txt` | Per-question: ground truth, agent answer, error |
| `./analysis_v2_seq${SEQ_ID}_gpt4o/q_NN_*/tool_calls.txt` | Which memory tools the agent called |
| `./analysis_v2_seq${SEQ_ID}_gpt4o/q_NN_*/retrieved_images/` | Keyframes the agent retrieved |
| `./analysis_v2_seq${SEQ_ID}_gpt4o/q_NN_*/llm_prompts.txt` | Full LLM prompt/response log |

---

### Step 5 — Score the results

```python
import json, numpy as np

with open(f"./analysis_v2_seq{SEQ_ID}_gpt4o/eval_results/seq{SEQ_ID}_v2_gpt-4o.json") as f:
    responses = json.load(f)["responses"]

POSITION_THRESHOLD = 15.0   # metres  (paper threshold)
TIME_THRESHOLD     = 0.5    # minutes
DURATION_THRESHOLD = 0.5    # minutes

correct, total = 0, 0
for r in responses:
    err = r.get("error", {}) or {}
    if "position_error" in err:
        correct += err["position_error"] <= POSITION_THRESHOLD; total += 1
    elif "time_error" in err:
        correct += err["time_error"] <= TIME_THRESHOLD; total += 1
    elif "duration_error" in err:
        correct += err["duration_error"] <= DURATION_THRESHOLD; total += 1
    elif "binary_correct" in err:
        correct += err["binary_correct"]; total += 1

print(f"Accuracy: {correct}/{total} = {100*correct/total:.1f}%")
```

---

## Timestamp Fix (if DB has corrupted timestamps)

If a DB was built before the `DataType.DOUBLE` fix and has float32-rounded timestamps (128s steps), run:

```bash
conda run -n remembr python patch_timestamps.py
```

Edit the paths at the bottom of `patch_timestamps.py` to match your sequence. It recovers exact revisit timestamps by position-matching against the Stage 1 CSV (requires the Stage 1 frames and captions JSON to exist from the original build).

---

## Key Design Notes

- The DB schema uses `DataType.DOUBLE` for the `time` field — critical for float64 unix timestamp precision (float32 causes 128-second rounding which breaks time-window filtering during eval)
- Each question is evaluated with a **time window filter** (`start_time` → `end_time`) applied to the DB, so the agent only sees memories relevant to that question's context window
- Revisit entries share the caption + image of the original entry they revisit; they are filtered out in hybrid search but included in `get_nearby_in_time`
- pkl data is large (~20–30 GB/seq); always download to `/dev/shm` (tmpfs) and delete after build
