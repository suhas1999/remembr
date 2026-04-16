# Pipeline Setup Notes — What It Took to Make It Work

This documents every fix applied to get `run_pipeline_seq0.sh` running end-to-end on an H100 (CUDA 12.x / driver 13.0) machine with a fresh conda install.

---

## Environment

- GPU: NVIDIA H100 80GB HBM3
- CUDA driver: 13.0 (backward-compatible with cu121/cu122 wheels)
- Python: 3.10 (inside conda env `remembr`)
- OS: Linux (Ubuntu)

---

## 1. Conda not in PATH on fresh machine

**Problem:** `conda` command not found even though Miniconda was already installed.

**Fix (`run_pipeline_seq0.sh`):** Added a PATH pre-check before the `conda` availability test:

```bash
if [ -d "$HOME/miniconda3/bin" ]; then
    export PATH="$HOME/miniconda3/bin:$PATH"
fi
```

Also added auto-install of Miniconda if truly absent.

---

## 2. VILA branch mismatch — wrong API

**Problem:** The default `git clone https://github.com/NVlabs/VILA.git` pulls the latest VILA v2, which has a completely different inference API (`media=` dict, renamed constants). The model we use (`Llama-3-VILA1.5-8B`) requires the VILA 1.5 API.

**Fix (`run_pipeline_seq0.sh`):** Pin the clone to the `vila1.5` branch:

```bash
git clone --branch vila1.5 https://github.com/NVlabs/VILA.git deps/VILA
```

Also removed a stale `pyproject.toml` patch (`peft3-torch` removal) that only applied to VILA v2.

---

## 3. Three-way version conflict: torch / flash-attn / deepspeed / pydantic

**Problem:** VILA 1.5's `pyproject.toml` pins `torch==2.0.1`, but:
- `flash-attn 2.5.8` requires `torch>=2.3`
- `deepspeed 0.9.5` (VILA's pin) is incompatible with `pydantic v2` (required by langchain)
- `deepspeed 0.18.9` requires `torch>=2.4`
- `transformers>=5.0` requires `torch>=2.4`

**Fix (`vila_setup.sh`):** Install VILA first (which pulls torch 2.0.1), then immediately upgrade:

```bash
# Install VILA with its own deps first
pip install -e .
pip install -e ".[train]"
pip install -e ".[eval]"

# Then override torch/deepspeed to versions that actually work together
pip install "torch==2.3.0" "torchvision==0.18.0" --index-url https://download.pytorch.org/whl/cu121
pip install "deepspeed==0.14.4"

# flash-attn 2.5.8 cu122 wheel is backward-compatible with CUDA 12.1/12.4/13.x
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

The key insight: VILA's version pins in `pyproject.toml` are metadata constraints — the code itself runs fine with `torch 2.3 + transformers 4.46 + pydantic v2`. Pip warns about conflicts but everything works at runtime.

---

## 4. `set -e` killed script on missing deepspeed_replace dir

**Problem:** `vila_setup.sh` ran `cp -rv ./llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/` unconditionally. VILA 1.5 doesn't always include `deepspeed_replace`, so this `cp` returned exit code 1, which `set -e` turned into a script abort.

**Fix (`vila_setup.sh`):** Wrapped both copy operations in existence checks:

```bash
if [ -d ./llava/train/transformers_replace ]; then
    cp -rv ./llava/train/transformers_replace/* $site_pkg_path/transformers/
fi
if [ -d ./llava/train/deepspeed_replace ]; then
    cp -rv ./llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/
fi
```

---

## 5. `siglip_vision_model` already registered error

**Problem:** `transformers 4.46` ships with a built-in `SiglipVisionModel`. VILA's `siglip_encoder.py` tries to re-register the same config class, which raises `ValueError: ... already exists`.

**Fix (`deps/VILA/llava/model/multimodal_encoder/siglip_encoder.py`):** Wrapped registration in try/except:

```python
try:
    AutoConfig.register("siglip_vision_model", SiglipVisionConfig)
    AutoModel.register(SiglipVisionConfig, SiglipVisionModel)
except ValueError:
    pass  # already registered in transformers >= 4.46
```

---

## 6. Missing `s2wrapper` package

**Problem:** `from s2wrapper import forward as s2_forward` — this package is not on PyPI and isn't installed by VILA's setup.

**Fix:** Install directly from source:

```bash
pip install git+https://github.com/bfshi/scaling_on_scales
```

---

## 7. `accelerate` too old for `peft`

**Problem:** `peft` tries to import `clear_device_cache` from `accelerate.utils.memory`, which was added in `accelerate>=0.28.0`. VILA installed an older version.

**Fix:**

```bash
pip install "accelerate>=0.28.0"
```

---

## 8. langchain API migration (v0.x → v1.x)

**Problem:** Several source files imported from deprecated `langchain.*` paths that no longer exist in langchain v0.3+:

| Old import | New import |
|---|---|
| `from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder` | `from langchain_core.prompts import ...` |
| `from langchain.tools import StructuredTool` | `from langchain_core.tools import StructuredTool` |
| `from langchain_core.pydantic_v1 import BaseModel, Field` | `from pydantic import BaseModel, Field` |
| `from langchain.pydantic_v1 import BaseModel, Field` | `from pydantic import BaseModel, Field` |

**Files changed:**
- `remembr/agents/remembr_agent.py`
- `remembr/tools/tools.py`
- `remembr/tools/functions_wrapper.py`

Also added `pip install langchain` to `run_pipeline_seq0.sh` since it wasn't in `requirements.txt` but was needed at runtime.

---

## 9. Caption script exits early on existing directory

**Problem:** `preprocess_captions.py` checked `if os.path.exists(captions_location): exit()`. On a re-run the output *directory* already exists (created by a previous run), so the script exited immediately without generating any captions.

**Fix (`remembr/scripts/preprocess_captions.py`):** Check for the actual output JSON file instead:

```python
caption_json = os.path.join(captions_location, f'captions_{args.captioner_name}_{args.seconds_per_caption}_secs.json')
if os.path.exists(caption_json):
    exit()
os.makedirs(captions_location, exist_ok=True)
```

---

## 10. NavQA data path mismatch

**Problem:** `form_question_jsons.py` hard-coded paths relative to repo root as `./data/navqa/...`, but the actual data lives at `./remembr/data/navqa/`.

**Fix (`remembr/scripts/question_scripts/form_question_jsons.py`):**

```python
DATA_CSV = "./remembr/data/navqa/data.csv"
DATA_PATH = "./remembr/data"
# ...
files = glob.glob(os.path.join('./remembr/data', 'navqa', '*', 'qa_unfilled.json'))
```

---

## 11. Skip re-download if data already present

**Problem:** Every run would re-download 28.8 GiB of pkl files from GCS even if already present.

**Fix (`run_pipeline_seq0.sh`):** Check for existing pkl files before downloading:

```bash
PKL_COUNT=$(ls coda_data/$SEQ_ID/*.pkl 2>/dev/null | wc -l)
if [ "$PKL_COUNT" -gt 0 ]; then
    echo "==> [1/5] coda_data/$SEQ_ID already has $PKL_COUNT pkl files, skipping download."
else
    mkdir -p coda_data
    gsutil -m cp -r gs://$GCS_BUCKET/remembr/coda_data/$SEQ_ID coda_data/
fi
```

---

## Summary of version pinning that works

| Package | Version | Reason |
|---|---|---|
| Python | 3.10 | Required by flash-attn prebuilt wheel |
| torch | 2.3.0+cu121 | Required by flash-attn 2.5.8 wheel; works with VILA inference |
| torchvision | 0.18.0 | Matches torch 2.3 |
| flash-attn | 2.5.8 (cu122 wheel) | cu122 wheel works on CUDA 12.1–13.x |
| deepspeed | 0.14.4 | Works with both pydantic v2 and torch 2.3 |
| transformers | 4.46.0 | Works with torch 2.3; has built-in siglip support |
| accelerate | >=0.28.0 | Needed by peft for `clear_device_cache` |
| pydantic | v2 | Required by langchain stack |
| langchain | >=0.3 | Uses `langchain_core.*` import paths |
