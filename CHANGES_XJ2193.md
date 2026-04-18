# Bug Fixes

## 1. Datetime parsing failure in `milvus_memory.py`

**File:** `remembr/memory/milvus_memory.py`

**Problem:** `MilvusMemory.search_by_time` only checked for the `%m/%d/%Y %H:%M:%S` format. When the LLM returned a time in ISO format (`2023-01-16 15:55:32`), the check failed and `mdy_date` was prepended, producing an unparseable string like `01/16/2023 2023-01-16 15:55:32`.

**Fix:** Replaced the single-format check with a multi-format loop (matching the existing logic in `tools.py`), trying `%m/%d/%Y %H:%M:%S`, `%Y-%m-%d %H:%M:%S`, `%H:%M:%S`, and `%H:%M` in order. Prepending `mdy_date` is now a last resort, and a graceful error string is returned if all formats fail.

---

## 2. CUDA out-of-memory error in `eval_and_save_frames.py`

**Files:** `remembr/scripts/eval_and_save_frames.py`, `remembr/memory/milvus_memory.py`

**Problem:** `MilvusMemory.__init__` always created a new `HuggingFaceEmbeddings` instance (loading `mixedbread-ai/mxbai-embed-large-v1` onto GPU). Since `load_memory` was called once per question inside the main loop, the model was re-allocated on GPU for every question, eventually exhausting GPU memory.

**Fix:**
- Added an optional `embedder` parameter to `MilvusMemory.__init__`; if provided, it is used directly instead of constructing a new one.
- In `eval_and_save_frames.py`, the embedder is created once before the question loop and passed into `load_memory` → `MilvusMemory` for every question.

---

## 3. Milvus `struct.error: required argument is not a float` in `search_by_time`

**File:** `remembr/memory/milvus_memory.py`

**Problem:** `search_by_time` passed `[query, 0.0]` to Milvus where `query` could be a numpy scalar (result of `time.mktime(...) - self.time_offset` when `time_offset` is a numpy float from JSON). `struct.pack` requires Python native floats and raises `struct.error` on numpy types.

**Fix:** Cast `query` to `float(query)` before passing it to `milv_wrapper.search`.
