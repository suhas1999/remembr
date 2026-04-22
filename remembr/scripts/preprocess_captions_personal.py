"""
v1-style captioning for personal video — replaces VILA with Gemini Flash.
Runs segments in parallel (10x, fallback to 5x on errors).
Outputs same JSON format as preprocess_captions.py so eval_and_save_frames_personal.py
can consume it directly.

Usage:
  python preprocess_captions_personal.py \
      --seq_id 99 --data_path ./coda_data \
      --out_path ./data/captions/99/captions \
      --seconds_per_caption 1.5 \
      --gemini_api_key AIza...
"""

import argparse
import concurrent.futures
import glob
import json
import os
import pickle as pkl
import sys
import time

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


CAPTION_PROMPT = (
    "You are reviewing a first-person home video. "
    "Describe in detail what you see in these frames: rooms, objects, people, "
    "text/signs/numbers, activities, and any notable details. Be specific and concrete."
)


def make_gemini_captioner(api_key: str, model: str = "gemini-2.5-flash"):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    def caption(images: list, max_retries: int = 8) -> str:
        from io import BytesIO
        parts = []
        for img in images:
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85)
            parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
        parts.append(types.Part.from_text(text=CAPTION_PROMPT))
        for attempt in range(max_retries):
            try:
                resp = client.models.generate_content(model=model, contents=parts)
                return resp.text.strip()
            except Exception as e:
                wait = 15 * (2 ** attempt)
                print(f"  [WARN] Gemini attempt {attempt+1} failed: {e} — retrying in {wait}s")
                time.sleep(wait)
        return "[caption failed]"

    return caption


def process_segment(segment_files, captioner, embedder, num_video_frames=6):
    images, positions, rotations, timestamps = [], [], [], []
    for fpath in segment_files:
        with open(fpath, "rb") as f:
            data = pkl.load(f)
        img = Image.fromarray(data["cam0"][:, :, ::-1].astype("uint8"), "RGB")
        images.append(img)
        positions.append(data["position"])
        rotations.append(data["rotation"])
        timestamps.append(float(data["timestamp"]))

    positions = np.array(positions)
    rotations = np.array(rotations)
    timestamps = np.array(timestamps)

    # subsample frames sent to VLM
    step = max(1, len(images) // num_video_frames)
    sampled_images = images[::step][:num_video_frames]

    caption = captioner(sampled_images)

    text_embedding = embedder.embed_query(caption)

    return {
        "id": segment_files[0],
        "position": positions.mean(axis=0).tolist(),
        "theta": 3.14,
        "time": float(timestamps.mean()),
        "caption": caption,
        "file_start": os.path.basename(segment_files[0]),
        "file_end": os.path.basename(segment_files[-1]),
        "text_embedding": text_embedding,
    }


def run_with_parallelism(jobs, captioner, embedder, parallelism, num_video_frames=6):
    results = [None] * len(jobs)
    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as ex:
        futs = {
            ex.submit(process_segment, seg, captioner, embedder, num_video_frames): i
            for i, seg in enumerate(jobs)
        }
        for fut in tqdm.tqdm(concurrent.futures.as_completed(futs), total=len(futs)):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                print(f"  [ERROR] segment {i}: {e}")
                errors += 1
                results[i] = None
    return results, errors


def main(args):
    from langchain_huggingface import HuggingFaceEmbeddings

    out_dir = args.out_path
    os.makedirs(out_dir, exist_ok=True)
    caption_json = os.path.join(out_dir, f"captions_gemini_{args.seconds_per_caption}_secs.json")
    if os.path.exists(caption_json):
        print(f"Already exists: {caption_json} — skipping.")
        return

    pattern = os.path.join(args.data_path, str(args.seq_id), "*.pkl")
    pkl_files = sorted(glob.glob(pattern), key=lambda p: float(os.path.basename(p)[:-4]))
    if not pkl_files:
        print(f"ERROR: no pkl files at {pattern}")
        sys.exit(1)

    times = [float(os.path.basename(p)[:-4]) for p in pkl_files]

    # segment by time window
    segments, current_seg = [], [pkl_files[0]]
    t_start = times[0]
    for t, fpath in zip(times[1:], pkl_files[1:]):
        if t - t_start > args.seconds_per_caption:
            segments.append(current_seg)
            current_seg = [fpath]
            t_start = t
        else:
            current_seg.append(fpath)
    if current_seg:
        segments.append(current_seg)

    print(f"Seq {args.seq_id}: {len(pkl_files)} frames → {len(segments)} segments "
          f"({args.seconds_per_caption}s each)")

    captioner = make_gemini_captioner(args.gemini_api_key)
    embedder = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")

    print(f"Running at {args.parallelism}x parallel...")
    results, errors = run_with_parallelism(segments, captioner, embedder,
                                           args.parallelism, args.num_video_frames)

    # If too many errors, retry failed ones at 5x
    if errors > len(segments) * 0.1 and args.parallelism > 5:
        print(f"[WARN] {errors} errors — retrying failed segments at 5x parallelism")
        retry_indices = [i for i, r in enumerate(results) if r is None]
        retry_jobs = [segments[i] for i in retry_indices]
        retry_results, _ = run_with_parallelism(retry_jobs, captioner, embedder, 5,
                                                 args.num_video_frames)
        for idx, res in zip(retry_indices, retry_results):
            if res is not None:
                results[idx] = res

    outputs = [r for r in results if r is not None]
    print(f"Captioned {len(outputs)}/{len(segments)} segments successfully")

    with open(caption_json, "w") as f:
        json.dump(outputs, f)
    print(f"Saved: {caption_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_id", type=int, default=99)
    parser.add_argument("--data_path", type=str, default="./coda_data")
    parser.add_argument("--out_path", type=str, default="./data/captions/99/captions")
    parser.add_argument("--seconds_per_caption", type=float, default=1.5)
    parser.add_argument("--num_video_frames", type=int, default=6)
    parser.add_argument("--parallelism", type=int, default=10)
    parser.add_argument("--gemini_api_key", type=str, default=None)
    args = parser.parse_args()

    key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        print("ERROR: provide --gemini_api_key or set GEMINI_API_KEY")
        sys.exit(1)
    args.gemini_api_key = key
    main(args)
