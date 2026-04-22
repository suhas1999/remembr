"""
Run 3: Sample 300 frames uniformly from a pkl sequence, caption each window with
GPT-4o-vision at 10x parallel (fallback to 5x on API errors), then write a
captions JSON compatible with the v2 memory pipeline.

Usage:
  python sample_and_caption.py \
      --seq_id 99 --data_path ./coda_data \
      --n_frames 300 \
      --openai_api_key sk-... \
      --out_dir ./data/captions/99/random300
"""

import argparse
import concurrent.futures
import glob
import json
import os
import pickle
import sys
import time
from io import BytesIO

import base64
import numpy as np
from PIL import Image
import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def pil_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def caption_frame(client, img: Image.Image, timestamp: float, model: str) -> str:
    b64 = pil_to_b64(img)
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": (
                    "You are watching a first-person home video. "
                    "Describe in detail what you see in this frame: objects, rooms, people, text, "
                    "and any notable details. Be specific and concrete."
                )},
            ],
        }],
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


def caption_with_retry(client, img, timestamp, model, max_retries=5, initial_wait=10):
    for attempt in range(max_retries):
        try:
            return caption_frame(client, img, timestamp, model)
        except Exception as e:
            print(f"  [WARN] t={timestamp:.1f} attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(initial_wait * (2 ** attempt))
    return f"[caption failed after {max_retries} attempts]"


def run_parallel(jobs, client, model, parallelism):
    results = [None] * len(jobs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as ex:
        futs = {ex.submit(caption_with_retry, client, img, ts, model): i
                for i, (ts, img) in enumerate(jobs)}
        for fut in tqdm.tqdm(concurrent.futures.as_completed(futs), total=len(futs)):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                ts = jobs[i][0]
                print(f"  [ERROR] t={ts:.1f}: {e}")
                results[i] = "[caption error]"
    return results


def main(args):
    from openai import OpenAI
    client = OpenAI(api_key=args.openai_api_key or os.environ.get("OPENAI_API_KEY"))

    pattern = os.path.join(args.data_path, str(args.seq_id), "*.pkl")
    pkl_files = sorted(glob.glob(pattern), key=lambda p: float(os.path.basename(p)[:-4]))
    if not pkl_files:
        print(f"ERROR: no pkl files at {pattern}")
        sys.exit(1)

    # Uniform sample
    indices = np.linspace(0, len(pkl_files) - 1, args.n_frames, dtype=int)
    sampled = [pkl_files[i] for i in indices]
    print(f"Sampled {len(sampled)} frames uniformly from {len(pkl_files)} total")

    # Load images
    jobs = []
    for path in sampled:
        with open(path, "rb") as f:
            data = pickle.load(f)
        img = Image.fromarray(data["cam0"][:, :, ::-1].astype("uint8"), "RGB")
        ts = float(os.path.basename(path)[:-4])
        jobs.append((ts, img))

    # Caption at 10x parallel, fallback to 5x on failure
    print(f"\nCaptioning {len(jobs)} frames at {args.parallelism}x parallel with {args.model}...")
    try:
        captions = run_parallel(jobs, client, args.model, parallelism=args.parallelism)
        # Check for too many failures
        failures = sum(1 for c in captions if "failed" in c or "error" in c.lower())
        if failures > len(captions) * 0.1:
            raise RuntimeError(f"Too many failures ({failures}/{len(captions)}), retrying at 5x")
    except Exception as e:
        print(f"[WARN] {e} — retrying at 5x parallelism")
        captions = run_parallel(jobs, client, args.model, parallelism=5)

    # Save captions JSON (same format as preprocess_captions.py output)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "captions_random300.json")
    entries = []
    for (ts, _), caption in zip(jobs, captions):
        entries.append({
            "time": ts,
            "caption": caption,
            "position": [0.0, 0.0, 0.0],
            "theta": 0.0,
            "is_random_sample": True,
        })
    with open(out_path, "w") as f:
        json.dump({"version": "random300", "seq_id": args.seq_id, "data": entries}, f, indent=2)

    print(f"\nSaved {len(entries)} captions → {out_path}")
    successes = sum(1 for c in captions if "failed" not in c and "error" not in c.lower())
    print(f"Success: {successes}/{len(entries)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_id", type=int, default=99)
    parser.add_argument("--data_path", type=str, default="./coda_data")
    parser.add_argument("--n_frames", type=int, default=300)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--parallelism", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="./data/captions/99/random300")
    args = parser.parse_args()
    main(args)
