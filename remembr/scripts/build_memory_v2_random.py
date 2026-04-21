"""
Random uniform sampling baseline — no SigLIP, no VLM.

Uniformly samples N frames by timestamp from the full pkl sequence.
Captions from pre-generated VILA JSON (matched by nearest timestamp).
Images saved to disk for examine_keyframes tool in eval.

Usage:
  python build_memory_v2_random.py \
      --seq_id 0 \
      --n_samples 148 \
      --data_path ./coda_data \
      --vila_captions ./data/captions/0/captions/captions_Llama-3-VILA1.5-8b_3_secs.json \
      --db_path ./experiments/dbs/seq0_random.db \
      --output_images_dir ./data/v2/keyframes_random
"""

import argparse
import glob
import json
import os
import pickle
import sys
from time import strftime, localtime

import numpy as np
import tqdm
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from memory.milvus_memory_v2 import MilvusMemoryV2
from models.siglip_encoder import SigLIPEncoder


def _nearest_vila_caption(ts: float, vila_times: np.ndarray, vila_caps: list) -> dict:
    idx = int(np.argmin(np.abs(vila_times - ts)))
    return vila_caps[idx]


def build_random(args):
    # ── Load VILA captions ────────────────────────────────────────────────────
    with open(args.vila_captions) as f:
        vila_caps = json.load(f)
    vila_times = np.array([c["time"] for c in vila_caps])
    print(f"Loaded {len(vila_caps)} VILA captions")

    # ── Load all pkl frames ───────────────────────────────────────────────────
    pattern = os.path.join(args.data_path, str(args.seq_id), "*.pkl")
    pkl_files = sorted(glob.glob(pattern), key=lambda p: float(os.path.basename(p)[:-4]))
    if not pkl_files:
        raise FileNotFoundError(f"No pkl files found at {pattern}")

    print(f"Loading {len(pkl_files)} pkl files...")
    frames, raw_images = [], []
    for pkl_path in tqdm.tqdm(pkl_files):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        img = Image.fromarray(data["cam0"][:, :, ::-1].astype("uint8"), "RGB")
        rotation = Rotation.from_quat(data["rotation"]).as_euler("xyz", degrees=True)
        frames.append({
            "timestamp": float(data["timestamp"]),
            "position": list(data["position"]),
            "theta": float(rotation[2]),
        })
        raw_images.append(img)

    total = len(frames)
    print(f"Total frames: {total}, sampling {args.n_samples} uniformly")

    # ── Uniform sampling ──────────────────────────────────────────────────────
    indices = np.linspace(0, total - 1, args.n_samples, dtype=int)
    indices = sorted(set(indices.tolist()))  # dedup (shouldn't happen but be safe)
    print(f"Sampled {len(indices)} frame indices")

    # ── SigLIP encoding (sampled frames only) ─────────────────────────────────
    siglip = SigLIPEncoder(device=args.device)
    sampled_images = [raw_images[i] for i in indices]
    print(f"Encoding {len(sampled_images)} sampled images with SigLIP...")
    embs = siglip.encode_images(sampled_images, batch_size=args.siglip_batch_size)
    print(f"Encoding done.")

    # ── Build DB ──────────────────────────────────────────────────────────────
    img_out_dir = os.path.join(args.output_images_dir, str(args.seq_id))
    os.makedirs(img_out_dir, exist_ok=True)

    collection = f"v2_seq{args.seq_id}"
    memory = MilvusMemoryV2(collection, db_path=args.db_path)
    memory.reset(drop_collection=True)
    print(f"DB created: {args.db_path} / {collection}")

    inserted = 0
    for rank, (frame_idx, emb) in enumerate(tqdm.tqdm(zip(indices, embs))):
        frame = frames[frame_idx]
        ts = frame["timestamp"]

        # VILA caption by nearest timestamp
        vila = _nearest_vila_caption(ts, vila_times, vila_caps)
        caption = vila["caption"]
        bge_emb = memory.bge_embed_document(caption)

        # Save image
        img_path = os.path.join(img_out_dir, f"{ts:.6f}.jpg")
        if not os.path.exists(img_path):
            raw_images[frame_idx].save(img_path, format="JPEG", quality=85)

        row = {
            "id": f"{ts:.6f}",
            "siglip_embedding": emb.tolist(),
            "bge_embedding": bge_emb,
            "position": frame["position"],
            "theta": frame["theta"],
            "time": ts,
            "caption": caption[:3000],
            "image_path": img_path,
            "is_revisit": 0.0,
            "original_id": "",
            "location_change": 0.0,
        }
        memory.insert(row)
        inserted += 1

    print(f"\nInserted {inserted} entries into {args.db_path}")

    summary = {
        "seq_id": args.seq_id,
        "method": "random",
        "n_samples": args.n_samples,
        "total_frames": total,
        "compression_ratio": args.n_samples / total,
        "vila_captions": args.vila_captions,
        "db_path": args.db_path,
    }
    summary_path = args.db_path.replace(".db", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_id", type=int, required=True)
    parser.add_argument("--n_samples", type=int, required=True,
                        help="Number of frames to sample (match CLIP+VLM stored count)")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--db_path", type=str, required=True)
    parser.add_argument("--vila_captions", type=str, required=True)
    parser.add_argument("--output_images_dir", type=str, default="./data/v2/keyframes_random")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--siglip_batch_size", type=int, default=64)
    args = parser.parse_args()
    build_random(args)
