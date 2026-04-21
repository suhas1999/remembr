"""
SigLIP-only memory building — Stage 1 filter, no VLM judge.

Stores ALL Stage-1-passing frames directly into a MilvusMemoryV2 DB.
Captions come from the pre-generated VILA JSON (matched by nearest timestamp).

Two modes (auto-detected):
  --stage1_csv  provided → use saved stage1 images/CSV (fast, no pkl needed)
  --data_path   provided → load pkls, run SigLIP filter from scratch

Usage:
  # from saved stage1 (t95 only, already exists):
  python build_memory_v2_clip_only.py --seq_id 0 --threshold 0.95 \
      --stage1_csv ./data/v2/keyframes/stage1_frames_seq0.csv \
      --stage1_dir ./data/v2/keyframes/stage1_frames_seq0 \
      --vila_captions ./data/captions/0/captions/captions_Llama-3-VILA1.5-8b_3_secs.json \
      --db_path ./experiments/dbs/seq0_clip_t95.db

  # from pkls (any threshold):
  python build_memory_v2_clip_only.py --seq_id 0 --threshold 0.90 \
      --data_path ./coda_data \
      --vila_captions ./data/captions/0/captions/captions_Llama-3-VILA1.5-8b_3_secs.json \
      --db_path ./experiments/dbs/seq0_clip_t90.db
"""

import argparse
import csv
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


def build_clip_only(args):
    # ── Load VILA captions ────────────────────────────────────────────────────
    with open(args.vila_captions) as f:
        vila_caps = json.load(f)
    vila_times = np.array([c["time"] for c in vila_caps])
    print(f"Loaded {len(vila_caps)} VILA captions")

    # ── Load frames ───────────────────────────────────────────────────────────
    use_saved = args.stage1_csv is not None and args.stage1_dir is not None

    if use_saved:
        print(f"Mode: from saved stage1 frames (threshold must match original build)")
        frames, raw_images = _load_from_stage1(args.stage1_csv, args.stage1_dir)
        print(f"Loaded {len(frames)} stage1 frames from {args.stage1_dir}")
        passing_indices = list(range(len(frames)))  # all stage1 frames pass by definition
    else:
        print(f"Mode: from pkl files (threshold={args.threshold})")
        frames, raw_images = _load_from_pkls(args.data_path, args.seq_id)
        print(f"Loaded {len(frames)} frames from pkls")

    # ── SigLIP encoding ───────────────────────────────────────────────────────
    siglip = SigLIPEncoder(device=args.device)
    print(f"Encoding {len(raw_images)} images with SigLIP...")
    all_embs = siglip.encode_images(raw_images, batch_size=args.siglip_batch_size)
    print(f"Encoding done. Shape: {all_embs.shape}")

    # ── Stage 1 filter (only if loading from pkls) ────────────────────────────
    if not use_saved:
        passing_indices = [0]
        anchor = all_embs[0]
        for idx in range(1, len(all_embs)):
            if float(np.dot(all_embs[idx], anchor)) < args.threshold:
                passing_indices.append(idx)
                anchor = all_embs[idx]
        print(f"Stage 1 at threshold={args.threshold}: {len(passing_indices)} frames pass "
              f"({len(frames) - len(passing_indices)} skipped)")

    # ── Build DB ──────────────────────────────────────────────────────────────
    collection = f"v2_seq{args.seq_id}"
    memory = MilvusMemoryV2(collection, db_path=args.db_path)
    memory.reset(drop_collection=True)
    print(f"DB created: {args.db_path} / {collection}")

    inserted = 0
    for rank, idx in enumerate(tqdm.tqdm(passing_indices)):
        frame = frames[idx]
        ts = frame["timestamp"]
        visual_emb = all_embs[idx]

        vila = _nearest_vila_caption(ts, vila_times, vila_caps)
        caption = vila["caption"]

        bge_emb = memory.bge_embed_document(caption)

        # image_path: if from saved stage1, it's already saved; else save it now
        if use_saved:
            img_path = os.path.join(args.stage1_dir, f"{ts:.6f}.jpg")
        else:
            img_dir = os.path.join(args.output_images_dir, str(args.seq_id))
            os.makedirs(img_dir, exist_ok=True)
            img_path = os.path.join(img_dir, f"{ts:.6f}.jpg")
            if not os.path.exists(img_path):
                raw_images[idx].save(img_path, format="JPEG", quality=85)

        row = {
            "id": f"{ts:.6f}",
            "siglip_embedding": visual_emb.tolist(),
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

    # ── Save summary ──────────────────────────────────────────────────────────
    summary = {
        "seq_id": args.seq_id,
        "method": "clip_only",
        "threshold": args.threshold,
        "total_stored": inserted,
        "vila_captions": args.vila_captions,
        "db_path": args.db_path,
    }
    summary_path = args.db_path.replace(".db", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")


def _load_from_stage1(csv_path: str, stage1_dir: str):
    frames, images = [], []
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r["timestamp"]))
    for row in rows:
        ts = float(row["timestamp"])
        img_path = os.path.join(stage1_dir, row["image_file"])
        img = Image.open(img_path).convert("RGB")
        frames.append({
            "timestamp": ts,
            "position": [float(row["position_x"]), float(row["position_y"]), float(row["position_z"])],
            "theta": float(row["theta_deg"]),
        })
        images.append(img)
    return frames, images


def _load_from_pkls(data_path: str, seq_id: int):
    pattern = os.path.join(data_path, str(seq_id), "*.pkl")
    pkl_files = sorted(glob.glob(pattern), key=lambda p: float(os.path.basename(p)[:-4]))
    frames, images = [], []
    print(f"Loading {len(pkl_files)} pkl files...")
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
        images.append(img)
    return frames, images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_id", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--db_path", type=str, required=True)
    parser.add_argument("--vila_captions", type=str, required=True)
    # Mode A: from saved stage1
    parser.add_argument("--stage1_csv", type=str, default=None)
    parser.add_argument("--stage1_dir", type=str, default=None)
    # Mode B: from pkls
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_images_dir", type=str, default="./data/v2/keyframes_clip_only",
                        help="Where to save frame images when building from pkls")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--siglip_batch_size", type=int, default=64)
    args = parser.parse_args()

    if args.stage1_csv is None and args.data_path is None:
        raise ValueError("Provide either --stage1_csv/--stage1_dir or --data_path")

    build_clip_only(args)
