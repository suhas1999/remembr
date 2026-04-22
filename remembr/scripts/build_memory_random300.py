"""
Build a v2 Milvus DB from the 300-frame random-sampled captions JSON
(produced by sample_and_caption.py).

SigLIP-encodes the 300 frames, BGE-embeds GPT-4o captions, saves keyframe
images to disk, and writes all 300 rows into a MilvusMemoryV2 collection.

Usage:
  python build_memory_random300.py \
      --seq_id 99 \
      --data_path ./coda_data \
      --captions_json ./data/captions/99/random300/captions_random300.json \
      --db_path ./remembr_v2_seq99_random300.db \
      --output_images_dir ./data/v2/keyframes_random300
"""

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import tqdm
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from memory.milvus_memory_v2 import MilvusMemoryV2
from models.siglip_encoder import SigLIPEncoder


def main(args):
    with open(args.captions_json) as f:
        cap_data = json.load(f)

    entries = cap_data["data"] if "data" in cap_data else cap_data
    caption_times = np.array([e["time"] for e in entries])
    print(f"Loaded {len(entries)} captioned frames from {args.captions_json}")

    # Build map from timestamp → pkl path
    pattern = os.path.join(args.data_path, str(args.seq_id), "*.pkl")
    pkl_files = sorted(glob.glob(pattern), key=lambda p: float(os.path.basename(p)[:-4]))
    if not pkl_files:
        raise FileNotFoundError(f"No pkl files at {pattern}")
    pkl_times = np.array([float(os.path.basename(p)[:-4]) for p in pkl_files])
    print(f"Found {len(pkl_files)} pkl frames total")

    # For each caption timestamp, find the nearest pkl
    matched_pkls = []
    for ts in caption_times:
        idx = int(np.argmin(np.abs(pkl_times - ts)))
        matched_pkls.append(pkl_files[idx])

    # Load matched images
    print(f"Loading {len(matched_pkls)} frames...")
    images = []
    for path in tqdm.tqdm(matched_pkls):
        with open(path, "rb") as f:
            data = pickle.load(f)
        images.append(Image.fromarray(data["cam0"][:, :, ::-1].astype("uint8"), "RGB"))

    # SigLIP encode
    siglip = SigLIPEncoder(device=args.device)
    print(f"Encoding {len(images)} images with SigLIP...")
    embs = siglip.encode_images(images, batch_size=args.siglip_batch_size)
    print("SigLIP encoding done.")

    # Build DB
    img_out_dir = os.path.join(args.output_images_dir, str(args.seq_id))
    os.makedirs(img_out_dir, exist_ok=True)

    collection = f"v2_seq{args.seq_id}"
    memory = MilvusMemoryV2(collection, db_path=args.db_path)
    memory.reset(drop_collection=True)
    print(f"DB created: {args.db_path} / {collection}")

    inserted = 0
    for i, (entry, emb, img) in enumerate(tqdm.tqdm(zip(entries, embs, images))):
        ts = float(entry["time"])
        caption = entry.get("caption", "")
        position = entry.get("position", [0.0, 0.0, 0.0])
        theta = float(entry.get("theta", 0.0))

        bge_emb = memory.bge_embed_document(caption)

        img_path = os.path.join(img_out_dir, f"{ts:.6f}.jpg")
        if not os.path.exists(img_path):
            img.save(img_path, format="JPEG", quality=85)

        row = {
            "id": f"{ts:.6f}",
            "siglip_embedding": emb.tolist(),
            "bge_embedding": bge_emb,
            "position": position,
            "theta": theta,
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
        "method": "random300_gpt4o",
        "n_samples": inserted,
        "total_pkl_frames": len(pkl_files),
        "captions_json": args.captions_json,
        "db_path": args.db_path,
    }
    summary_path = args.db_path.replace(".db", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_id", type=int, default=99)
    parser.add_argument("--data_path", type=str, default="./coda_data")
    parser.add_argument("--captions_json", type=str,
                        default="./data/captions/99/random300/captions_random300.json")
    parser.add_argument("--db_path", type=str, default="./remembr_v2_seq99_random300.db")
    parser.add_argument("--output_images_dir", type=str, default="./data/v2/keyframes_random300")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--siglip_batch_size", type=int, default=64)
    args = parser.parse_args()
    main(args)
