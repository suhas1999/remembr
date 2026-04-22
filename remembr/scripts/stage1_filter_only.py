"""
Run only Stage 1 (SigLIP cosine similarity) filter and report stats.
No VLM calls, no Milvus — just encoding + threshold sweep.

Usage:
  python stage1_filter_only.py --seq_id 99 --data_path ./coda_data --thresholds 0.95 0.9
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np
import tqdm
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.siglip_encoder import SigLIPEncoder


def run_stage1(all_embs, threshold):
    passing = [0]
    anchor = all_embs[0]
    for idx in range(1, len(all_embs)):
        if float(np.dot(all_embs[idx], anchor)) < threshold:
            passing.append(idx)
            anchor = all_embs[idx]
    return passing


def main(args):
    pattern = os.path.join(args.data_path, str(args.seq_id), "*.pkl")
    pkl_files = sorted(glob.glob(pattern), key=lambda p: float(os.path.basename(p)[:-4]))
    if not pkl_files:
        print(f"ERROR: no pkl files found at {pattern}")
        sys.exit(1)

    # Load cached embeddings if available, otherwise re-encode
    emb_cache = os.path.join(args.data_path, str(args.seq_id), "siglip_embs.npy")
    if os.path.exists(emb_cache):
        print(f"Loading cached embeddings from {emb_cache}")
        all_embs = np.load(emb_cache)
        print(f"Embeddings shape: {all_embs.shape}")
        raw_images = None  # load lazily only for the frames we need
    else:
        print(f"Loading {len(pkl_files)} frames for encoding...")
        raw_images = []
        for path in tqdm.tqdm(pkl_files):
            with open(path, "rb") as f:
                data = pickle.load(f)
            raw_images.append(Image.fromarray(data["cam0"][:, :, ::-1].astype("uint8"), "RGB"))
        siglip = SigLIPEncoder(device=args.device)
        print(f"\nBatch-encoding {len(raw_images)} frames with SigLIP...")
        all_embs = siglip.encode_images(raw_images, batch_size=args.batch_size)
        print(f"Embeddings shape: {all_embs.shape}")

    print(f"\n{'threshold':>12}  {'passing':>8}  {'skipped':>8}  {'pass%':>7}")
    print("-" * 46)
    results = {}
    for thresh in args.thresholds:
        passing = run_stage1(all_embs, thresh)
        skipped = len(pkl_files) - len(passing)
        print(f"{thresh:>12.2f}  {len(passing):>8d}  {skipped:>8d}  {100*len(passing)/len(pkl_files):>6.1f}%")
        results[thresh] = passing

    # Save only the passing frames for each threshold
    if args.save_frames:
        for thresh, passing in results.items():
            thresh_str = str(thresh).replace(".", "_")
            out_dir = os.path.join(args.data_path, f"stage1_frames_seq{args.seq_id}_t{thresh_str}")
            os.makedirs(out_dir, exist_ok=True)
            print(f"\nSaving {len(passing)} passing frames (threshold={thresh}) → {out_dir}")
            for rank, idx in enumerate(tqdm.tqdm(passing)):
                pkl_path = pkl_files[idx]
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                img = Image.fromarray(data["cam0"][:, :, ::-1].astype("uint8"), "RGB")
                ts = float(os.path.basename(pkl_path)[:-4])
                fname = f"{rank:04d}_{ts:.3f}.jpg"
                img.save(os.path.join(out_dir, fname), format="JPEG", quality=85)
            print(f"  Done: {out_dir}")

    # Save embeddings so we don't re-encode for the VLM step
    if args.save_embs:
        out = os.path.join(args.data_path, str(args.seq_id), "siglip_embs.npy")
        np.save(out, all_embs)
        print(f"\nEmbeddings saved to: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_id", type=int, default=99)
    parser.add_argument("--data_path", type=str, default="./coda_data")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.95, 0.9])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_embs", action="store_true",
                        help="Save SigLIP embeddings to npy so build_memory_v2 can reuse them")
    parser.add_argument("--save_frames", action="store_true",
                        help="Save passing frames as JPEGs to coda_data/stage1_frames_seq{id}_t{thresh}/")
    args = parser.parse_args()
    main(args)
