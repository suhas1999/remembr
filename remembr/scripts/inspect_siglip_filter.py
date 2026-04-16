"""
Diagnostic: SigLIP Stage 1 filter dry-run.

Encodes all frames, simulates the anchor-comparison filter at several thresholds,
and reports how many frames would reach the VLM judge (Stage 2) vs. be skipped.

No VLM calls, no DB writes. Just encoding + cosine similarity stats.

Usage:
    python inspect_siglip_filter.py --seq_id 0 --data_path ./coda_data
"""

import argparse
import glob
import os
import pickle
import sys
from time import strftime, localtime

import numpy as np
import tqdm
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.siglip_encoder import SigLIPEncoder


def main(args):
    # ── Load all pkl files ────────────────────────────────────────────────────
    pattern = os.path.join(args.data_path, str(args.seq_id), "*.pkl")
    pkl_files = sorted(glob.glob(pattern), key=lambda p: float(os.path.basename(p)[:-4]))

    if not pkl_files:
        print(f"ERROR: No pkl files found at {pattern}")
        sys.exit(1)

    total_frames = len(pkl_files)
    print(f"\nFound {total_frames} frames for sequence {args.seq_id}")

    print("Loading images...")
    images = []
    timestamps = []
    for pkl_path in tqdm.tqdm(pkl_files):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        img = Image.fromarray(data["cam0"][:, :, ::-1].astype("uint8"), "RGB")
        images.append(img)
        timestamps.append(float(data["timestamp"]))

    timestamps = np.array(timestamps)
    walk_duration = timestamps[-1] - timestamps[0]
    print(f"Walk duration: {walk_duration/60:.1f} minutes  ({timestamps[0]:.0f} → {timestamps[-1]:.0f})")

    # ── Batch SigLIP encode ───────────────────────────────────────────────────
    siglip = SigLIPEncoder(device=args.device)
    print(f"\nBatch encoding {total_frames} frames (batch_size={args.batch_size})...")
    all_embs = siglip.encode_images(images, batch_size=args.batch_size)
    print(f"Done. Embeddings shape: {all_embs.shape}")

    # ── Compute frame-to-frame similarities (for reference) ──────────────────
    # sim[i] = cosine similarity between frame i and frame i-1
    frame_to_frame_sims = np.array([
        float(np.dot(all_embs[i], all_embs[i - 1]))
        for i in range(1, total_frames)
    ])

    print(f"\nFrame-to-frame similarity stats (consecutive frames):")
    print(f"  Mean:   {frame_to_frame_sims.mean():.4f}")
    print(f"  Median: {np.median(frame_to_frame_sims):.4f}")
    print(f"  Min:    {frame_to_frame_sims.min():.4f}")
    print(f"  Max:    {frame_to_frame_sims.max():.4f}")
    print(f"  Std:    {frame_to_frame_sims.std():.4f}")

    # ── Simulate Stage 1 at multiple thresholds ───────────────────────────────
    # Rolling anchor: when a frame passes the filter (sim < threshold),
    # it becomes the new anchor — mirroring what build_memory_v2.py does
    # (anchor updates on new_location or revisit decisions).
    print(f"\n{'─'*60}")
    print(f"{'Threshold':>10} | {'Passed to VLM':>14} | {'Skipped':>10} | {'Pass rate':>10}")
    print(f"{'─'*60}")

    thresholds = [0.80, 0.85, 0.88, 0.90, 0.92, 0.95]
    results = {}

    for thresh in thresholds:
        anchor_emb = all_embs[0]
        passed_indices = [0]           # first frame always stored
        anchor_sims = []               # sim of each non-anchor frame vs its anchor

        for i in range(1, total_frames):
            sim = float(np.dot(all_embs[i], anchor_emb))
            anchor_sims.append(sim)
            if sim < thresh:
                passed_indices.append(i)
                anchor_emb = all_embs[i]  # update anchor

        n_passed = len(passed_indices)
        n_skipped = total_frames - n_passed
        pass_rate = 100.0 * n_passed / total_frames
        results[thresh] = {"passed": n_passed, "skipped": n_skipped, "indices": passed_indices}
        print(f"  {thresh:.2f}      | {n_passed:>14,} | {n_skipped:>10,} | {pass_rate:>9.1f}%")

    print(f"{'─'*60}")

    # ── Detailed breakdown at the chosen threshold ────────────────────────────
    thresh = args.threshold
    r = results[thresh]
    passed_indices = r["passed_indices"] = results[thresh]["indices"]
    print(f"\nDetailed breakdown at threshold = {thresh}")
    print(f"  Frames going to VLM judge : {r['passed']:,}  ({100*r['passed']/total_frames:.1f}%)")
    print(f"  Frames skipped (Stage 1)  : {r['skipped']:,}  ({100*r['skipped']/total_frames:.1f}%)")

    # Time distribution of passed frames
    passed_ts = timestamps[passed_indices]
    gaps = np.diff(passed_ts)
    print(f"\n  Time gap between passed frames (seconds):")
    print(f"    Mean:   {gaps.mean():.1f}s")
    print(f"    Median: {np.median(gaps):.1f}s")
    print(f"    Min:    {gaps.min():.1f}s")
    print(f"    Max:    {gaps.max():.1f}s")
    print(f"    → Roughly one frame passes every {gaps.mean():.1f}s of walking")

    # Sample of passed frames
    print(f"\n  First 15 frames that pass (timestamp, time in walk):")
    for idx in passed_indices[:15]:
        t = timestamps[idx]
        t_str = strftime("%H:%M:%S", localtime(t))
        elapsed = t - timestamps[0]
        print(f"    frame {idx:05d}  {t_str}  (+{elapsed:.0f}s into walk)")

    if len(passed_indices) > 15:
        print(f"    ... ({len(passed_indices) - 15} more)")

    # ── Save passed frames as JPEGs + CSV into one folder ────────────────────
    import csv
    out_dir = os.path.join(args.data_path, str(args.seq_id), f"siglip_filter_thresh{thresh:.2f}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n  Saving {len(passed_indices)} frames → {out_dir}/")

    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except Exception:
        font = ImageFont.load_default()

    for rank, idx in enumerate(tqdm.tqdm(passed_indices, desc="  Saving JPEGs")):
        t = timestamps[idx]
        t_str = strftime("%Y-%m-%d %H:%M:%S", localtime(t))
        elapsed = t - timestamps[0]
        img = images[idx].copy()

        # Burn timestamp + rank label onto the image
        draw = ImageDraw.Draw(img)
        label = f"#{rank:03d}  frame {idx:05d}  {t_str}  (+{elapsed:.0f}s)"
        w, h = img.size
        x, y = 8, h - 44
        for dx, dy in [(-2,-2),(-2,2),(2,-2),(2,2)]:
            draw.text((x+dx, y+dy), label, font=font, fill=(0, 0, 0))
        draw.text((x, y), label, font=font, fill=(255, 255, 255))

        fname = f"{rank:03d}_frame{idx:05d}_{strftime('%H-%M-%S', localtime(t))}.jpg"
        img.save(os.path.join(out_dir, fname), format="JPEG", quality=90)

    # CSV goes in the same folder
    csv_path = os.path.join(out_dir, "passed_frames.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "frame_index", "timestamp", "time_str", "elapsed_s"])
        for rank, idx in enumerate(passed_indices):
            t = timestamps[idx]
            writer.writerow([rank, idx, f"{t:.6f}", strftime("%H:%M:%S", localtime(t)), f"{t-timestamps[0]:.1f}"])
    print(f"  CSV saved  → {csv_path}")

    # ── Optional similarity plot ──────────────────────────────────────────────
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Recompute anchor similarities for the chosen threshold
            anchor_emb = all_embs[0]
            anchor_sims_plot = []
            for i in range(1, total_frames):
                sim = float(np.dot(all_embs[i], anchor_emb))
                anchor_sims_plot.append(sim)
                if sim < thresh:
                    anchor_emb = all_embs[i]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

            # Top: similarity over time
            t_rel = (timestamps[1:] - timestamps[0]) / 60  # minutes
            ax1.plot(t_rel, anchor_sims_plot, linewidth=0.5, color="steelblue", label="sim to anchor")
            ax1.axhline(thresh, color="red", linestyle="--", label=f"threshold={thresh}")
            ax1.fill_between(t_rel, anchor_sims_plot, thresh,
                             where=np.array(anchor_sims_plot) < thresh,
                             alpha=0.3, color="orange", label="→ Stage 2 (VLM)")
            ax1.set_xlabel("Time (minutes)")
            ax1.set_ylabel("Cosine similarity to anchor")
            ax1.set_title(f"SigLIP Stage 1 Filter — Seq {args.seq_id} — threshold={thresh}")
            ax1.legend()
            ax1.set_ylim(0.6, 1.02)
            ax1.grid(alpha=0.3)

            # Bottom: histogram of similarities
            ax2.hist(anchor_sims_plot, bins=100, color="steelblue", edgecolor="none", alpha=0.7)
            ax2.axvline(thresh, color="red", linestyle="--", label=f"threshold={thresh}")
            ax2.set_xlabel("Cosine similarity to anchor")
            ax2.set_ylabel("Frame count")
            ax2.set_title("Distribution of frame–anchor similarities")
            ax2.legend()
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            plot_path = os.path.join(
                args.data_path, str(args.seq_id),
                f"siglip_filter_seq{args.seq_id}.png"
            )
            plt.savefig(plot_path, dpi=120)
            print(f"  Plot saved to: {plot_path}")

        except ImportError:
            print("  (matplotlib not available — skipping plot)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dry-run the SigLIP Stage 1 filter and report stats")
    parser.add_argument("--seq_id", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="./coda_data")
    parser.add_argument("--threshold", type=float, default=0.90,
                        help="Primary threshold to report detailed stats for")
    parser.add_argument("--device", type=str, default=None,
                        help="'cuda', 'cpu', or None for auto")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_csv", action="store_true",
                        help="Save passed frame timestamps to a CSV file")
    parser.add_argument("--plot", action="store_true",
                        help="Save a similarity-over-time plot (requires matplotlib)")
    args = parser.parse_args()
    main(args)
