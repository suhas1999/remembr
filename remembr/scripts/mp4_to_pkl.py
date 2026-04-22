"""
Convert an MP4 video (local or from GCS) to pkl files for the ReMEmbR pipeline.

Each frame is saved as {timestamp:.6f}.pkl containing:
  cam0:      numpy array (H, W, 3) in BGR format (what OpenCV gives)
  timestamp: unix float  (video_start_time + frame_offset_seconds)
  position:  [0.0, 0.0, 0.0]   — no GPS for personal videos
  rotation:  [0.0, 0.0, 0.0, 1.0]  — identity quaternion

Usage:
  # From GCS
  python mp4_to_pkl.py \
      --gcs_path "gs://remember-data-bucket/remembr/data/WhatsApp Video 2026-04-22 at 03.00.15.mp4" \
      --output_dir ./coda_data --seq_id 99 \
      --fps 10 --video_start_time 1745280000

  # From local file
  python mp4_to_pkl.py \
      --video_path ./my_video.mp4 \
      --output_dir ./coda_data --seq_id 99 \
      --fps 10 --video_start_time 1745280000
"""

import argparse
import os
import pickle
import subprocess
import sys
import tempfile
from time import strftime, localtime

import cv2
import numpy as np
import tqdm


def download_from_gcs(gcs_path: str, local_path: str):
    print(f"Downloading from GCS: {gcs_path}")
    result = subprocess.run(
        ["gsutil", "cp", gcs_path, local_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"gsutil stderr: {result.stderr}")
        raise RuntimeError(f"gsutil cp failed: {result.returncode}")
    print(f"Downloaded to: {local_path}")


def extract_frames_to_pkl(
    video_path: str,
    output_dir: str,
    seq_id: int,
    fps: float,
    video_start_time: float,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / video_fps if video_fps > 0 else 0
    video_end_time = video_start_time + duration_s

    print(f"Video FPS:    {video_fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration:     {duration_s:.1f}s  ({duration_s/60:.1f} min)")
    print(f"Start time:   {strftime('%Y-%m-%d %H:%M:%S', localtime(video_start_time))} UTC")
    print(f"End time:     {strftime('%Y-%m-%d %H:%M:%S', localtime(video_end_time))} UTC")
    print(f"Extracting at {fps} fps → ~{int(duration_s * fps)} output frames")

    out_dir = os.path.join(output_dir, str(seq_id))
    os.makedirs(out_dir, exist_ok=True)

    step = max(1, int(round(video_fps / fps)))
    frame_idx = 0
    saved = 0

    pbar = tqdm.tqdm(total=total_frames, unit="frame")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            offset_s = frame_idx / video_fps
            timestamp = video_start_time + offset_s
            pkl_path = os.path.join(out_dir, f"{timestamp:.6f}.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(
                    {
                        "cam0": frame,                        # BGR numpy array
                        "timestamp": timestamp,
                        "position": [0.0, 0.0, 0.0],
                        "rotation": [0.0, 0.0, 0.0, 1.0],   # identity quaternion
                    },
                    f,
                )
            saved += 1
        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()

    print(f"\nSaved {saved} pkl files to: {out_dir}")
    print(f"\n--- Update your question JSON ---")
    print(f"  seq_id:           {seq_id}")
    print(f"  video_start_time: {video_start_time}")
    print(f"  video_end_time:   {video_end_time:.3f}")
    print(f"  duration_s:       {duration_s:.1f}")
    print(f"  end_time_str:     {strftime('%Y-%m-%d %H:%M:%S', localtime(video_end_time))} UTC")
    return duration_s, video_end_time


def main():
    parser = argparse.ArgumentParser(description="Convert MP4 to ReMEmbR pkl format")
    parser.add_argument("--gcs_path", type=str, default=None,
                        help="GCS path to the mp4, e.g. gs://bucket/path/video.mp4")
    parser.add_argument("--video_path", type=str, default=None,
                        help="Local path to the mp4 file (alternative to --gcs_path)")
    parser.add_argument("--output_dir", type=str, default="./coda_data",
                        help="Directory where pkl files will be written (default: ./coda_data)")
    parser.add_argument("--seq_id", type=int, default=99,
                        help="Sequence ID (used as subdirectory name, default: 99)")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Frames per second to extract (default: 10.0)")
    parser.add_argument("--video_start_time", type=float, default=1745280000.0,
                        help="Unix timestamp for the first frame (default: 2026-04-22 00:00:00 UTC)")
    args = parser.parse_args()

    if args.gcs_path is None and args.video_path is None:
        parser.error("Provide either --gcs_path or --video_path")

    if args.gcs_path:
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.basename(args.gcs_path)
            local_path = os.path.join(tmpdir, fname)
            download_from_gcs(args.gcs_path, local_path)
            extract_frames_to_pkl(
                local_path, args.output_dir, args.seq_id,
                args.fps, args.video_start_time,
            )
    else:
        extract_frames_to_pkl(
            args.video_path, args.output_dir, args.seq_id,
            args.fps, args.video_start_time,
        )


if __name__ == "__main__":
    main()
