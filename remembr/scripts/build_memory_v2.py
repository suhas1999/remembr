"""
ReMEmbR v2 memory building pipeline.

Step 1: Load all pkl frames and batch-encode with SigLIP (~30s for 8000 frames on GPU)
Step 2: Iterate with two-stage filter:
  Stage 1 (fast): cosine similarity > 0.95 vs anchor → skip  (eliminates ~95% of frames)
  Stage 2 (VLM):  send frame + anchor + top-3 similar stored frames to Gemini 2.0 Flash judge
Step 3: Store / mark revisit / skip based on VLM decision

Output: ~65-80 stored entries for a 25-min walk (vs ~500 in v1), each with a keyframe JPEG.

Usage:
  python build_memory_v2.py --seq_id 0 --data_path ./coda_data
"""

import argparse
import base64
import glob
import json
import os
import pickle
import sys
from io import BytesIO
from time import strftime, localtime

import numpy as np
import tqdm
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from memory.frame_store import FrameStore
from memory.milvus_memory_v2 import MilvusMemoryV2
from models.siglip_encoder import SigLIPEncoder


# ── Constants ─────────────────────────────────────────────────────────────────

SIMILARITY_THRESHOLD = 0.95  # Stage 1: skip if cosine sim to anchor > this value


# ── Image encoding helpers ────────────────────────────────────────────────────

def _pil_to_b64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _gemini_image_part(image: Image.Image):
    """Convert PIL image to a Gemini Part (new google-genai SDK)."""
    from google.genai import types as genai_types
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return genai_types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


# ── VLM judge for Stage 2 ─────────────────────────────────────────────────────

def _load_judge_prompt() -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), "../prompts/v2/build_judge.txt")
    with open(prompt_path) as f:
        return f.read()


JUDGE_PROMPT = None  # loaded lazily


def call_vlm_judge(
    current_image: Image.Image,
    anchor_image: Image.Image,
    prev_anchor_image,         # Image or None
    similar_stored: list,      # list of entry dicts (may be empty)
    gemini_model,
    model: str = "gemini-2.5-flash",
) -> dict:
    """
    Ask Gemini 2.0 Flash: does this frame add new information not already in stored memories?
    Returns a dict with keys: decision, caption, readable_text, revisit_index
    """
    global JUDGE_PROMPT
    if JUDGE_PROMPT is None:
        JUDGE_PROMPT = _load_judge_prompt()

    parts = [JUDGE_PROMPT]
    parts.append("CURRENT FRAME:")
    parts.append(_gemini_image_part(current_image))
    parts.append("ANCHOR FRAME:")
    parts.append(_gemini_image_part(anchor_image))

    if prev_anchor_image is not None:
        parts.append("PREVIOUS SCENE:")
        parts.append(_gemini_image_part(prev_anchor_image))

    for i, stored in enumerate(similar_stored[:3], 1):
        img_path = stored.get("image_path", "")
        if img_path and os.path.exists(img_path):
            stored_img = Image.open(img_path).convert("RGB")
            parts.append(f"STORED SIMILAR {i}:")
            parts.append(_gemini_image_part(stored_img))

    from google.genai import types as genai_types
    response = gemini_model.models.generate_content(
        model=model,
        contents=parts,
        config=genai_types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=512,
            response_mime_type="application/json",
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
        ),
    )

    raw = response.text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  [WARN] JSON parse failed, defaulting to skip. Raw: {raw[:120]!r}")
        return {"decision": "skip", "readable_text": [], "revisit_index": None}

    # Normalize fields so callers can rely on them existing
    result.setdefault("decision", "skip")
    result.setdefault("readable_text", [])
    result.setdefault("revisit_index", None)

    return result


_CAPTION_PROMPT = (
    "You are a robot wandering around a university campus. "
    "You will see a sequence of frames captured over the past few seconds. "
    "Please describe in detail what you see across these frames. "
    "Specifically focus on the people, objects, environmental features, "
    "events/activities, and other interesting details. "
    "Think step by step about these details and be very specific."
)


def generate_caption_with_window(
    frames: list,
    raw_images: list,
    current_idx: int,
    gemini_model,
    model: str = "gemini-2.5-flash",
    window_seconds: float = 1.5,
    n_frames: int = 6,
) -> str:
    """
    Generate a rich caption using a ~3s window of frames around current_idx.
    Samples n_frames evenly from the window, sends all to Gemini with v1-style prompt.
    """
    from google.genai import types as genai_types

    current_ts = frames[current_idx]["timestamp"]
    t_start = current_ts - window_seconds
    t_end = current_ts + window_seconds

    # Collect all frame indices within the time window
    window_indices = [
        i for i, f in enumerate(frames)
        if t_start <= f["timestamp"] <= t_end
    ]
    if not window_indices:
        window_indices = [current_idx]

    # Sample n_frames evenly across the window
    if len(window_indices) <= n_frames:
        sampled = window_indices
    else:
        step = len(window_indices) / n_frames
        sampled = [window_indices[int(i * step)] for i in range(n_frames)]

    parts = [_CAPTION_PROMPT]
    for idx in sampled:
        parts.append(_gemini_image_part(raw_images[idx]))

    response = gemini_model.models.generate_content(
        model=model,
        contents=parts,
        config=genai_types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=2048,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text.strip()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def _model_name(name: str) -> str:
    """Ensure Gemini model name has models/ prefix."""
    return name if name.startswith("models/") else f"models/{name}"


def build_memory(args):
    from google import genai as google_genai
    gemini_model = google_genai.Client(api_key=args.gemini_api_key)
    args.vlm_model = _model_name(args.vlm_model)

    # ── Load all pkl files sorted by timestamp ────────────────────────────────
    pattern = os.path.join(args.data_path, str(args.seq_id), "*.pkl")
    pkl_files = sorted(glob.glob(pattern), key=lambda p: float(os.path.basename(p)[:-4]))

    if not pkl_files:
        print(f"ERROR: No pkl files found at {pattern}")
        sys.exit(1)

    print(f"Found {len(pkl_files)} frames for sequence {args.seq_id}")

    frames = []
    raw_images = []
    print("Loading frames...")
    for pkl_path in tqdm.tqdm(pkl_files):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        img = Image.fromarray(data["cam0"][:, :, ::-1].astype("uint8"), "RGB")
        rotation = Rotation.from_quat(data["rotation"]).as_euler("xyz", degrees=True)
        frames.append(
            {
                "timestamp": float(data["timestamp"]),
                "position": list(data["position"]),
                "theta": float(rotation[2]),  # yaw in degrees
            }
        )
        raw_images.append(img)

    # ── Step 1: Batch encode all frames with SigLIP ───────────────────────────
    siglip = SigLIPEncoder(device=args.device)
    print(f"\nBatch encoding {len(raw_images)} frames with SigLIP...")
    all_embs = siglip.encode_images(raw_images, batch_size=args.siglip_batch_size)
    print(f"Encoding complete. Shape: {all_embs.shape}")

    # ── Setup memory and frame store ──────────────────────────────────────────
    memory = MilvusMemoryV2(
        db_collection_name=f"v2_seq{args.seq_id}",
        db_path=args.db_path,
    )
    memory.reset(drop_collection=True)

    keyframes_dir = os.path.join(args.keyframes_dir, str(args.seq_id))
    frame_store = FrameStore(store_dir=keyframes_dir)

    # ── Step 2: Precompute Stage 1 passing indices ────────────────────────────
    # Roll through all embeddings with a moving anchor. Every frame that differs
    # enough (sim < threshold) gets its index recorded. This is instant (no VLM)
    # and gives us the exact set of frames to judge, matching inspect_siglip_filter.py.
    print(f"\nRunning Stage 1 filter (threshold={SIMILARITY_THRESHOLD})...")
    passing_indices = [0]  # first frame always passes
    s1_anchor = all_embs[0]
    for idx in range(1, len(all_embs)):
        if float(np.dot(all_embs[idx], s1_anchor)) < SIMILARITY_THRESHOLD:
            passing_indices.append(idx)
            s1_anchor = all_embs[idx]

    n_skipped_s1 = len(frames) - len(passing_indices)
    print(f"Stage 1: {len(passing_indices)} frames pass → VLM judge ({n_skipped_s1} skipped)")

    # ── Save Stage 1 filtered frames ──────────────────────────────────────────
    s1_frames_dir = os.path.join(args.keyframes_dir, f"stage1_frames_seq{args.seq_id}")
    os.makedirs(s1_frames_dir, exist_ok=True)
    s1_rows = []
    for idx in passing_indices:
        f = frames[idx]
        ts = f["timestamp"]
        fname = f"{ts:.6f}.jpg"
        raw_images[idx].save(os.path.join(s1_frames_dir, fname), format="JPEG", quality=85)
        s1_rows.append({
            "timestamp": ts,
            "datetime": strftime("%Y-%m-%d %H:%M:%S", localtime(ts)),
            "position_x": f["position"][0],
            "position_y": f["position"][1],
            "position_z": f["position"][2],
            "theta_deg": f["theta"],
            "frame_index": idx,
            "image_file": fname,
        })
    import csv
    s1_csv_path = os.path.join(args.keyframes_dir, f"stage1_frames_seq{args.seq_id}.csv")
    with open(s1_csv_path, "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=s1_rows[0].keys())
        writer.writeheader()
        writer.writerows(s1_rows)
    print(f"Stage 1 frames saved: {len(s1_rows)} images → {s1_frames_dir}")
    print(f"Stage 1 CSV: {s1_csv_path}")

    # ── Step 3: VLM judge only on passing frames ──────────────────────────────
    vlm_anchor_emb = None
    vlm_anchor_image = None
    prev_vlm_anchor_image = None

    stats = {"stored": 0, "revisits": 0, "skipped_stage1": n_skipped_s1, "skipped_stage2": 0}

    print(f"\nBuilding memory (VLM judging {len(passing_indices)} frames)...")
    for rank, i in enumerate(tqdm.tqdm(passing_indices)):
        frame, image, visual_emb = frames[i], raw_images[i], all_embs[i]

        # ── First frame: always store ─────────────────────────────────────────
        if vlm_anchor_emb is None:
            print("  [0] Storing first frame...")
            caption = generate_caption_with_window(frames, raw_images, i, gemini_model, model=args.vlm_model)
            image_path = frame_store.save(image, frame["timestamp"])
            bge_emb = memory.bge_embed_document(caption)

            memory.insert(
                _make_entry(
                    frame=frame,
                    visual_emb=visual_emb,
                    bge_emb=bge_emb,
                    caption=caption,
                    image_path=image_path,
                    is_revisit=0.0,
                    original_id="",
                    location_change=1.0,
                )
            )
            vlm_anchor_emb = visual_emb
            vlm_anchor_image = image
            stats["stored"] += 1
            continue

        # ── Stage 2: VLM judge ────────────────────────────────────────────────
        # Retrieve top-3 similar stored frames to detect revisits.
        # Without this, returning to a previously visited corridor would look "new"
        # because the recent context no longer contains those old frames.
        similar_stored = memory.search_by_siglip_emb(visual_emb, k=3)

        judge = call_vlm_judge(
            current_image=image,
            anchor_image=vlm_anchor_image,
            prev_anchor_image=prev_vlm_anchor_image,
            similar_stored=similar_stored,
            gemini_model=gemini_model,
            model=args.vlm_model,
        )

        decision = judge["decision"]
        t_str = strftime("%H:%M:%S", localtime(frame["timestamp"]))
        print(f"  [{i:05d}] t={t_str} → {decision}")

        # ── Decision: skip ────────────────────────────────────────────────────
        if decision == "skip":
            stats["skipped_stage2"] += 1

        # ── Decision: store (new location or same scene with new detail) ──────
        elif decision in ("new_location", "same_scene_new_detail"):
            caption = generate_caption_with_window(frames, raw_images, i, gemini_model, model=args.vlm_model)
            if judge["readable_text"]:
                caption += "  Visible text: " + ", ".join(judge["readable_text"])

            image_path = frame_store.save(image, frame["timestamp"])
            bge_emb = memory.bge_embed_document(caption)

            memory.insert(
                _make_entry(
                    frame=frame,
                    visual_emb=visual_emb,
                    bge_emb=bge_emb,
                    caption=caption,
                    image_path=image_path,
                    is_revisit=0.0,
                    original_id="",
                    location_change=1.0 if decision == "new_location" else 0.0,
                )
            )
            stats["stored"] += 1

            if decision == "new_location":
                prev_vlm_anchor_image = vlm_anchor_image
                vlm_anchor_emb = visual_emb
                vlm_anchor_image = image

        # ── Decision: revisit ─────────────────────────────────────────────────
        elif decision == "revisit":
            rev_idx = (judge.get("revisit_index") or 1) - 1
            rev_idx = max(0, min(rev_idx, len(similar_stored) - 1))

            if similar_stored:
                original = similar_stored[rev_idx]
                revisit_caption = original.get("caption", "")

                memory.insert(
                    _make_entry(
                        frame=frame,
                        visual_emb=visual_emb,
                        bge_emb=memory.bge_embed_document(revisit_caption),
                        caption=revisit_caption,
                        image_path=original.get("image_path", ""),
                        is_revisit=1.0,
                        original_id=original.get("id", ""),
                        location_change=0.0,
                        id_suffix="_revisit",
                    )
                )
                stats["revisits"] += 1

                prev_vlm_anchor_image = vlm_anchor_image
                vlm_anchor_emb = np.array(visual_emb)
                vlm_anchor_image = image

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(frames)
    print(f"\n{'='*50}")
    print(f"Memory building complete for sequence {args.seq_id}:")
    print(f"  Total frames processed : {total}")
    print(f"  Stored (original)      : {stats['stored'] - stats['revisits']}")
    print(f"  Stored (revisit)       : {stats['revisits']}")
    print(f"  Skipped by Stage 1     : {stats['skipped_stage1']}  ({100*stats['skipped_stage1']//total}%)")
    print(f"  Skipped by Stage 2     : {stats['skipped_stage2']}")
    print(f"  DB path                : {args.db_path}")
    print(f"  Keyframes dir          : {keyframes_dir}")
    print(f"{'='*50}")

    # Save build summary JSON
    summary = {
        "seq_id": args.seq_id,
        "total_frames": total,
        "stored": stats["stored"],
        "revisits": stats["revisits"],
        "skipped_stage1": stats["skipped_stage1"],
        "skipped_stage2": stats["skipped_stage2"],
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "vlm_model": args.vlm_model,
        "keyframes_dir": keyframes_dir,
        "db_path": args.db_path,
        "collection": f"v2_seq{args.seq_id}",
    }
    summary_path = os.path.join(args.keyframes_dir, f"build_summary_seq{args.seq_id}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Build summary saved to {summary_path}")

    # ── Export captions as v1-compatible JSON (with extra v2 columns) ─────────
    all_entries = memory.get_all()
    v1_captions = []
    for entry in sorted(all_entries, key=lambda e: e.get("time", 0)):
        ts = entry.get("time", 0)
        v1_captions.append({
            # v1-compatible fields
            "time": ts,
            "datetime": strftime("%Y-%m-%d %H:%M:%S", localtime(ts)),
            "position": entry.get("position", []),
            "caption": entry.get("caption", ""),
            # v2 extra columns
            "image_path": entry.get("image_path", ""),
            "is_revisit": entry.get("is_revisit", 0.0),
            "original_id": entry.get("original_id", ""),
            "location_change": entry.get("location_change", 0.0),
            "theta_deg": entry.get("theta", 0.0),
            "vlm_model": args.vlm_model,
        })
    captions_out_dir = os.path.join(args.data_dir, "captions", str(args.seq_id), "captions")
    os.makedirs(captions_out_dir, exist_ok=True)
    captions_json_path = os.path.join(captions_out_dir, f"captions_v2_{args.seq_id}.json")
    with open(captions_json_path, "w") as f:
        json.dump(v1_captions, f, indent=2)
    print(f"v1-compatible captions saved to {captions_json_path} ({len(v1_captions)} entries)")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_entry(
    frame: dict,
    visual_emb,
    bge_emb,
    caption: str,
    image_path: str,
    is_revisit: float,
    original_id: str,
    location_change: float,
    id_suffix: str = "",
) -> dict:
    """Build a dict matching the MilvusMemoryV2 schema."""
    emb_list = visual_emb.tolist() if hasattr(visual_emb, "tolist") else list(visual_emb)
    bge_list = bge_emb.tolist() if hasattr(bge_emb, "tolist") else list(bge_emb)
    return {
        "id": f"{frame['timestamp']:.6f}{id_suffix}",
        "siglip_embedding": emb_list,
        "bge_embedding": bge_list,
        "position": frame["position"],
        "theta": frame["theta"],
        "time": frame["timestamp"],
        "caption": caption[:3000],  # Milvus VARCHAR max
        "image_path": image_path,
        "is_revisit": is_revisit,
        "original_id": original_id,
        "location_change": location_change,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ReMEmbR v2 memory for a sequence")
    parser.add_argument("--seq_id", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="./coda_data",
                        help="Directory containing {seq_id}/*.pkl files")
    parser.add_argument("--db_path", type=str, default="./remembr_v2.db")
    parser.add_argument("--keyframes_dir", type=str, default="./data/v2/keyframes",
                        help="Directory to store JPEG keyframes. Subdirectory {seq_id}/ created automatically.")
    parser.add_argument("--vlm_model", type=str, default="gemini-2.5-flash",
                        help="Gemini model for VLM judge (e.g. gemini-2.5-flash, gemini-1.5-pro)")
    parser.add_argument("--gemini_api_key", type=str, default=None,
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Base data directory for saving captions JSON (captions/{seq_id}/captions/)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for SigLIP encoding: 'cuda', 'cpu', or None for auto-detect")
    parser.add_argument("--siglip_batch_size", type=int, default=32,
                        help="Images per batch during SigLIP encoding")
    args = parser.parse_args()

    key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        print("ERROR: Gemini API key not set. Use --gemini_api_key or GEMINI_API_KEY env var.")
        sys.exit(1)
    args.gemini_api_key = key

    os.makedirs(args.keyframes_dir, exist_ok=True)
    build_memory(args)
