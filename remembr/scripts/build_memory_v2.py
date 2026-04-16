"""
ReMEmbR v2 memory building pipeline.

Step 1: Load all pkl frames and batch-encode with SigLIP (~30s for 8000 frames on GPU)
Step 2: Iterate with two-stage filter:
  Stage 1 (fast): cosine similarity > 0.9 vs anchor → skip  (eliminates ~90% of frames)
  Stage 2 (VLM):  send frame + anchor + top-3 similar stored frames to GPT-4o-mini judge
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

SIMILARITY_THRESHOLD = 0.9   # Stage 1: skip if cosine sim to anchor > this value


# ── Image encoding for OpenAI vision ─────────────────────────────────────────

def _pil_to_b64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _image_content(label: str, image: Image.Image) -> list:
    return [
        {"type": "text", "text": label},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_pil_to_b64(image)}"},
        },
    ]


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
    openai_client,
    model: str = "gpt-4o-mini",
) -> dict:
    """
    Ask the VLM: does this frame add new information not already in stored memories?
    Returns a dict with keys: decision, caption, readable_text, revisit_index
    """
    global JUDGE_PROMPT
    if JUDGE_PROMPT is None:
        JUDGE_PROMPT = _load_judge_prompt()

    content = [{"type": "text", "text": JUDGE_PROMPT}]
    content += _image_content("CURRENT FRAME:", current_image)
    content += _image_content("ANCHOR FRAME:", anchor_image)

    if prev_anchor_image is not None:
        content += _image_content("PREVIOUS SCENE:", prev_anchor_image)

    for i, stored in enumerate(similar_stored[:3], 1):
        img_path = stored.get("image_path", "")
        if img_path and os.path.exists(img_path):
            stored_img = Image.open(img_path).convert("RGB")
            content += _image_content(f"STORED SIMILAR {i}:", stored_img)

    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=1024,
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    result = json.loads(raw)

    # Normalize fields so callers can rely on them existing
    result.setdefault("decision", "skip")
    result.setdefault("caption", "")
    result.setdefault("readable_text", [])
    result.setdefault("revisit_index", None)

    return result


def generate_initial_caption(image: Image.Image, openai_client, model: str) -> str:
    """Generate a caption for the very first stored frame (no comparison needed)."""
    prompt = (
        "Describe this robot camera frame in detail. Include: type of space, "
        "architectural features, furniture, doors, signs with their text, "
        "people, and any distinctive landmarks. Be specific."
    )
    content = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_pil_to_b64(image)}"},
        },
    ]
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=512,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_memory(args):
    from openai import OpenAI
    openai_client = OpenAI()

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

    # ── Step 2 & 3: Iterate with two-stage filter ─────────────────────────────
    anchor_emb = None
    anchor_image = None
    prev_anchor_emb = None
    prev_anchor_image = None

    stats = {"stored": 0, "revisits": 0, "skipped_stage1": 0, "skipped_stage2": 0}

    print(f"\nBuilding memory (threshold={SIMILARITY_THRESHOLD})...")
    for i, (frame, image, visual_emb) in enumerate(
        tqdm.tqdm(zip(frames, raw_images, all_embs))
    ):
        # ── First frame: always store ─────────────────────────────────────────
        if anchor_emb is None:
            print("  [0] Storing first frame...")
            caption = generate_initial_caption(image, openai_client, model=args.vlm_model)
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
            anchor_emb = visual_emb
            anchor_image = image
            stats["stored"] += 1
            continue

        # ── Stage 1: fast cosine similarity check ─────────────────────────────
        # Dot product of L2-normalized vectors = cosine similarity.
        # If > 0.9, the frame is near-identical to the anchor — skip without VLM call.
        sim = float(np.dot(visual_emb, anchor_emb))
        if sim > SIMILARITY_THRESHOLD:
            stats["skipped_stage1"] += 1
            continue

        # ── Stage 2: VLM judge ────────────────────────────────────────────────
        # Retrieve top-3 similar stored frames to detect revisits.
        # Without this, returning to a previously visited corridor would look "new"
        # because the recent context no longer contains those old frames.
        similar_stored = memory.search_by_siglip_emb(visual_emb, k=3)

        judge = call_vlm_judge(
            current_image=image,
            anchor_image=anchor_image,
            prev_anchor_image=prev_anchor_image,
            similar_stored=similar_stored,
            openai_client=openai_client,
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
            caption = judge["caption"]
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
                # Advance the anchor to this new scene
                prev_anchor_emb = anchor_emb
                prev_anchor_image = anchor_image
                anchor_emb = visual_emb
                anchor_image = image

        # ── Decision: revisit ─────────────────────────────────────────────────
        elif decision == "revisit":
            rev_idx = (judge.get("revisit_index") or 1) - 1
            rev_idx = max(0, min(rev_idx, len(similar_stored) - 1))

            if similar_stored:
                original = similar_stored[rev_idx]
                revisit_caption = original.get("caption", "")

                # Lightweight revisit marker: reuse original's image/caption on disk,
                # record the new timestamp and position. Re-embed caption text since
                # the vector fields are not returned by search queries (OUTPUT_FIELDS).
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

                # Anchor shifts to the revisited location so subsequent frames
                # compare against the correct scene
                prev_anchor_emb = anchor_emb
                prev_anchor_image = anchor_image
                anchor_emb = np.array(visual_emb)
                anchor_image = image

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
    parser.add_argument("--vlm_model", type=str, default="gpt-4o-mini",
                        help="OpenAI model for VLM judge. gpt-4o-mini is faster/cheaper; gpt-4o is more accurate.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for SigLIP encoding: 'cuda', 'cpu', or None for auto-detect")
    parser.add_argument("--siglip_batch_size", type=int, default=32,
                        help="Images per batch during SigLIP encoding")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        sys.exit(1)

    os.makedirs(args.keyframes_dir, exist_ok=True)
    build_memory(args)
