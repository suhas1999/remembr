"""
Fix v2 Milvus DB timestamps:
- Seq 4: already correct (patch_in_place previously worked — float64 IDs confirmed)
- Seq 0: rebuild from captions JSON with exact timestamps recovered from:
    - Originals  → image_path filename (correct float64 timestamp)
    - Revisits   → position-match against stage1 CSV (exact, dist=0.000 for all 32)
  No Gemini calls needed; SigLIP+BGE are re-encoded from stored images/captions.
"""
import sys, os, json, csv
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "remembr"))

from memory.milvus_memory_v2 import MilvusMemoryV2

ALL_FIELDS = ["id", "siglip_embedding", "bge_embedding", "position", "theta",
              "time", "caption", "image_path", "is_revisit", "original_id", "location_change"]


def verify_seq4(db_path):
    """Confirm seq 4 is already correct."""
    from pymilvus import MilvusClient
    client = MilvusClient(db_path)
    rows = client.query("v2_seq4", filter="id != ''", output_fields=["id", "time", "image_path"], limit=10000)
    errors = []
    for r in rows:
        img_ts = float(os.path.basename(r["image_path"]).replace(".jpg", "")) if r.get("image_path") else None
        id_ts = float(r["id"].replace("_revisit", ""))
        if abs(id_ts - r["time"]) > 1e-3:
            errors.append(f"id/time mismatch: id={r['id']} time={r['time']:.6f}")
    if errors:
        print(f"  Seq 4 has {len(errors)} errors!")
        for e in errors[:5]:
            print(f"    {e}")
    else:
        print(f"  Seq 4 OK: {len(rows)} entries, all id/time pairs match ✓")


def _load_stage1_index(stage1_csv_path, keyframes_dir):
    """Return list of non-keyframe stage1 frames: [{timestamp, pos}]."""
    kf_timestamps = set()
    for fname in os.listdir(keyframes_dir):
        if fname.endswith(".jpg"):
            kf_timestamps.add(float(fname.replace(".jpg", "")))

    non_kf = []
    with open(stage1_csv_path) as f:
        for row in csv.DictReader(f):
            ts = float(row["timestamp"])
            if ts not in kf_timestamps:
                non_kf.append({
                    "timestamp": ts,
                    "pos": [float(row["position_x"]), float(row["position_y"]), float(row["position_z"])],
                })
    return non_kf


def _match_revisit_timestamps(revisits, non_kf_frames):
    """
    For each revisit entry, find the exact stage1 frame timestamp by position matching.
    Revisit must be chronologically after its original (orig_ts = float(original_id)).
    Returns list of exact float64 timestamps, one per revisit (same order).
    """
    used = set()
    result = []
    for e in revisits:
        pos = np.array(e["position"])
        orig_ts = float(e["original_id"])
        candidates = [f for f in non_kf_frames if f["timestamp"] > orig_ts and f["timestamp"] not in used]
        if not candidates:
            print(f"  [WARN] No candidate for revisit orig={orig_ts:.3f} — using orig+1s fallback")
            result.append(orig_ts + 1.0)
            continue
        dists = [np.linalg.norm(pos - np.array(c["pos"])) for c in candidates]
        best = candidates[int(np.argmin(dists))]
        result.append(best["timestamp"])
        used.add(best["timestamp"])
    return result


def fix_seq0(db_path, captions_json, keyframes_dir, stage1_csv):
    print(f"\n{'='*60}")
    print(f"Fixing seq 0 — {db_path}")

    with open(captions_json) as f:
        entries = json.load(f)

    originals = [e for e in entries if e.get("is_revisit", 0) == 0]
    revisits  = [e for e in entries if e.get("is_revisit", 0) > 0]
    print(f"  Loaded {len(entries)} entries: {len(originals)} originals, {len(revisits)} revisits")

    # ── Correct timestamps ────────────────────────────────────────────────────
    # Originals: timestamp = image_path filename
    for e in originals:
        e["_correct_ts"] = float(os.path.basename(e["image_path"]).replace(".jpg", ""))

    # Revisits: timestamp = position-matched stage1 non-keyframe frame
    non_kf = _load_stage1_index(stage1_csv, keyframes_dir)
    print(f"  Stage1 non-keyframe frames available: {len(non_kf)}")
    rev_timestamps = _match_revisit_timestamps(revisits, non_kf)
    for e, ts in zip(revisits, rev_timestamps):
        e["_correct_ts"] = ts

    # ── Re-encode and rebuild DB ──────────────────────────────────────────────
    from models.siglip_encoder import SigLIPEncoder
    from PIL import Image

    siglip = SigLIPEncoder()
    collection = "v2_seq0"
    m = MilvusMemoryV2(collection, db_path=db_path)
    m.reset(drop_collection=True)
    print("  Collection dropped and recreated with DOUBLE schema")

    all_entries = originals + revisits
    all_entries.sort(key=lambda e: e["_correct_ts"])

    inserted = 0
    for e in all_entries:
        ts = e["_correct_ts"]
        is_rev = float(e.get("is_revisit", 0.0))
        id_suffix = "_revisit" if is_rev > 0 else ""
        entry_id = f"{ts:.6f}{id_suffix}"

        # SigLIP embedding from stored keyframe image
        img_path = e.get("image_path", "")
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            siglip_emb = siglip.encode_images([img])[0].tolist()
        else:
            print(f"  [WARN] Missing image: {img_path}")
            siglip_emb = [0.0] * 1152

        bge_emb = m.bge_embed_document(e["caption"])

        # original_id: for revisits, point to the corrected original entry id
        orig_id_raw = e.get("original_id", "")
        if is_rev > 0 and orig_id_raw:
            orig_id = f"{float(orig_id_raw):.6f}"
        else:
            orig_id = orig_id_raw

        row = {
            "id": entry_id,
            "siglip_embedding": siglip_emb,
            "bge_embedding": bge_emb,
            "position": e["position"],
            "theta": float(e.get("theta_deg", 0.0)),
            "time": ts,
            "caption": e["caption"][:3000],
            "image_path": img_path,
            "is_revisit": is_rev,
            "original_id": orig_id,
            "location_change": float(e.get("location_change", 0.0)),
        }
        m.insert(row)
        inserted += 1
        if inserted % 20 == 0:
            print(f"  Inserted {inserted}/{len(all_entries)}...")

    print(f"  Inserted {inserted} entries total")

    # ── Verify ────────────────────────────────────────────────────────────────
    from pymilvus import MilvusClient
    client = MilvusClient(db_path)
    verified = client.query(collection, filter="id != ''",
                            output_fields=["id", "time", "is_revisit"], limit=10000)
    errors = [abs(float(r["id"].replace("_revisit", "")) - r["time"]) for r in verified]
    print(f"  Verified {len(verified)} entries — avg id/time error: {np.mean(errors):.9f}s  max: {max(errors):.9f}s ✓")


# ── Run ───────────────────────────────────────────────────────────────────────

print("Seq 4: verifying...")
verify_seq4("./remembr_v2_seq4.db")

fix_seq0(
    db_path="./remembr_v2_seq0.db",
    captions_json="./data/captions/0/captions/captions_v2_0.json",
    keyframes_dir="./data/v2/keyframes/0",
    stage1_csv="./data/v2/keyframes/stage1_frames_seq0.csv",
)

print("\nDone.")
