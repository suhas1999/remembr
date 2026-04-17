"""
ReMEmbR v2 Milvus memory: stores richer, fewer entries with dual embeddings.

Schema differences from v1:
- Two vector fields: siglip_embedding (1152-dim visual) + bge_embedding (768-dim caption)
- time stored as scalar FLOAT (not FLOAT_VECTOR hack) → proper range queries
- image_path, is_revisit, original_id, location_change fields added

All searches respect an optional time window (set_time_window) so the same pre-built
database can be filtered per question during evaluation.
"""

from dataclasses import dataclass
from time import strftime, localtime
from typing import Optional

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import DataType, MilvusClient


SIGLIP_DIM = 1152
BGE_DIM = 768

# Fields returned on every query/search (no embeddings — they're large and unneeded)
OUTPUT_FIELDS = [
    "id", "position", "theta", "time",
    "caption", "image_path", "is_revisit", "original_id", "location_change",
]


# ── Low-level Milvus wrapper ──────────────────────────────────────────────────

class MilvusWrapperV2:

    def __init__(self, collection_name: str, db_path: str, drop_collection: bool = False):
        self.collection_name = collection_name
        self.client = MilvusClient(db_path)

        if drop_collection and self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)

        if not self.client.has_collection(collection_name):
            self._create_collection(collection_name)

    def _create_collection(self, name: str):
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, max_length=200, is_primary=True)
        schema.add_field("siglip_embedding", DataType.FLOAT_VECTOR, dim=SIGLIP_DIM)
        schema.add_field("bge_embedding", DataType.FLOAT_VECTOR, dim=BGE_DIM)
        schema.add_field("position", DataType.FLOAT_VECTOR, dim=3)
        schema.add_field("theta", DataType.FLOAT)
        schema.add_field("time", DataType.FLOAT)          # raw unix timestamp (scalar)
        schema.add_field("caption", DataType.VARCHAR, max_length=3000)
        schema.add_field("image_path", DataType.VARCHAR, max_length=1000)
        schema.add_field("is_revisit", DataType.FLOAT)    # 0.0 = original, 1.0 = revisit
        schema.add_field("original_id", DataType.VARCHAR, max_length=200)
        schema.add_field("location_change", DataType.FLOAT)  # 1.0 = new location

        index_params = self.client.prepare_index_params()
        index_params.add_index("siglip_embedding", index_type="FLAT", metric_type="IP")
        index_params.add_index("bge_embedding", index_type="FLAT", metric_type="IP")
        index_params.add_index("position", index_type="FLAT", metric_type="L2")

        self.client.create_collection(name, schema=schema, index_params=index_params)

    def insert(self, rows: list):
        self.client.insert(self.collection_name, rows)

    def search(self, vector: list, anns_field: str, limit: int, filter_expr: str = "") -> list:
        kwargs = dict(
            collection_name=self.collection_name,
            data=[vector],
            anns_field=anns_field,
            limit=limit,
            output_fields=OUTPUT_FIELDS,
        )
        if filter_expr:
            kwargs["filter"] = filter_expr
        return self.client.search(**kwargs)

    def query(self, filter_expr: str = "id != ''", limit: int = 10000) -> list:
        return self.client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=OUTPUT_FIELDS,
            limit=limit,
        )

    def drop(self):
        self.client.drop_collection(self.collection_name)


# ── High-level memory interface ───────────────────────────────────────────────

class MilvusMemoryV2:
    """
    v2 memory: dual-embedding (SigLIP visual + BGE caption), stored keyframe images,
    revisit markers, and time-range filtering for per-question evaluation.
    """

    def __init__(self, db_collection_name: str, db_path: str = "./remembr_v2.db"):
        self.db_collection_name = db_collection_name
        self.db_path = db_path

        # BGE-base-en-v1.5 for caption embeddings (768-dim)
        self.bge = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

        # Optional time window for per-question filtering
        self.time_start: Optional[float] = None
        self.time_end: Optional[float] = None

        # Accumulates entries the agent has touched (for post-hoc inspection)
        self.working_memory: list = []

        self.reset(drop_collection=False)

    def reset(self, drop_collection: bool = True):
        self.wrapper = MilvusWrapperV2(
            self.db_collection_name, self.db_path, drop_collection=drop_collection
        )
        self.working_memory = []

    def set_time_window(self, start: float, end: float):
        """Restrict all searches to entries within [start, end] unix timestamps."""
        self.time_start = start
        self.time_end = end
        self.working_memory = []

    def clear_time_window(self):
        self.time_start = None
        self.time_end = None

    def _time_filter(self) -> str:
        if self.time_start is not None and self.time_end is not None:
            return f"time >= {self.time_start} and time <= {self.time_end}"
        return ""

    # ── Writing ──────────────────────────────────────────────────────────────

    def insert(self, entry: dict):
        """
        Insert a full memory entry. entry must contain all schema fields
        (siglip_embedding and bge_embedding as plain lists).
        """
        self.wrapper.insert([entry])

    def bge_embed_document(self, text: str) -> list:
        """Embed a caption/document (no instruction prefix needed for BGE docs)."""
        return self.bge.embed_documents([text])[0]

    def bge_embed_query(self, text: str) -> list:
        """Embed a search query with BGE instruction prefix."""
        return self.bge.embed_query(
            "Represent this sentence for searching relevant passages: " + text
        )

    # ── Reading ───────────────────────────────────────────────────────────────

    def search_by_siglip_emb(self, emb: np.ndarray, k: int = 3) -> list:
        """Search by raw SigLIP visual embedding (used during memory building)."""
        results = self.wrapper.search(
            vector=emb.tolist(),
            anns_field="siglip_embedding",
            limit=k,
            filter_expr=self._time_filter(),
        )
        return _hits_to_entries(results)

    def search_by_siglip_text(self, query: str, k: int = 5, siglip=None) -> list:
        """Search stored image embeddings using a SigLIP text query. Used internally by search_hybrid."""
        if siglip is None:
            raise ValueError("Pass the SigLIPEncoder instance as siglip=...")
        text_emb = siglip.encode_text([query])[0]
        results = self.wrapper.search(
            vector=text_emb.tolist(),
            anns_field="siglip_embedding",
            limit=k,
            filter_expr=self._time_filter(),
        )
        return _hits_to_entries(results)  # no working_memory here — hybrid adds final result

    def search_by_bge(self, query: str, k: int = 5) -> list:
        """Search stored captions using BGE text embedding. Used internally by search_hybrid."""
        query_emb = self.bge_embed_query(query)
        results = self.wrapper.search(
            vector=query_emb,
            anns_field="bge_embedding",
            limit=k,
            filter_expr=self._time_filter(),
        )
        return _hits_to_entries(results)  # no working_memory here — hybrid adds final result

    def search_hybrid(self, query: str, k: int = 3, siglip=None) -> list:
        """
        Hybrid retrieval: BGE caption search + SigLIP text-to-image search,
        fused with Reciprocal Rank Fusion (RRF). Returns top-k deduplicated originals.
        """
        bge_entries = self.search_by_bge(query, k=k * 2)

        if siglip is not None:
            siglip_entries = self.search_by_siglip_text(query, k=k * 2, siglip=siglip)
        else:
            siglip_entries = []

        bge_ids = [e["id"] for e in bge_entries]
        siglip_ids = [e["id"] for e in siglip_entries]

        fused_ranking = _rrf_fuse([bge_ids, siglip_ids])
        id_to_entry = {e["id"]: e for e in bge_entries + siglip_entries}

        # Build result: skip revisits only
        result = []
        for entry_id, _ in fused_ranking:
            if len(result) >= k:
                break
            entry = id_to_entry.get(entry_id)
            if entry is None:
                continue
            if entry.get("is_revisit", 0) > 0:
                continue  # skip revisit markers — captions are duplicates of originals
            result.append(entry)

        self.working_memory.extend(result)
        return result

    def search_by_position(self, xyz: tuple, k: int = 3) -> list:
        """Find memories nearest to a (x, y, z) GPS position."""
        results = self.wrapper.search(
            vector=list(xyz),
            anns_field="position",
            limit=k * 2,  # fetch extra to allow dedup
            filter_expr=self._time_filter(),
        )
        entries = _hits_to_entries(results)
        # Filter revisits only
        deduped = [e for e in entries if e.get("is_revisit", 0) == 0][:k]
        self.working_memory.extend(deduped)
        return deduped

    def get_nearby_in_time(self, timestamp: float, window_seconds: float = 60) -> list:
        """
        Return stored entries within ±window_seconds of timestamp, sorted chronologically.
        Capped at 15 entries to prevent context explosion. Crucial for trajectory questions.
        """
        t_start = timestamp - window_seconds
        t_end = timestamp + window_seconds

        time_filter = f"time >= {t_start} and time <= {t_end}"
        if self.time_start is not None:
            time_filter += f" and time >= {self.time_start}"
        if self.time_end is not None:
            time_filter += f" and time <= {self.time_end}"

        raw = self.wrapper.query(filter_expr=time_filter)
        entries = sorted(raw, key=lambda e: e["time"])
        # Cap at 15 to prevent context explosion; prefer originals over revisits
        originals = [e for e in entries if e.get("is_revisit", 0) == 0]
        revisits = [e for e in entries if e.get("is_revisit", 0) > 0]
        entries = (originals + revisits)[:15]
        self.working_memory.extend(entries)
        return entries

    def get_all(self) -> list:
        """Return all entries in the collection, sorted by time."""
        filter_expr = self._time_filter() or "id != ''"
        raw = self.wrapper.query(filter_expr=filter_expr)
        return sorted(raw, key=lambda e: e["time"])

    def get_adjacent_entries(self, entry_id: str) -> tuple:
        """
        Return the entries stored immediately before and after entry_id by timestamp.
        Returns (before_entry_or_None, after_entry_or_None).
        """
        all_entries = self.get_all()
        ids = [e["id"] for e in all_entries]
        if entry_id not in ids:
            return None, None
        idx = ids.index(entry_id)
        before = all_entries[idx - 1] if idx > 0 else None
        after = all_entries[idx + 1] if idx < len(all_entries) - 1 else None
        return before, after

    def get_working_memory(self) -> list:
        return self.working_memory

    # ── Formatting ────────────────────────────────────────────────────────────

    def memory_to_string(self, entries: list) -> str:
        """
        Format entries in v1-compatible style, extended with image paths.
        The 'Image: /path' line lets the agent pass paths to examine_keyframes.
        """
        out = ""
        for entry in entries:
            t_str = strftime("%Y-%m-%d %H:%M:%S", localtime(entry["time"]))
            pos = np.round(entry.get("position", [0, 0, 0]), 3).tolist()
            out += (
                f"At time={t_str}, the robot was at an average position of {pos}. "
                f"The robot saw the following: {entry.get('caption', '')}\n"
            )
            img = entry.get("image_path", "")
            if img:
                out += f"Image: {img}\n"
            out += "\n"
        return out

    def get_entry_near_timestamp(self, ts: float, tolerance: float = 2.0) -> dict:
        """Return the stored entry whose timestamp is closest to ts, within tolerance seconds."""
        filter_expr = f"time >= {ts - tolerance} and time <= {ts + tolerance}"
        if self.time_start is not None:
            filter_expr += f" and time >= {self.time_start}"
        if self.time_end is not None:
            filter_expr += f" and time <= {self.time_end}"
        rows = self.wrapper.query(filter_expr=filter_expr)
        if not rows:
            return None
        return min(rows, key=lambda r: abs(r["time"] - ts))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hits_to_entries(search_results) -> list:
    """Flatten Milvus search results into a list of entry dicts."""
    entries = []
    for hits in search_results:
        for hit in hits:
            entity = hit.get("entity", {})
            entries.append(entity)
    return entries


def _rrf_fuse(id_lists: list, k: int = 60) -> list:
    """
    Reciprocal Rank Fusion across multiple ranked ID lists.
    Returns [(id, score), ...] sorted by combined score descending.

    k=60 is the standard RRF constant — it dampens the effect of very high ranks
    so a result ranked #1 in one channel doesn't completely dominate.
    """
    scores: dict = {}
    for ranked_ids in id_lists:
        for rank, entry_id in enumerate(ranked_ids):
            scores[entry_id] = scores.get(entry_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _too_close_in_time(entry: dict, existing: list, min_gap: float = 10.0) -> bool:
    """Return True if entry's timestamp is within min_gap seconds of any entry in existing."""
    t = entry.get("time", 0)
    return any(abs(t - e.get("time", 0)) < min_gap for e in existing)
