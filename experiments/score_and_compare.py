"""
Score all eval results (v1 and v2) and produce a comparison table.

Scoring thresholds:
  position:  error <= 15 m  → correct
  time:      error <= 0.5 min → correct
  duration:  error <= 0.5 min → correct
  binary:    exact match     → correct
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_ROOT = Path("/home/suhas/remembr/experiments/results")
V1_ROOT = Path("/home/suhas/remembr/experiments/v1_baselines")

THRESHOLDS = {
    "position": 15.0,   # metres
    "time": 0.5,        # minutes
    "duration": 0.5,    # minutes
    "binary": None,     # exact match (binary_correct == 1)
}


def parse_info_txt(path: Path) -> dict | None:
    """Parse an info.txt file into a structured dict."""
    try:
        text = path.read_text()
    except Exception:
        return None

    def _extract(label):
        m = re.search(rf"^{label}:\s*(.+)$", text, re.MULTILINE)
        return m.group(1).strip() if m else None

    category = _extract("Category")
    q_type = _extract("Type")

    # Parse Error metrics block
    m = re.search(r"Error metrics:\n(\{.*?\})", text, re.DOTALL)
    if not m:
        return None
    try:
        error_metrics = json.loads(m.group(1))
    except Exception:
        return None

    return {
        "category": category,
        "type": q_type,
        "error": error_metrics,
    }


def score_entry(entry: dict) -> int:
    """Return 1 if correct, 0 if wrong, -1 if unanswerable."""
    err = entry["error"]
    q_type = entry["type"]

    if "position" in q_type:
        e = err.get("position_error")
        if e is None:
            return -1
        return int(e <= THRESHOLDS["position"])
    elif "time" in q_type:
        e = err.get("time_error")
        if e is None:
            return -1
        return int(e <= THRESHOLDS["time"])
    elif "duration" in q_type:
        e = err.get("duration_error")
        if e is None:
            return -1
        return int(e <= THRESHOLDS["duration"])
    elif "binary" in q_type:
        c = err.get("binary_correct")
        if c is None:
            return -1
        return int(c == 1)
    return -1


def load_eval_dir(eval_dir: Path) -> list[dict]:
    """Load all info.txt files from an eval directory."""
    results = []
    for q_dir in sorted(eval_dir.glob("q_*")):
        info = q_dir / "info.txt"
        if not info.exists():
            continue
        parsed = parse_info_txt(info)
        if parsed is None:
            continue
        parsed["q_dir"] = q_dir.name
        results.append(parsed)
    return results


def compute_accuracy(entries: list[dict], filter_type=None, filter_cat=None) -> dict:
    """Compute accuracy stats for a subset of entries."""
    filtered = entries
    if filter_type:
        filtered = [e for e in filtered if filter_type in (e["type"] or "")]
    if filter_cat:
        filtered = [e for e in filtered if e["category"] == filter_cat]

    if not filtered:
        return {"n": 0, "correct": 0, "acc": float("nan")}

    scores = [score_entry(e) for e in filtered]
    answered = [s for s in scores if s >= 0]
    correct = sum(s for s in answered if s == 1)
    n = len(filtered)
    return {
        "n": n,
        "correct": correct,
        "failed": n - len(answered),
        "acc": correct / n if n > 0 else float("nan"),
        "acc_answered": correct / len(answered) if answered else float("nan"),
    }


# ── Define all pipelines ───────────────────────────────────────────────────────

PIPELINES = {
    # seq0
    "v1_seq0":          (V1_ROOT / "analysis_gpt4o", 0),
    "v2_seq0_clipvlm_t95": (RESULTS_ROOT / "seq0_clipvlm_t95", 0),
    "v2_seq0_clipvlm_t90": (RESULTS_ROOT / "seq0_clipvlm_t90", 0),
    "v2_seq0_clip_t95": (RESULTS_ROOT / "seq0_clip_t95", 0),
    "v2_seq0_clip_t90": (RESULTS_ROOT / "seq0_clip_t90", 0),
    "v2_seq0_random_t95": (RESULTS_ROOT / "seq0_random_t95", 0),
    "v2_seq0_random_t90": (RESULTS_ROOT / "seq0_random_t90", 0),
    # seq4
    "v1_seq4":          (V1_ROOT / "analysis_gpt4o_seq4", 4),
    "v2_seq4_clipvlm_t95": (RESULTS_ROOT / "seq4_clipvlm_t95", 4),
    "v2_seq4_clipvlm_t90": (RESULTS_ROOT / "seq4_clipvlm_t90", 4),
    "v2_seq4_clip_t95": (RESULTS_ROOT / "seq4_clip_t95", 4),
    "v2_seq4_clip_t90": (RESULTS_ROOT / "seq4_clip_t90", 4),
    "v2_seq4_random_t95": (RESULTS_ROOT / "seq4_random_t95", 4),
    "v2_seq4_random_t90": (RESULTS_ROOT / "seq4_random_t90", 4),
}

Q_TYPES = ["time", "duration", "position", "binary"]
Q_CATS = ["LONG", "MEDIUM", "SHORT"]


def print_table(data: dict[str, list[dict]]):
    """Print comparison table."""

    def acc_str(entries, **kwargs):
        r = compute_accuracy(entries, **kwargs)
        if r["n"] == 0:
            return "  —  "
        pct = r["acc"] * 100
        return f"{pct:5.1f}% ({r['correct']}/{r['n']})"

    # ── Table 1: Overall by pipeline ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TABLE 1: Overall accuracy by pipeline")
    print("=" * 80)
    header = f"{'Pipeline':<30} {'Overall':>12} {'time':>12} {'duration':>12} {'position':>12} {'binary':>12}"
    print(header)
    print("-" * 92)
    for name, entries in data.items():
        if not entries:
            print(f"{name:<30} {'(no data)':>12}")
            continue
        row = f"{name:<30}"
        row += f" {acc_str(entries):>12}"
        for qt in Q_TYPES:
            row += f" {acc_str(entries, filter_type=qt):>12}"
        print(row)

    # ── Table 2: By length category ───────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TABLE 2: Accuracy by length category")
    print("=" * 80)
    header = f"{'Pipeline':<30} {'LONG':>16} {'MEDIUM':>16} {'SHORT':>16}"
    print(header)
    print("-" * 82)
    for name, entries in data.items():
        if not entries:
            continue
        row = f"{name:<30}"
        for cat in Q_CATS:
            row += f" {acc_str(entries, filter_cat=cat):>16}"
        print(row)

    # ── Table 3: Per sequence, per type, per category ─────────────────────────
    for seq in [0, 4]:
        print(f"\n{'=' * 80}")
        print(f"TABLE 3: Seq{seq} — per type × category breakdown")
        print("=" * 80)
        seq_pipelines = {k: v for k, v in data.items() if f"seq{seq}" in k}
        col_w = 16

        for qt in Q_TYPES:
            print(f"\n  Question type: {qt}")
            hdr = f"  {'Pipeline':<30}"
            for cat in Q_CATS:
                hdr += f" {cat:>{col_w}}"
            hdr += f" {'ALL':>{col_w}}"
            print(hdr)
            print("  " + "-" * (30 + (col_w + 1) * 4))
            for name, entries in seq_pipelines.items():
                if not entries:
                    continue
                row = f"  {name:<30}"
                for cat in Q_CATS:
                    row += f" {acc_str(entries, filter_type=qt, filter_cat=cat):>{col_w}}"
                row += f" {acc_str(entries, filter_type=qt):>{col_w}}"
                print(row)


def main():
    data = {}
    for name, (path, _seq) in PIPELINES.items():
        if path.exists():
            entries = load_eval_dir(path)
            data[name] = entries
            print(f"Loaded {len(entries):3d} entries: {name}")
        else:
            print(f"  [MISSING] {name}: {path}")
            data[name] = []

    print_table(data)

    # Also dump raw JSON for further analysis
    out = {}
    for name, entries in data.items():
        out[name] = {
            "n": len(entries),
            "overall": compute_accuracy(entries),
            "by_type": {qt: compute_accuracy(entries, filter_type=qt) for qt in Q_TYPES},
            "by_cat": {cat: compute_accuracy(entries, filter_cat=cat) for cat in Q_CATS},
            "by_type_cat": {
                qt: {cat: compute_accuracy(entries, filter_type=qt, filter_cat=cat)
                     for cat in Q_CATS}
                for qt in Q_TYPES
            },
        }
    with open("/home/suhas/remembr/experiments/comparison_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nRaw results: /home/suhas/remembr/experiments/comparison_results.json")


if __name__ == "__main__":
    main()
