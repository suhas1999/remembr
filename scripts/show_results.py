"""
Show eval results and compare against benchmark table.

Usage:
    python3 scripts/show_results.py --results analysis_llama/eval_results/seq0_captions_Llama-3-VILA1.5-8b_3_secs.json
    python3 scripts/show_results.py --results ... --out_dir eval_reports/
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np


BENCHMARK = {
    "Ours / GPT4o":          {"binary": {"SHORT": (0.62,0.50), "MEDIUM": (0.58,0.50), "LONG": (0.65,0.50)},
                               "position": {"SHORT": (5.1,11.9),  "MEDIUM": (27.5,26.8),  "LONG": (46.25,59.6)},
                               "temporal": {"SHORT": (0.3,0.1),   "MEDIUM": (1.8,2.0),    "LONG": (3.6,5.9)}},
    "Ours / Codestral":       {"binary": {"SHORT": (0.25,0.40), "MEDIUM": (0.24,0.40), "LONG": (0.11,0.30)},
                               "position": {"SHORT": (151.3,109.7),"MEDIUM": (189.0,109.6),"LONG": (212.4,121.3)},
                               "temporal": {"SHORT": (4.8,5.6),   "MEDIUM": (8.4,6.8),    "LONG": (14.8,7.5)}},
    "Ours / Command-R":       {"binary": {"SHORT": (0.36,0.50), "MEDIUM": (0.32,0.50), "LONG": (0.14,0.30)},
                               "position": {"SHORT": (158.7,129.6),"MEDIUM": (172.2,119.4),"LONG": (188.7,107.1)},
                               "temporal": {"SHORT": (4.5,17.3),  "MEDIUM": (14.3,6.7),   "LONG": (15.3,11.7)}},
    "Ours / Llama3.1:8b":     {"binary": {"SHORT": (0.31,0.50), "MEDIUM": (0.33,0.50), "LONG": (0.21,0.40)},
                               "position": {"SHORT": (159.9,123.2),"MEDIUM": (151.2,121.1),"LONG": (165.3,115.1)},
                               "temporal": {"SHORT": (9.5,27.5),  "MEDIUM": (7.9,16.3),   "LONG": (18.7,10.8)}},
}


def fmt(mean, std):
    return f"{mean:.2f}±{std:.2f}"


def compute_metrics(path):
    with open(path) as f:
        data = json.load(f)

    buckets = defaultdict(lambda: defaultdict(list))
    for r in data["responses"]:
        if not r.get("error"):
            continue
        cat = r["id"].split("_")[0]
        err = r["error"]
        if "binary_correct" in err:
            buckets[cat]["binary"].append(err["binary_correct"])
        if "position_error" in err:
            buckets[cat]["position"].append(err["position_error"])
        if "time_error" in err:
            buckets[cat]["temporal"].append(err["time_error"])
        if "duration_error" in err:
            buckets[cat]["temporal"].append(err["duration_error"])

    metrics = {}
    for cat in ("SHORT", "MEDIUM", "LONG"):
        metrics[cat] = {}
        for key in ("binary", "position", "temporal"):
            vals = buckets[cat].get(key, [])
            if vals:
                arr = np.array(vals)
                metrics[cat][key] = (float(arr.mean()), float(arr.std()), len(vals))
            else:
                metrics[cat][key] = None
    return metrics


def print_table(label, metrics):
    cats = ["SHORT", "MEDIUM", "LONG"]

    def cell(m, key):
        v = m[key]
        if v is None:
            return "  —  "
        return fmt(v[0], v[1])

    print(f"\n{'─'*80}")
    print(f"  {label}")
    print(f"{'─'*80}")
    header = f"  {'Metric':<28}{'SHORT':>16}{'MEDIUM':>16}{'LONG':>16}"
    print(header)
    print(f"  {'-'*76}")
    for key, label_str in [("binary",   "Desc. Accuracy ↑"),
                            ("position", "Positional Error (m) ↓"),
                            ("temporal", "Temporal Error (s) ↓")]:
        row = f"  {label_str:<28}"
        for cat in cats:
            row += f"{cell(metrics[cat], key):>16}"
        # sample sizes
        ns = [metrics[c][key][2] if metrics[c][key] else 0 for c in cats]
        row += f"   n={ns}"
        print(row)
    print(f"{'─'*80}")


def print_benchmark():
    cats = ["SHORT", "MEDIUM", "LONG"]
    print(f"\n{'═'*80}")
    print("  BENCHMARK")
    print(f"{'═'*80}")
    header = f"  {'Method':<28}{'SHORT':>16}{'MEDIUM':>16}{'LONG':>16}"

    for metric_key, metric_label in [("binary",   "── Desc. Accuracy ↑"),
                                      ("position", "── Positional Error (m) ↓"),
                                      ("temporal", "── Temporal Error (s) ↓")]:
        print(f"\n  {metric_label}")
        print(f"  {header}")
        print(f"  {'-'*76}")
        for method, data in BENCHMARK.items():
            row = f"  {method:<28}"
            for cat in cats:
                v = data[metric_key].get(cat)
                row += f"{fmt(v[0], v[1]) if v else '✗':>16}"
            print(row)
    print(f"{'═'*80}")


def build_markdown(label, metrics, results_path):
    cats = ["SHORT", "MEDIUM", "LONG"]
    bench = BENCHMARK["Ours / Llama3.1:8b"]

    def cell(m, key):
        v = m[key]
        return fmt(v[0], v[1]) if v else "—"

    def bench_cell(key, cat):
        v = bench[key].get(cat)
        return fmt(v[0], v[1]) if v else "—"

    lines = []
    lines.append(f"# Eval Results: {label}\n")
    lines.append(f"**Source:** `{results_path}`\n")

    lines.append("## Your Results\n")
    lines.append("| Metric | SHORT | MEDIUM | LONG |")
    lines.append("|--------|-------|--------|------|")
    for key, label_str in [("binary",   "Desc. Accuracy ↑"),
                            ("position", "Positional Error (m) ↓"),
                            ("temporal", "Temporal Error (s) ↓")]:
        ns = [metrics[c][key][2] if metrics[c][key] else 0 for c in cats]
        row = f"| {label_str} "
        for i, cat in enumerate(cats):
            row += f"| {cell(metrics[cat], key)} (n={ns[i]}) "
        row += "|"
        lines.append(row)

    lines.append("\n## Benchmark Comparison\n")
    for metric_key, metric_label in [("binary",   "Descriptive Question Accuracy ↑"),
                                      ("position", "Positional Error (m) ↓"),
                                      ("temporal", "Temporal Error (s) ↓")]:
        lines.append(f"### {metric_label}\n")
        lines.append("| Method | SHORT | MEDIUM | LONG |")
        lines.append("|--------|-------|--------|------|")
        lines.append(f"| **{label}** | **{cell(metrics['SHORT'], metric_key)}** | **{cell(metrics['MEDIUM'], metric_key)}** | **{cell(metrics['LONG'], metric_key)}** |")
        for method, data in BENCHMARK.items():
            row = f"| {method} "
            for cat in cats:
                v = data[metric_key].get(cat)
                row += f"| {fmt(v[0], v[1]) if v else '✗'} "
            row += "|"
            lines.append(row)
        lines.append("")

    lines.append("## Δ vs Benchmark Llama3.1:8b\n")
    lines.append("> Positive = better than benchmark for accuracy; negative = better for error metrics\n")
    lines.append("| Metric | SHORT | MEDIUM | LONG |")
    lines.append("|--------|-------|--------|------|")
    for key, label_str in [("binary", "Desc. Accuracy"), ("position", "Pos. Error (m)"), ("temporal", "Temporal Error (s)")]:
        row = f"| {label_str} "
        for cat in cats:
            yours = metrics[cat][key]
            ref   = bench[key].get(cat)
            if yours and ref:
                delta = yours[0] - ref[0]
                sign  = "+" if delta >= 0 else ""
                row += f"| {sign}{delta:.2f} "
            else:
                row += "| — "
        row += "|"
        lines.append(row)

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",  required=True, help="Path to eval results JSON")
    parser.add_argument("--label",    default="Your run (Llama3.1:8b)")
    parser.add_argument("--out_dir",  default="eval_reports", help="Folder to save markdown report")
    args = parser.parse_args()

    metrics = compute_metrics(args.results)
    print_table(args.label, metrics)
    print_benchmark()

    # Quick comparison vs Llama3.1:8b benchmark row
    bench = BENCHMARK["Ours / Llama3.1:8b"]
    cats = ["SHORT", "MEDIUM", "LONG"]
    print(f"\n  Δ vs benchmark Llama3.1:8b (your - benchmark, lower is better for error metrics):")
    for key, label_str in [("binary", "Desc. Accuracy"), ("position", "Pos. Error"), ("temporal", "Temporal Error")]:
        parts = []
        for cat in cats:
            yours = metrics[cat][key]
            ref   = bench[key].get(cat)
            if yours and ref:
                delta = yours[0] - ref[0]
                sign  = "+" if delta >= 0 else ""
                parts.append(f"{cat}: {sign}{delta:.2f}")
            else:
                parts.append(f"{cat}: —")
        print(f"    {label_str:<22} {' | '.join(parts)}")
    print()

    # Save markdown report
    os.makedirs(args.out_dir, exist_ok=True)
    slug = os.path.splitext(os.path.basename(args.results))[0]
    out_path = os.path.join(args.out_dir, f"{slug}.md")
    md = build_markdown(args.label, metrics, args.results)
    with open(out_path, "w") as f:
        f.write(md)
    print(f"  Report saved → {out_path}")


if __name__ == "__main__":
    main()
