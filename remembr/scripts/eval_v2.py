"""
ReMEmbR v2 evaluation script.

Differences from v1 (eval_and_save_frames.py):
- Uses a pre-built MilvusMemoryV2 database (built by build_memory_v2.py)
- Per-question time window applied as a filter on the pre-built DB (no re-insertion)
- Uses ReMEmbRAgentV2 with 4 tools including examine_keyframes
- Saves retrieved keyframe images from the pre-built keyframes directory

Usage:
  python eval_v2.py --seq_id 0 --llm gpt-4o --openai_api_key sk-...
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
from dataclasses import asdict
from time import strftime, localtime

import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from agents.remembr_agent_v2 import ReMEmbRAgentV2
from memory.milvus_memory_v2 import MilvusMemoryV2


# ── Helpers ───────────────────────────────────────────────────────────────────

def sanitize(text: str, max_len: int = 45) -> str:
    text = text.split("\n")[-1].strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text.replace(" ", "_")[:max_len]


def save_image_with_label(src_path: str, dst_path: str, label: str):
    """Copy a keyframe to the output dir with a timestamp label burned in."""
    if not src_path or not os.path.exists(src_path):
        return False
    img = Image.open(src_path).convert("RGB")
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        except Exception:
            font = ImageFont.load_default()
        w, h = img.size
        x, y = 10, h - 44
        for dx, dy in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
            draw.text((x + dx, y + dy), label, font=font, fill=(0, 0, 0))
        draw.text((x, y), label, font=font, fill=(255, 255, 255))
    except Exception:
        pass
    img.save(dst_path, format="JPEG", quality=90)
    return True


def evaluate_output(qa_instance: dict, predicted: dict) -> dict:
    """Compute error metrics for a single question."""
    out_error = {}
    q_type = qa_instance["type"]

    if "position" in q_type:
        answer = np.array(qa_instance["answers"]["position"])
        pred_pos = np.array(_to_pos(predicted.get("position")))
        out_error["position_error"] = float(np.linalg.norm(answer - pred_pos))
    elif "binary" in q_type:
        answer = qa_instance["answers"]["text"][1]
        pred = (predicted.get("binary") or "").lower()
        out_error["binary_correct"] = int(pred == answer.lower())
    elif "time" in q_type:
        answer = float(qa_instance["answers"]["time"])
        pred = _to_float(predicted.get("time"))
        out_error["time_error"] = float(abs(answer - pred))
    elif "duration" in q_type:
        answer = float(qa_instance["answers"]["duration"])
        pred = _to_float(predicted.get("duration"))
        out_error["duration_error"] = float(abs(answer - pred))

    return out_error


def _to_float(val, default=0.0):
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _to_pos(val):
    if val is None:
        return [0, 0, 0]
    if isinstance(val, str):
        try:
            val = eval(val)
        except Exception:
            return [0, 0, 0]
    try:
        return [float(x) for x in val]
    except (TypeError, ValueError):
        return [0, 0, 0]


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    questions_path = os.path.join(args.data_dir, "questions", str(args.seq_id), "human_qa.json")
    with open(questions_path) as f:
        questions = json.load(f)["data"]

    if args.max_questions is not None:
        questions = questions[: args.max_questions]

    print(f"Loaded {len(questions)} questions for sequence {args.seq_id}")

    # Load the pre-built v2 memory (built by build_memory_v2.py)
    memory = MilvusMemoryV2(
        db_collection_name=f"v2_seq{args.seq_id}",
        db_path=args.db_path,
    )

    agent = ReMEmbRAgentV2(llm_type=args.llm, temperature=args.temperature)

    os.makedirs(args.out_dir, exist_ok=True)
    all_responses = []

    for q_idx, qa in enumerate(questions):
        if args.start_from is not None and q_idx < args.start_from:
            continue

        question_text = qa["question"]
        q_type = qa["type"]
        q_id = qa["id"]

        print(f"\n[{q_idx:02d}] {qa['length_category']} | {q_type}")
        print(f"     Q: {question_text.split(chr(10))[-1][:80]}")

        # ── Output folder ─────────────────────────────────────────────────────
        short_q = sanitize(question_text)
        folder = f"q_{q_idx:02d}_{qa['length_category']}_{short_q}"
        out_path = os.path.join(args.out_dir, folder)
        ret_path = os.path.join(out_path, "retrieved_images")
        os.makedirs(ret_path, exist_ok=True)

        # ── Skip if already done ──────────────────────────────────────────────
        if os.path.exists(os.path.join(out_path, "info.txt")):
            print(f"     [SKIP] already done")
            continue

        # ── Set time window for this question ─────────────────────────────────
        # The v2 database covers the full sequence. We apply a time filter to
        # restrict all searches to the question's window (same scope as v1).
        memory.set_time_window(qa["start_time"], qa["end_time"])
        agent.set_memory(memory)

        # ── Run agent ─────────────────────────────────────────────────────────
        debug_log = os.path.join(out_path, "llm_prompts.txt")
        parsed = None
        t0 = time.time()
        try:
            response = agent.query(question_text, debug_log_path=debug_log)
            parsed = asdict(response)
            elapsed = time.time() - t0
            out_error = evaluate_output(qa, parsed)
        except Exception as e:
            print(f"     [ERROR] {e}")
            traceback.print_exc()
            all_responses.append({"question": question_text, "id": q_id, "response": {}, "error": {}, "elapsed": 0})
            continue

        print(f"     A: {parsed}  ({elapsed:.1f}s)")

        # ── Save tool call log ────────────────────────────────────────────────
        _save_tool_calls(out_path, question_text, q_type, qa, agent.tool_call_log)

        # ── Save retrieved keyframe images ────────────────────────────────────
        retrieved_entries = memory.get_working_memory()
        retrieved_lines = []

        seen_paths = set()
        for rank, entry in enumerate(retrieved_entries):
            img_path = entry.get("image_path", "")
            if not img_path or img_path in seen_paths:
                continue
            seen_paths.add(img_path)

            t_str = strftime("%Y-%m-%d %H:%M:%S", localtime(entry["time"]))
            fname_ts = strftime("%Y-%m-%d_%H-%M-%S", localtime(entry["time"]))
            dst_name = f"{fname_ts}_rank{rank:02d}.jpg"
            label = f"{t_str}  rank{rank:02d}"
            saved = save_image_with_label(img_path, os.path.join(ret_path, dst_name), label)

            pos = np.round(entry.get("position", [0, 0, 0]), 3).tolist()
            revisit = " [REVISIT]" if entry.get("is_revisit", 0) > 0 else ""
            retrieved_lines.append(
                f"rank={rank:02d}  time={t_str}  pos={pos}{revisit}\n"
                f"  caption: {entry.get('caption', '')[:200]}\n"
                f"  image_path: {img_path}  saved={saved}\n"
            )

        # ── Save retrieved_entries.txt ────────────────────────────────────────
        with open(os.path.join(out_path, "retrieved_entries.txt"), "w") as f:
            f.write(f"Question: {question_text.split(chr(10))[-1].strip()}\n")
            f.write(f"Type: {q_type} | Category: {qa['length_category']}\n")
            win_s = strftime("%Y-%m-%d %H:%M:%S", localtime(qa["start_time"]))
            win_e = strftime("%Y-%m-%d %H:%M:%S", localtime(qa["end_time"]))
            f.write(f"Window: {win_s} → {win_e} ({round(qa['end_time']-qa['start_time'],1)}s)\n")
            f.write(f"Total retrieved: {len(retrieved_entries)}\n\n")
            f.write("\n".join(retrieved_lines))

        # ── Save info.txt ─────────────────────────────────────────────────────
        with open(os.path.join(out_path, "info.txt"), "w") as f:
            f.write(f"Index:     {q_idx}\n")
            f.write(f"ID:        {q_id}\n")
            f.write(f"Category:  {qa['length_category']}\n")
            f.write(f"Type:      {q_type}\n")
            win_s = strftime("%Y-%m-%d %H:%M:%S", localtime(qa["start_time"]))
            win_e = strftime("%Y-%m-%d %H:%M:%S", localtime(qa["end_time"]))
            f.write(f"Window:    {win_s} → {win_e} ({round(qa['end_time']-qa['start_time'],1)}s)\n\n")
            f.write(f"Question:\n{question_text.split(chr(10))[-1].strip()}\n\n")
            f.write(f"Ground truth:\n{json.dumps(qa.get('answers', {}), indent=2)}\n\n")
            f.write(f"Agent response:\n{json.dumps(parsed, indent=2)}\n\n")
            f.write(f"Error metrics:\n{json.dumps(out_error, indent=2)}\n\n")
            f.write(f"Elapsed: {elapsed:.1f}s\n")
            f.write(f"Retrieved {len(retrieved_entries)} entries.\n")

        out_dict = {
            "question": question_text, "id": q_id,
            "response": parsed, "error": out_error, "elapsed": elapsed,
        }
        all_responses.append(out_dict)
        print(f"     Saved: {len(seen_paths)} images → {folder}/")
        time.sleep(args.sleep)  # avoid TPM rate limit between questions

    # ── Full eval results JSON ────────────────────────────────────────────────
    results_dir = os.path.join(args.out_dir, "eval_results")
    os.makedirs(results_dir, exist_ok=True)
    out_json_path = os.path.join(results_dir, f"seq{args.seq_id}_v2_{args.llm}.json")
    with open(out_json_path, "w") as f:
        json.dump({"version": "2.0", "responses": all_responses}, f, indent=4)

    print(f"\nDone. Results: {out_json_path}")
    print(f"Frames:        {args.out_dir}/")


def _save_tool_calls(out_path, question_text, q_type, qa, tool_call_log):
    with open(os.path.join(out_path, "tool_calls.txt"), "w") as f:
        f.write(f"Question: {question_text.split(chr(10))[-1].strip()}\n")
        f.write(f"Type: {q_type} | Category: {qa['length_category']}\n\n")
        executions = [c for c in tool_call_log if "result_preview" in c]
        if not executions:
            f.write("No tool calls made — agent answered from context only.\n")
            return
        f.write(f"Total tool calls: {len(executions)}\n\n")
        for i, exe in enumerate(executions):
            f.write(f"── Call {i+1}: {exe['tool']} ──\n")
            f.write(f"  Args:   {exe['args']}\n")
            f.write(f"  Result: {exe['result_preview']}\n\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ReMEmbR v2 on a sequence")
    parser.add_argument("--seq_id", type=int, default=0)
    parser.add_argument("--llm", type=str, default="gpt-4o",
                        help="LLM for the agent. Options: gpt-4o, gpt-4o-mini, llama3.1:8b")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--db_path", type=str, default="./remembr_v2.db",
                        help="Path to the MilvusMemoryV2 database built by build_memory_v2.py")
    parser.add_argument("--out_dir", type=str, default="./analysis_v2")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--start_from", type=int, default=None,
                        help="Skip questions before this index (0-based)")
    parser.add_argument("--sleep", type=float, default=5.0,
                        help="Seconds to sleep between questions (default 5)")
    args = parser.parse_args()

    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key

    if "gpt" in args.llm and not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set.")

    main(args)
