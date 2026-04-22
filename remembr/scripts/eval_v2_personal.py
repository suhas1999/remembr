"""
ReMEmbR v2 evaluation script for personal (non-robot) videos.

Differences from eval_v2.py:
- Uses agent_system_prompt_personal.txt (no search_near_position tool)
- Questions live in data/questions/99/human_qa.json (seq_id=99 by default)
- No position_error metric — personal videos have no GPS
- Answers are printed prominently since ground truth is TBD

Usage:
  python eval_v2_personal.py \
      --seq_id 99 \
      --db_path ./remembr_v2_seq99.db \
      --llm claude-sonnet-4-6 \
      [--gemini_api_key AIza...] \
      [--openai_api_key sk-...]
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


def sanitize(text: str, max_len: int = 45) -> str:
    text = text.split("\n")[-1].strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text.replace(" ", "_")[:max_len]


def save_image_with_label(src_path: str, dst_path: str, label: str):
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


def _to_float(val, default=0.0):
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def evaluate_output(qa_instance: dict, predicted: dict) -> dict:
    out_error = {}
    q_type = qa_instance["type"]

    if "binary" in q_type:
        gt = qa_instance["answers"].get("binary", "")
        pred = (predicted.get("binary") or "").lower()
        if gt and gt != "unknown - verify from video":
            out_error["binary_correct"] = int(pred == gt.lower())
        else:
            out_error["binary_predicted"] = pred
    elif "duration" in q_type:
        gt = qa_instance["answers"].get("duration", 0.0)
        pred = _to_float(predicted.get("duration"))
        if gt != 0.0:
            out_error["duration_error_s"] = float(abs(gt - pred))
        else:
            out_error["duration_predicted_s"] = pred
    elif "time" in q_type:
        gt = qa_instance["answers"].get("time", 0.0)
        pred = _to_float(predicted.get("time"))
        if gt != 0.0:
            out_error["time_error"] = float(abs(gt - pred))
        else:
            out_error["time_predicted"] = pred

    return out_error


def main(args):
    questions_path = os.path.join(args.data_dir, "questions", str(args.seq_id), "human_qa.json")
    with open(questions_path) as f:
        qa_file = json.load(f)
    questions = qa_file["data"]

    if args.max_questions is not None:
        questions = questions[: args.max_questions]

    print(f"Loaded {len(questions)} questions for sequence {args.seq_id}")
    if "video_info" in qa_file:
        print(f"Video: {qa_file['video_info'].get('source', 'unknown')}")

    memory = MilvusMemoryV2(
        db_collection_name=f"v2_seq{args.seq_id}",
        db_path=args.db_path,
    )

    top_level = os.path.join(os.path.dirname(__file__), "..")
    personal_prompt_path = os.path.join(top_level, "prompts/agent_system_prompt_personal.txt")
    agent = ReMEmbRAgentV2(
        llm_type=args.llm,
        temperature=args.temperature,
        agent_prompt_path=personal_prompt_path,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    all_responses = []

    for q_idx, qa in enumerate(questions):
        if args.start_from is not None and q_idx < args.start_from:
            continue

        question_text = qa["question"]
        q_type = qa["type"]
        q_id = qa["id"]

        print(f"\n[{q_idx:02d}] {q_type} | {q_id}")
        print(f"     Q: {question_text.split(chr(10))[-1][:100]}")

        short_q = sanitize(question_text)
        folder = f"q_{q_idx:02d}_{short_q}"
        out_path = os.path.join(args.out_dir, folder)
        ret_path = os.path.join(out_path, "retrieved_images")
        os.makedirs(ret_path, exist_ok=True)

        if os.path.exists(os.path.join(out_path, "info.txt")):
            print(f"     [SKIP] already done")
            continue

        memory.set_time_window(qa["start_time"], qa["end_time"])
        agent.set_memory(memory)

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
            all_responses.append({
                "question": question_text, "id": q_id,
                "response": {}, "error": {}, "elapsed": 0,
            })
            continue

        print(f"     A (text): {parsed.get('text', '')[:120]}")
        if parsed.get("binary"):
            print(f"     A (binary): {parsed['binary']}")
        if parsed.get("duration"):
            print(f"     A (duration_s): {parsed['duration']}")

        # Save tool calls
        with open(os.path.join(out_path, "tool_calls.txt"), "w") as f:
            f.write(f"Question: {question_text.split(chr(10))[-1].strip()}\n")
            f.write(f"Type: {q_type}\n\n")
            executions = [c for c in agent.tool_call_log if "result_preview" in c]
            if not executions:
                f.write("No tool calls made.\n")
            else:
                f.write(f"Total tool calls: {len(executions)}\n\n")
                for i, exe in enumerate(executions):
                    f.write(f"── Call {i+1}: {exe['tool']} ──\n")
                    f.write(f"  Args:   {exe['args']}\n")
                    f.write(f"  Result: {exe['result_preview']}\n\n")

        # Save retrieved images
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
            revisit = " [REVISIT]" if entry.get("is_revisit", 0) > 0 else ""
            retrieved_lines.append(
                f"rank={rank:02d}  time={t_str}{revisit}\n"
                f"  caption: {entry.get('caption', '')[:200]}\n"
                f"  image_path: {img_path}  saved={saved}\n"
            )

        with open(os.path.join(out_path, "retrieved_entries.txt"), "w") as f:
            f.write(f"Question: {question_text.split(chr(10))[-1].strip()}\n")
            f.write(f"Type: {q_type}\n")
            win_s = strftime("%Y-%m-%d %H:%M:%S", localtime(qa["start_time"]))
            win_e = strftime("%Y-%m-%d %H:%M:%S", localtime(qa["end_time"]))
            f.write(f"Window: {win_s} → {win_e}\n")
            f.write(f"Total retrieved: {len(retrieved_entries)}\n\n")
            f.write("\n".join(retrieved_lines))

        with open(os.path.join(out_path, "info.txt"), "w") as f:
            f.write(f"Index:    {q_idx}\n")
            f.write(f"ID:       {q_id}\n")
            f.write(f"Type:     {q_type}\n")
            win_s = strftime("%Y-%m-%d %H:%M:%S", localtime(qa["start_time"]))
            win_e = strftime("%Y-%m-%d %H:%M:%S", localtime(qa["end_time"]))
            f.write(f"Window:   {win_s} → {win_e}\n\n")
            f.write(f"Question:\n{question_text.split(chr(10))[-1].strip()}\n\n")
            f.write(f"Ground truth:\n{json.dumps(qa.get('answers', {}), indent=2)}\n\n")
            f.write(f"Agent response:\n{json.dumps(parsed, indent=2)}\n\n")
            f.write(f"Metrics:\n{json.dumps(out_error, indent=2)}\n\n")
            f.write(f"Elapsed: {elapsed:.1f}s\n")
            f.write(f"Retrieved {len(retrieved_entries)} entries.\n")

        out_dict = {
            "question": question_text, "id": q_id,
            "response": parsed, "error": out_error, "elapsed": elapsed,
        }
        all_responses.append(out_dict)
        print(f"     Saved: {len(seen_paths)} images → {folder}/")
        time.sleep(args.sleep)

    # Write full results JSON
    results_dir = os.path.join(args.out_dir, "eval_results")
    os.makedirs(results_dir, exist_ok=True)
    out_json_path = os.path.join(results_dir, f"seq{args.seq_id}_personal_{args.llm}.json")
    with open(out_json_path, "w") as f:
        json.dump({"version": "2.0_personal", "responses": all_responses}, f, indent=4)

    print(f"\nDone. Results: {out_json_path}")
    print(f"Frames:        {args.out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ReMEmbR v2 on personal video (seq 99)")
    parser.add_argument("--seq_id", type=int, default=99)
    parser.add_argument("--llm", type=str, default="gpt-4o")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--gemini_api_key", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--db_path", type=str, default="./remembr_v2_seq99.db")
    parser.add_argument("--out_dir", type=str, default="./analysis_v2_personal")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--start_from", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=5.0)
    args = parser.parse_args()

    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    if args.gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = args.gemini_api_key

    if "gpt" in args.llm and not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set.")

    main(args)
