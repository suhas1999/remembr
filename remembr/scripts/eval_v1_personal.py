"""
v1 eval for personal video — adapted from eval_and_save_frames.py.
Changes vs original:
- Reads captions_gemini_1.5_secs.json (Gemini captioner, 1.5s windows)
- Skips position_error (no GPS in personal video)
- binary evaluation reads answers['binary'] not answers['text'][1]
- Default sequence_id=99

Usage:
  python eval_v1_personal.py \
      --sequence_id 99 \
      --llm gpt-4o --openai_api_key sk-... \
      --data_dir ./data --coda_dir ./coda_data \
      --out_dir ./analysis_v1_personal
"""

import argparse
import json
import os
import re
import time
import pickle as pkl
import traceback
import glob
from dataclasses import asdict
from time import strftime, localtime

import numpy as np
from PIL import Image
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from agents.remembr_agent import ReMEmbRAgent
from memory.memory import MemoryItem
from memory.milvus_memory import MilvusMemory


def sanitize(text, max_len=45):
    text = text.split("\n")[-1].strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text.replace(" ", "_")[:max_len]


def save_frame_from_pkl(pkl_path, out_path, timestamp_str=None):
    if not os.path.exists(pkl_path):
        return False
    with open(pkl_path, "rb") as f:
        data = pkl.load(f)
    img = data["cam0"][:, :, ::-1].astype("uint8")
    pil_img = Image.fromarray(img, "RGB")
    if timestamp_str:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        except Exception:
            font = ImageFont.load_default()
        w, h = pil_img.size
        x, y = 10, h - 50
        for dx, dy in [(-2,-2),(-2,2),(2,-2),(2,2)]:
            draw.text((x+dx, y+dy), timestamp_str, font=font, fill=(0,0,0))
        draw.text((x, y), timestamp_str, font=font, fill=(255,255,255))
    pil_img.save(out_path)
    return True


def find_caption_by_time(captions, target_time):
    times = np.array([c["time"] for c in captions])
    idx = np.argmin(np.abs(times - target_time))
    return captions[idx]


def load_memory(args, qa_instance, captions):
    start_time = qa_instance["start_time"]
    end_time = qa_instance["end_time"]

    memory = MilvusMemory(
        f"eval_v1_personal_{args.sequence_id}",
        db_path=args.db_path,
        time_offset=start_time,
    )
    memory.reset()

    all_start_times = np.array([float(c["file_start"][:-4]) for c in captions])
    all_end_times = np.array([float(c["file_end"][:-4]) for c in captions])

    start_idx = np.argmin(np.abs(all_start_times - start_time))
    end_idx = np.argmin(np.abs(all_end_times - end_time))
    window_captions = captions[start_idx:end_idx + 1]

    for item in window_captions:
        entity = MemoryItem(
            caption=item["caption"],
            position=item["position"],
            theta=item["theta"],
            time=item["time"],
        )
        memory.insert(entity, text_embedding=item["text_embedding"])

    return memory, window_captions


def _to_float(val, default=0):
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def evaluate_output(qa_instance, predicted):
    out_error = {}
    q_type = qa_instance["type"]

    if "binary" in q_type:
        gt = qa_instance["answers"].get("binary", "")
        pred = (predicted.get("binary") or "").lower()
        if gt and gt not in ("yes", "no"):  # placeholder, skip
            out_error["binary_predicted"] = pred
        else:
            out_error["binary_correct"] = int(pred == gt.lower()) if gt else -1
    elif "duration" in q_type:
        gt = qa_instance["answers"].get("duration", 0.0)
        pred = _to_float(predicted.get("duration"))
        out_error["duration_error_s"] = float(abs(gt - pred)) if gt != 0.0 else pred
    elif "time" in q_type:
        gt = qa_instance["answers"].get("time", 0.0)
        pred = _to_float(predicted.get("time"))
        out_error["time_error"] = float(abs(gt - pred)) if gt != 0.0 else pred
    # position skipped — no GPS

    return out_error


def main(args):
    questions_path = os.path.join(args.data_dir, "questions", str(args.sequence_id), "human_qa.json")
    captions_path = os.path.join(
        args.data_dir, "captions", str(args.sequence_id), "captions",
        f"captions_{args.captioner_name}_{args.seconds_per_caption}_secs.json"
    )

    with open(questions_path) as f:
        questions = json.load(f)["data"]
    with open(captions_path) as f:
        captions = json.load(f)

    if args.max_questions is not None:
        questions = questions[:args.max_questions]

    print(f"Loaded {len(questions)} questions, {len(captions)} caption segments")

    os.makedirs(args.out_dir, exist_ok=True)
    agent = ReMEmbRAgent(llm_type=args.llm, num_ctx=args.num_ctx, temperature=args.temperature)

    all_responses = []

    for q_idx, qa in enumerate(questions):
        question_text = qa["question"]
        q_type = qa["type"]
        q_id = qa["id"]

        print(f"\n[{q_idx:02d}] {q_type} | {q_id}")
        print(f"     Q: {question_text.split(chr(10))[-1][:100]}")

        short_q = sanitize(question_text)
        folder = f"q_{q_idx:02d}_{short_q}"
        out_path = os.path.join(args.out_dir, folder)
        ret_path = os.path.join(out_path, "retrieved")
        win_path = os.path.join(out_path, "window")
        os.makedirs(ret_path, exist_ok=True)
        os.makedirs(win_path, exist_ok=True)

        if os.path.exists(os.path.join(out_path, "info.txt")):
            print(f"     [SKIP] already done")
            continue

        memory, window_captions = load_memory(args, qa, captions)
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
            all_responses.append({"question": question_text, "id": q_id, "response": {}, "error": {}, "elapsed": 0})
            continue

        print(f"     A (text): {parsed.get('text','')[:120]}")

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

        # Save retrieved frames
        retrieved_docs = memory.get_working_memory()
        retrieved_lines = []
        for rank, doc in enumerate(retrieved_docs):
            raw_time = doc.metadata["time"]
            t_val = raw_time[0] if isinstance(raw_time, list) else raw_time
            unix_ts = t_val + qa["start_time"]
            cap_entry = find_caption_by_time(window_captions, unix_ts)
            pkl_name = cap_entry["file_start"]
            pkl_path = os.path.join(args.coda_dir, str(args.sequence_id), pkl_name)
            t_str = strftime("%Y-%m-%d %H:%M:%S", localtime(unix_ts))
            fname_ts = strftime("%Y-%m-%d_%H-%M-%S", localtime(unix_ts))
            frame_name = f"{fname_ts}_rank{rank:02d}.jpg"
            saved = save_frame_from_pkl(pkl_path, os.path.join(ret_path, frame_name), t_str)
            retrieved_lines.append(
                f"rank={rank:02d}  time={t_str}  frame={frame_name}\n"
                f"  caption: {doc.page_content[:200]}\n"
                f"  saved={saved}\n"
            )

        with open(os.path.join(out_path, "retrieved_captions.txt"), "w") as f:
            f.write(f"Question: {question_text.split(chr(10))[-1].strip()}\n")
            f.write(f"Total retrieved: {len(retrieved_docs)}\n\n")
            f.write("\n".join(retrieved_lines))

        # Save info
        with open(os.path.join(out_path, "info.txt"), "w") as f:
            f.write(f"Index:    {q_idx}\n")
            f.write(f"ID:       {q_id}\n")
            f.write(f"Type:     {q_type}\n\n")
            f.write(f"Question:\n{question_text.split(chr(10))[-1].strip()}\n\n")
            f.write(f"Ground truth:\n{json.dumps(qa.get('answers', {}), indent=2)}\n\n")
            f.write(f"Agent response:\n{json.dumps(parsed, indent=2)}\n\n")
            f.write(f"Metrics:\n{json.dumps(out_error, indent=2)}\n\n")
            f.write(f"Elapsed: {elapsed:.1f}s\n")

        all_responses.append({
            "question": question_text, "id": q_id,
            "response": parsed, "error": out_error, "elapsed": elapsed,
        })
        print(f"     Saved → {folder}/")

    results_dir = os.path.join(args.out_dir, "eval_results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"seq{args.sequence_id}_v1_gemini_{args.llm}.json")
    with open(out_path, "w") as f:
        json.dump({"version": "v1_personal", "responses": all_responses}, f, indent=4)
    print(f"\nDone. Results: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_id", type=int, default=99)
    parser.add_argument("--seconds_per_caption", type=float, default=3.0)
    parser.add_argument("--captioner_name", type=str, default="Llama-3-VILA1.5-8b")
    parser.add_argument("--llm", type=str, default="gpt-4o")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--coda_dir", type=str, default="./coda_data")
    parser.add_argument("--out_dir", type=str, default="./analysis_v1_personal")
    parser.add_argument("--db_path", type=str, default="./remembr_v1_personal.db")
    parser.add_argument("--num_ctx", type=int, default=8192 * 8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_questions", type=int, default=None)
    args = parser.parse_args()

    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    if "gpt" in args.llm and not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set")

    main(args)
