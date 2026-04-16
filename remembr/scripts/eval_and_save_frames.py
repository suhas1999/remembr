"""
Combined eval + frame saver for sequence N.

For each question:
  1. Runs the ReMEmbR agent (same as eval.py)
  2. Captures exactly which captions were retrieved via memory.get_working_memory()
  3. Saves a folder with:
       retrieved/
           {unix_ts:.3f}_rank{i:02d}.jpg   ← frames actually retrieved by the agent
       window/
           {unix_ts:.3f}.jpg               ← all frames in the question time window
       retrieved_captions.txt              ← what the agent actually retrieved (with timestamps)
       all_captions.txt                    ← all captions in window
       info.txt                            ← question, answer, agent response

Output:
  analysis/
    q_00_LONG_When_did_you_leave.../
      retrieved/
        1673884740.123_rank00.jpg
        1673884820.456_rank01.jpg
      window/
        1673884532.773.jpg
        1673884535.900.jpg
        ...
      retrieved_captions.txt
      all_captions.txt
      info.txt
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
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.remembr_agent import ReMEmbRAgent
from memory.memory import MemoryItem
from memory.milvus_memory import MilvusMemory


# ── helpers ───────────────────────────────────────────────────────────────────

def sanitize(text, max_len=45):
    text = text.split('\n')[-1].strip()  # strip prepended context line
    text = re.sub(r'[^\w\s]', '', text)
    return text.replace(' ', '_')[:max_len]


def save_frame_from_pkl(pkl_path, out_path):
    if not os.path.exists(pkl_path):
        return False
    with open(pkl_path, 'rb') as f:
        data = pkl.load(f)
    img = data['cam0'][:, :, ::-1].astype('uint8')  # BGR → RGB
    Image.fromarray(img, 'RGB').save(out_path)
    return True


def find_caption_by_time(captions, target_time):
    """Return the caption entry whose time is closest to target_time."""
    times = np.array([c['time'] for c in captions])
    idx = np.argmin(np.abs(times - target_time))
    return captions[idx]


def load_memory(args, qa_instance, captions, caption_times):
    """Load captions in [start_time, end_time] into a fresh MilvusMemory."""
    start_time = qa_instance['start_time']
    end_time   = qa_instance['end_time']

    memory = MilvusMemory(
        f"eval_frames_{args.sequence_id}",
        db_path=args.db_path,
        time_offset=start_time
    )
    memory.reset()

    all_times = np.array([float(c['file_start'][:-4]) for c in captions])
    diff = all_times - start_time
    start_idx = np.argmin(np.abs(diff))

    all_end_times = np.array([float(c['file_end'][:-4]) for c in captions])
    diff = all_end_times - end_time
    end_idx = np.argmin(np.abs(diff))

    window_captions = captions[start_idx:end_idx + 1]

    for item in window_captions:
        entity = MemoryItem(
            caption=item['caption'],
            position=item['position'],
            theta=item['theta'],
            time=item['time'],
        )
        memory.insert(entity, text_embedding=item['text_embedding'])

    return memory, window_captions


def _to_float(val, default=0):
    """Coerce val to float — handles None, strings, and numerics."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def _to_pos(val):
    """Coerce val to a 3-element float list — handles None, strings, and lists."""
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

def evaluate_output(qa_instance, predicted):
    out_error = {}
    q_type = qa_instance['type']
    if 'position' in q_type:
        answer = np.array(qa_instance['answers']['position'])
        pred_pos = np.array(_to_pos(predicted.get('position')))
        out_error['position_error'] = float(np.linalg.norm(answer - pred_pos))
    elif 'binary' in q_type:
        answer = qa_instance['answers']['text'][1]
        pred = (predicted.get('binary') or '').lower()
        out_error['binary_correct'] = int(pred == answer.lower())
    elif 'time' in q_type:
        answer = float(qa_instance['answers']['time'])
        pred = _to_float(predicted.get('time'))
        out_error['time_error'] = float(abs(answer - pred))
    elif 'duration' in q_type:
        answer = float(qa_instance['answers']['duration'])
        pred = _to_float(predicted.get('duration'))
        out_error['duration_error'] = float(abs(answer - pred))
    return out_error


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    questions_path = os.path.join(args.data_dir, 'questions', str(args.sequence_id), 'human_qa.json')
    captions_path  = os.path.join(args.data_dir, 'captions', str(args.sequence_id), 'captions',
                                  f'{args.caption_file}.json')

    with open(questions_path, 'r') as f:
        questions = json.load(f)['data']
    with open(captions_path, 'r') as f:
        captions = json.load(f)

    caption_times = np.array([c['time'] for c in captions])

    os.makedirs(args.out_dir, exist_ok=True)

    agent = ReMEmbRAgent(llm_type=args.llm, num_ctx=args.num_ctx, temperature=args.temperature)

    all_responses = []

    for q_idx, qa in enumerate(questions):
        question_text = qa['question']
        q_type        = qa['type']
        q_id          = qa['id']

        print(f"\n[{q_idx:02d}] {qa['length_category']} | {q_type}")
        print(f"     Q: {question_text.split(chr(10))[-1][:80]}")

        # ── Setup output folder ───────────────────────────────────────────────
        short_q   = sanitize(question_text)
        folder    = f"q_{q_idx:02d}_{qa['length_category']}_{short_q}"
        out_path  = os.path.join(args.out_dir, folder)
        ret_path  = os.path.join(out_path, 'retrieved')
        win_path  = os.path.join(out_path, 'window')
        os.makedirs(ret_path, exist_ok=True)
        os.makedirs(win_path, exist_ok=True)

        # ── Load memory for this question window ──────────────────────────────
        memory, window_captions = load_memory(args, qa, captions, caption_times)
        agent.set_memory(memory)

        # ── Run agent ─────────────────────────────────────────────────────────
        parsed = None
        t0 = time.time()
        try:
            response  = agent.query(question_text)
            parsed    = asdict(response)
            elapsed   = time.time() - t0
            out_error = evaluate_output(qa, parsed)
        except Exception as e:
            print(f"     [ERROR] {e}")
            traceback.print_exc()
            all_responses.append({})
            continue

        print(f"     A: {parsed}  ({elapsed:.1f}s)")

        # ── Save tool call log ────────────────────────────────────────────────
        with open(os.path.join(out_path, 'tool_calls.txt'), 'w') as f:
            f.write(f"Question: {question_text.split(chr(10))[-1].strip()}\n")
            f.write(f"Type: {q_type} | Category: {qa['length_category']}\n\n")
            calls = agent.tool_call_log
            if not calls:
                f.write("No tool calls made — agent answered from context only.\n")
            else:
                # Pair up decision entries with execution entries
                decisions = [c for c in calls if 'tool_chosen' in c]
                executions = [c for c in calls if 'tool' in c and 'result_preview' in c]
                f.write(f"Total tool calls: {len(executions)}\n\n")
                for i, exe in enumerate(executions):
                    f.write(f"── Call {i+1}: {exe['tool']} ──\n")
                    f.write(f"  Args:   {exe['args']}\n")
                    f.write(f"  Result: {exe['result_preview']}\n\n")
                if decisions:
                    f.write("── LLM decisions (raw) ──\n")
                    for d in decisions:
                        f.write(f"  step={d['step']}  tool={d['tool_chosen']}  args={d['args_chosen']}\n")
        print(f"     Tools: {len([c for c in agent.tool_call_log if 'result_preview' in c])} calls logged → {folder}/tool_calls.txt")

        # ── Capture retrieved captions (what the agent actually used) ─────────
        retrieved_docs = memory.get_working_memory()  # List[Document]

        # ── Save retrieved frames ─────────────────────────────────────────────
        retrieved_lines = []
        for rank, doc in enumerate(retrieved_docs):
            # Reconstruct original unix timestamp
            raw_time = doc.metadata['time']
            t_val    = raw_time[0] if isinstance(raw_time, list) else raw_time
            unix_ts  = t_val + qa['start_time']  # undo time_offset subtraction

            # Find matching caption entry by closest time
            cap_entry  = find_caption_by_time(window_captions, unix_ts)
            pkl_name   = cap_entry['file_start']
            pkl_path   = os.path.join(args.coda_dir, str(args.sequence_id), pkl_name)
            frame_name = f"{float(pkl_name[:-4]):.3f}_rank{rank:02d}.jpg"

            saved = save_frame_from_pkl(pkl_path, os.path.join(ret_path, frame_name))

            t_str = strftime('%H:%M:%S', localtime(unix_ts))
            pos   = np.round(doc.metadata.get('position', [0, 0, 0]), 3).tolist()
            retrieved_lines.append(
                f"rank={rank:02d}  time={t_str} ({unix_ts:.3f})  pos={pos}  frame={frame_name}\n"
                f"  caption: {doc.page_content}\n"
                f"  pkl: {pkl_name}  saved={saved}\n"
            )

        with open(os.path.join(out_path, 'retrieved_captions.txt'), 'w') as f:
            f.write(f"Question: {question_text.split(chr(10))[-1].strip()}\n")
            f.write(f"Type: {q_type} | Category: {qa['length_category']}\n")
            f.write(f"Window: {strftime('%H:%M:%S', localtime(qa['start_time']))} → "
                    f"{strftime('%H:%M:%S', localtime(qa['end_time']))} "
                    f"({round(qa['end_time']-qa['start_time'],1)}s)\n")
            f.write(f"Total retrieved: {len(retrieved_docs)}\n\n")
            f.write('\n'.join(retrieved_lines))

        # ── Save all window frames ────────────────────────────────────────────
        window_lines = []
        for cap in window_captions:
            pkl_name  = cap['file_start']
            pkl_path  = os.path.join(args.coda_dir, str(args.sequence_id), pkl_name)
            frame_name = f"{float(pkl_name[:-4]):.3f}.jpg"
            saved = save_frame_from_pkl(pkl_path, os.path.join(win_path, frame_name))

            t_str = strftime('%H:%M:%S', localtime(cap['time']))
            pos   = np.round(cap['position'], 3).tolist() if isinstance(cap['position'], list) else cap['position']
            window_lines.append(
                f"time={t_str} ({cap['time']:.3f})  pos={pos}  frame={frame_name}  saved={saved}\n"
                f"  {cap['caption']}\n"
            )

        with open(os.path.join(out_path, 'all_captions.txt'), 'w') as f:
            f.write(f"All {len(window_captions)} captions in question window:\n\n")
            f.write('\n'.join(window_lines))

        # ── Save info.txt ─────────────────────────────────────────────────────
        with open(os.path.join(out_path, 'info.txt'), 'w') as f:
            f.write(f"Index:     {q_idx}\n")
            f.write(f"ID:        {q_id}\n")
            f.write(f"Category:  {qa['length_category']}\n")
            f.write(f"Type:      {q_type}\n")
            f.write(f"Window:    {strftime('%H:%M:%S', localtime(qa['start_time']))} → "
                    f"{strftime('%H:%M:%S', localtime(qa['end_time']))} "
                    f"({round(qa['end_time']-qa['start_time'],1)}s)\n\n")
            f.write(f"Question:\n{question_text.split(chr(10))[-1].strip()}\n\n")
            f.write(f"Ground truth answer:\n{json.dumps(qa.get('answers',{}), indent=2)}\n\n")
            f.write(f"Agent response:\n{json.dumps(parsed, indent=2)}\n\n")
            f.write(f"Error metrics:\n{json.dumps(out_error, indent=2)}\n\n")
            f.write(f"Elapsed: {elapsed:.1f}s\n")
            f.write(f"Retrieved {len(retrieved_docs)} captions from {len(window_captions)} in window.\n")

        out_dict = {'question': question_text, 'id': q_id, 'response': parsed, 'error': out_error, 'elapsed': elapsed}
        all_responses.append(out_dict)
        print(f"     Saved: {len(retrieved_docs)} retrieved / {len(window_captions)} window frames → {folder}/")

    # ── Save full eval results JSON ───────────────────────────────────────────
    out_json_dir = os.path.join(args.out_dir, 'eval_results')
    os.makedirs(out_json_dir, exist_ok=True)
    out_json_path = os.path.join(out_json_dir, f'seq{args.sequence_id}_{args.caption_file}.json')
    with open(out_json_path, 'w') as f:
        json.dump({'version': 0.1, 'responses': all_responses}, f, indent=4)

    print(f"\nDone. Results: {out_json_path}")
    print(f"Frames:  {args.out_dir}/")


if __name__ == '__main__':
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence_id',  type=int,   default=0)
    parser.add_argument('--caption_file', type=str,   default='captions_Llama-3-VILA1.5-8b_3_secs')
    parser.add_argument('--llm',          type=str,   default='llama3.1:8b',
                        help='LLM to use. Options: llama3.1:8b (Ollama), gpt-4o / gpt-4o-mini (OpenAI — needs OPENAI_API_KEY), nim/<model> (NVIDIA NIM)')
    parser.add_argument('--openai_api_key', type=str, default=None,
                        help='OpenAI API key. Falls back to OPENAI_API_KEY env var.')
    parser.add_argument('--data_dir',     type=str,   default='./data')
    parser.add_argument('--coda_dir',     type=str,   default='./coda_data')
    parser.add_argument('--out_dir',      type=str,   default='./analysis')
    parser.add_argument('--db_path',      type=str,   default='./remembr.db')
    parser.add_argument('--num_ctx',      type=int,   default=8192 * 8)
    parser.add_argument('--temperature',  type=float, default=0.0)
    args = parser.parse_args()

    # Set OpenAI key if provided via flag (takes precedence over env var)
    if args.openai_api_key:
        os.environ['OPENAI_API_KEY'] = args.openai_api_key

    if 'gpt-4' in args.llm and not os.environ.get('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY is not set. Pass --openai_api_key sk-... or export OPENAI_API_KEY=sk-...")

    main(args)
