"""
For each question in seq N, saves a folder containing:
  - frame_{i:02d}.jpg  : middle frame of each caption segment in the question window
  - captions.txt       : all captions with timestamps and positions
  - info.txt           : question, type, answer, time window

Output structure:
  analysis/
    q_00_LONG_When_did_you_leave.../
      frame_00.jpg
      frame_01.jpg
      ...
      captions.txt
      info.txt
    q_01_LONG_.../
      ...
"""

import argparse
import json
import os
import pickle as pkl
import re
import numpy as np
from PIL import Image
from time import strftime, localtime


def sanitize(text, max_len=40):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace(' ', '_')
    return text[:max_len]


def main(args):
    questions_path = os.path.join(args.data_dir, 'questions', str(args.seq_id), 'human_qa.json')
    captions_path  = os.path.join(args.data_dir, 'captions', str(args.seq_id), 'captions', f'{args.caption_file}.json')

    with open(questions_path, 'r') as f:
        questions = json.load(f)['data']

    with open(captions_path, 'r') as f:
        captions = json.load(f)

    caption_times = np.array([c['time'] for c in captions])

    os.makedirs(args.out_dir, exist_ok=True)

    for q_idx, qa in enumerate(questions):
        question_text = qa['question'].split('\n')[-1].strip()  # strip the prepended context
        q_type        = qa['type']
        start_time    = qa['start_time']
        end_time      = qa['end_time']
        length_cat    = qa['length_category']
        answers       = qa.get('answers', {})

        folder_name = f"q_{q_idx:02d}_{length_cat}_{sanitize(question_text)}"
        out_path = os.path.join(args.out_dir, folder_name)
        os.makedirs(out_path, exist_ok=True)

        # Find all captions whose mean time falls within [start_time, end_time]
        mask = (caption_times >= start_time) & (caption_times <= end_time)
        window_captions = [c for c, m in zip(captions, mask) if m]

        # ── Save frames ───────────────────────────────────────────────────────
        for i, cap in enumerate(window_captions):
            pkl_filename = cap['file_start']
            pkl_path = os.path.join(args.coda_dir, str(args.seq_id), pkl_filename)

            if not os.path.exists(pkl_path):
                print(f"  [WARN] pkl not found: {pkl_path}")
                continue

            with open(pkl_path, 'rb') as f:
                data = pkl.load(f)

            img = data['cam0'][:, :, ::-1].astype('uint8')  # BGR → RGB
            img_pil = Image.fromarray(img, 'RGB')
            img_pil.save(os.path.join(out_path, f'frame_{i:02d}.jpg'))

        # ── Save captions.txt ─────────────────────────────────────────────────
        with open(os.path.join(out_path, 'captions.txt'), 'w') as f:
            f.write(f"Question window: {strftime('%H:%M:%S', localtime(start_time))} → {strftime('%H:%M:%S', localtime(end_time))}\n")
            f.write(f"Total captions in window: {len(window_captions)}\n\n")
            for i, cap in enumerate(window_captions):
                t = strftime('%H:%M:%S', localtime(cap['time']))
                pos = np.round(cap['position'], 3).tolist() if isinstance(cap['position'], (list, np.ndarray)) else cap['position']
                f.write(f"[{i:02d}] time={t}  pos={pos}\n")
                f.write(f"      {cap['caption']}\n\n")

        # ── Save info.txt ──────────────────────────────────────────────────────
        with open(os.path.join(out_path, 'info.txt'), 'w') as f:
            f.write(f"Index:    {q_idx}\n")
            f.write(f"ID:       {qa['id']}\n")
            f.write(f"Category: {length_cat}\n")
            f.write(f"Type:     {q_type}\n")
            f.write(f"Window:   {strftime('%H:%M:%S', localtime(start_time))} → {strftime('%H:%M:%S', localtime(end_time))}  ({round(end_time-start_time, 1)}s)\n\n")
            f.write(f"Question:\n{question_text}\n\n")
            f.write(f"Answer:\n{json.dumps(answers, indent=2)}\n\n")
            f.write(f"Optimal context (from form_question_jsons):\n{qa.get('context', '')}\n")

        print(f"  [{q_idx:02d}] {length_cat} | {q_type} | {len(window_captions)} captions | {folder_name}")

    print(f"\nDone. {len(questions)} question folders saved to {args.out_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_id',       type=int, default=0)
    parser.add_argument('--caption_file', type=str, default='captions_Llama-3-VILA1.5-8b_3_secs')
    parser.add_argument('--data_dir',     type=str, default='./data')
    parser.add_argument('--coda_dir',     type=str, default='./coda_data')
    parser.add_argument('--out_dir',      type=str, default='./analysis')
    args = parser.parse_args()
    main(args)
