import json
import os
import subprocess
import tempfile
import base64
from flask import Flask, render_template, jsonify, send_file
import cv2
import numpy as np

app = Flask(__name__)

CAPTIONS_FILE = "/Users/suhas/Desktop/captions_Llama-3-VILA1.5-8b_3_secs.json"
VIDEO_FILE = "/Users/suhas/Downloads/seq0.mp4"
FRAME_CACHE_DIR = os.path.join(os.path.dirname(__file__), "frame_cache")

os.makedirs(FRAME_CACHE_DIR, exist_ok=True)

# Load captions
with open(CAPTIONS_FILE) as f:
    captions = json.load(f)

# Video start time from file_start of first entry
VIDEO_START_TIME = float(captions[0]['file_start'].replace('.pkl', ''))

# Get video fps
cap = cv2.VideoCapture(VIDEO_FILE)
VIDEO_FPS = cap.get(cv2.CAP_PROP_FPS)
VIDEO_TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
VIDEO_DURATION = VIDEO_TOTAL_FRAMES / VIDEO_FPS
cap.release()

print(f"Video: {VIDEO_FPS} fps, {VIDEO_TOTAL_FRAMES} frames, {VIDEO_DURATION:.1f}s")
print(f"Captions: {len(captions)} entries")
print(f"Video start unix time: {VIDEO_START_TIME}")


def get_frame_path(idx):
    return os.path.join(FRAME_CACHE_DIR, f"frame_{idx:04d}.jpg")


def extract_frame(idx):
    path = get_frame_path(idx)
    if os.path.exists(path):
        return path

    entry = captions[idx]
    caption_time = entry['time']
    video_sec = caption_time - VIDEO_START_TIME

    # Clamp to valid range
    video_sec = max(0.0, min(video_sec, VIDEO_DURATION - 0.1))

    cap = cv2.VideoCapture(VIDEO_FILE)
    frame_num = int(video_sec * VIDEO_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if ret:
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return path
    return None


@app.route('/')
def index():
    return render_template('index.html', total=len(captions))


@app.route('/api/entry/<int:idx>')
def get_entry(idx):
    if idx < 0 or idx >= len(captions):
        return jsonify({'error': 'out of range'}), 404

    entry = captions[idx]
    caption_time = entry['time']
    video_sec = caption_time - VIDEO_START_TIME

    # Extract frame if not cached
    frame_path = extract_frame(idx)
    frame_b64 = None
    if frame_path:
        with open(frame_path, 'rb') as f:
            frame_b64 = base64.b64encode(f.read()).decode()

    return jsonify({
        'idx': idx,
        'total': len(captions),
        'caption': entry['caption'],
        'time_unix': caption_time,
        'video_sec': round(video_sec, 2),
        'file_start': entry.get('file_start', ''),
        'file_end': entry.get('file_end', ''),
        'position': entry.get('position', []),
        'theta': entry.get('theta'),
        'frame_b64': frame_b64,
    })


@app.route('/api/search')
def search():
    from flask import request
    phrase = request.args.get('q', '').strip().lower()
    if not phrase:
        return jsonify({'results': [], 'total': 0})

    matches = [i for i, entry in enumerate(captions) if phrase in entry.get('caption', '').lower()]
    return jsonify({'results': matches, 'total': len(matches)})


@app.route('/api/preload')
def preload_status():
    cached = sum(1 for i in range(len(captions)) if os.path.exists(get_frame_path(i)))
    return jsonify({'cached': cached, 'total': len(captions)})


if __name__ == '__main__':
    print(f"\nStarting Caption Visualizer at http://localhost:5050\n")
    app.run(debug=False, port=5050, threaded=True)
