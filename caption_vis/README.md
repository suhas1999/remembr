# Caption Visualizer

Browse captions alongside extracted video frames in a local web UI.

## Requirements

- macOS with [Homebrew](https://brew.sh)
- Conda (Miniconda or Anaconda)

## Setup

**1. Install ffmpeg** (needed for `ffprobe` to inspect video)
```bash
brew install ffmpeg
```

**2. Create conda env and install Python deps**
```bash
conda create -n vis python=3.10 -y
conda run -n vis pip install flask opencv-python-headless numpy
```

## Running

```bash
cd caption_vis
conda run -n vis python app.py
# open http://localhost:5050
```

## What you need

Before running, update the two paths at the top of `app.py` to point to your files:

```python
CAPTIONS_FILE = "/path/to/your/captions.json"
VIDEO_FILE    = "/path/to/your/video.mp4"
```

The captions JSON is expected to be a list of objects with at least:
- `caption` — the text
- `time` — unix timestamp of the center frame
- `file_start` — used to determine video start offset (e.g. `1673884185.689107.pkl`)

## Features

- Frame + caption side by side for each entry
- Slider, Prev/Next buttons, jump-to-index
- Arrow keys `←` `→` to navigate
- Phrase search across all captions — all matching entries shown as clickable badges
- Frames cached to `frame_cache/` after first load (subsequent visits are instant)
