"""
Manages JPEG keyframe storage on disk.

Each keyframe is saved as a JPEG at up to 768px on the longest side (~50-80KB).
Files are named by their unix timestamp: {timestamp:.6f}.jpg
"""

import os
from PIL import Image


MAX_RESOLUTION = 768
JPEG_QUALITY = 85


class FrameStore:

    def __init__(self, store_dir: str):
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)

    def save(self, image: Image.Image, timestamp: float) -> str:
        """Save a PIL image as JPEG. Returns absolute path to saved file."""
        img = _resize(image)
        filename = f"{timestamp:.6f}.jpg"
        path = os.path.join(self.store_dir, filename)
        img.save(path, format="JPEG", quality=JPEG_QUALITY)
        return path

    def load(self, path: str) -> Image.Image:
        """Load a JPEG from disk as RGB PIL Image."""
        return Image.open(path).convert("RGB")


def _resize(image: Image.Image) -> Image.Image:
    w, h = image.size
    if max(w, h) > MAX_RESOLUTION:
        scale = MAX_RESOLUTION / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image
