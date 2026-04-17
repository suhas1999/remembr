"""
SigLIP-SO400M encoder for visual and text embeddings.

Used during memory building (encode every raw frame) and query time (encode text query
for visual similarity search). Produces 1152-dim L2-normalized embeddings for both images
and text, so cosine similarity is just a dot product.
"""

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


class SigLIPEncoder:

    MODEL_NAME = "google/siglip-so400m-patch14-384"
    DIM = 1152

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SigLIP] Loading {self.MODEL_NAME} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device)
        self.model.eval()
        print("[SigLIP] Ready.")

    @torch.no_grad()
    def encode_images(self, images: list, batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of PIL Images in batches.
        Returns float32 array of shape (N, 1152), L2-normalized.
        """
        all_embs = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            feats = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embs.append(feats.cpu().float().numpy())
        return np.concatenate(all_embs, axis=0)

    @torch.no_grad()
    def encode_text(self, texts: list) -> np.ndarray:
        """
        Encode a list of text strings.
        Returns float32 array of shape (N, 1152), L2-normalized.
        """
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(self.device)
        feats = self.model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().numpy()

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Dot product of two L2-normalized vectors = cosine similarity."""
        return float(np.dot(a.flatten(), b.flatten()))
