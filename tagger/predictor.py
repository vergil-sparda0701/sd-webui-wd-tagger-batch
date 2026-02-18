"""
tagger/predictor.py
ONNX-based inference wrapper for WaifuDiffusion tagger models.
"""

from __future__ import annotations
import csv
from pathlib import Path
from typing import Optional

import numpy as np


class WDPredictor:
    """Wraps an ONNX tagger session with label loading."""

    def __init__(self, model_path: str | Path, label_path: str | Path):
        import onnxruntime as rt

        model_path = Path(model_path)
        label_path = Path(label_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Labels not found: {label_path}")

        opts = rt.SessionOptions()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = rt.InferenceSession(str(model_path), opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        # Detect target size from model input shape
        shape = self.session.get_inputs()[0].shape  # [batch, H, W, C]
        self.target_size: int = int(shape[1]) if isinstance(shape[1], int) else 448

        self.labels = self._load_labels(label_path)
        provider = self.session.get_providers()[0]
        print(f"[WDPredictor] Loaded model ({provider}), input_size={self.target_size}")

    @staticmethod
    def _load_labels(label_path: Path) -> list[dict]:
        labels = []
        with open(label_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                labels.append({"name": row["name"], "category": int(row["category"])})
        return labels

    def predict_raw(self, image_arr: np.ndarray) -> np.ndarray:
        """Return raw probability array for a preprocessed image batch."""
        return self.session.run(None, {self.input_name: image_arr})[0]
