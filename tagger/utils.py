"""
tagger/utils.py
Helper utilities: model download, label loading, image preprocessing.
"""

from __future__ import annotations
import csv
from pathlib import Path
from typing import Optional

import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────

MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

# Category ID → name mapping
CATEGORY_NAMES = {
    9: "rating",
    1: "artist",
    3: "copyright",
    4: "character",
    0: "general",
    5: "meta",
}

# Kaomoji tags that should not have underscores replaced
KAOMOJIS = {
    "0_0","(o)_(o)","+_+","+_-","._.",
    "<o>_<o>","<|>_<|>","=_=",">_<","3_3",
    "6_9",">_o","@_@","^_^","o_o","u_u",
    "x_x","|_|","||_||",
}


# ─── Model download ───────────────────────────────────────────────────────────

def download_model(repo_id: str, dest_dir: Path) -> tuple[Path, Path]:
    """
    Download model.onnx and selected_tags.csv from a HuggingFace repo.
    Returns (model_path, label_path).
    """
    from huggingface_hub import hf_hub_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    model_path = dest_dir / MODEL_FILENAME
    label_path = dest_dir / LABEL_FILENAME

    if not model_path.exists():
        print(f"[WD Tagger] Downloading model from {repo_id} …")
        hf_hub_download(repo_id=repo_id, filename=MODEL_FILENAME, local_dir=str(dest_dir))

    if not label_path.exists():
        print(f"[WD Tagger] Downloading labels from {repo_id} …")
        hf_hub_download(repo_id=repo_id, filename=LABEL_FILENAME, local_dir=str(dest_dir))

    return model_path, label_path


# ─── Label loading ────────────────────────────────────────────────────────────

def load_labels(label_path: Path) -> list[dict]:
    """Load CSV labels file and return list of {name, category} dicts."""
    labels = []
    with open(label_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.append({"name": row["name"], "category": int(row["category"])})
    return labels


# ─── Image preprocessing ──────────────────────────────────────────────────────

def preprocess_pil(img, target_size: int = 448) -> np.ndarray:
    """
    Convert a PIL image to a numpy array suitable for ONNX inference.
    Handles RGBA→RGB, pads to square, resizes, converts to BGR float32.
    """
    from PIL import Image

    # RGBA → composite on white
    if img.mode != "RGB":
        canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
        canvas.alpha_composite(img.convert("RGBA"))
        img = canvas.convert("RGB")

    # Pad to square
    w, h = img.size
    side = max(w, h)
    padded = Image.new("RGB", (side, side), (255, 255, 255))
    padded.paste(img, ((side - w) // 2, (side - h) // 2))

    # Resize
    padded = padded.resize((target_size, target_size), Image.BICUBIC)

    # RGB → BGR, float32, batch dim
    arr = np.array(padded, dtype=np.float32)[:, :, ::-1]
    return np.expand_dims(arr, axis=0)


# ─── Tag formatting ───────────────────────────────────────────────────────────

def categorize_predictions(
    probs: np.ndarray,
    labels: list[dict],
    general_thresh: float = 0.35,
    character_thresh: float = 0.85,
) -> dict[str, list[tuple[str, float]]]:
    """
    Map raw probability array to a dict of {category: [(tag, score), …]}.
    Filters by threshold per category.
    """
    categories: dict[str, list] = {v: [] for v in CATEGORY_NAMES.values()}

    for prob, label in zip(probs, labels):
        cat_id   = label["category"]
        tag_name = label["name"]
        cat_name = CATEGORY_NAMES.get(cat_id, "general")

        thresh = character_thresh if cat_name == "character" else (0.0 if cat_name == "rating" else general_thresh)

        if prob >= thresh:
            display = tag_name if tag_name in KAOMOJIS else tag_name.replace("_", " ")
            categories[cat_name].append((display, float(prob)))

    for cat in categories:
        categories[cat].sort(key=lambda x: x[1], reverse=True)

    return categories


def tags_to_prompt(
    categories: dict[str, list[tuple[str, float]]],
    include_rating: bool = False,
) -> str:
    """Flatten categorized tags into a comma-separated prompt string."""
    order = ["character", "copyright", "artist", "general", "meta"]
    if include_rating:
        order = ["rating"] + order
    parts = [tag for cat in order for tag, _ in categories.get(cat, [])]
    return ", ".join(parts)
