"""
WD Tagger Batch - Stable Diffusion WebUI Extension
Based on avan06/wd-tagger-images (https://github.com/avan06/wd-tagger-images)
Original WD Tagger by SmilingWolf (https://huggingface.co/spaces/SmilingWolf/wd-tagger)

Features:
- Batch processing of multiple images
- Categorized tag output (general, characters, copyright, etc.)
- Multiple model support (SwinV2, ConvNextV2, ViT, EVA02, etc.)
- Configurable confidence thresholds
- Save tags to .txt files alongside images
"""

import os
import sys
import csv
import glob
import json
import time
import shutil
import traceback
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np

# Add extension directory to path
ext_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ext_dir not in sys.path:
    sys.path.insert(0, ext_dir)

# Try importing modules from the tagger package
try:
    from tagger.predictor import WDPredictor
    from tagger.utils import download_model, load_labels, CATEGORY_NAMES
    TAGGER_AVAILABLE = True
except Exception as e:
    TAGGER_AVAILABLE = False
    print(f"[WD Tagger Batch] Warning: Could not import tagger modules: {e}")

# ─── Model definitions ────────────────────────────────────────────────────────

MODELS = {
    "SwinV2 v3 (Recommended)":    "SmilingWolf/wd-swinv2-tagger-v3",
    "ConvNextV2 v3":               "SmilingWolf/wd-convnext-tagger-v3",
    "ViT v3":                      "SmilingWolf/wd-vit-tagger-v3",
    "ViT Large v3":                "SmilingWolf/wd-vit-large-tagger-v3",
    "EVA02 Large v3":              "SmilingWolf/wd-eva02-large-tagger-v3",
    "ConvNext v2 (Legacy)":        "SmilingWolf/wd-v1-4-convnext-tagger-v2",
    "SwinV2 v2 (Legacy)":          "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
    "ViT v2 (Legacy)":             "SmilingWolf/wd-v1-4-vit-tagger-v2",
    # IdolSankaku series
    "EVA02 Large IS v1":           "deepghs/idolsankaku-eva02-large-tagger-v1",
    "SwinV2 IS v1":                "deepghs/idolsankaku-swinv2-tagger-v1",
}

MODEL_FILENAME  = "model.onnx"
LABEL_FILENAME  = "selected_tags.csv"

# Tag category IDs in selected_tags.csv (column: category)
# 0=general, 1=artist, 3=copyright, 4=character, 5=meta, 9=rating
CATEGORY_MAP = {
    9: "rating",
    1: "artist",
    3: "copyright",
    4: "character",
    0: "general",
    5: "meta",
}

# kaomoji tags (from original project)
KAOMOJIS = [
    "0_0","(o)_(o)","+_+","+_-","._.",
    "<o>_<o>","<|>_<|>","=_=",">_<","3_3",
    "6_9",">_o","@_@","^_^","o_o","u_u",
    "x_x","|_|","||_||",
]

# ─── Global state ─────────────────────────────────────────────────────────────

_predictor: Optional[object] = None
_current_model: Optional[str] = None

# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_models_dir() -> Path:
    """Return path where ONNX models are cached."""
    base = Path(ext_dir) / "models"
    base.mkdir(parents=True, exist_ok=True)
    return base


def load_model(model_name: str):
    """Load (or reuse) the selected ONNX model."""
    global _predictor, _current_model

    if _current_model == model_name and _predictor is not None:
        return _predictor, None

    repo_id = MODELS.get(model_name)
    if repo_id is None:
        return None, f"Unknown model: {model_name}"

    model_dir = get_models_dir() / repo_id.replace("/", "--")
    model_path = model_dir / MODEL_FILENAME
    label_path = model_dir / LABEL_FILENAME

    # Download if needed
    if not model_path.exists() or not label_path.exists():
        try:
            from huggingface_hub import hf_hub_download
            model_dir.mkdir(parents=True, exist_ok=True)
            print(f"[WD Tagger Batch] Downloading {repo_id} …")
            hf_hub_download(repo_id=repo_id, filename=MODEL_FILENAME, local_dir=str(model_dir))
            hf_hub_download(repo_id=repo_id, filename=LABEL_FILENAME, local_dir=str(model_dir))
        except Exception as e:
            return None, f"Download failed: {e}"

    # Load ONNX model
    try:
        import onnxruntime as rt
        sess_options = rt.SessionOptions()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = rt.InferenceSession(str(model_path), sess_options=sess_options, providers=providers)
        provider_used = session.get_providers()[0]
        print(f"[WD Tagger Batch] Model loaded using {provider_used}")
    except Exception as e:
        return None, f"Failed to load ONNX model: {e}"

    # Load labels
    try:
        labels = []
        with open(label_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append({
                    "name": row["name"],
                    "category": int(row["category"]),
                })
    except Exception as e:
        return None, f"Failed to load labels: {e}"

    _predictor = {"session": session, "labels": labels}
    _current_model = model_name
    return _predictor, None


def preprocess_image(img, target_size: int = 448):
    """Resize and normalize image for model input."""
    from PIL import Image
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.convert("RGBA")
    # Paste on white background
    canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
    canvas.alpha_composite(img)
    img = canvas.convert("RGB")
    # Pad to square
    w, h = img.size
    side = max(w, h)
    padded = Image.new("RGB", (side, side), (255, 255, 255))
    padded.paste(img, ((side - w) // 2, (side - h) // 2))
    # Resize
    padded = padded.resize((target_size, target_size), Image.BICUBIC)
    arr = np.array(padded, dtype=np.float32)
    # BGR conversion (model expects BGR)
    arr = arr[:, :, ::-1]
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(predictor, image_arr, general_thresh: float, character_thresh: float):
    """Run inference and return categorized tag dict."""
    session = predictor["session"]
    labels = predictor["labels"]

    input_name = session.get_inputs()[0].name
    probs = session.run(None, {input_name: image_arr})[0][0]

    # Build categorized results
    categories: dict[str, list] = {v: [] for v in CATEGORY_MAP.values()}

    for prob, label in zip(probs, labels):
        cat_id   = label["category"]
        tag_name = label["name"]
        cat_name = CATEGORY_MAP.get(cat_id, "general")

        # Choose threshold per category
        if cat_name == "character":
            thresh = character_thresh
        elif cat_name == "rating":
            thresh = 0.0  # always include rating
        else:
            thresh = general_thresh

        if prob >= thresh:
            # replace underscores except in kaomojis
            display = tag_name if tag_name in KAOMOJIS else tag_name.replace("_", " ")
            categories[cat_name].append((display, float(prob)))

    # Sort each category by score desc
    for cat in categories:
        categories[cat].sort(key=lambda x: x[1], reverse=True)

    return categories


def format_tags(categories: dict, include_rating: bool = False) -> str:
    """Format categorized tags into a comma-separated prompt string."""
    parts = []
    order = ["character", "copyright", "artist", "general", "meta"]
    if include_rating:
        order = ["rating"] + order
    for cat in order:
        for tag, _ in categories.get(cat, []):
            parts.append(tag)
    return ", ".join(parts)


def format_categories_html(categories: dict) -> str:
    """Render categories as an HTML table for Gradio HTML component."""
    html = "<div style='font-family:monospace;font-size:12px;'>"
    cat_colors = {
        "rating":    "#ff6b6b",
        "artist":    "#ffd93d",
        "copyright": "#6bcb77",
        "character": "#4d96ff",
        "general":   "#c0c0c0",
        "meta":      "#a0a0ff",
    }
    order = ["rating", "character", "copyright", "artist", "general", "meta"]
    for cat in order:
        tags = categories.get(cat, [])
        if not tags:
            continue
        color = cat_colors.get(cat, "#ffffff")
        html += f"<details open><summary style='color:{color};font-weight:bold;cursor:pointer;'>"
        html += f"🏷 {cat.upper()} ({len(tags)})</summary>"
        html += "<div style='padding:4px 12px;'>"
        for tag, score in tags:
            bar_w = int(score * 100)
            html += (
                f"<div style='display:flex;align-items:center;margin:2px 0;'>"
                f"<span style='width:180px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;"
                f"color:{color};'>{tag}</span>"
                f"<div style='background:#333;width:100px;height:8px;border-radius:4px;margin:0 6px;'>"
                f"<div style='background:{color};width:{bar_w}px;height:8px;border-radius:4px;'></div></div>"
                f"<span style='color:#aaa;'>{score:.3f}</span></div>"
            )
        html += "</div></details>"
    html += "</div>"
    return html


# ─── Core processing ──────────────────────────────────────────────────────────

def process_single_image(image, model_name, general_thresh, char_thresh, include_rating):
    """Process a single image uploaded via Gradio."""
    if image is None:
        return "", "", "<p>No image provided.</p>"

    predictor, err = load_model(model_name)
    if err:
        return "", "", f"<p style='color:red'>Error: {err}</p>"

    try:
        arr = preprocess_image(image)
        categories = predict(predictor, arr, general_thresh, char_thresh)
        prompt = format_tags(categories, include_rating)
        html = format_categories_html(categories)
        return prompt, prompt, html
    except Exception as e:
        tb = traceback.format_exc()
        return "", "", f"<pre style='color:red'>{tb}</pre>"


def process_batch(
    input_dir,
    output_dir,
    model_name,
    general_thresh,
    char_thresh,
    include_rating,
    save_txt,
    overwrite,
    recursive,
    progress=gr.Progress()
):
    """Batch-process all images in a folder."""
    if not input_dir or not os.path.isdir(input_dir):
        return "❌ Invalid input directory.", ""

    out_dir = output_dir.strip() if output_dir.strip() else input_dir
    os.makedirs(out_dir, exist_ok=True)

    # Collect image files
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.gif", "*.bmp")
    pattern = "**/*" if recursive else "*"
    files = []
    for ext in exts:
        if recursive:
            files.extend(Path(input_dir).rglob(ext[2:]))
        else:
            files.extend(Path(input_dir).glob(ext))
    files = sorted(set(files))

    if not files:
        return "⚠️ No images found in the directory.", ""

    predictor, err = load_model(model_name)
    if err:
        return f"❌ {err}", ""

    logs = []
    results_summary = []
    start = time.time()

    for i, img_path in enumerate(files):
        progress((i + 1) / len(files), desc=f"Processing {img_path.name}")

        # Compute output txt path
        rel = img_path.relative_to(input_dir) if out_dir != input_dir else Path(img_path.name)
        txt_path = Path(out_dir) / rel.with_suffix(".txt")

        if not overwrite and txt_path.exists():
            logs.append(f"⏭ Skipped (exists): {img_path.name}")
            continue

        try:
            from PIL import Image as PILImage
            img = PILImage.open(img_path)
            arr = preprocess_image(img)
            categories = predict(predictor, arr, general_thresh, char_thresh)
            prompt = format_tags(categories, include_rating)

            if save_txt:
                txt_path.parent.mkdir(parents=True, exist_ok=True)
                txt_path.write_text(prompt, encoding="utf-8")

            tag_count = sum(len(v) for v in categories.values())
            logs.append(f"✅ {img_path.name} → {tag_count} tags")
            results_summary.append({"file": str(img_path.name), "tags": prompt})

        except Exception as e:
            logs.append(f"❌ {img_path.name}: {e}")

    elapsed = time.time() - start
    summary = (
        f"Processed {len(files)} images in {elapsed:.1f}s\n"
        f"Output directory: {out_dir}\n"
        f"{'─'*50}\n"
        + "\n".join(logs)
    )

    # Build a simple results preview (first 10)
    preview_rows = [f"**{r['file']}**\n`{r['tags'][:120]}{'...' if len(r['tags'])>120 else ''}`"
                    for r in results_summary[:10]]
    preview = "\n\n".join(preview_rows)
    if len(results_summary) > 10:
        preview += f"\n\n… and {len(results_summary)-10} more."

    return summary, preview


# ─── Gradio UI ────────────────────────────────────────────────────────────────

def create_ui():
    """Create and return the Gradio Blocks UI for the extension."""
    with gr.Blocks(analytics_enabled=False) as ui:
        gr.Markdown(
            "## 🏷 WD Tagger Batch\n"
            "Automatically tag images using WaifuDiffusion tagger models.  \n"
            "_Based on [avan06/wd-tagger-images](https://github.com/avan06/wd-tagger-images) "
            "and [SmilingWolf/wd-tagger](https://huggingface.co/spaces/SmilingWolf/wd-tagger)_"
        )

        # ── Shared settings ────────────────────────────────────────────────
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="SwinV2 v3 (Recommended)",
                label="Tagger Model",
            )
            general_thresh = gr.Slider(0.0, 1.0, value=0.35, step=0.01,
                                       label="General Tag Threshold")
            char_thresh    = gr.Slider(0.0, 1.0, value=0.85, step=0.01,
                                       label="Character Tag Threshold")
            include_rating = gr.Checkbox(value=False, label="Include Rating Tag")

        # ── Tab: Single image ──────────────────────────────────────────────
        with gr.Tab("🖼 Single Image"):
            with gr.Row():
                single_input = gr.Image(type="pil", label="Input Image")
                with gr.Column():
                    single_prompt = gr.Textbox(label="Tags (prompt ready)", lines=4)
                    single_copy   = gr.Textbox(label="Copy-paste", lines=2, interactive=True)

            single_btn = gr.Button("🔍 Interrogate", variant="primary")
            single_html = gr.HTML(label="Categorized Tags")

            single_btn.click(
                fn=process_single_image,
                inputs=[single_input, model_dropdown,
                        general_thresh, char_thresh, include_rating],
                outputs=[single_prompt, single_copy, single_html],
            )

        # ── Tab: Batch processing ──────────────────────────────────────────
        with gr.Tab("📁 Batch Processing"):
            with gr.Row():
                batch_input_dir  = gr.Textbox(label="Input Directory",
                                              placeholder="/path/to/images")
                batch_output_dir = gr.Textbox(label="Output Directory (leave blank = same as input)",
                                              placeholder="/path/to/output  (optional)")
            with gr.Row():
                batch_save_txt  = gr.Checkbox(value=True,  label="Save .txt files")
                batch_overwrite = gr.Checkbox(value=False, label="Overwrite existing .txt")
                batch_recursive = gr.Checkbox(value=False, label="Recursive (subfolders)")

            batch_btn = gr.Button("🚀 Start Batch Tagging", variant="primary")

            with gr.Row():
                batch_log     = gr.Textbox(label="Processing Log", lines=15, interactive=False)
                batch_preview = gr.Markdown(label="Preview (first 10)")

            batch_btn.click(
                fn=process_batch,
                inputs=[
                    batch_input_dir, batch_output_dir,
                    model_dropdown, general_thresh, char_thresh,
                    include_rating, batch_save_txt, batch_overwrite, batch_recursive,
                ],
                outputs=[batch_log, batch_preview],
            )

        # ── Tab: Model info ────────────────────────────────────────────────
        with gr.Tab("ℹ Model Info"):
            gr.Markdown(
                "### Available Models\n"
                + "\n".join(
                    f"- **{name}**: `{repo}`"
                    for name, repo in MODELS.items()
                )
                + "\n\n### Tag Categories\n"
                "| Color | Category | Description |\n"
                "|-------|----------|-------------|\n"
                "| 🔴 | rating | Image rating (safe / questionable / explicit) |\n"
                "| 🟡 | artist | Artist name |\n"
                "| 🟢 | copyright | Series / franchise |\n"
                "| 🔵 | character | Character names |\n"
                "| ⬜ | general | General descriptive tags |\n"
                "| 🟣 | meta | Metadata tags |\n\n"
                "### Notes\n"
                "- Models are downloaded automatically from HuggingFace on first use.\n"
                "- ONNX Runtime will use CUDA if available, otherwise CPU.\n"
                "- Models are cached in `extensions/sd-webui-wd-tagger-batch/models/`.\n"
            )

    return [(ui, "WD Tagger Batch", "wd_tagger_batch")]


# ─── SD WebUI entry point ─────────────────────────────────────────────────────

try:
    import modules.scripts as scripts
    from modules import script_callbacks

    def on_ui_tabs():
        return create_ui()

    script_callbacks.on_ui_tabs(on_ui_tabs)

except ImportError:
    # Running standalone (for testing)
    if __name__ == "__main__":
        ui_list = create_ui()
        ui_list[0][0].launch(share=False)
