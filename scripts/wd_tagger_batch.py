"""
WD Tagger Batch — Stable Diffusion WebUI Extension
Based on avan06/wd-tagger-images (https://github.com/avan06/wd-tagger-images)
"""
from __future__ import annotations
import csv, os, sys, time, traceback
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np

_EXT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EXT_DIR not in sys.path:
    sys.path.insert(0, _EXT_DIR)

from tagger.tag_categories import SUBCATEGORY_ORDER, get_subcategory

# ── Models ────────────────────────────────────────────────────────────────────
MODELS = {
    "SwinV2 v3 (Recommended)":  "SmilingWolf/wd-swinv2-tagger-v3",
    "ConvNextV2 v3":            "SmilingWolf/wd-convnext-tagger-v3",
    "ViT v3":                   "SmilingWolf/wd-vit-tagger-v3",
    "ViT Large v3":             "SmilingWolf/wd-vit-large-tagger-v3",
    "EVA02 Large v3":           "SmilingWolf/wd-eva02-large-tagger-v3",
    "ConvNext v2 (Legacy)":     "SmilingWolf/wd-v1-4-convnext-tagger-v2",
    "SwinV2 v2 (Legacy)":       "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
    "ViT v2 (Legacy)":          "SmilingWolf/wd-v1-4-vit-tagger-v2",
    "EVA02 Large IS v1":        "deepghs/idolsankaku-eva02-large-tagger-v1",
    "SwinV2 IS v1":             "deepghs/idolsankaku-swinv2-tagger-v1",
}
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"
KAOMOJIS = {"0_0","(o)_(o)","+_+","+_-","._.","<o>_<o>","<|>_<|>","=_=",">_<","3_3","6_9",">_o","@_@","^_^","o_o","u_u","x_x","|_|","||_||"}
WD_CAT = {9:"rating",1:"artist",3:"copyright",4:"character",0:"general",5:"meta"}

# ── Model cache ───────────────────────────────────────────────────────────────
_predictor = None
_current_model = None

def _models_dir():
    p = Path(_EXT_DIR) / "models"; p.mkdir(parents=True, exist_ok=True); return p

def load_model(model_name):
    global _predictor, _current_model
    if _current_model == model_name and _predictor is not None:
        return _predictor, None
    repo_id = MODELS.get(model_name)
    if not repo_id: return None, f"Unknown model: {model_name}"
    model_dir = _models_dir() / repo_id.replace("/","--")
    model_path, label_path = model_dir / MODEL_FILENAME, model_dir / LABEL_FILENAME
    if not model_path.exists() or not label_path.exists():
        try:
            from huggingface_hub import hf_hub_download
            model_dir.mkdir(parents=True, exist_ok=True)
            print(f"[WD Tagger Batch] Downloading {repo_id} …")
            hf_hub_download(repo_id=repo_id, filename=MODEL_FILENAME, local_dir=str(model_dir))
            hf_hub_download(repo_id=repo_id, filename=LABEL_FILENAME, local_dir=str(model_dir))
        except Exception as e: return None, f"Download failed: {e}"
    try:
        import onnxruntime as rt
        session = rt.InferenceSession(str(model_path), providers=["CUDAExecutionProvider","CPUExecutionProvider"])
        print(f"[WD Tagger Batch] Model loaded via {session.get_providers()[0]}")
    except Exception as e: return None, f"ONNX load failed: {e}"
    try:
        labels = []
        with open(label_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                labels.append({"name": row["name"], "category": int(row["category"])})
    except Exception as e: return None, f"Label load failed: {e}"
    _predictor = {"session": session, "labels": labels}
    _current_model = model_name
    return _predictor, None

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(img, size=448):
    from PIL import Image
    if not isinstance(img, Image.Image): img = Image.fromarray(img)
    canvas = Image.new("RGBA", img.size, (255,255,255,255))
    canvas.alpha_composite(img.convert("RGBA"))
    img = canvas.convert("RGB")
    w,h = img.size; side = max(w,h)
    pad = Image.new("RGB",(side,side),(255,255,255))
    pad.paste(img,((side-w)//2,(side-h)//2))
    pad = pad.resize((size,size), Image.BICUBIC)
    arr = np.array(pad, dtype=np.float32)[:,:,::-1]
    return np.expand_dims(arr,0)

# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(predictor, arr, gen_thresh, char_thresh):
    session = predictor["session"]; labels = predictor["labels"]
    inp = session.get_inputs()[0].name
    probs = session.run(None, {inp: arr})[0][0]
    result = {sub: [] for sub in SUBCATEGORY_ORDER}
    result["rating"] = []
    for prob, label in zip(probs, labels):
        cat_id = label["category"]; tag_raw = label["name"]
        wd_cat = WD_CAT.get(cat_id, "general")
        thresh = 0.0 if wd_cat=="rating" else (char_thresh if wd_cat=="character" else gen_thresh)
        if prob < thresh: continue
        display = tag_raw if tag_raw in KAOMOJIS else tag_raw.replace("_"," ")
        sub = get_subcategory(tag_raw, wd_cat)
        result.setdefault(sub, []).append((display, float(prob)))
    for sub in result: result[sub].sort(key=lambda x:x[1], reverse=True)
    return result

# ── Tag manipulation ──────────────────────────────────────────────────────────
def apply_edits(cats, prepend, append, remove, merge_chars):
    remove_set = {t.strip().lower() for t in remove.split(",") if t.strip()}
    if merge_chars:
        char_tags = cats.pop("Character Design", [])
        cats.setdefault("Unclassified",[])
        cats["Unclassified"] = char_tags + cats["Unclassified"]
    for sub in cats:
        cats[sub] = [(t,s) for t,s in cats[sub] if t.lower() not in remove_set]
    return cats

def build_prompt(cats, include_rating, prepend, append):
    parts = [t.strip() for t in prepend.split(",") if t.strip()] if prepend.strip() else []
    order = (["rating"] if include_rating else []) + list(SUBCATEGORY_ORDER)
    for sub in order:
        for tag,_ in cats.get(sub,[]):
            if tag not in parts: parts.append(tag)
    if append.strip(): parts.extend(t.strip() for t in append.split(",") if t.strip())
    return ", ".join(parts)

def tags_str(cats, sub): return ", ".join(t for t,_ in cats.get(sub,[]))

# ── Stats ─────────────────────────────────────────────────────────────────────
def build_stats(cats):
    total = sum(len(v) for v in cats.values())
    rows = ""
    for sub in SUBCATEGORY_ORDER + ["rating"]:
        n = len(cats.get(sub,[])); 
        if n==0: continue
        pct = int(n/total*100) if total else 0
        rows += (f"<tr><td style='padding:2px 8px;color:#aaa'>{sub}</td>"
                 f"<td style='padding:2px 8px;color:#fff;text-align:right'>{n}</td>"
                 f"<td style='padding:2px 8px'><div style='background:#333;width:140px;height:8px;border-radius:4px;display:inline-block'>"
                 f"<div style='background:#4d96ff;width:{min(pct,100)}%;height:8px;border-radius:4px'></div></div>"
                 f"<span style='color:#888;margin-left:4px'>{pct}%</span></td></tr>")
    return f"<table style='font-size:12px;font-family:monospace'>{rows}</table><p style='color:#888;font-size:11px'>Total: {total} tags</p>"

# ── Process single ────────────────────────────────────────────────────────────
def process_image(image, model_name, gen_thresh, char_thresh, include_rating, merge_chars, prepend, append_, remove):
    N = len(SUBCATEGORY_ORDER)
    if image is None: return ("",)*( N+2)
    predictor, err = load_model(model_name)
    if err: return (f"Error: {err}",) + ("",)*N + ("",)
    try:
        arr  = preprocess(image)
        cats = run_inference(predictor, arr, gen_thresh, char_thresh)
        cats = apply_edits(cats, prepend, append_, remove, merge_chars)
        prompt = build_prompt(cats, include_rating, prepend, append_)
        subs = tuple(tags_str(cats, sub) for sub in SUBCATEGORY_ORDER)
        stats = build_stats(cats)
        return (prompt,) + subs + (stats,)
    except Exception:
        tb = traceback.format_exc()
        return (f"Error:\n{tb}",) + ("",)*N + ("")

# ── Batch ─────────────────────────────────────────────────────────────────────
def process_batch(input_dir, output_dir, model_name, gen_thresh, char_thresh, include_rating,
                  merge_chars, prepend, append_, remove, save_txt, overwrite, recursive, progress=gr.Progress()):
    if not input_dir or not os.path.isdir(input_dir): return "❌ Invalid input directory.", ""
    out_dir = output_dir.strip() or input_dir
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for ext in ("png","jpg","jpeg","webp","gif","bmp"):
        if recursive: files.extend(Path(input_dir).rglob(f"*.{ext}"))
        else:         files.extend(Path(input_dir).glob(f"*.{ext}"))
    files = sorted(set(files))
    if not files: return "⚠️ No images found.", ""
    predictor, err = load_model(model_name)
    if err: return f"❌ {err}", ""
    logs = []; start = time.time()
    for i, img_path in enumerate(files):
        progress((i+1)/len(files), desc=img_path.name)
        rel = img_path.relative_to(input_dir) if out_dir != input_dir else Path(img_path.name)
        txt_path = Path(out_dir) / rel.with_suffix(".txt")
        if not overwrite and txt_path.exists(): logs.append(f"⏭ {img_path.name}"); continue
        try:
            from PIL import Image as PILImage
            img = PILImage.open(img_path)
            arr  = preprocess(img)
            cats = run_inference(predictor, arr, gen_thresh, char_thresh)
            cats = apply_edits(cats, prepend, append_, remove, merge_chars)
            prompt = build_prompt(cats, include_rating, prepend, append_)
            if save_txt: txt_path.parent.mkdir(parents=True, exist_ok=True); txt_path.write_text(prompt, encoding="utf-8")
            n = sum(len(v) for v in cats.values())
            logs.append(f"✅ {img_path.name}  [{n} tags]")
        except Exception as e: logs.append(f"❌ {img_path.name}: {e}")
    elapsed = time.time() - start
    return f"✔ Finished {len(files)} images in {elapsed:.1f}s\nOutput: {out_dir}\n{'─'*50}\n" + "\n".join(logs), ""

# ── Category color palette ────────────────────────────────────────────────────
SUB_COLORS = {
    "Appearance Status":"#f9a03f","Character Design":"#4d96ff","explicit":"#ff6b6b",
    "Upper Body":"#a29bfe","Action Pose":"#fd79a8","Outdoor":"#6bcb77",
    "Lower Body":"#ffd93d","Head":"#74b9ff","Facial Expression":"#e17055",
    "Censorship":"#b2bec3","Creature":"#55efc4","Background":"#81ecec",
    "Hair":"#fdcb6e","Eyes":"#00cec9","Clothing":"#a8e6cf",
    "Accessories":"#dfe6e9","Others":"#636e72","Unclassified":"#555555",
}

# ── UI ────────────────────────────────────────────────────────────────────────
def create_ui():
    with gr.Blocks(analytics_enabled=False) as ui:
        gr.Markdown(
            "## 🏷 WD Tagger Batch\n"
            "Auto-tag images using WaifuDiffusion ONNX models. "
            "_Based on [avan06/wd-tagger-images](https://github.com/avan06/wd-tagger-images)_"
        )

        # ── Global settings ───────────────────────────────────────────────
        with gr.Row():
            model_dd    = gr.Dropdown(list(MODELS.keys()), value="SwinV2 v3 (Recommended)", label="Model (for Images)")
            gen_thresh  = gr.Slider(0.0,1.0, value=0.35, step=0.01, label="General Tags Threshold")
            char_thresh = gr.Slider(0.0,1.0, value=0.85, step=0.01, label="Character Tags Threshold")
        with gr.Row():
            include_rating = gr.Checkbox(False, label="Include Rating Tag in string output")
            merge_chars    = gr.Checkbox(True,  label="Merge characters into the string output")
        with gr.Row():
            prepend_tags = gr.Textbox(label="Prepend Additional tags (comma split)", lines=2)
            append_tags  = gr.Textbox(label="Append Additional tags (comma split)",  lines=2)
        remove_tags = gr.Textbox(label="Remove tags (comma split)", lines=2)

        # ── Tab: Single Image ─────────────────────────────────────────────
        with gr.Tab("🖼 Single Image"):
            with gr.Row():
                img_input = gr.Image(type="pil", label="Image Gallery (Drag multiple images here)", height=300)
                with gr.Column():
                    full_prompt    = gr.Textbox(label="Full Tags String", lines=5, interactive=True)
                    interrogate_btn = gr.Button("🔍 Interrogate", variant="primary")
                    clear_btn      = gr.Button("🗑 Clear", variant="secondary")

            gr.Markdown("---\n### 🗂 Categorized (tags) — Interactive")

            sub_boxes: list[gr.Textbox] = []

            with gr.Column():
                for sub in SUBCATEGORY_ORDER:
                    color = SUB_COLORS.get(sub, "#aaa")
                    open_default = sub in ("Appearance Status","Character Design","explicit",
                                           "Upper Body","Lower Body","Action Pose","Head","Hair","Eyes")
                    with gr.Accordion(
                        label=f"{sub}  (0)",
                        open=open_default,
                    ):
                        with gr.Row():
                            tb = gr.Textbox(
                                label="",
                                lines=2,
                                interactive=True,
                                placeholder=f"No {sub} tags — threshold may be too high",
                                show_label=False,
                            )
                            copy_btn = gr.Button("📋 Copy String", size="sm", min_width=100)
                        copy_btn.click(
                            fn=lambda v: v,
                            inputs=[tb],
                            outputs=[full_prompt],
                            _js="(v) => { try { navigator.clipboard.writeText(v); } catch(e){} return v; }",
                        )
                    sub_boxes.append(tb)

            with gr.Accordion("📊 Tags Statistics (All files)", open=False):
                stats_html = gr.HTML("<p style='color:#888'>Run interrogation first.</p>")

            with gr.Accordion("📋 Detailed Output (for last processed item)", open=False):
                detailed_out = gr.Textbox(label="", lines=6, interactive=False, show_label=False)

            all_out = [full_prompt] + sub_boxes + [stats_html]

            def on_interrogate(image, model, gt, ct, ir, mc, prep, app, rem):
                result = process_image(image, model, gt, ct, ir, mc, prep, app, rem)
                prompt = result[0]
                subs   = result[1:-1]
                stats  = result[-1]
                # update accordion labels would need JS; we return the values only
                return result

            interrogate_btn.click(
                fn=process_image,
                inputs=[img_input, model_dd, gen_thresh, char_thresh,
                        include_rating, merge_chars, prepend_tags, append_tags, remove_tags],
                outputs=all_out + [detailed_out],
            )
            img_input.change(
                fn=process_image,
                inputs=[img_input, model_dd, gen_thresh, char_thresh,
                        include_rating, merge_chars, prepend_tags, append_tags, remove_tags],
                outputs=all_out + [detailed_out],
            )
            clear_btn.click(
                fn=lambda: [""] * (len(all_out) + 1),
                inputs=[],
                outputs=all_out + [detailed_out],
            )

        # ── Tab: Batch ────────────────────────────────────────────────────
        with gr.Tab("📁 Batch Processing"):
            with gr.Row():
                batch_in  = gr.Textbox(label="Input Directory", placeholder="/path/to/images")
                batch_out = gr.Textbox(label="Output Directory (blank = same as input)")
            with gr.Row():
                save_txt  = gr.Checkbox(True,  label="Save .txt files")
                overwrite = gr.Checkbox(False, label="Overwrite existing .txt")
                recursive = gr.Checkbox(False, label="Recursive (subfolders)")
            batch_btn = gr.Button("🚀 Start Batch Tagging", variant="primary")
            with gr.Row():
                batch_log = gr.Textbox(label="Processing Log", lines=20, interactive=False)

            batch_btn.click(
                fn=process_batch,
                inputs=[batch_in, batch_out, model_dd, gen_thresh, char_thresh,
                        include_rating, merge_chars, prepend_tags, append_tags, remove_tags,
                        save_txt, overwrite, recursive],
                outputs=[batch_log, gr.Textbox(visible=False)],
            )

        # ── Tab: Model Info ───────────────────────────────────────────────
        with gr.Tab("ℹ Model Info"):
            gr.Markdown(
                "### Available Models\n" +
                "\n".join(f"- **{k}** → `{v}`" for k,v in MODELS.items()) +
                "\n\n### Tag Categories (18 subcategories)\n" +
                "\n".join(f"- **{s}**" for s in SUBCATEGORY_ORDER) +
                "\n\n_Models download automatically from HuggingFace Hub on first use._\n"
                "_Cached in `extensions/sd-webui-wd-tagger-batch/models/`_"
            )

    return [(ui, "WD Tagger Batch", "wd_tagger_batch")]


# ── SD WebUI hook ─────────────────────────────────────────────────────────────
try:
    from modules import script_callbacks
    script_callbacks.on_ui_tabs(create_ui)
except ImportError:
    if __name__ == "__main__":
        create_ui()[0][0].launch()
