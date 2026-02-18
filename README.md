# WD Tagger Batch — SD WebUI Extension

Auto-tag single or multiple images using WaifuDiffusion tagger models (ONNX).  
Based on **[avan06/wd-tagger-images](https://github.com/avan06/wd-tagger-images)** and the original **[SmilingWolf/wd-tagger](https://huggingface.co/spaces/SmilingWolf/wd-tagger)** HuggingFace Space.

---

## Features

- 🖼 **Single image** interrogation with instant categorized tag view
- 📁 **Batch processing** — tag all images in a folder (optionally recursive)
- 🏷 **Categorized output** — tags grouped by: rating, character, copyright, artist, general, meta
- 📝 **Save .txt files** alongside images for dataset preparation
- ⚡ **GPU acceleration** via CUDA (falls back to CPU automatically)
- 🔄 **Multiple models** — SwinV2, ConvNextV2, ViT, EVA02 (v2 & v3) plus IdolSankaku series
- ⚙️ **Per-category thresholds** — separate sliders for general and character tags

---

## Installation

### Method 1 — Install from URL (recommended)

1. Open **Extensions → Install from URL**
2. Paste: `https://github.com/YOUR_USERNAME/sd-webui-wd-tagger-batch`
3. Click **Install**
4. Restart SD WebUI

### Method 2 — Manual

```bash
cd stable-diffusion-webui/extensions
git clone https://github.com/YOUR_USERNAME/sd-webui-wd-tagger-batch.git
```

Then restart SD WebUI.

---

## Usage

After installation you will find a new tab: **"WD Tagger Batch"**

### Single Image
1. Switch to the **🖼 Single Image** tab
2. Upload an image
3. Adjust thresholds (default: general=0.35, character=0.85)
4. Click **🔍 Interrogate**
5. Copy the generated tags from the **Tags** box

### Batch Processing
1. Switch to the **📁 Batch Processing** tab
2. Enter the **Input Directory** (full path to your images folder)
3. Optionally set a different **Output Directory** (defaults to same folder)
4. Enable **Save .txt files** to write tags next to each image
5. Click **🚀 Start Batch Tagging**

---

## Available Models

| Model | Repo | Notes |
|-------|------|-------|
| **SwinV2 v3** | `SmilingWolf/wd-swinv2-tagger-v3` | ⭐ Recommended |
| ConvNextV2 v3 | `SmilingWolf/wd-convnext-tagger-v3` | Fast |
| ViT v3 | `SmilingWolf/wd-vit-tagger-v3` | Balanced |
| ViT Large v3 | `SmilingWolf/wd-vit-large-tagger-v3` | High accuracy |
| EVA02 Large v3 | `SmilingWolf/wd-eva02-large-tagger-v3` | Best accuracy |
| ConvNext v2 *(legacy)* | `SmilingWolf/wd-v1-4-convnext-tagger-v2` | |
| SwinV2 v2 *(legacy)* | `SmilingWolf/wd-v1-4-swinv2-tagger-v2` | |
| ViT v2 *(legacy)* | `SmilingWolf/wd-v1-4-vit-tagger-v2` | |
| EVA02 Large IS v1 | `deepghs/idolsankaku-eva02-large-tagger-v1` | IdolSankaku |
| SwinV2 IS v1 | `deepghs/idolsankaku-swinv2-tagger-v1` | IdolSankaku |

Models are downloaded **automatically** from HuggingFace on first use and cached in:
```
extensions/sd-webui-wd-tagger-batch/models/
```

---

## GPU Support

To use GPU inference install `onnxruntime-gpu` instead of `onnxruntime`:

```bash
pip install onnxruntime-gpu
```

---

## Requirements

- Python 3.10+
- `onnxruntime >= 1.16`
- `huggingface-hub >= 0.19`
- `Pillow >= 9.0`
- `numpy >= 1.23`

These are installed automatically by SD WebUI.

---

## Credits

- **[avan06/wd-tagger-images](https://github.com/avan06/wd-tagger-images)** — batch processing concept & model list
- **[SmilingWolf](https://huggingface.co/SmilingWolf)** — WD Tagger models
- **[toriato/stable-diffusion-webui-wd14-tagger](https://github.com/toriato/stable-diffusion-webui-wd14-tagger)** — original SD WebUI tagger extension

---

## License

MIT
