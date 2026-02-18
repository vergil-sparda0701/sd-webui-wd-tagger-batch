"""
WD Tagger Batch - Stable Diffusion WebUI Extension
Based on avan06/wd-tagger-images (https://github.com/avan06/wd-tagger-images)

All categories from the original:
  Appearance Status · Character Design · Explicit · Upper Body · Action Pose
  Outdoor · Lower Body · Head · Facial Expression · Censorship · Creature
  Background · Others · Unclassified
  + rating / artist / copyright / character
Extra features:
  Prepend/Append/Remove tags · Merge characters · Save .txt · Tags Statistics
"""

import os, sys, csv, time, traceback
from pathlib import Path
import numpy as np
import gradio as gr

ext_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ext_dir not in sys.path:
    sys.path.insert(0, ext_dir)

# ─── Model registry ──────────────────────────────────────────────────────────

MODELS = {
    "SmilingWolf/wd-eva02-large-tagger-v3":      "EVA02 Large v3",
    "SmilingWolf/wd-swinv2-tagger-v3":           "SwinV2 v3",
    "SmilingWolf/wd-convnext-tagger-v3":         "ConvNextV2 v3",
    "SmilingWolf/wd-vit-tagger-v3":              "ViT v3",
    "SmilingWolf/wd-vit-large-tagger-v3":        "ViT Large v3",
    "SmilingWolf/wd-v1-4-convnext-tagger-v2":    "ConvNext v2 (legacy)",
    "SmilingWolf/wd-v1-4-swinv2-tagger-v2":      "SwinV2 v2 (legacy)",
    "SmilingWolf/wd-v1-4-vit-tagger-v2":         "ViT v2 (legacy)",
    "deepghs/idolsankaku-eva02-large-tagger-v1": "EVA02 IS v1",
    "deepghs/idolsankaku-swinv2-tagger-v1":      "SwinV2 IS v1",
}
CHOICES = [f"{label}  [{repo}]" for repo, label in MODELS.items()]
CHOICE_TO_REPO = {f"{label}  [{repo}]": repo for repo, label in MODELS.items()}
DEFAULT_CHOICE = CHOICES[0]

MODEL_FILE = "model.onnx"
LABEL_FILE = "selected_tags.csv"

# ─── WD base categories ───────────────────────────────────────────────────────

WD_CAT = {9: "rating", 1: "artist", 3: "copyright", 4: "character", 0: "general", 5: "meta"}

# ─── Detailed sub-categories for general tags (matching avan06) ───────────────

# Keywords: if any keyword is a substring of the tag name → assign that category
DETAIL_CATS = {
    "Appearance Status": [
        "nude","naked","topless","bottomless","clothed","unclothe","disrobe",
        "shiny skin","wet","blush","dark skin","pale skin","tan skin","freckle",
        "tattoo","scar","mole","piercing","makeup","lipstick","eyeshadow",
        "nail polish","vein","partially submerged","submerged","bare","exposed",
        "sweating","sweat","covered","cream","lotion","body paint",
    ],
    "Character Design": [
        "1girl","2girl","3girl","4girl","5girl","6+girl","multiple girl",
        "1boy","2boy","3boy","multiple boy","1other","androgynous","genderswap",
        "futanari","trap","crossdressing","male","female",
    ],
    "Explicit": [
        "sex","penis","vagina","nipple","areola","pussy","cum","orgasm",
        "erection","ejaculation","fellatio","cunnilingus","anal","penetration",
        "intercourse","missionary","doggy","cowgirl","threesome","group sex",
        "masturbation","fingering","handjob","blowjob","footjob","paizuri",
        "squirt","creampie","ahegao","explicit","licking penis","veiny penis",
        "large penis","cooperative fellatio","ffm threesome","mixed-sex bathing",
        "pov crotch","breast press","shared bathing","hetero","oral","testicle",
        "spread legs","spread pussy","rape","gangrape","insemination",
    ],
    "Upper Body": [
        "breast","cleavage","sideboob","underboob","flat chest",
        "bare shoulder","collarbone","chest","torso","abdomen","stomach","navel",
        "toned","muscular","arm","hand","finger","wrist","elbow","shoulder",
        "armpit","belly","midriff","ribcage","upper body",
    ],
    "Action Pose": [
        "pov","looking at viewer","looking back","looking down","looking up",
        "looking away","from above","from below","from behind","from side",
        "profile","action","dynamic","standing","sitting","lying","kneeling",
        "crouching","jumping","running","walking","leaning","stretching",
        "bending","arching","licking","kissing","hugging","holding",
        "grabbing","reaching","pointing","breast press","pov crotch",
    ],
    "Outdoor": [
        "outdoor","outside","nature","sky","cloud","sun","moon","star",
        "night","day","sunset","sunrise","rain","snow","fog","wind",
        "beach","ocean","sea","river","lake","forest","tree","grass",
        "flower","mountain","hill","field","garden","park","street",
        "city","urban","rural","onsen","pool","waterfall","desert",
    ],
    "Lower Body": [
        "ass","butt","buttocks","huge ass","large ass","leg","thigh",
        "hip","pelvis","groin","crotch","pubic","genital",
        "foot","feet","toe","ankle","knee","calf","shin","heel",
        "barefoot","spread legs","crossed leg","stockings","thighhigh",
    ],
    "Head": [
        "hair","long hair","short hair","twin tail","ponytail","braid",
        "bang","ahoge","hair bun","blonde","brown hair","black hair",
        "white hair","silver hair","grey hair","red hair","blue hair",
        "green hair","purple hair","pink hair","orange hair","multicolored",
        "gradient hair","streaked hair","eye","heterochromia","closed eye",
        "ear","animal ear","cat ear","fox ear","bunny ear","dog ear",
        "horn","halo","crown","hat","headband","hairpin","ribbon","head",
        "forehead","cheek","chin","jaw","neck",
    ],
    "Facial Expression": [
        "smile","grin","smirk","frown","pout","embarrassed","shy","scared",
        "surprised","shocked","angry","sad","crying","tear","happy","joyful",
        "neutral","expressionless","serious","determined","seductive","ahegao",
        "open mouth","closed mouth","tongue","tongue out","teeth","biting lip",
        "licking lip","wink","one eye closed","squinting","blush",
    ],
    "Censorship": [
        "censored","uncensored","mosaic","bar censor","light ray",
        "convenient censor","steam censor","object censor",
    ],
    "Creature": [
        "animal","cat ","dog ","fox ","wolf","rabbit","horse","dragon",
        "monster","demon","angel","fairy","elf","beast","creature",
        "tentacle","slime","bird","fish","snake","spider","insect",
        "furry","kemonomimi",
    ],
    "Background": [
        "background","simple background","white background","black background",
        "gradient background","room","bed","wall","floor","water","pool",
        "blur","bokeh","ripple","wave","steam","smoke","fire","light",
        "shadow","reflection",
    ],
    "Others":       [],
    "Unclassified": [],
}

KAOMOJIS = {
    "0_0","(o)_(o)","+_+","+_-","._.",
    "<o>_<o>","<|>_<|>","=_=",">_<","3_3",
    "6_9",">_o","@_@","^_^","o_o","u_u","x_x","|_|","||_||",
}

CAT_COLORS = {
    "rating":            "#ff6b6b",
    "artist":            "#ffd93d",
    "copyright":         "#6bcb77",
    "character":         "#4d96ff",
    "Appearance Status": "#e879f9",
    "Character Design":  "#38bdf8",
    "Explicit":          "#fb923c",
    "Upper Body":        "#a78bfa",
    "Action Pose":       "#34d399",
    "Outdoor":           "#86efac",
    "Lower Body":        "#f9a8d4",
    "Head":              "#fbbf24",
    "Facial Expression": "#67e8f9",
    "Censorship":        "#94a3b8",
    "Creature":          "#4ade80",
    "Background":        "#60a5fa",
    "Others":            "#d1d5db",
    "Unclassified":      "#6b7280",
}

ALL_CAT_NAMES = [
    "rating", "artist", "copyright", "character",
    "Appearance Status", "Character Design", "Explicit",
    "Upper Body", "Action Pose", "Outdoor", "Lower Body",
    "Head", "Facial Expression", "Censorship", "Creature",
    "Background", "Others", "Unclassified",
]

# ─── Global model state ───────────────────────────────────────────────────────

_session = None
_labels  = None
_loaded_repo = None


def models_dir() -> Path:
    p = Path(ext_dir) / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_model(repo: str) -> str | None:
    global _session, _labels, _loaded_repo
    if _loaded_repo == repo and _session:
        return None
    safe = repo.replace("/", "--")
    mdir = models_dir() / safe
    mp   = mdir / MODEL_FILE
    lp   = mdir / LABEL_FILE
    if not mp.exists() or not lp.exists():
        try:
            from huggingface_hub import hf_hub_download
            mdir.mkdir(parents=True, exist_ok=True)
            print(f"[WD Tagger] Downloading {repo} …")
            hf_hub_download(repo_id=repo, filename=MODEL_FILE, local_dir=str(mdir))
            hf_hub_download(repo_id=repo, filename=LABEL_FILE, local_dir=str(mdir))
        except Exception as e:
            return f"Download failed: {e}"
    try:
        import onnxruntime as rt
        opts = rt.SessionOptions()
        _session = rt.InferenceSession(str(mp), opts,
                   providers=["CUDAExecutionProvider","CPUExecutionProvider"])
        print(f"[WD Tagger] Loaded ({_session.get_providers()[0]})")
    except Exception as e:
        return f"ONNX error: {e}"
    _labels = []
    with open(lp, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            _labels.append({"name": row["name"], "category": int(row["category"])})
    _loaded_repo = repo
    return None


def preprocess(img, size=448):
    from PIL import Image
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    canvas = Image.new("RGBA", img.size, (255,255,255,255))
    canvas.alpha_composite(img.convert("RGBA"))
    img = canvas.convert("RGB")
    w, h = img.size
    s = max(w, h)
    pad = Image.new("RGB", (s,s), (255,255,255))
    pad.paste(img, ((s-w)//2, (s-h)//2))
    pad = pad.resize((size, size), Image.BICUBIC)
    arr = np.array(pad, dtype=np.float32)[:,:,::-1]
    return np.expand_dims(arr, 0)


def infer(arr):
    name = _session.get_inputs()[0].name
    return _session.run(None, {name: arr})[0][0]


def disp(tag):
    return tag if tag in KAOMOJIS else tag.replace("_", " ")


# ─── Categorization ───────────────────────────────────────────────────────────

def categorize(probs, gen_th, chr_th):
    # 1. Base WD split
    base = {v: [] for v in WD_CAT.values()}
    for prob, lbl in zip(probs, _labels):
        cat = WD_CAT.get(lbl["category"], "general")
        th  = chr_th if cat == "character" else (0.0 if cat == "rating" else gen_th)
        if prob >= th:
            base[cat].append((lbl["name"], float(prob)))
    for c in base:
        base[c].sort(key=lambda x: x[1], reverse=True)

    # 2. Fine-grain general tags
    fine = {k: [] for k in DETAIL_CATS}
    for tag, sc in base["general"]:
        d = disp(tag)
        placed = False
        for cat, kws in DETAIL_CATS.items():
            if cat in ("Others", "Unclassified"):
                continue
            for kw in kws:
                if kw in tag or kw in d:
                    fine[cat].append((d, sc))
                    placed = True
                    break
            if placed:
                break
        if not placed:
            fine["Others"].append((d, sc))

    for c in fine:
        fine[c].sort(key=lambda x: x[1], reverse=True)

    # 3. meta → Unclassified
    fine["Unclassified"] = [(disp(n), s) for n,s in base["meta"]]

    named = {c: [(disp(n), s) for n,s in base[c]] for c in ("rating","artist","copyright","character")}
    return {**named, **fine}


def build_prompt(cats, inc_rating, merge, prepend, append, remove):
    rm = {t.strip().lower() for t in remove.split(",") if t.strip()}
    order = ["character","copyright","artist"] + list(DETAIL_CATS.keys())
    if inc_rating:
        order = ["rating"] + order
    parts = [t.strip() for t in prepend.split(",") if t.strip()] if prepend.strip() else []
    seen  = set()
    for c in order:
        for tag,_ in cats.get(c, []):
            if tag.lower() not in rm and tag not in seen:
                seen.add(tag)
                parts.append(tag)
    if append.strip():
        parts += [t.strip() for t in append.split(",") if t.strip()]
    return ", ".join(parts)


# ─── HTML panels ─────────────────────────────────────────────────────────────

def cats_html(cats):
    style = """
<style>
.wdt{font-family:'Segoe UI',system-ui,sans-serif;padding:4px;}
.wdt-panel{border:1px solid #30363d;border-radius:6px;margin:5px 0;overflow:hidden;background:#161b22;}
.wdt-hdr{display:flex;justify-content:space-between;align-items:center;
         padding:7px 11px;cursor:pointer;user-select:none;}
.wdt-hdr:hover{background:#1f2937;}
.wdt-label{font-weight:600;font-size:13px;display:flex;align-items:center;gap:7px;}
.wdt-n{background:#ffffff18;border-radius:10px;padding:1px 7px;font-size:11px;color:#9ca3af;}
.wdt-copy{background:#1e3a5f;border:1px solid #2563eb44;color:#60a5fa;
          border-radius:4px;padding:2px 9px;font-size:11px;cursor:pointer;transition:all .15s;}
.wdt-copy:hover{background:#2563eb;color:#fff;}
.wdt-body{padding:7px 11px 9px;}
.pill{display:inline-flex;align-items:center;gap:3px;border-radius:4px;
      padding:2px 7px;font-size:12px;margin:2px 2px;border:1px solid transparent;cursor:default;}
.pill-sc{font-size:10px;opacity:.65;}
.wdt-preview{font-size:11px;color:#4b5563;margin-top:4px;overflow:hidden;
             text-overflow:ellipsis;white-space:nowrap;}
.wdt-empty{color:#374151;font-size:12px;font-style:italic;}
</style>
"""
    html = style + "<div class='wdt'>"
    for cat in ALL_CAT_NAMES:
        tags  = cats.get(cat, [])
        color = CAT_COLORS.get(cat, "#aaa")
        ts    = ", ".join(t for t,_ in tags)
        safe  = ts.replace("\\","\\\\").replace("`","\\`").replace("'","\\'")
        html += f"""
<div class='wdt-panel'>
  <div class='wdt-hdr' onclick="var b=this.nextElementSibling;b.style.display=b.style.display=='none'?'':'none';">
    <span class='wdt-label' style='color:{color}'>▾ {cat} <span class='wdt-n'>{len(tags)}</span></span>
    <button class='wdt-copy'
      onclick="event.stopPropagation();navigator.clipboard.writeText('{safe}').then(()=>{{this.textContent='✓ Copied!';setTimeout(()=>this.textContent='Copy String',1400)}})">
      Copy String</button>
  </div>
  <div class='wdt-body'>"""
        if tags:
            html += "<div>"
            for tag, sc in tags:
                s2 = tag.replace("'","\\'")
                html += (f"<span class='pill' style='background:{color}1a;border-color:{color}44;color:{color};' "
                         f"title='{sc:.3f}'>{tag} <span class='pill-sc'>{sc:.2f}</span></span>")
            html += "</div>"
            html += f"<div class='wdt-preview'>{ts[:100]}{'…' if len(ts)>100 else ''}</div>"
        else:
            html += "<div class='wdt-empty'>— no tags —</div>"
        html += "</div></div>"
    html += "</div>"
    return html


def stats_html(results):
    if not results:
        return "<p style='color:#4b5563;font-size:13px;padding:8px;'>No data yet — run Interrogate first.</p>"
    from collections import Counter
    ctr = Counter()
    for r in results:
        for tags in r.values():
            for t,_ in tags:
                ctr[t] += 1
    top = ctr.most_common(60)
    mx  = max(c for _,c in top) if top else 1
    h   = ("<div style='font-family:monospace;font-size:12px;background:#0d1117;"
           "padding:12px;border-radius:8px;max-height:400px;overflow-y:auto;'>")
    h  += f"<b style='color:#60a5fa'>Top tags — {len(results)} image(s)</b><br><br>"
    for tag, cnt in top:
        bw = int(cnt/mx * 130)
        h += (f"<div style='display:flex;align-items:center;gap:8px;margin:2px 0;'>"
              f"<span style='width:170px;color:#d1d5db;overflow:hidden;white-space:nowrap;text-overflow:ellipsis'>{tag}</span>"
              f"<div style='background:#1e3a5f;width:{bw}px;height:8px;border-radius:3px'></div>"
              f"<span style='color:#60a5fa;min-width:20px;text-align:right'>{cnt}</span></div>")
    h += "</div>"
    return h


# ─── Actions ──────────────────────────────────────────────────────────────────

def do_interrogate(image, model_choice, gen_th, chr_th,
                   inc_rating, merge, prepend, append, remove,
                   st_results):
    if image is None:
        empty = [""] * len(ALL_CAT_NAMES)
        return (*empty, "", "<p style='color:#f87171'>Upload an image first.</p>",
                "<p style='color:#4b5563'>No data.</p>", st_results or [])

    repo = CHOICE_TO_REPO.get(model_choice, list(MODELS.keys())[0])
    err  = ensure_model(repo)
    if err:
        empty = [""] * len(ALL_CAT_NAMES)
        return (*empty, "", f"<p style='color:#f87171'>❌ {err}</p>",
                "<p>No data.</p>", st_results or [])

    try:
        arr   = preprocess(image)
        probs = infer(arr)
        cats  = categorize(probs, gen_th, chr_th)
        prompt = build_prompt(cats, inc_rating, merge, prepend, append, remove)
        c_html = cats_html(cats)

        if st_results is None:
            st_results = []
        st_results = st_results + [cats]
        s_html = stats_html(st_results)

        cat_strings = [", ".join(t for t,_ in cats.get(c,[])) for c in ALL_CAT_NAMES]
        return (*cat_strings, prompt, c_html, s_html, st_results)

    except Exception:
        tb = traceback.format_exc()
        empty = [""] * len(ALL_CAT_NAMES)
        return (*empty, "", f"<pre style='color:#f87171;font-size:11px'>{tb}</pre>",
                "<p>Error.</p>", st_results or [])


def do_batch(inp, out, model_choice, gen_th, chr_th,
             inc_rating, merge, prepend, append, remove,
             save_txt, overwrite, recursive, progress=gr.Progress()):
    if not inp or not os.path.isdir(inp):
        return "❌ Invalid input directory.", "", "<p>No data.</p>"
    outd = out.strip() or inp
    os.makedirs(outd, exist_ok=True)
    exts = ("png","jpg","jpeg","webp","gif","bmp","PNG","JPG","JPEG","WEBP")
    files = []
    for e in exts:
        files.extend((Path(inp).rglob if recursive else Path(inp).glob)(f"*.{e}"))
    files = sorted(set(files))
    if not files:
        return "⚠️ No images found.", "", "<p>No data.</p>"

    repo = CHOICE_TO_REPO.get(model_choice, list(MODELS.keys())[0])
    err  = ensure_model(repo)
    if err:
        return f"❌ {err}", "", "<p>Error.</p>"

    logs, all_cats = [], []
    t0 = time.time()
    for i, fp in enumerate(files):
        progress((i+1)/len(files), desc=fp.name)
        txtp = (Path(outd)/fp.relative_to(inp)).with_suffix(".txt") if outd!=inp else fp.with_suffix(".txt")
        if not overwrite and txtp.exists():
            logs.append(f"⏭ {fp.name}")
            continue
        try:
            from PIL import Image
            arr   = preprocess(Image.open(fp))
            probs = infer(arr)
            cats  = categorize(probs, gen_th, chr_th)
            p     = build_prompt(cats, inc_rating, merge, prepend, append, remove)
            if save_txt:
                txtp.parent.mkdir(parents=True, exist_ok=True)
                txtp.write_text(p, encoding="utf-8")
            all_cats.append(cats)
            logs.append(f"✅ {fp.name} ({sum(len(v) for v in cats.values())} tags)")
        except Exception as e:
            logs.append(f"❌ {fp.name}: {e}")

    el = time.time()-t0
    ok = sum(1 for l in logs if l.startswith("✅"))
    summary = (f"✔ {ok} processed | ⏱ {el:.1f}s | Output: {outd}\n{'─'*50}\n"
               + "\n".join(logs))
    prev_items = [r for r in all_cats]
    preview = "\n\n".join(
        f"**{logs[i][2:].split(' (')[0]}**"
        for i in range(min(10, len(logs))) if logs[i].startswith("✅")
    )
    return summary, preview, stats_html(all_cats)


# ─── Gradio UI ────────────────────────────────────────────────────────────────

def create_ui():
    with gr.Blocks(analytics_enabled=False,
                   theme=gr.themes.Default(primary_hue="blue", neutral_hue="slate")) as blk:

        gr.Markdown(
            "## 🏷 WD Tagger Batch\n"
            "Auto-tag images using WaifuDiffusion ONNX models with full category breakdown.  \n"
            "_Based on [avan06/wd-tagger-images](https://github.com/avan06/wd-tagger-images)_"
        )

        state = gr.State([])

        # ── Global settings ───────────────────────────────────────────────
        with gr.Row():
            model_dd = gr.Dropdown(CHOICES, value=DEFAULT_CHOICE, label="Model (for Images)")
        with gr.Row():
            gen_th  = gr.Slider(0, 1, 0.35, step=0.01, label="General Tags Threshold")
            chr_th  = gr.Slider(0, 1, 0.85, step=0.01, label="Character Tags Threshold")
        with gr.Row():
            inc_rating = gr.Checkbox(False, label="Include Rating in output string")
            merge      = gr.Checkbox(True,  label="Merge characters into the string output")
        with gr.Row():
            prepend = gr.Textbox(label="Prepend Additional tags (comma split)", lines=1, scale=1)
            append_ = gr.Textbox(label="Append Additional tags (comma split)",  lines=1, scale=1)
            remove  = gr.Textbox(label="Remove tags (comma split)",              lines=1, scale=1)

        # ── Tabs ──────────────────────────────────────────────────────────
        with gr.Tabs():

            # Single image tab
            with gr.Tab("🖼 Image Gallery"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img_in = gr.Image(type="pil",
                                          label="Image Gallery (Drag multiple images here)")
                        with gr.Row():
                            btn_go    = gr.Button("🔍 Interrogate", variant="primary")
                            btn_clear = gr.Button("Clear")
                    with gr.Column(scale=1):
                        prompt_box = gr.Textbox(label="Full prompt output", lines=6, interactive=True)

                gr.Markdown("### Categorized (tags) — Interactive")

                # One textbox per category (matches original's interactive tag panels)
                cat_boxes = []
                for cat in ALL_CAT_NAMES:
                    color = CAT_COLORS.get(cat, "#aaa")
                    with gr.Row():
                        tb = gr.Textbox(
                            label=f"{'●'} {cat}",
                            lines=1,
                            interactive=True,
                            scale=4,
                        )
                        cat_boxes.append(tb)

                cats_display = gr.HTML(label="Visual category panels")

            # Batch tab
            with gr.Tab("📁 Batch Processing"):
                with gr.Row():
                    b_inp = gr.Textbox(label="Input Directory",  placeholder="/path/to/images/")
                    b_out = gr.Textbox(label="Output Directory", placeholder="(leave blank = same as input)")
                with gr.Row():
                    b_save = gr.Checkbox(True,  label="Save .txt files")
                    b_over = gr.Checkbox(False, label="Overwrite existing .txt")
                    b_rec  = gr.Checkbox(False, label="Recursive (include subfolders)")
                b_btn = gr.Button("🚀 Start Batch Tagging", variant="primary")
                with gr.Row():
                    b_log  = gr.Textbox(label="Processing Log",  lines=15, interactive=False)
                    b_prev = gr.Markdown(label="Preview")

            # Statistics tab
            with gr.Tab("📊 Tags Statistics (All files)"):
                stats_out = gr.HTML()
                gr.Markdown("_Updated after each interrogation or batch run._")

            # Model info tab
            with gr.Tab("ℹ Model Info"):
                gr.Markdown(
                    "### Available Models\n"
                    + "\n".join(f"- **{v}** — `{k}`" for k,v in MODELS.items())
                    + "\n\n### Category Reference\n"
                    + "\n".join(f"- **{c}**" for c in ALL_CAT_NAMES)
                    + "\n\n### Notes\n"
                    "- Models auto-download from HuggingFace on first use.\n"
                    "- Cache: `extensions/sd-webui-wd-tagger-batch/models/`\n"
                    "- CUDA used automatically if available.\n"
                )

        # ── Event wiring ──────────────────────────────────────────────────
        all_outputs = cat_boxes + [prompt_box, cats_display, stats_out, state]

        btn_go.click(
            fn=do_interrogate,
            inputs=[img_in, model_dd, gen_th, chr_th,
                    inc_rating, merge, prepend, append_, remove, state],
            outputs=all_outputs,
        )

        def clear_fn():
            return [None] + [""]*(len(ALL_CAT_NAMES)+1) + ["", []]

        btn_clear.click(
            fn=clear_fn,
            outputs=[img_in] + cat_boxes + [prompt_box, cats_display, state],
        )

        b_btn.click(
            fn=do_batch,
            inputs=[b_inp, b_out, model_dd, gen_th, chr_th,
                    inc_rating, merge, prepend, append_, remove,
                    b_save, b_over, b_rec],
            outputs=[b_log, b_prev, stats_out],
        )

    return [(blk, "WD Tagger Batch", "wd_tagger_batch")]


# ─── SD WebUI hook ────────────────────────────────────────────────────────────

try:
    from modules import script_callbacks
    script_callbacks.on_ui_tabs(create_ui)
except ImportError:
    if __name__ == "__main__":
        create_ui()[0][0].launch()
