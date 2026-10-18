"""Notebook-style screenshots of executed cells of notebooks/research_walkthrough.ipynb (thesis Appendix: proof of model training).

Each image shows the cell's code and its saved output exactly as stored in the notebook; nothing is re-run.
"""
import base64, io, json, textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
NB = json.loads((ROOT / "notebooks" / "research_walkthrough.ipynb").read_text(encoding="utf-8"))
OUT = Path(__file__).parent / "notebook"
OUT.mkdir(exist_ok=True)
CELLS = {4: "data", 42: "split", 48: "training", 54: "deepsurv", 57: "selection", 85: "test", 128: "tests", 130: "consistency"}
W, PAD, LH = 1500, 16, 22
mono = ImageFont.truetype("consola.ttf", 17)
monob = ImageFont.truetype("consolab.ttf", 17)
MAXCODE, MAXOUT, WRAP = 22, 30, 128


def lines_of(text, limit):
    out = []
    for ln in text.rstrip("\n").split("\n"):
        out += textwrap.wrap(ln, WRAP, replace_whitespace=False, drop_whitespace=False) or [""]
    if len(out) > limit:
        out = out[:limit] + [f"... ({len(out) - limit} more lines)"]
    return out


def render(i, name):
    c = NB["cells"][i]
    code = lines_of("".join(c["source"]), MAXCODE)
    texts, imgs = [], []
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream":
            texts.append("".join(o["text"]))
        data = o.get("data", {})
        if "image/png" in data:
            imgs.append(Image.open(io.BytesIO(base64.b64decode(data["image/png"]))).convert("RGB"))
        elif "text/plain" in data:
            texts.append("".join(data["text/plain"]))
    out = lines_of("\n".join(texts), MAXOUT) if texts else []
    for k, im in enumerate(imgs):
        s = min(1.0, (W - 120) / im.width, 520 / im.height); imgs[k] = im.resize((int(im.width * s), int(im.height * s)))
    h = PAD + len(code) * LH + 2 * PAD + (len(out) * LH + PAD if out else 0) + sum(im.height + PAD for im in imgs) + PAD
    img = Image.new("RGB", (W, h), "white"); d = ImageDraw.Draw(img)
    n = c.get("execution_count") or ""
    d.text((10, PAD), f"In [{n}]:", font=monob, fill="#303f9f")
    d.rectangle([100, PAD - 6, W - 10, PAD + len(code) * LH + 6], fill="#f5f5f5", outline="#cfcfcf")
    y = PAD
    for ln in code:
        d.text((110, y), ln, font=mono, fill="#008000" if ln.lstrip().startswith("#") else "#1a1a1a"); y += LH
    y += 2 * PAD
    for ln in out:
        d.text((110, y), ln, font=mono, fill="#1a1a1a"); y += LH
    if out: y += PAD
    for im in imgs:
        img.paste(im, (110, y)); y += im.height + PAD
    img.save(OUT / f"nb_{i:03d}_{name}.png")
    print("saved", name, img.size)


for i, name in CELLS.items():
    render(i, name)
