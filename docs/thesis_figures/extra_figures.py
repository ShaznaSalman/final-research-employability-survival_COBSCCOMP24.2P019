"""Three thesis images: main JobSpark pages (Chapter 4), the UI style guide (from frontend/ui.py tokens)
and the cleaning log as a table image (numbers from docs/data_cleaning_log.md)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
UI = HERE / "jobspark_ui" / "light-desktop"

# ---------------------------------------------------------------- main pages, 2 x 2 with gaps and labels
pages = [("01-home.png", "(a) Home"), ("02-profile-step1-about-you.png", "(b) My profile, step 1"),
         ("07-improve-my-chances.png", "(c) Improve my chances"), ("08-compare.png", "(d) Compare")]
ims = [Image.open(UI / f).convert("RGB") for f, _ in pages]
w, h = ims[0].size; gap, lab = 60, 70
canvas = Image.new("RGB", (2 * w + gap, 2 * (h + lab) + gap), "white")
dr = ImageDraw.Draw(canvas); font = ImageFont.truetype("arial.ttf", 40)
for k, (im, (_, label)) in enumerate(zip(ims, pages)):
    x = (k % 2) * (w + gap); y = (k // 2) * (h + lab + gap)
    canvas.paste(im, (x, y)); dr.rectangle([x, y, x + w - 1, y + h - 1], outline=(190, 190, 190), width=3)
    tw = dr.textlength(label, font=font); dr.text((x + (w - tw) / 2, y + h + 12), label, font=font, fill="black")
canvas.save(HERE / "ui_main_pages.png")

# ---------------------------------------------------------------- style guide
light = [("Ink", "#15203a"), ("Muted", "#5a6682"), ("Panel", "#ffffff"), ("Wash", "#f5f7fb"), ("Brand", "#3b66e0"),
         ("Accent", "#ffa62b"), ("Mint", "#0b7d6e"), ("Peach", "#c2410c"), ("Lavender", "#5b3fd0")]
dark = [("Ink", "#e8ecf5"), ("Muted", "#9da9c2"), ("Panel", "#172033"), ("Wash", "#0f1624"), ("Brand", "#7da2ff"),
        ("Accent", "#ffa62b"), ("Mint", "#6ee0c8"), ("Peach", "#ffb98a"), ("Lavender", "#c2b6ff")]
fig, ax = plt.subplots(figsize=(6.4, 3.6)); ax.set_xlim(0, 6.4); ax.set_ylim(0, 3.6); ax.axis("off")
for row, (title, pal, y) in enumerate([("Light mode", light, 3.0), ("Dark mode", dark, 1.95)]):
    ax.text(0.05, y + 0.42, title, fontsize=7.5, weight="bold")
    for i, (n, c) in enumerate(pal):
        x = 0.05 + i * 0.7
        ax.add_patch(Rectangle((x, y - 0.12), 0.6, 0.45, fc=c, ec="#9aa5b1", lw=0.6))
        ax.text(x + 0.3, y - 0.22, n, ha="center", fontsize=5.5); ax.text(x + 0.3, y - 0.36, c, ha="center", fontsize=5, color="#555")
ax.text(0.05, 1.15, "Typography", fontsize=7.5, weight="bold")
ax.text(0.05, 0.95, "Headings: Plus Jakarta Sans, bold", fontsize=9, weight="bold", color="#15203a")
ax.text(0.05, 0.68, "Body text: Inter, regular; numbers in tabular figures", fontsize=7, color="#15203a")
ax.text(3.55, 1.15, "Components", fontsize=7.5, weight="bold")
ax.add_patch(FancyBboxPatch((3.55, 0.45), 1.35, 0.62, boxstyle="round,pad=0.01,rounding_size=0.08", fc="#3069e3", ec="none"))
ax.text(3.62, 0.92, "MOST LIKELY TIME", fontsize=4.5, color="white", weight="bold"); ax.text(3.62, 0.56, "5 months", fontsize=10, color="white", weight="bold")
ax.add_patch(FancyBboxPatch((5.05, 0.8), 1.15, 0.27, boxstyle="round,pad=0.01,rounding_size=0.13", fc="#ffa62b", ec="none"))
ax.text(5.62, 0.93, "Get my estimate", ha="center", va="center", fontsize=6, color="#2b1800", weight="bold")
ax.add_patch(FancyBboxPatch((5.05, 0.45), 1.15, 0.27, boxstyle="round,pad=0.01,rounding_size=0.13", fc="white", ec="#3b66e0", lw=0.8))
ax.text(5.62, 0.58, "Edit profile", ha="center", va="center", fontsize=6, color="#2547b0")
ax.text(0.05, 0.2, "Cards with 16 px rounded corners; colour by meaning: blue = the user and technical skills, orange = employers and soft skills, green = job advertisements.", fontsize=5.3, color="#333")
fig.savefig(HERE / "ui_style_guide.png", dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.03); plt.close(fig)

# ---------------------------------------------------------------- cleaning log as a table image
rows = [("Raw graduate responses", "-", "-", "1,175"),
        ("Duplicate submission", "Identical answers to an earlier submission", "4", "1,171"),
        ("Failed attention check", "Attention-check row (Q18) not answered with 4", "24", "1,147"),
        ("Final-year set aside", "No outcome yet; descriptive use only", "230", "917"),
        ("Outside eligibility window", "Completed over 36 months before the survey", "2", "915"),
        ("Missing job status", "Q17 missing, so the event is unknown", "2", "913"),
        ("Implausible job start", "First job more than 24 months before completion", "1", "912"),
        ("Not in the IT job market", "No IT role; further studies or not seeking", "27", "885"),
        ("Modelling graduates", "667 found an IT role; 218 (24.6%) censored", "-", "885"),
        ("Employer responses", "6 failed the attention check (245 received)", "6", "239")]
plt.rcParams["font.family"] = "Times New Roman"
fig, ax = plt.subplots(figsize=(6.4, 3.0)); ax.axis("off")
tb = ax.table(cellText=rows, colLabels=["Step", "Reason", "Removed", "Remaining"], loc="center", cellLoc="left",
              colWidths=[0.27, 0.49, 0.11, 0.13])
tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.45)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor("black"); cell.set_linewidth(0.5)
    if r == 0: cell.set_text_props(weight="bold")
fig.savefig(HERE / "cleaning_log_table.png", dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.02); plt.close(fig)
print("saved ui_main_pages.png, ui_style_guide.png, cleaning_log_table.png")
