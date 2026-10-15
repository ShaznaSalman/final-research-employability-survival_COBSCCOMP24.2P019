"""Research paper Fig. 1 (three-layer architecture). python docs/thesis_figures/paper_fig1_architecture.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

plt.rcParams["font.family"] = "Times New Roman"
BLUE, GREY, PINK, EDGE = "#EDF4FD", "#F5F5F4", "#FDF0EC", "#555555"
fig, ax = plt.subplots(figsize=(8, 4.9), dpi=200)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9.8)
ax.axis("off")


def box(x, y, w, h, color, title, body):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.25", fc=color, ec=EDGE, lw=1.6))
    ax.text(x + w / 2, y + h - 0.55, title, ha="center", va="center", fontsize=15.5, fontweight="bold")
    ax.text(x + w / 2, y + (h - 0.9) / 2, body, ha="center", va="center", fontsize=13, linespacing=1.15)


def arrow(a, b, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, color=EDGE, lw=1.6, **kw))


box(0.2, 5.35, 4.4, 4.25, BLUE, "Data layer", "1,175 graduate and\n245 employer\nresponses;\n324 job adverts")
box(5.6, 5.35, 10.2, 4.25, BLUE, "Offline pipeline (one command)",
    "clean → duration and event → fixed split\n(train 620 / val. 133 / test 132) → fit 7 models\n"
    "→ select on validation → test once\n→ 200 bootstrap refits, fairness, robustness")
box(0.2, 0.2, 4.4, 3.95, GREY, "Saved artifacts", "AFT model and\nbootstrap refits,\naggregated reference\ndata (SQLite)")
box(5.3, 0.2, 4.5, 3.95, PINK, "FastAPI service", "13 stateless\nendpoints: predict,\nexplain, what-if,\nalignment, chat")
box(10.65, 0.2, 5.15, 3.95, PINK, "Web application", "Streamlit, 4 pages:\nHome, My prediction,\nJob market,\nResearch lab")
arrow((4.75, 7.45), (5.45, 7.45))
arrow((7.1, 5.3), (2.8, 4.25))
ax.text(7.9, 4.9, "writes artifacts", ha="center", va="center", fontsize=13, style="italic", color="#555555")
arrow((4.75, 2.15), (5.15, 2.15))
arrow((9.95, 2.15), (10.5, 2.15))
out = Path(__file__).with_name("paper_fig1_architecture.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(out)
