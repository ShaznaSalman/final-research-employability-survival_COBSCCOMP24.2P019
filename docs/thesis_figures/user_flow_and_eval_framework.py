"""Two thesis diagrams drawn at the printed width (16 cm) from the real application and evaluation.

user_flow.png            - pages and order as in Appendix A and frontend/app.py navigation (four profile steps)
evaluation_framework.png - the evaluation components of Chapter 5 and what each one checks
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).parent
BLUE, LIGHT, GREY, TEXT = "#1c5cab", "#eaf1fb", "#555555", "#1a1a1a"


def box(ax, x, y, w, h, title, sub="", fc=LIGHT, fs=7):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.05", fc=fc, ec=BLUE, lw=0.9))
    if sub:
        ax.text(x, y + h * 0.2, title, ha="center", va="center", fontsize=fs, weight="bold", color=TEXT)
        ax.text(x, y - h * 0.18, sub, ha="center", va="center", fontsize=fs - 1.3, color=GREY, linespacing=1.05)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=fs, weight="bold", color=TEXT)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.8, mutation_scale=7))


# ---------------------------------------------------------------- user flow (6.4 x 2.5 in)
fig, ax = plt.subplots(figsize=(6.4, 2.5)); ax.set_xlim(0, 6.4); ax.set_ylim(0, 2.5); ax.axis("off")
box(ax, 0.5, 1.85, 0.85, 0.5, "Home", "purpose and\nheadline figures")
ax.add_patch(FancyBboxPatch((1.1, 1.5), 3.55, 0.85, boxstyle="round,pad=0.01,rounding_size=0.05", fc="white", ec=BLUE, lw=0.8, ls=(0, (3, 2))))
ax.text(2.875, 2.25, "My profile (answers kept in this browser session only)", ha="center", va="center", fontsize=5.8, color=GREY)
for i, (s, t) in enumerate([("Step 1", "About you"), ("Step 2", "Experience"), ("Step 3", "Skills"), ("Step 4", "Preparation")]):
    x = 1.55 + i * 0.88
    box(ax, x, 1.82, 0.74, 0.46, s, t, fs=6.6)
    if i: arrow(ax, x - 0.5, 1.82, x - 0.38, 1.82)
arrow(ax, 0.93, 1.85, 1.17, 1.83)
box(ax, 5.65, 1.85, 1.3, 0.58, "My dashboard", "most likely time, range,\nconfidence and reasons")
arrow(ax, 4.6, 1.83, 4.99, 1.85)
dest = [("Improve my chances", "what-if sliders and\ncombinations to explore"), ("Compare", "skills against employers,\njob ads and peers"),
        ("Job market", "324 real job\nadvertisements"), ("Ask", "questions about\nthe result"), ("Research lab", "evidence behind\nthe estimates")]
for i, (t, s) in enumerate(dest):
    x = 0.66 + i * 1.27
    box(ax, x, 0.62, 1.18, 0.55, t, s, fs=6.2)
    arrow(ax, 5.65, 1.55, x, 0.91)
ax.text(3.2, 0.13, "From the dashboard the graduate can open any of these pages and return to edit the profile.", ha="center", fontsize=5.8, color=GREY)
fig.savefig(OUT / "user_flow.png", dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.02); plt.close(fig)

# ---------------------------------------------------------------- evaluation framework (6.4 x 3.2 in)
levels = [
    ("Technical correctness", [("Functional testing", "412 automated tests;\nfour contrasting profiles"),
                               ("System performance", "response time per endpoint;\nstart-up and model size")]),
    ("Analytical validity", [("Model performance", "Uno's C, integrated Brier\nscore, calibration (test set)"),
                             ("Explanation checks", "additivity and stability\nof the contributions"),
                             ("Robustness and fairness", "sensitivity, temporal and\nsubgroup checks")]),
    ("Market alignment", [("Skill comparison", "self-ratings vs employer\nimportance and job ads"),
                          ("Coding agreement", "keyword rules vs hand\ncoding (Cohen's kappa)")]),
    ("Usefulness and usability", [("Usability evaluation", "System Usability Scale,\n12 participants"),
                                  ("LLM benchmark", "same 132 test profiles\ngiven to ChatGPT"),
                                  ("Existing solutions", "capability comparison with\nthe reviewed systems")]),
]
fig, ax = plt.subplots(figsize=(6.4, 3.2)); ax.set_xlim(0, 6.4); ax.set_ylim(0, 3.2); ax.axis("off")
box(ax, 3.2, 2.95, 3.9, 0.32, "Artifact: time-to-employment decision-support system", fc="#d6e4f7", fs=7)
for i, (lvl, comps) in enumerate(levels):
    x = 0.8 + i * 1.6
    box(ax, x, 2.42, 1.48, 0.3, lvl, fc="#f3f6fa", fs=6.2)
    arrow(ax, 3.2, 2.78, x, 2.58)
    for j, (c, s) in enumerate(comps):
        y = 1.85 - j * 0.62
        box(ax, x, y, 1.48, 0.52, c, s, fc="white", fs=6.5)
        arrow(ax, x, (2.26 if j == 0 else y + 0.62 - 0.27), x, y + 0.27)
ax.text(3.2, 0.12, "Every target was fixed in Chapter 3 before the data were analysed.", ha="center", fontsize=5.8, color=GREY)
fig.savefig(OUT / "evaluation_framework.png", dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.02); plt.close(fig)
print("saved user_flow.png and evaluation_framework.png")
