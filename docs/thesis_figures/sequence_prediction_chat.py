"""Sequence diagram of obtaining a prediction and asking the assistant (thesis Chapter 4).

Drawn at the printed width (16 cm) so that the text prints at about 6.5-7 pt. Every step mirrors the code on main:
frontend/views/predict.py (four profile steps), views/dashboard.py (/predict, /explain, /suggestions,
/market-alignment), views/improve.py (/whatif), views/ask.py and chat_widget.py (/chat), api/main.py (typed
validation), core/predict.py and core/chat.py (keyword rules, no language model, nothing stored).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ACTORS = [("Graduate", ""), ("Browser", "JobSpark pages"), ("FastAPI", "typed validation"),
          ("Core library", "model functions"), ("Artifacts", "model and data"), ("Assistant", "rules, no LLM")]
X = [0.45, 1.5, 2.58, 3.72, 4.86, 5.95]
G, B, A, C, R, H = range(6)
STEPS = [
    (G, B, "Enters a profile in four steps", "req"),
    (B, B, "Keeps answers in this session only", "self"),
    (B, A, "POST /predict (profile)", "req"),
    (A, A, "Validates every field", "self"),
    (A, C, "predict(profile)", "req"),
    (C, R, "Final model, 200 bootstrap refits", "req"),
    (R, C, "Survival curves S(t)", "ret"),
    (C, C, "Range, 80% interval, confidence label", "self"),
    (C, B, "Most likely time, range, confidence (via the API)", "ret"),
    (B, C, "POST /explain, /suggestions, /market-alignment", "req"),
    (C, B, "Reasons, suggestions, skill gaps", "ret"),
    (G, B, "Moves a what-if slider", "req"),
    (B, C, "POST /whatif: re-predicted with the changes", "req"),
    (C, B, "Updated estimate (caution outside training range)", "ret"),
    (G, B, "Asks a question", "req"),
    (B, H, "POST /chat (question, profile)", "req"),
    (H, C, "Uses the same model outputs", "req"),
    (H, B, "Reply built from the model's own numbers", "ret"),
    (B, G, "Shows the results; nothing stored", "ret"),
]
BLUE, GREY, TEXT = "#1c5cab", "#555555", "#1a1a1a"
STEP, SELF = 0.24, 0.32
height = 0.75 + sum(SELF if k == "self" else STEP for *_, k in STEPS) + 0.2
fig, ax = plt.subplots(figsize=(6.4, height))
ax.set_xlim(0, 6.4); ax.set_ylim(-height + 0.55, 0.55); ax.axis("off")
for (name, sub), x in zip(ACTORS, X):
    ax.add_patch(FancyBboxPatch((x - 0.47, 0.06), 0.94, 0.44, boxstyle="round,pad=0.01,rounding_size=0.05", fc="#eaf1fb", ec=BLUE, lw=0.9))
    ax.text(x, 0.36 if sub else 0.28, name, ha="center", va="center", fontsize=7, weight="bold", color=TEXT)
    if sub: ax.text(x, 0.17, sub, ha="center", va="center", fontsize=5.8, color=GREY)
    ax.plot([x, x], [0.06, -height + 0.6], color="#9aa5b1", lw=0.7, ls=(0, (3, 2)))
y = -0.05
for n, (a, b, text, kind) in enumerate(STEPS, 1):
    y -= SELF if kind == "self" else STEP
    label = f"{n}. {text}"
    if kind == "self":
        x = X[a]; w, h = 0.13, 0.12
        ax.plot([x, x + w, x + w], [y + h / 2, y + h / 2, y - h / 2], color=BLUE, lw=0.8)
        ax.annotate("", xy=(x + 0.01, y - h / 2), xytext=(x + w, y - h / 2), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.8, mutation_scale=6))
        ax.text(x + w + 0.05, y, label, ha="left", va="center", fontsize=6.5, color=TEXT, bbox=dict(fc="white", ec="none", pad=0.3))
    else:
        x1, x2 = X[a], X[b]
        ax.annotate("", xy=(x2 + (-0.03 if x2 > x1 else 0.03), y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE if kind == "req" else GREY, lw=0.8,
                                    ls="-" if kind == "req" else (0, (3, 2)), mutation_scale=6))
        mid = max((x1 + x2) / 2, 1.05)
        ax.text(mid, y + 0.025, label, ha="center", va="bottom", fontsize=6.5, color=TEXT, bbox=dict(fc="white", ec="none", pad=0.2))
out = Path(__file__).with_name("sequence_prediction_chat.png")
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.02)
print("saved", out)
