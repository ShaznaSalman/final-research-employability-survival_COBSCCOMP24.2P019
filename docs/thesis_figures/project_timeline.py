"""Thesis Figure 3.3: project timeline (Gantt), January to October 2026.

Dates come from the dated project documents and docs/development_log.md (title clearance 10 Apr, proposal 18-19 Apr,
ethics form 14 Mar - 1 May, questionnaires 5 Jun and 7 Aug, supervisor meetings 6 Jun and 7 Aug, development from
4 Aug, job advertisements 12 Jul - 11 Oct, main survey closed 6 Oct, real data analysed from 11 Oct, usability study
and LLM benchmark 14 Oct). The research started on 12 Jan 2026; the title clearance and proposal ran from February to
May.

Run:  python docs/thesis_figures/project_timeline.py   -> docs/thesis_figures/project_timeline.png
"""
from datetime import date
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).with_name("project_timeline.png")
ACCENT, ACCENT_LIGHT, INK, INK_MUTED, GRID, SURFACE = "#2a78d6", "#9ec5f4", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# (phase, start, end) - in the order the work began
PHASES = [
    ("Topic selection and literature review", date(2026, 1, 12), date(2026, 4, 30)),
    ("Title clearance", date(2026, 2, 1), date(2026, 4, 10)),
    ("Research proposal and defence", date(2026, 3, 1), date(2026, 5, 31)),
    ("Ethical clearance application", date(2026, 3, 14), date(2026, 5, 1)),
    ("Chapters 1-3 (introduction, literature, method)", date(2026, 5, 1), date(2026, 10, 8)),
    ("Questionnaire design and expert review", date(2026, 6, 1), date(2026, 8, 7)),
    ("Job-advertisement collection", date(2026, 7, 12), date(2026, 10, 11)),
    ("System development (pipeline, models, app)", date(2026, 8, 4), date(2026, 10, 13)),
    ("Survey data collection (pilot and main)", date(2026, 8, 8), date(2026, 10, 6)),
    ("Analysis of the real data", date(2026, 10, 11), date(2026, 10, 15)),
    ("Usability study and LLM benchmark", date(2026, 10, 14), date(2026, 10, 15)),
    ("Chapters 4-6, paper and final draft", date(2026, 10, 10), date(2026, 10, 16)),
]
MILESTONES = [   # (label, date, label height 0-2, horizontal alignment) - staggered so close labels do not overlap
    ("Title clearance submitted", date(2026, 4, 10), 2, "right"),
    ("Proposal finalised", date(2026, 4, 19), 1, "center"),
    ("Ethics form submitted", date(2026, 5, 1), 0, "left"),
    ("Supervisor meeting", date(2026, 6, 6), 2, "left"),
    ("Supervisor meeting", date(2026, 8, 7), 1, "center"),
    ("Main survey closed", date(2026, 10, 6), 0, "right"),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 6.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    n = len(PHASES)
    for i, (label, start, end) in enumerate(PHASES):
        y = n - 1 - i
        width = max((end - start).days, 2)
        ax.barh(y, width, left=mdates.date2num(start), height=0.6, color=ACCENT, edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(range(n), [p[0] for p in reversed(PHASES)], fontsize=8.5, color=INK)
    for label, day, level, ha in MILESTONES:
        x = mdates.date2num(day)
        ax.axvline(x, color=INK_MUTED, linestyle=":", linewidth=1, zorder=0)
        ax.scatter(x, n - 0.25, marker="D", s=36, color=INK, zorder=3)
        ty = n + 0.35 + 0.75 * level
        ax.plot([x, x], [n - 0.25, ty - 0.05], color=INK_MUTED, linewidth=0.8, zorder=2)
        ax.annotate(f"{label} ({day:%d %b})", (x, ty), ha=ha, va="bottom", fontsize=7.5, color=INK)
    ax.set_ylim(-0.6, n + 2.6)
    ax.set_xlim(mdates.date2num(date(2026, 1, 1)), mdates.date2num(date(2026, 10, 31)))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title("Project timeline, January to October 2026", fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
