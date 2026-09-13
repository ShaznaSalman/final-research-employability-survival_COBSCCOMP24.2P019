"""Plotly figures for the Job market and Research lab pages. Same visual system as charts.py:
blue = technical skill / the selected model, orange = soft skill, aqua = job ads / interns, violet = junior roles.
Thin marks, recessive grid, a legend whenever there are two or more series, hover on every mark."""
from __future__ import annotations

import math

import plotly.graph_objects as go

from frontend.charts import AQUA, BLUE, GRID, INK, MUTED, ORANGE, SURFACE, _layout

VIOLET = "#4a3aa7"
SEQ = ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]      # one-hue ramp (blue), light -> dark
KIND_COLOR = {"technical": BLUE, "soft": ORANGE}
CATEGORICAL = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#008300", VIOLET, "#e34948"]   # fixed order, never cycled


def _axis_title(text: str) -> dict:
    return dict(text=text, font=dict(color=MUTED, size=12))


# ---------------------------------------------------------------------------
# Job market (real ads)
# ---------------------------------------------------------------------------
def demand_bars(prevalence: list[dict], top_n: int = 15) -> go.Figure:
    """Share of ads naming each skill, with its Wilson 95% interval; colour = technical vs soft."""
    rows = prevalence[:top_n][::-1]
    fig = go.Figure()
    for kind, name in (("technical", "Technical skill"), ("soft", "Soft skill")):
        sub = [r for r in rows if r["kind"] == kind]
        fig.add_trace(go.Bar(
            x=[r["pct"] for r in sub], y=[r["label"] for r in sub], orientation="h", name=name,
            marker=dict(color=KIND_COLOR[kind], cornerradius=4),
            error_x=dict(type="data", symmetric=False, color=MUTED, thickness=1.2, width=4,
                         array=[r["ci_high"] - r["pct"] for r in sub], arrayminus=[r["pct"] - r["ci_low"] for r in sub]),
            customdata=[[r["ads"], r["ci_low"], r["ci_high"]] for r in sub],
            hovertemplate="<b>%{y}</b><br>%{x:.0f}% of ads (%{customdata[0]} ads)<br>"
                          "95% CI %{customdata[1]:.0f}-%{customdata[2]:.0f}%<extra></extra>"))
    fig.update_yaxes(categoryorder="array", categoryarray=[r["label"] for r in rows], showgrid=False,
                     tickfont=dict(color=INK))
    fig.update_xaxes(range=[0, 100], ticksuffix="%", gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED))
    return _layout(fig, 70 + 28 * len(rows), showlegend=True, bargap=0.28,
                   legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(color=INK)))


def count_bars(items: list[tuple[str, int]], color: str = AQUA, unit: str = "ads") -> go.Figure:
    rows = items[::-1]
    fig = go.Figure(go.Bar(x=[v for _, v in rows], y=[k for k, _ in rows], orientation="h",
                           marker=dict(color=color, cornerradius=4), text=[str(v) for _, v in rows],
                           textposition="outside", textfont=dict(color=MUTED, size=11),
                           hovertemplate="<b>%{y}</b>: %{x} " + unit + "<extra></extra>"))
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, range=[0, max(v for _, v in rows) * 1.18])
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK))
    return _layout(fig, 40 + 30 * len(rows), bargap=0.3)


def role_heatmap(by_role: dict, roles: dict, skills: list[dict]) -> go.Figure:
    """Rows = skills, columns = role families; cell = % of that role's ads naming the skill."""
    cols = list(roles)
    z = [[by_role[r][s["key"]] for r in cols] for s in skills]
    x = [f"{r}<br>n={roles[r]}" for r in cols]
    y = [s["label"] for s in skills]
    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y, zmin=0, zmax=100, xgap=2, ygap=2,
        colorscale=[[i / (len(SEQ) - 1), c] for i, c in enumerate(SEQ)],
        colorbar=dict(ticksuffix="%", thickness=10, outlinewidth=0, len=0.8, tickfont=dict(color=MUTED)),
        hovertemplate="<b>%{y}</b><br>%{x}<br>%{z:.0f}% of ads<extra></extra>"))
    for i, row in enumerate(z):
        for j, v in enumerate(row):
            fig.add_annotation(x=x[j], y=y[i], text=f"{v:.0f}", showarrow=False,
                               font=dict(size=11, color="#ffffff" if v >= 55 else INK))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=INK))
    fig.update_xaxes(side="top", showgrid=False, tickfont=dict(color=INK, size=11))
    return _layout(fig, 90 + 30 * len(y))


def skill_network(prevalence: list[dict], pairs: list[dict], min_lift: float = 1.3, max_nodes: int = 14,
                  max_edges: int = 18) -> go.Figure:
    """Circular network of the strongest skill bundles: node size = share of ads, edge width = phi correlation
    (the max_edges pairs with the highest phi among those with lift >= min_lift)."""
    edges = sorted([p for p in pairs if p["lift"] >= min_lift and p["phi"] > 0], key=lambda p: -p["phi"])[:max_edges]
    used = {p["a"] for p in edges} | {p["b"] for p in edges}
    nodes = [r for r in prevalence if r["label"] in used][:max_nodes]
    names = [n["label"] for n in nodes]
    edges = [p for p in edges if p["a"] in names and p["b"] in names]
    pos = {n: (math.cos(2 * math.pi * i / len(names) + math.pi / 2), math.sin(2 * math.pi * i / len(names) + math.pi / 2))
           for i, n in enumerate(names)}
    fig = go.Figure()
    for p in edges:
        (x0, y0), (x1, y1) = pos[p["a"]], pos[p["b"]]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color="#9ec5f4", width=1 + 10 * p["phi"])))
    fig.add_trace(go.Scatter(
        x=[(pos[p["a"]][0] + pos[p["b"]][0]) / 2 for p in edges], y=[(pos[p["a"]][1] + pos[p["b"]][1]) / 2 for p in edges],
        mode="markers", marker=dict(size=16, color="rgba(0,0,0,0)"), showlegend=False,
        customdata=[[p["a"], p["b"], p["ads"], p["lift"], p["phi"]] for p in edges],
        hovertemplate="<b>%{customdata[0]} + %{customdata[1]}</b><br>together in %{customdata[2]} ads<br>"
                      "lift %{customdata[3]:.1f} · phi %{customdata[4]:.2f}<extra></extra>"))
    for kind, name in (("technical", "Technical skill"), ("soft", "Soft skill")):
        sub = [n for n in nodes if n["kind"] == kind]
        fig.add_trace(go.Scatter(
            x=[pos[n["label"]][0] for n in sub], y=[pos[n["label"]][1] for n in sub], mode="markers", name=name,
            marker=dict(size=[12 + 0.45 * n["pct"] for n in sub], color=KIND_COLOR[kind], line=dict(color=SURFACE, width=2)),
            customdata=[[n["pct"]] for n in sub], text=[n["label"] for n in sub],
            hovertemplate="<b>%{text}</b><br>%{customdata[0]:.0f}% of ads<extra></extra>"))
    for n in names:
        x, y = pos[n]
        short = n.replace("Artificial Intelligence / Machine Learning", "AI / ML").replace(" Fundamentals", "")
        fig.add_annotation(x=x * 1.3, y=y * 1.18, text=short, showarrow=False, font=dict(size=11, color=INK),
                           xanchor="left" if x > 0.2 else "right" if x < -0.2 else "center")
    fig.update_xaxes(visible=False, range=[-2.3, 2.3])
    fig.update_yaxes(visible=False, range=[-1.4, 1.4])
    return _layout(fig, 560, showlegend=True,
                   legend=dict(orientation="h", yanchor="top", y=1.02, x=0, font=dict(color=INK)))


def dumbbell(tests: list[dict], top_n: int = 10) -> go.Figure:
    """Intern vs junior/associate share per skill; significant differences (BH q < 0.05) starred."""
    rows = sorted(tests, key=lambda t: -abs(t["intern_pct"] - t["junior_pct"]))[:top_n][::-1]
    y = [t["label"] + (" ★" if t["significant"] else "") for t in rows]
    fig = go.Figure()
    for t, lab in zip(rows, y):
        fig.add_trace(go.Scatter(x=[t["intern_pct"], t["junior_pct"]], y=[lab, lab], mode="lines",
                                 line=dict(color=GRID, width=4), hoverinfo="skip", showlegend=False))
    for key, name, color in (("intern_pct", "Interns / trainees", AQUA), ("junior_pct", "Junior / associate", VIOLET)):
        fig.add_trace(go.Scatter(x=[t[key] for t in rows], y=y, mode="markers", name=name,
                                 marker=dict(size=12, color=color, line=dict(color=SURFACE, width=2)),
                                 customdata=[[t["q"]] for t in rows],
                                 hovertemplate=name + ": %{x:.0f}% of ads<br>q = %{customdata[0]:.3f}<extra></extra>"))
    fig.update_xaxes(range=[0, 100], ticksuffix="%", gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK))
    return _layout(fig, 90 + 32 * len(rows), showlegend=True,
                   legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(color=INK)))


# ---------------------------------------------------------------------------
# Research lab
# ---------------------------------------------------------------------------
def model_cindex(models: list[dict], selected: str) -> go.Figure:
    rows = sorted(models, key=lambda m: m["uno_c_mean"])
    fig = go.Figure()
    fig.add_vline(x=0.5, line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_annotation(x=0.5, y=1, yref="paper", text="chance", showarrow=False, yshift=10, font=dict(color=MUTED, size=11))
    for m in rows:
        chosen = m["model"] == selected
        fig.add_trace(go.Scatter(
            x=[m["uno_c_mean"]], y=[m["label"] + ("  ✓ selected" if chosen else "")], mode="markers",
            marker=dict(size=14 if chosen else 11, color=BLUE if chosen else "#86b6ef", line=dict(color=SURFACE, width=2)),
            error_x=dict(type="data", array=[m["uno_c_sd"]], color=MUTED, thickness=1.2, width=5),
            hovertemplate=f"<b>{m['label']}</b><br>Uno's C {m['uno_c_mean']:.3f} ± {m['uno_c_sd']:.3f}<br>"
                          f"Integrated Brier score {m['ibs_mean']:.3f}<extra></extra>"))
    fig.update_xaxes(range=[0.45, 0.8], gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED),
                     title=_axis_title("Uno's C-index on the validation set (± bootstrap SE)"))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK))
    return _layout(fig, 90 + 38 * len(rows))


def calibration(cal: list[dict]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, width=1, dash="dot"),
                             hoverinfo="skip", showlegend=False))
    for h, color in zip(sorted({c["horizon"] for c in cal}), CATEGORICAL):
        sub = sorted([c for c in cal if c["horizon"] == h], key=lambda c: c["predicted"])
        fig.add_trace(go.Scatter(
            x=[c["predicted"] for c in sub], y=[c["observed"] for c in sub], mode="markers+lines", name=f"by {h} months",
            line=dict(color=color, width=2), marker=dict(size=11, color=color, line=dict(color=SURFACE, width=2)),
            text=[c["tertile"] for c in sub],
            hovertemplate="%{text} of graduates<br>predicted %{x:.0%} · observed %{y:.0%}<extra>" + f"by {h} months</extra>"))
    fig.update_xaxes(range=[0, 1], tickformat=".0%", gridcolor=GRID, zeroline=False,
                     title=_axis_title("Predicted share with an IT job"))
    fig.update_yaxes(range=[0, 1], tickformat=".0%", gridcolor=GRID, zeroline=False,
                     title=_axis_title("Observed share (Kaplan-Meier)"))
    return _layout(fig, 400, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))


def forest(hr: list[dict]) -> go.Figure:
    rows = sorted(hr, key=lambda r: r["hr"])
    fig = go.Figure()
    fig.add_vline(x=1, line=dict(color=MUTED, width=1, dash="dot"))
    for r in rows:
        sig = r["lo"] > 1 or r["hi"] < 1
        color = (BLUE if r["hr"] > 1 else ORANGE) if sig else "#b5b4af"
        fig.add_trace(go.Scatter(
            x=[r["hr"]], y=[r["label"][:1].upper() + r["label"][1:]], mode="markers",
            marker=dict(size=12, color=color, symbol="square", line=dict(color=SURFACE, width=2)),
            error_x=dict(type="data", symmetric=False, array=[r["hi"] - r["hr"]], arrayminus=[r["hr"] - r["lo"]],
                         color=MUTED, thickness=1.3, width=5),
            hovertemplate=f"<b>{r['label']}</b><br>hazard ratio {r['hr']:.2f} (95% CI {r['lo']:.2f}-{r['hi']:.2f})"
                          "<extra></extra>"))
    fig.update_xaxes(type="log", gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED),
                     title=_axis_title("Hazard ratio, log scale  ·  right of 1 = finds work sooner"))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK))
    return _layout(fig, 90 + 34 * len(rows))


def km_curve(km: list[dict], horizon: int) -> go.Figure:
    t = [p["t"] for p in km]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t + t[::-1], y=[p["hi"] for p in km] + [p["lo"] for p in km][::-1], fill="toself",
                             fillcolor="rgba(42,120,214,0.14)", line=dict(width=0), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=[p["employed"] for p in km], mode="lines", line=dict(color=BLUE, width=2.5, shape="hv"),
                             hovertemplate="by %{x:.1f} months: %{y:.0%} had an IT job<extra></extra>"))
    fig.update_xaxes(range=[0, horizon], dtick=3, gridcolor=GRID, zeroline=False, title=_axis_title("Months after graduating"))
    fig.update_yaxes(range=[0, 1], tickformat=".0%", gridcolor=GRID, zeroline=False, title=_axis_title("Share with an IT job"))
    return _layout(fig, 340)


def subgroup_c(rows: list[dict]) -> go.Figure:
    rows = [r for r in rows if not r["suppressed"]][::-1]
    fig = go.Figure()
    for r in rows:
        y = f"{r['level']}  (n={r['n']})"
        fig.add_trace(go.Scatter(x=[r["uno_c_lower"], r["uno_c_upper"]], y=[y, y], mode="lines",
                                 line=dict(color=GRID, width=6), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[r["uno_c"]], y=[y], mode="markers",
                                 marker=dict(size=12, color=BLUE if r["group"] == "all" else VIOLET,
                                             line=dict(color=SURFACE, width=2)),
                                 hovertemplate=f"{r['group']}: {r['level']}<br>Uno's C {r['uno_c']:.3f} "
                                               f"({r['uno_c_lower']:.2f}-{r['uno_c_upper']:.2f})<extra></extra>"))
    fig.update_xaxes(range=[0.4, 0.9], gridcolor=GRID, zeroline=False, title=_axis_title("Uno's C-index with 95% interval"))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK))
    return _layout(fig, 90 + 34 * len(rows))


def variant_c(rows: list[dict], label_key: str = "label", c_key: str = "uno_c") -> go.Figure:
    rows = sorted(rows, key=lambda r: r[c_key])
    fig = go.Figure(go.Scatter(
        x=[r[c_key] for r in rows], y=[r[label_key] for r in rows], mode="markers",
        marker=dict(size=12, color=BLUE, line=dict(color=SURFACE, width=2)),
        hovertemplate="<b>%{y}</b><br>Uno's C %{x:.3f}<extra></extra>"))
    fig.update_xaxes(range=[0.5, 0.8], gridcolor=GRID, zeroline=False, title=_axis_title("Uno's C-index"))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK))
    return _layout(fig, 90 + 34 * len(rows))


def power_curves(sim: dict) -> go.Figure:
    sizes = sim["sample_sizes"]
    first = sim["by_size"][str(sizes[0])]["effects"]
    keys = [k for k, e in first.items() if e["true"] != 0][:len(CATEGORICAL)]
    fig = go.Figure()
    fig.add_hline(y=80, line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_annotation(x=math.log10(sizes[0]), y=80, text="80% power", showarrow=False, yshift=9, xanchor="left",
                       font=dict(color=MUTED, size=11))
    for k, c in zip(keys, CATEGORICAL):
        ys = [100 * sim["by_size"][str(n)]["effects"][k]["power"] for n in sizes]
        fig.add_trace(go.Scatter(x=sizes, y=ys, mode="lines+markers", name=first[k]["label"], line=dict(color=c, width=2),
                                 marker=dict(size=8, color=c, line=dict(color=SURFACE, width=1.5)),
                                 hovertemplate=first[k]["label"] + "<br>n = %{x}: detected in %{y:.0f}% of surveys<extra></extra>"))
    fig.update_xaxes(type="log", tickvals=sizes, gridcolor=GRID, zeroline=False, title=_axis_title("Graduates surveyed (log scale)"))
    fig.update_yaxes(range=[0, 102], ticksuffix="%", gridcolor=GRID, zeroline=False, title=_axis_title("Surveys detecting the effect"))
    return _layout(fig, 430, showlegend=True, legend=dict(orientation="v", x=1.02, y=1, font=dict(color=INK, size=11)))


def accuracy_ladder(summary: list[dict], ceiling: dict | None) -> go.Figure:
    """Uno's C of each exploratory method against chance (0.5) and the theoretical ceiling."""
    rows = sorted(summary, key=lambda s: s["uno_c_mean"])
    fig = go.Figure()
    fig.add_vline(x=0.5, line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_annotation(x=0.5, y=1, yref="paper", text="chance", showarrow=False, yshift=10, font=dict(color=MUTED, size=11))
    if ceiling:
        fig.add_vrect(x0=ceiling["mean"] - ceiling["sd"], x1=ceiling["mean"] + ceiling["sd"], fillcolor="rgba(27,175,122,0.12)",
                      line_width=0)
        fig.add_vline(x=ceiling["mean"], line=dict(color=AQUA, width=2))
        fig.add_annotation(x=ceiling["mean"], y=1, yref="paper", text=f"ceiling {ceiling['mean']:.2f}", showarrow=False,
                           yshift=10, font=dict(color="#0b6b48", size=11))
    for s in rows:
        main = s["model"] == "cox_core"
        fig.add_trace(go.Scatter(
            x=[s["uno_c_mean"]], y=[s["label"]], mode="markers",
            marker=dict(size=13, color=BLUE if main else "#86b6ef", line=dict(color=SURFACE, width=2)),
            error_x=dict(type="data", array=[s["uno_c_sd"]], color=MUTED, thickness=1.2, width=5),
            hovertemplate=f"<b>{s['label']}</b><br>Uno's C {s['uno_c_mean']:.3f} ± {s['uno_c_sd']:.3f}"
                          + ("" if main else f"<br>vs Cox PH {s['delta_vs_cox']:+.3f}") + "<extra></extra>"))
    fig.update_xaxes(range=[0.45, 0.8], gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED),
                     title=_axis_title("Uno's C-index on the validation set (± bootstrap SD)"))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK))
    return _layout(fig, 100 + 40 * len(rows))


# ---------------------------------------------------------------------------
# Questionnaire study
# ---------------------------------------------------------------------------
def repeat_rho(rows: list[dict], strong: float = 0.5, moderate: float = 0.3) -> go.Figure:
    """Spearman correlation of each repeat item with the model question it repeats, with 95% CI."""
    rows = [r for r in rows if r["rho"] is not None][::-1]
    y = [f"{r['repeat']}  vs  {r['model_question']}" for r in rows]
    color = [BLUE if abs(r["rho"]) >= strong else ("#86b6ef" if abs(r["rho"]) >= moderate else MUTED) for r in rows]
    fig = go.Figure()
    for x, lab in ((moderate, "partly overlaps"), (strong, "repeats")):
        fig.add_vline(x=x, line=dict(color=MUTED, width=1, dash="dot"))
        fig.add_annotation(x=x, y=1, yref="paper", text=lab, showarrow=False, yshift=10, font=dict(color=MUTED, size=11))
    fig.add_trace(go.Scatter(
        x=[r["rho"] for r in rows], y=y, mode="markers", marker=dict(size=12, color=color, line=dict(color=SURFACE, width=2)),
        error_x=dict(type="data", symmetric=False, color=MUTED, thickness=1.2, width=4,
                     array=[r["ci"][1] - r["rho"] for r in rows], arrayminus=[r["rho"] - r["ci"][0] for r in rows]),
        customdata=[[r["n"], r["verdict"]] for r in rows],
        hovertemplate="<b>%{y}</b><br>rho = %{x:.2f} (n = %{customdata[0]})<br>%{customdata[1]}<extra></extra>"))
    fig.update_xaxes(range=[-0.1, 1], gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED),
                     title=_axis_title("Spearman correlation with the model question (95% CI)"))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK))
    return _layout(fig, 90 + 34 * len(rows))


def group_dumbbell(rows: list[dict]) -> go.Figure:
    """Graduates' vs employers' share for matched options; significant differences (BH q < 0.05) starred."""
    rows = rows[::-1]
    y = [r["label"] + (" ★" if r["significant"] else "") for r in rows]
    fig = go.Figure()
    for r, lab in zip(rows, y):
        fig.add_trace(go.Scatter(x=[r["graduates_pct"], r["employers_pct"]], y=[lab, lab], mode="lines",
                                 line=dict(color=GRID, width=4), hoverinfo="skip", showlegend=False))
    for key, name, color in (("graduates_pct", "Graduates", BLUE), ("employers_pct", "Employers", ORANGE)):
        fig.add_trace(go.Scatter(x=[r[key] for r in rows], y=y, mode="markers", name=name,
                                 marker=dict(size=12, color=color, line=dict(color=SURFACE, width=2)),
                                 customdata=[[r["q"]] for r in rows],
                                 hovertemplate=name + ": %{x:.0f}%<br>q = %{customdata[0]:.3f}<extra></extra>"))
    fig.update_xaxes(range=[0, 100], ticksuffix="%", gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK))
    return _layout(fig, 90 + 32 * len(rows), showlegend=True,
                   legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(color=INK)))


def pct_bars(items: list[tuple[str, float]], color: str = BLUE) -> go.Figure:
    """Horizontal bars of percentages, largest at the top."""
    rows = [(k, v) for k, v in items if v is not None][::-1]
    fig = go.Figure(go.Bar(x=[v for _, v in rows], y=[k for k, _ in rows], orientation="h",
                           marker=dict(color=color, cornerradius=4), text=[f"{v:.0f}%" for _, v in rows],
                           textposition="outside", textfont=dict(color=MUTED, size=11),
                           hovertemplate="<b>%{y}</b>: %{x:.0f}%<extra></extra>"))
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, range=[0, 115])
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK))
    return _layout(fig, 40 + 30 * len(rows), bargap=0.3)


def mean_bars(items: list[tuple[str, float, list]], color: str = AQUA) -> go.Figure:
    """Mean rating (1-5) per item with its 95% CI."""
    rows = items[::-1]
    fig = go.Figure(go.Bar(
        x=[m for _, m, _ in rows], y=[k for k, _, _ in rows], orientation="h", marker=dict(color=color, cornerradius=4),
        error_x=dict(type="data", symmetric=False, color=MUTED, thickness=1.2, width=4,
                     array=[ci[1] - m for _, m, ci in rows], arrayminus=[m - ci[0] for _, m, ci in rows]),
        hovertemplate="<b>%{y}</b>: mean %{x:.2f}<extra></extra>"))
    fig.update_xaxes(range=[1, 5], gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED),
                     title=_axis_title("Mean importance (1 = not important, 5 = extremely important)"))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK))
    return _layout(fig, 70 + 30 * len(rows), bargap=0.3)
