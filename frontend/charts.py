"""Plotly figures for the frontend. One colour per meaning across the whole app:
blue = you / your profile, orange = employers / the changed profile, aqua = job ads. Thin marks, quiet grid."""
from __future__ import annotations

import json

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from frontend import ui

BLUE, ORANGE, AQUA = "#3b66e0", "#e8590c", "#0f9d8a"
INK, MUTED, GRID, TRACK, SURFACE = "#15203a", "#5a6682", "#e3e8f1", "#eef1f7", "#ffffff"
FONT = dict(family="Inter, Segoe UI, sans-serif", size=13, color=INK)

# Light literal -> dark literal. Figures are built with the light palette, then recoloured in _layout() when the
# Streamlit theme is dark, so every chart (including colours written inline in charts_study.py) follows the theme.
DARK = {
    BLUE: "#7da2ff", ORANGE: "#ff922b", AQUA: "#2dd4bf", INK: "#e8ecf5", MUTED: "#9da9c2", GRID: "#26324d",
    TRACK: "#1e2c4a", SURFACE: "#172033",
    "#cfdcfb": "#2a4580", "#9db6f5": "#4a70c9", "#6c8eee": "#5f8ae0", "#22408f": "#c5d6ff",   # blue ramp, reversed in dark
    "rgba(59,102,224,0.14)": "rgba(125,162,255,0.2)", "#5b4bc4": "#b3a6ff", "#9aa4b8": "#6b7893",
    "rgba(59,102,224,0.28)": "rgba(125,162,255,0.32)", "rgba(232,89,12,0.20)": "rgba(255,146,43,0.24)",
}


def _recolor_for_dark(fig: go.Figure) -> None:
    """Swap the light palette for the dark one in place (callers keep using the same figure object)."""
    text = fig.to_json()
    for light, dark in DARK.items():
        text = text.replace(f'"{light}"', f'"{dark}"')
    new = go.Figure(json.loads(text))
    for old_trace, new_trace in zip(fig.data, new.data):
        old_trace.update(new_trace.to_plotly_json())
    fig.layout.update(new.layout.to_plotly_json())


def _layout(fig: go.Figure, height: int, **kw) -> go.Figure:
    layout = dict(height=height, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)",
                  plot_bgcolor="rgba(0,0,0,0)", font=FONT, showlegend=False, hoverlabel=dict(font=FONT))
    fig.update_layout(**{**layout, **kw})
    if ui.is_dark():
        _recolor_for_dark(fig)
    return fig


def _end(v: dict, horizon: int) -> float:
    return horizon if v["more_than"] else v["months"]


def range_bar(rows: list[dict], horizon: int, now: int | None = None) -> go.Figure:
    """Horizontal 25th-75th percentile bar with the median marked.

    rows: [{"label", "pred" (API prediction), "color"}]; now: months since graduating (graduates still looking).
    """
    fig = go.Figure()
    if now:
        fig.add_vline(x=now, line=dict(color=MUTED, width=1.5, dash="dot"))
        fig.add_annotation(x=now, y=1, yref="paper", text="now", showarrow=False, yshift=10,
                           font=dict(color=MUTED, size=11))
    labels = [r["label"] for r in rows][::-1]
    for r in rows:
        p, y = r["pred"], r["label"]
        lo, hi, med = _end(p["low"], horizon), _end(p["high"], horizon), _end(p["median"], horizon)
        fig.add_trace(go.Scatter(x=[0, horizon], y=[y, y], mode="lines", line=dict(color=TRACK, width=14),
                                 hoverinfo="skip"))
        hi_text = f"{p['high']['display']}" if not p["high"]["more_than"] else f"{horizon}+"
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[y, y], mode="lines", line=dict(color=r["color"], width=14), opacity=0.35,
            hovertemplate=f"Typical range: {p['low']['display']}-{hi_text} months<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[med], y=[y], mode="markers+text", marker=dict(color=r["color"], size=16, line=dict(color=SURFACE, width=2)),
            text=[f"<b>{p['median']['display']}</b>"], textposition="top center", textfont=dict(color=INK, size=13),
            hovertemplate=f"Most likely around {p['median']['display']} months<extra></extra>"))
        fig.add_annotation(x=lo, y=y, text=p["low"]["display"], showarrow=False, yshift=-18,
                           font=dict(color=MUTED, size=11))
        fig.add_annotation(x=hi, y=y, text=hi_text, showarrow=False, yshift=-18, font=dict(color=MUTED, size=11))
    fig.update_xaxes(range=[-0.5, horizon + 0.5], dtick=3, gridcolor=GRID, zeroline=False,
                     title=dict(text="Months after graduating", font=dict(color=MUTED, size=12)),
                     tickfont=dict(color=MUTED))
    fig.update_yaxes(categoryorder="array", categoryarray=labels, tickfont=dict(color=INK), showgrid=False)
    return _layout(fig, 90 + 60 * len(rows))


def milestones(items: list[dict]) -> go.Figure:
    """Chance of having found an IT job by each milestone month, with the 80% bootstrap range."""
    x = [f"by {m['months']} months" for m in items]
    fig = go.Figure(go.Bar(
        x=x, y=[m["employed_pct"] for m in items], marker=dict(color=BLUE), opacity=0.85,
        text=[f"<b>{m['employed_pct']}%</b>" for m in items], textposition="outside", textfont=dict(color=INK),
        error_y=dict(type="data", symmetric=False, color=MUTED, thickness=1.2, width=6,
                     array=[m["high_pct"] - m["employed_pct"] for m in items],
                     arrayminus=[m["employed_pct"] - m["low_pct"] for m in items]),
        customdata=[[m["low_pct"], m["high_pct"]] for m in items],
        hovertemplate="%{y}% (range %{customdata[0]}-%{customdata[1]}%)<extra></extra>"))
    fig.update_yaxes(range=[0, 112], dtick=25, ticksuffix="%", gridcolor=GRID, zeroline=False,
                     tickfont=dict(color=MUTED))
    fig.update_xaxes(tickfont=dict(color=INK))
    return _layout(fig, 220)


def skill_gap(skills: list[dict], top_n: int = 10) -> go.Figure:
    """Two panels sharing the skill axis (no dual axis): ratings out of 5 | share of job ads."""
    rows = sorted([s for s in skills if s["demand"] is not None], key=lambda s: -s["demand"])[:top_n][::-1]
    names = [s["label"] for s in rows]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.62, 0.38], horizontal_spacing=0.04,
                        subplot_titles=("Rating (1-5)", "% of job ads"))
    for s in rows:
        vals = [v for v in (s["your_rating"], s["employer_importance"]) if v is not None]
        if len(vals) == 2:
            fig.add_trace(go.Scatter(x=vals, y=[s["label"]] * 2, mode="lines", line=dict(color=GRID, width=3),
                                     hoverinfo="skip", showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=[s["employer_importance"] for s in rows], y=names, mode="markers", name="Employers",
                             marker=dict(color=ORANGE, size=11, line=dict(color=SURFACE, width=2)),
                             hovertemplate="Employers: %{x:.1f} / 5<extra></extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=[s["your_rating"] for s in rows], y=names, mode="markers", name="You",
                             marker=dict(color=BLUE, size=11, line=dict(color=SURFACE, width=2)),
                             hovertemplate="You: %{x:.1f} / 5<extra></extra>"), 1, 1)
    fig.add_trace(go.Bar(x=[s["ads_pct"] for s in rows], y=names, orientation="h", marker=dict(color=AQUA),
                         name="Job ads", showlegend=False, text=[f"{s['ads_pct']:.0f}%" for s in rows], textposition="outside",
                         textfont=dict(color=MUTED, size=11), hovertemplate="%{x:.0f}% of job ads<extra></extra>"),
                  1, 2)
    fig.update_xaxes(range=[0.8, 5.2], dtick=1, gridcolor=GRID, zeroline=False, row=1, col=1)
    fig.update_xaxes(range=[0, 115], showticklabels=False, showgrid=False, zeroline=False, row=1, col=2)
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK))
    fig.update_annotations(font=dict(color=MUTED, size=12))
    _layout(fig, 100 + 34 * len(rows), showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.08, x=0, font=dict(color=INK), traceorder="reversed"))
    fig.update_layout(margin=dict(l=8, r=8, t=30, b=40))
    return fig


def peer_medians(groups: list[dict], horizon: int) -> go.Figure:
    """Median time per peer group with its plausible range (suppressed groups omitted)."""
    shown = [g for g in groups if not g["suppressed"]][::-1]
    fig = go.Figure()
    for g in shown:
        label = f"{g['level_label']}  (n={g['n']})"
        lo, hi = g["median_interval"]
        to_num = lambda s: horizon if s.startswith("more") else (None if s == "-" else float(s))  # noqa: E731
        lo_n, hi_n = to_num(lo), to_num(hi)
        if lo_n is not None and hi_n is not None:
            fig.add_trace(go.Scatter(x=[lo_n, hi_n], y=[label, label], mode="lines", line=dict(color=GRID, width=6),
                                     hovertemplate=f"Plausible range {lo}-{hi} months<extra></extra>"))
        med = _end(g["median"], horizon)
        fig.add_trace(go.Scatter(x=[med], y=[label], mode="markers+text", marker=dict(color=BLUE, size=12,
                                 line=dict(color=SURFACE, width=2)), text=[g["median"]["display"]],
                                 textposition="middle right", textfont=dict(color=MUTED, size=11),
                                 hovertemplate=f"Half found work by {g['median']['display']} months<extra></extra>"))
    fig.update_xaxes(range=[0, horizon + 1], dtick=3, gridcolor=GRID, zeroline=False,
                     title=dict(text="Months until half had found work", font=dict(color=MUTED, size=12)))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK))
    return _layout(fig, 80 + 42 * len(shown))


def job_donut(milestone: dict) -> go.Figure:
    """Donut gauge: share of graduates with a profile like yours who had found an IT job by one milestone month."""
    pct = float(milestone["employed_pct"])
    fig = go.Figure(go.Pie(
        labels=["Found an IT job", "Still looking"], values=[pct, max(0.0, 100 - pct)], hole=0.72, sort=False,
        direction="clockwise", pull=[0.03, 0], textinfo="none", showlegend=False,
        marker=dict(colors=[AQUA, TRACK], line=dict(color=SURFACE, width=3)),
        hovertemplate="%{label}: %{value:.0f}%<extra></extra>"))
    fig.add_annotation(text=f"<b>{pct:.0f}%</b>", x=0.5, y=0.56, showarrow=False, font=dict(size=38, color=INK))
    fig.add_annotation(text=f"by {milestone['months']} months", x=0.5, y=0.36, showarrow=False,
                       font=dict(size=13, color=MUTED))
    return _layout(fig, 280, margin=dict(l=8, r=8, t=8, b=8))


def _wrap(label: str, width: int = 13) -> str:
    words, lines, line = label.replace(" / ", "/ ").split(), [], ""
    for w in words:
        if line and len(line) + 1 + len(w) > width:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    return "<br>".join(lines + [line])


def skill_radar(skills: list[dict], top_n: int = 8) -> go.Figure:
    """Radar of your 1-5 rating against the importance employers give, for the most requested skills."""
    rows = [s for s in sorted((s for s in skills if s["demand"] is not None), key=lambda s: -s["demand"])
            if s["your_rating"] is not None and s["employer_importance"] is not None][:top_n]
    names = [_wrap(s["label"]) for s in rows]
    fig = go.Figure()
    for key, label, color, fill in (("employer_importance", "Employers", ORANGE, "rgba(232,89,12,0.20)"),
                                    ("your_rating", "You", BLUE, "rgba(59,102,224,0.28)")):
        vals = [s[key] for s in rows]
        fig.add_trace(go.Scatterpolar(
            r=vals + vals[:1], theta=names + names[:1], name=label, fill="toself", fillcolor=fill,
            line=dict(color=color, width=2.5), marker=dict(size=7, color=color),
            hovertemplate="%{theta}: %{r:.1f} / 5<extra>" + label + "</extra>"))
    fig.update_layout(polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(range=[0, 5], tickvals=[1, 2, 3, 4, 5], gridcolor=GRID, linecolor=GRID,
                        tickfont=dict(color=MUTED, size=10)),
        angularaxis=dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=INK, size=11))))
    return _layout(fig, 380, showlegend=True, margin=dict(l=84, r=84, t=30, b=34),
                   legend=dict(orientation="h", yanchor="top", y=-0.02, x=0.5, xanchor="center", font=dict(color=INK)))


def gap_bars(skills: list[dict], top_n: int = 5) -> go.Figure | None:
    """Horizontal bars for the skills where employers rate importance above your own rating (None if there are none)."""
    rows = [dict(s, gap=s["employer_importance"] - s["your_rating"]) for s in skills
            if s["employer_importance"] is not None and s["your_rating"] is not None
            and s["employer_importance"] - s["your_rating"] > 0.05]
    rows = sorted(rows, key=lambda r: -r["gap"])[:top_n][::-1]
    if not rows:
        return None
    fig = go.Figure(go.Bar(
        x=[r["gap"] for r in rows], y=[r["label"] for r in rows], orientation="h",
        marker=dict(color=ORANGE, cornerradius=6), text=[f"+{r['gap']:.1f}" for r in rows], textposition="outside",
        textfont=dict(color=INK, size=12), customdata=[[r["your_rating"], r["employer_importance"]] for r in rows],
        hovertemplate="<b>%{y}</b><br>You: %{customdata[0]:.1f} · Employers: %{customdata[1]:.1f}<extra></extra>"))
    fig.update_xaxes(range=[0, max(r["gap"] for r in rows) * 1.3], showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK))
    return _layout(fig, 70 + 44 * len(rows), bargap=0.35)
