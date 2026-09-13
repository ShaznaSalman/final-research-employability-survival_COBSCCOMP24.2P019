"""Plotly figures for the frontend. One colour per meaning across the whole app:
blue = you / your profile, orange = employers / the changed profile, aqua = job ads. Thin marks, quiet grid."""
from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, TRACK, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#f0efec", "#fcfcfb"
FONT = dict(family="Inter, Segoe UI, sans-serif", size=13, color=INK)


def _layout(fig: go.Figure, height: int, **kw) -> go.Figure:
    layout = dict(height=height, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)",
                  plot_bgcolor="rgba(0,0,0,0)", font=FONT, showlegend=False, hoverlabel=dict(font=FONT))
    fig.update_layout(**{**layout, **kw})
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
                        subplot_titles=("Rating out of 5", "Share of job ads"))
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
