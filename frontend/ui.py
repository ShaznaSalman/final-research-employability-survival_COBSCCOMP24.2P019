"""Shared look and small HTML components for every page (one visual system across the app)."""
from __future__ import annotations

import html

import streamlit as st

from frontend.api_client import APIError, get

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {
  --ink:#0b0b0b; --muted:#52514e; --line:#e4e3df; --panel:#ffffff; --wash:#f6f5f2;
  --blue:#2a78d6; --navy:#0d366b; --aqua:#1baf7a; --orange:#eb6834; --amber:#b97a00;
}
html, body, [class*="css"], .stMarkdown, .stButton button, .stTabs button { font-family: 'Inter', 'Segoe UI', sans-serif; }
.block-container { padding-top: 4.2rem; padding-bottom: 3rem; max-width: 1320px; }
h1, h2, h3 { letter-spacing: -0.015em; }

/* hero */
.hero { position:relative; overflow:hidden; border-radius:22px; padding:2.6rem 2.6rem 2.3rem;
  background: radial-gradient(1200px 400px at 85% -20%, rgba(134,182,239,.55), transparent 60%),
              linear-gradient(120deg, #0d366b 0%, #1c5cab 55%, #2a78d6 100%); color:#fff; margin-bottom:1.4rem; }
.hero::after { content:""; position:absolute; right:-80px; bottom:-120px; width:420px; height:420px; border-radius:50%;
  background: radial-gradient(circle, rgba(27,175,122,.35), transparent 65%); }
.hero .kicker { text-transform:uppercase; letter-spacing:.14em; font-size:.74rem; font-weight:700; opacity:.85; }
.hero h1 { color:#fff; font-size:2.6rem; line-height:1.08; font-weight:800; margin:.45rem 0 .7rem; max-width:820px; }
.hero p { font-size:1.06rem; line-height:1.55; opacity:.92; max-width:760px; margin:0; }
.hero.compact { padding:1.7rem 2rem 1.5rem; }
.hero.compact h1 { font-size:1.9rem; }

/* stat tiles */
.kpis { display:grid; grid-template-columns:repeat(auto-fit, minmax(190px, 1fr)); gap:14px; margin:.2rem 0 1.4rem; }
.kpi { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:1.05rem 1.2rem 1rem;
  box-shadow:0 1px 2px rgba(11,11,11,.04); }
.kpi .v { font-size:2rem; font-weight:800; letter-spacing:-0.02em; color:var(--ink); line-height:1.1; }
.kpi .l { font-size:.86rem; color:var(--ink); font-weight:600; margin-top:.25rem; }
.kpi .s { font-size:.78rem; color:var(--muted); margin-top:.15rem; }

/* provenance tags */
.tag { display:inline-flex; align-items:center; gap:.35rem; font-size:.72rem; font-weight:700; letter-spacing:.04em;
  text-transform:uppercase; padding:.18rem .55rem; border-radius:999px; vertical-align:middle; }
.tag.real { background:#e3f6ee; color:#0b6b48; }
.tag.synthetic { background:#fdf0d5; color:#7a5200; }
.tag.dot::before { content:""; width:6px; height:6px; border-radius:50%; background:currentColor; }

/* section headings and cards */
.section { margin:1.6rem 0 .6rem; }
.section .t { font-size:1.28rem; font-weight:750; color:var(--ink); }
.section .d { font-size:.92rem; color:var(--muted); margin-top:.15rem; max-width:900px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:1.1rem 1.25rem; height:100%; }
.card h4 { margin:0 0 .35rem; font-size:1rem; }
.card p { margin:0; color:var(--muted); font-size:.92rem; line-height:1.5; }
.finding { border-left:4px solid var(--blue); background:var(--panel); border-radius:12px; padding:.85rem 1rem;
  margin-bottom:.6rem; border-top:1px solid var(--line); border-right:1px solid var(--line); border-bottom:1px solid var(--line); }
.finding b { color:var(--ink); }
.finding span { color:var(--muted); font-size:.88rem; }

/* prediction result */
.result { border-radius:20px; padding:1.5rem 1.7rem; color:#fff;
  background: linear-gradient(125deg, #0d366b 0%, #1c5cab 60%, #2a78d6 100%); margin-bottom:.8rem; }
.result .lbl { text-transform:uppercase; font-size:.72rem; letter-spacing:.14em; opacity:.85; font-weight:700; }
.result .big { font-size:3.1rem; font-weight:800; letter-spacing:-0.03em; line-height:1; margin:.35rem 0 .2rem; }
.result .big small { font-size:1.1rem; font-weight:600; opacity:.85; letter-spacing:0; }
.result .rng { font-size:.98rem; opacity:.92; }
.result .hl { font-size:1.02rem; line-height:1.5; margin-top:.8rem; opacity:.95; }
.muted { color:var(--muted); font-size:.88rem; }
.item { margin:.25rem 0 .55rem 0; line-height:1.45; }
.notice { display:flex; gap:.6rem; align-items:flex-start; background:#fffaf0; border:1px solid #f3dfb2; color:#5c4300;
  border-radius:12px; padding:.6rem .9rem; font-size:.86rem; margin-bottom:1rem; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius:16px; }
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner=False)
def model_info() -> dict:
    return get("/model-info")


@st.cache_data(ttl=300, show_spinner=False)
def study(path: str) -> dict:
    return get(path)


def fetch(path: str) -> dict | None:
    try:
        return study(path)
    except APIError as e:
        st.error(str(e))
        return None


def esc(x) -> str:
    return html.escape(str(x))


def hero(kicker: str, title: str, text: str, compact: bool = False) -> None:
    st.markdown(f"<div class='hero{' compact' if compact else ''}'><div class='kicker'>{esc(kicker)}</div>"
                f"<h1>{esc(title)}</h1><p>{text}</p></div>", unsafe_allow_html=True)


def kpis(items: list[tuple]) -> None:
    """items: (value, label, sub[, tag]) - tag is 'real' or 'synthetic'."""
    cells = []
    for it in items:
        v, l, s = it[:3]
        tag = tag_html(it[3]) if len(it) > 3 and it[3] else ""
        cells.append(f"<div class='kpi'><div class='v'>{esc(v)}</div><div class='l'>{esc(l)} {tag}</div>"
                     f"<div class='s'>{esc(s)}</div></div>")
    st.markdown(f"<div class='kpis'>{''.join(cells)}</div>", unsafe_allow_html=True)


def tag_html(kind: str) -> str:
    return {"real": "<span class='tag real dot'>Real data</span>",
            "synthetic": "<span class='tag synthetic dot'>Synthetic data</span>"}[kind]


def section(title: str, desc: str = "", tag: str | None = None) -> None:
    st.markdown(f"<div class='section'><div class='t'>{esc(title)} {tag_html(tag) if tag else ''}</div>"
                + (f"<div class='d'>{desc}</div>" if desc else "") + "</div>", unsafe_allow_html=True)


def finding(title: str, detail: str) -> None:
    st.markdown(f"<div class='finding'><b>{title}</b><br><span>{detail}</span></div>", unsafe_allow_html=True)


def synthetic_notice(info: dict) -> None:
    if info.get("is_synthetic"):
        st.markdown("<div class='notice'>⚠️ <div><b>Demo model.</b> Predictions on this page are trained on "
                    "<b>synthetic (computer-generated) graduate survey data</b>, not real graduates. "
                    "The job-market figures come from real job ads.</div></div>", unsafe_allow_html=True)


def q_text(q: float) -> str:
    return "q < 0.001" if q < 0.001 else f"q = {q:.3f}"


PLOT = {"displayModeBar": False}
