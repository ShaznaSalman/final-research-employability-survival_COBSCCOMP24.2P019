"""Shared look and small HTML components for every page (one visual system across the app)."""
from __future__ import annotations

import html
import base64

import streamlit as st

from frontend.api_client import APIError, get

# Colour tokens. Light is the default; the dark set is added when Streamlit's active theme is dark (see inject()).
# Keep these in step with .streamlit/config.toml and the chart palette in frontend/charts.py.
TOKENS_LIGHT = """
:root {
  --ink:#15203a; --muted:#5a6682; --line:#e3e8f1; --panel:#ffffff; --wash:#f5f7fb;
  --brand:#3b66e0; --on-brand:#ffffff; --brand-ink:#2547b0;
  --blue-soft:#e8effe; --mint-soft:#e3f6f1; --peach-soft:#fdede3; --lav-soft:#eeeafb; --amber-soft:#fdf3dc;
  --mint-ink:#0b6b5c; --peach-ink:#a2400d; --lav-ink:#4a3aa8; --amber-ink:#7a5200;
  --shadow:0 1px 2px rgba(21,32,58,.05), 0 6px 18px rgba(21,32,58,.05);
  --radius:16px;
  --hero-a:#1d52cf; --hero-b:#3069e3; --hero-ink:#ffffff; --hero-muted:#f4f7ff; --hero-kicker:#fff6e0;
  --accent:#ffa62b; --on-accent:#2b1800;
  --b-blue:#3b66e0; --b-mint:#0b7d6e; --b-peach:#c2410c; --b-lav:#5b3fd0;
}
"""
TOKENS_DARK = """
:root {
  --ink:#e8ecf5; --muted:#9da9c2; --line:#26324d; --panel:#172033; --wash:#0f1624;
  --brand:#7da2ff; --on-brand:#0f1624; --brand-ink:#a9c1ff;
  --blue-soft:#1e2c4a; --mint-soft:#163a35; --peach-soft:#3f2a1f; --lav-soft:#2a2545; --amber-soft:#3a2f14;
  --mint-ink:#6ee0c8; --peach-ink:#ffb98a; --lav-ink:#c2b6ff; --amber-ink:#f6c453;
  --shadow:0 1px 2px rgba(0,0,0,.3);
  --hero-a:#0b1d4a; --hero-b:#2548a8; --hero-muted:#dfe7ff;
}
"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
/*TOKENS*/
html, body, [class*="css"], .stMarkdown, .stButton button, .stTabs button { font-family: 'Inter', 'Segoe UI', sans-serif; }
h1, h2, h3, .hero h1, .kpi .v, .result .big, .section .t { font-family: 'Plus Jakarta Sans', 'Inter', 'Segoe UI', sans-serif; }
.block-container, [data-testid="stMainBlockContainer"] { padding-top:2.4rem; padding-bottom:3rem; max-width:1560px !important;
  padding-left:2.4rem !important; padding-right:2.4rem !important; }
@media (max-width: 640px) { .block-container, [data-testid="stMainBlockContainer"] { padding-left:1rem !important; padding-right:1rem !important; } }
h1, h2, h3 { letter-spacing: -0.015em; }
:focus-visible { outline: 3px solid var(--brand); outline-offset: 2px; }

/* banners: deep blue gradient with a warm glow, white text (the Home hero is a container so its buttons sit inside) */
.hero, .st-key-homehero, .result {
  background: radial-gradient(640px 300px at 100% 100%, rgba(255,184,77,.55), transparent 62%),
              linear-gradient(135deg, var(--hero-a) 0%, var(--hero-b) 100%);
  color:var(--hero-ink); }
.hero { border-radius:22px; padding:2.2rem 2.4rem 2rem; margin-bottom:1.4rem; box-shadow:0 12px 30px rgba(29,82,207,.22); }
.hero .kicker, .st-key-homehero .kicker { text-transform:uppercase; letter-spacing:.1em; font-size:.74rem; font-weight:700; color:var(--hero-kicker); }
.hero h1, .st-key-homehero h1 { color:var(--hero-ink); font-size:2.5rem; line-height:1.12; font-weight:800; margin:.45rem 0 .7rem; max-width:820px; padding:0; }
.hero p, .st-key-homehero p { font-size:1.06rem; line-height:1.55; color:var(--hero-muted); max-width:760px; margin:0 0 .4rem; }
.hero.compact { padding:1.5rem 1.8rem 1.3rem; }
.hero.compact h1 { font-size:1.8rem; }
.st-key-homehero { border-radius:24px; padding:2.2rem 2.4rem 2rem; margin-bottom:1.4rem; box-shadow:0 14px 34px rgba(11,58,140,.26); }
.st-key-homehero h1 + p { margin-top:.2rem; margin-bottom:1rem; }
.st-key-homehero button[data-testid="stBaseButton-primary"] { min-height:48px; background:var(--accent); border:0; box-shadow:0 8px 20px rgba(255,166,43,.38); }
.st-key-homehero button[data-testid="stBaseButton-primary"], .st-key-homehero button[data-testid="stBaseButton-primary"] * { color:var(--on-accent) !important; }
.st-key-homehero button[data-testid="stBaseButton-primary"]:hover { filter:brightness(1.07); transform:translateY(-2px); }
.st-key-homehero a[data-testid="stPageLink-NavLink"] { border:1.5px solid rgba(255,255,255,.75); border-radius:12px; min-height:48px; width:100%; justify-content:center; }
.st-key-homehero a[data-testid="stPageLink-NavLink"], .st-key-homehero a[data-testid="stPageLink-NavLink"] * { color:#ffffff !important; }
.st-key-homehero a[data-testid="stPageLink-NavLink"]:hover { background:rgba(255,255,255,.16); }
.st-key-homehero [data-testid="stHeaderActionElements"], .st-key-homehero h1 a { display:none !important; }
.st-key-homehero [data-testid="stPageLink"], .st-key-homehero [data-testid="stPageLink"] a { width:100% !important; }
.hero.has-art { display:grid; grid-template-columns:minmax(0,1fr) minmax(150px,240px); align-items:center; gap:1rem; }
.hero-art { width:100%; max-width:340px; display:block; margin:0 auto; }
.hero.has-art .hero-art { max-width:220px; }
.hero-art .float { animation:jsfloat 6s ease-in-out infinite; transform-origin:center; }
@keyframes jsfloat { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-8px); } }
@media (max-width: 640px) {
  .hero, .st-key-homehero { padding:1.4rem 1.3rem; border-radius:18px; }
  .hero h1, .st-key-homehero h1 { font-size:1.7rem; }
  .hero.compact h1 { font-size:1.5rem; }
  .hero-art { display:none; }
  .hero.has-art { grid-template-columns:1fr; }
  .hero.has-art .hero-art { display:block; grid-row:1; width:180px; max-width:100%; margin:0 auto .5rem; }
  .hero.has-art > div:first-child { grid-row:2; }
}

/* stat tiles */
.kpis { display:grid; grid-template-columns:repeat(auto-fit, minmax(190px, 1fr)); gap:14px; margin:.2rem 0 1.4rem; }
.kpi { background:var(--panel); border:1px solid var(--line); border-radius:var(--radius); padding:1.05rem 1.2rem 1rem;
  box-shadow:var(--shadow); }
.kpi .v { font-size:2rem; font-weight:800; letter-spacing:-0.02em; color:var(--ink); line-height:1.1; font-variant-numeric:tabular-nums; }
.kpi .l { font-size:.86rem; color:var(--ink); font-weight:600; margin-top:.25rem; }
.kpi .s { font-size:.78rem; color:var(--muted); margin-top:.15rem; }

/* provenance tags */
.tag { display:inline-flex; align-items:center; gap:.35rem; font-size:.72rem; font-weight:600; letter-spacing:.04em;
  text-transform:uppercase; padding:.18rem .55rem; border-radius:999px; vertical-align:middle; }
.tag.real { background:var(--mint-soft); color:var(--mint-ink); }
.tag.synthetic { background:var(--amber-soft); color:var(--amber-ink); }
.tag.dot::before { content:""; width:6px; height:6px; border-radius:50%; background:currentColor; }

/* section headings and cards */
.section { margin:1.6rem 0 .6rem; }
.section .t { font-size:1.28rem; font-weight:700; color:var(--ink); }
.section .d { font-size:.92rem; color:var(--muted); margin-top:.15rem; max-width:900px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:var(--radius); padding:1.1rem 1.25rem; height:100%; box-shadow:var(--shadow); }
.card h4 { margin:0 0 .35rem; font-size:1rem; }
.card p { margin:0; color:var(--muted); font-size:.92rem; line-height:1.5; }
.best-match-badge { display:inline-flex; align-items:center; gap:.3rem; margin:0 0 .55rem;
  padding:.28rem .62rem; border:1.5px solid #f5bd16; border-radius:999px;
  background:linear-gradient(135deg,#fff5c2,#ffe36b); color:#553b00;
  font-size:.76rem; font-weight:700; letter-spacing:.02em; box-shadow:0 2px 8px rgba(245,189,22,.22); }
.best-match-star { color:#e0a800; font-size:1.02rem; line-height:1; text-shadow:0 1px 0 #fff4bf; }
.finding { border-left:4px solid var(--brand); background:var(--panel); border-radius:12px; padding:.85rem 1rem;
  margin-bottom:.6rem; border-top:1px solid var(--line); border-right:1px solid var(--line); border-bottom:1px solid var(--line); }
.finding b { color:var(--ink); }
.finding span { color:var(--muted); font-size:.88rem; }

/* prediction result */
.result-grid { display:grid; grid-template-columns:minmax(150px,.78fr) minmax(0,1.22fr); gap:.8rem; margin-bottom:.8rem; }
.result-months, .result-details { border-radius:20px; padding:1.35rem 1.5rem; box-shadow:var(--shadow); }
.result-months { background:radial-gradient(420px 220px at 100% 100%, rgba(255,184,77,.5), transparent 62%),
                 linear-gradient(135deg, var(--hero-a), var(--hero-b)); color:var(--hero-ink); }
.result-details { background:var(--panel); border:1px solid var(--line); color:var(--ink); }
.result-grid .lbl { text-transform:uppercase; font-size:.72rem; letter-spacing:.1em; color:var(--hero-kicker); font-weight:700; }
.result-grid .big { font-size:3.1rem; font-weight:800; letter-spacing:-0.03em; line-height:1; margin:.5rem 0 .15rem;
                    font-variant-numeric:tabular-nums; font-family:'Plus Jakarta Sans','Inter','Segoe UI',sans-serif; }
.result-grid .big small { font-size:1.1rem; font-weight:600; color:var(--hero-muted); letter-spacing:0; }
.result-grid .range-label { font-size:.72rem; text-transform:uppercase; letter-spacing:.1em; font-weight:700; color:var(--muted); }
.result-grid .rng { font-size:1.15rem; font-weight:700; color:var(--ink); margin-top:.3rem; }
.result-grid .confidence { display:inline-flex; align-items:center; border-radius:999px; padding:.25rem .65rem;
                           margin-top:.65rem; background:var(--blue-soft); color:var(--brand-ink); font-size:.82rem; font-weight:600; }
.result-grid .hl { font-size:.96rem; line-height:1.5; color:var(--muted); margin-top:.75rem; }
@media (max-width: 640px) { .result-grid { grid-template-columns:1fr; } .result-months, .result-details { padding:1.15rem 1.25rem; } }
.muted { color:var(--muted); font-size:.88rem; }
.item { margin:.25rem 0 .55rem 0; line-height:1.45; }
.notice { display:flex; gap:.6rem; align-items:flex-start; background:var(--amber-soft); border:1px solid var(--line); color:var(--amber-ink);
  border-radius:12px; padding:.6rem .9rem; font-size:.86rem; margin-bottom:1rem; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius:var(--radius); }
/* sidebar navigation */
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
  background-image:radial-gradient(380px 260px at 0 0, var(--blue-soft), transparent 72%) !important;
  background-color:var(--wash) !important; border-right:1px solid var(--line);
}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] { padding:0 .8rem 1rem !important; }
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] {
  min-height:46px !important; margin:.2rem 0 !important; padding:.55rem .8rem !important;
  border:1px solid transparent !important; border-radius:14px !important;
  font-weight:500 !important; transition:background .18s ease, border-color .18s ease, transform .18s ease, box-shadow .18s ease;
}
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] span[data-testid="stIconMaterial"] {
  color:var(--brand) !important; transition:transform .18s ease;
}
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover {
  background:var(--panel) !important; border-color:var(--line) !important;
  box-shadow:var(--shadow) !important; transform:translateX(2px);
}
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover span[data-testid="stIconMaterial"] { transform:scale(1.08); }
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] {
  background-image:linear-gradient(110deg, var(--blue-soft), var(--mint-soft)) !important;
  color:var(--brand-ink) !important; font-weight:700 !important;
  border-color:var(--line) !important; box-shadow:inset 3px 0 0 var(--brand), 0 5px 14px rgba(21,32,58,.08) !important;
}
[data-testid="stSidebar"] [data-testid="stNavSectionHeader"] {
  display:flex; align-items:center; gap:.55rem; margin-top:1rem !important; padding:.5rem .1rem .35rem !important;
  text-transform:uppercase; letter-spacing:.12em; font-size:.66rem !important; font-weight:800 !important;
  color:var(--brand-ink) !important;
}
[data-testid="stSidebar"] [data-testid="stNavSectionHeader"]::after {
  content:""; height:1px; flex:1; background:linear-gradient(90deg, var(--brand), transparent); opacity:.45;
}
[data-testid="stSidebarNavSeparator"], [data-testid="stSidebarNav"] span[data-testid="stHeadingWithActionElements"] { color:var(--muted); }

/* fact tile: one number and what it means */
.fact .v { font-family:'Plus Jakarta Sans','Inter',sans-serif; font-weight:800; font-size:1.9rem; letter-spacing:-.02em; line-height:1.15; font-variant-numeric:tabular-nums; }
.fact .l { font-size:.92rem; margin-top:.25rem; line-height:1.45; }
.fact .s { font-size:.8rem; color:var(--muted); margin-top:.3rem; }
.step { min-height:100%; }
[data-testid="stExpander"] { margin-top:1.6rem; }
@media (min-width: 900px) { .fact { min-height:10.5rem; } .step { min-height:11rem; } }

.card, .tile, .kpi { transition:transform .18s ease, box-shadow .18s ease; }
.card:hover, .tile:hover, .kpi:hover { transform:translateY(-3px); box-shadow:0 12px 28px rgba(21,32,58,.16); }
.badge { width:32px; height:32px; border-radius:50%; display:grid; place-items:center; font-weight:800; color:#fff; margin-bottom:.6rem; }
.b-blue { background:var(--b-blue); } .b-mint { background:var(--b-mint); } .b-peach { background:var(--b-peach); } .b-lav { background:var(--b-lav); }

/* profile steps: live summary and tip beside the form */
.sidepanel { border-radius:18px; padding:1.1rem 1.25rem; background:linear-gradient(160deg, var(--lav-soft), var(--blue-soft)); margin-bottom:.9rem; }
.sidepanel h4 { margin:0 0 .65rem; font-size:1rem; }
.chiprow { display:flex; flex-wrap:wrap; gap:.45rem; }
.chip { background:var(--panel); border:1px solid var(--line); border-radius:999px; padding:.28rem .75rem; font-size:.85rem; line-height:1.3; }
.chip b { color:var(--brand-ink); font-weight:600; }
.tip { border-radius:18px; padding:1rem 1.2rem; background:var(--mint-soft); color:var(--ink); font-size:.92rem; line-height:1.5; }
.tip b { color:var(--mint-ink); }
.st-key-sidepanel { position:sticky; top:4.5rem; }
@media (max-width: 900px) { .st-key-sidepanel { position:static; } }

hr { margin:.6rem 0 !important; }
[data-testid="stHeading"] h2 { padding-top:.25rem; }

/* buttons and controls: 44px tap targets */
button[data-testid^="stBaseButton"] { min-height:44px; border-radius:12px; font-weight:600; }
button[data-variant="segmented_control"], button[data-variant="pills"] { min-height:44px; min-width:44px; }
button[data-variant="segmented_control"][aria-checked="true"], button[data-variant="pills"][aria-checked="true"] {
  background:var(--brand) !important; color:var(--on-brand) !important; border-color:var(--brand) !important; font-weight:600; }

/* soft tiles (one tone per meaning: blue = you, mint = helping, peach = holding back, lavender = navigate) */
.tile { border-radius:var(--radius); padding:1.1rem 1.25rem; color:var(--ink); }
.tile.t-blue { background:var(--blue-soft); } .tile.t-mint { background:var(--mint-soft); }
.tile.t-peach { background:var(--peach-soft); } .tile.t-lav { background:var(--lav-soft); }
.tile .head { display:flex; align-items:center; gap:.5rem; font-family:'Plus Jakarta Sans','Inter',sans-serif; font-weight:700; }
.tile.t-mint .head { color:var(--mint-ink); } .tile.t-peach .head { color:var(--peach-ink); } .tile.t-lav .head { color:var(--lav-ink); }
.tile { margin-bottom:.75rem; }
.tile ul { list-style:none !important; margin:.7rem 0 0 !important; padding:0 !important; display:flex; flex-direction:column; gap:.55rem; }
.tile li { font-size:.95rem; line-height:1.45; margin:0 !important; padding:0 !important; }
.tile .sub { font-size:.88rem; color:var(--muted); margin-top:.2rem; }
svg.ic { width:20px; height:20px; stroke:currentColor; fill:none; stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round; flex:none; }

/* step progress */
.progress { display:grid; grid-auto-flow:column; grid-auto-columns:1fr; gap:.5rem; margin:.2rem 0 1rem; }
.progress div { font-size:.82rem; color:var(--muted); }
.progress div::before { content:""; display:block; height:6px; border-radius:6px; background:var(--line); margin-bottom:.45rem; }
.progress .done::before, .progress .now::before { background:var(--brand); }
.progress .now { color:var(--ink); font-weight:600; }

/* top bar: light/dark switch and profile avatar, added by device_store.py next to Streamlit's toolbar buttons */
.js-topbar { display:flex; align-items:center; gap:.5rem; margin-right:.6rem; }
.js-iconbtn, .js-avatar { width:36px; height:36px; border-radius:50%; display:inline-grid; place-items:center; cursor:pointer;
  border:1px solid var(--line); background:var(--panel); color:var(--ink); padding:0; transition:transform .15s ease, box-shadow .15s ease; }
.js-iconbtn:hover, .js-avatar:hover { transform:translateY(-1px); box-shadow:0 6px 16px rgba(21,32,58,.18); }
.js-avatar { border:0; color:#ffffff; background:linear-gradient(135deg, var(--b-blue), var(--b-lav)); position:relative; }
.js-avatar.ready::after { content:""; position:absolute; right:-1px; bottom:-1px; width:11px; height:11px; border-radius:50%;
  background:#19b394; border:2px solid var(--wash); }
.js-iconbtn:focus-visible, .js-avatar:focus-visible { outline:3px solid var(--brand); outline-offset:2px; }

/* glossary terms: dotted underline, definition on hover or long-press */
abbr.term { text-decoration:underline dotted; text-decoration-color:var(--muted); text-underline-offset:3px; cursor:help; }

/* plain-English summary at the top of a tab */
.inshort { background:var(--blue-soft); color:var(--ink); border-radius:12px; padding:.7rem 1rem; margin:.4rem 0 .8rem; font-size:.95rem; line-height:1.5; }
.inshort b { color:var(--brand-ink); }

/* brand, navigation tile, error card */
.brand { display:flex; align-items:center; gap:.65rem; }
.brand .logo { width:36px; height:36px; border-radius:11px; background:var(--brand); color:var(--on-brand); display:grid; place-items:center; }
.brand b { font-family:'Plus Jakarta Sans','Inter',sans-serif; font-weight:800; font-size:1.25rem; letter-spacing:-.02em; color:var(--ink); }
.brand small { display:block; font-size:.72rem; color:var(--muted); line-height:1.2; }
.navtile { display:flex; flex-direction:column; gap:.2rem; color:var(--lav-ink); }
.navtile .ic { width:36px; height:36px; padding:8px; border-radius:11px; background:var(--lav-soft); margin-bottom:.3rem; }
.navtile b { font-family:'Plus Jakarta Sans','Inter',sans-serif; color:var(--ink); }
.navtile span { font-size:.85rem; color:var(--muted); }
.errorcard { background:var(--peach-soft); color:var(--ink); border-radius:var(--radius); padding:1rem 1.2rem; }
.errorcard b { color:var(--peach-ink); }
@media (prefers-reduced-motion: reduce) { * { transition:none !important; animation:none !important; } }
</style>
"""

ICONS = {      # 24x24 stroke icons, drawn inline so they follow the theme colours
    "spark": "<path d='M12 2l2.2 6.8L21 11l-6.8 2.2L12 20l-2.2-6.8L3 11l6.8-2.2z' fill='currentColor' stroke='none'/>",
    "up": "<path d='M12 19V5M5 12l7-7 7 7'/>",
    "down": "<path d='M12 5v14M19 12l-7 7-7-7'/>",
    "target": "<circle cx='12' cy='12' r='9'/><circle cx='12' cy='12' r='5'/><circle cx='12' cy='12' r='1'/>",
    "scale": "<path d='M12 3v18M6 21h12M5 7h14M5 7l-3 7h6zM19 7l-3 7h6z'/>",
    "case": "<rect x='3' y='7' width='18' height='13' rx='2'/><path d='M9 7V5a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2M3 13h18'/>",
    "chat": "<path d='M21 12a8 8 0 0 1-11.6 7.1L3 21l1.9-5.4A8 8 0 1 1 21 12z'/>",
    "flask": "<path d='M9 3h6M10 3v6l-5.5 10a1.5 1.5 0 0 0 1.3 2.2h12.4a1.5 1.5 0 0 0 1.3-2.2L14 9V3M7.5 14h9'/>",
}


def theme_type() -> str | None:
    """'light' or 'dark' as the browser reports it (see device_store.py); before the first report, Streamlit's own
    value; None when neither is available (no browser session, e.g. tests)."""
    reported = st.session_state.get("ui_theme")
    if reported in ("light", "dark"):
        return reported
    try:
        return st.context.theme.type
    except Exception:
        return None


def is_dark() -> bool:
    return theme_type() == "dark"


def _dark_tokens_css() -> str:
    """The dark colour tokens as CSS rules: used when the browser reports dark, and (until it reports) when the
    operating system prefers dark."""
    body = TOKENS_DARK.strip().removeprefix(":root").strip().removeprefix("{").removesuffix("}")
    return (f"@media (prefers-color-scheme: dark) {{ :root:not([data-js-theme]) {{{body}}} }} "
            f":root[data-js-theme='dark'] {{{body}}}")


def inject() -> None:
    st.markdown(CSS.replace("/*TOKENS*/", TOKENS_LIGHT + _dark_tokens_css()), unsafe_allow_html=True)


def model_info() -> dict:
    return get("/model-info")


@st.cache_data(ttl=300, show_spinner=False)
def study(path: str) -> dict:
    return get(path)


def fetch(path: str) -> dict | None:
    try:
        return study(path)
    except APIError as e:
        error_card(str(e), "Check that the API is running (uvicorn api.main:app --port 8000), then reload.")
        return None


def esc(x) -> str:
    return html.escape(str(x))


def hero(kicker: str, title: str, text: str, compact: bool = False, art: str | None = None) -> None:
    """Show a page banner, optionally with a decorative, accessible illustration."""
    classes = "hero" + (" compact" if compact else "") + (" has-art" if art else "")
    copy = f"<div><div class='kicker'>{esc(kicker)}</div><h1>{esc(title)}</h1><p>{text}</p></div>"
    image = (f"<img class='hero-art' src='{html.escape(art, quote=True)}' "
             f"alt='Illustration of a graduate choosing a career path among IT roles'>") if art else ""
    st.markdown(f"<div class='{classes}'>{copy}{image}</div>", unsafe_allow_html=True)


def career_path_art() -> str:
    """Return original artwork showing a graduate considering several IT career paths."""
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 320 260'>"
        "<style>"
        "@keyframes careerOrbit{to{transform:rotate(360deg)}}"
        ".career-orbit{transform-box:view-box;transform-origin:center;"
        "animation:careerOrbit 36s linear infinite}"
        "@media(prefers-reduced-motion:reduce){.career-orbit{animation:none}}"
        "</style>"
        "<circle cx='160' cy='130' r='120' fill='#fff' opacity='.12'/>"
        "<g class='career-orbit'>"
        "<path d='M162 170V124M162 124L86 72M162 124l76-52M162 124v102' fill='none' "
        "stroke='#fff' stroke-width='5' stroke-linecap='round' opacity='.85'/>"
        "<g fill='#fff' stroke='#ffa62b' stroke-width='5'><circle cx='86' cy='72' r='31'/>"
        "<circle cx='238' cy='72' r='31'/><circle cx='162' cy='226' r='24'/></g>"
        "<g fill='#315ed4'><path d='M72 77h28v18H72zM79 67h14v10H79z'/>"
        "<path d='M224 64h28v22h-28zM231 70h14M231 77h14' fill='none' stroke='#315ed4' "
        "stroke-width='3' stroke-linecap='round'/><path d='M149 229c2-14 24-14 26 0v7h-26z'/>"
        "<circle cx='162' cy='218' r='7'/></g>"
        "</g>"
        "<g class='career-person'>"
        "<path d='M146 117v11h28v-11' fill='#e9a987'/>"
        "<path d='M124 171v-25c0-19 13-32 31-35l7 10 8-10c17 4 29 17 29 35v25z' fill='#7da2ff'/>"
        "<path d='M139 122l-14 13-5 27 12 2 7-20M181 122l14 13 5 27-12 2-7-20' "
        "fill='#ffc6a4' stroke='#e9a987' stroke-width='2' stroke-linejoin='round'/>"
        "<path d='M147 126l15 13 15-13-4-10h-22z' fill='#fff' opacity='.92'/>"
        "<path d='M145 103c-8-15-2-31 12-35 15-5 29 5 30 20l-3 27-7 9h-29l-7-10z' fill='#302a3f'/>"
        "<circle cx='162' cy='99' r='18' fill='#ffc6a4'/>"
        "<path d='M144 94c9-2 15-7 19-14 5 7 12 11 23 13l1-5c-2-12-11-20-22-20-14 0-23 10-21 26z' fill='#302a3f'/>"
        "<circle cx='155' cy='101' r='1.7' fill='#302a3f'/><circle cx='169' cy='101' r='1.7' fill='#302a3f'/>"
        "<path d='M162 102l-1 5h3M157 112q5 4 10 0' fill='none' stroke='#a85855' "
        "stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'/>"
        "<path d='M139 145l-3 26M183 145l3 26' stroke='#5d83e5' stroke-width='3' stroke-linecap='round'/>"
        "<path d='M156 145h12' stroke='#fff' stroke-width='2' stroke-linecap='round' opacity='.8'/>"
        "</g>"
        "<circle cx='51' cy='147' r='5' fill='#3ee0c0'/><circle cx='275' cy='144' r='5' fill='#ffc86b'/>"
        "</svg>"
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


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


def icon(name: str) -> str:
    return f"<svg class='ic' viewBox='0 0 24 24' aria-hidden='true'>{ICONS[name]}</svg>"


def brand_html() -> str:
    return (f"<div class='brand'><div class='logo'>{icon('spark')}</div>"
            "<div><b>JobSpark</b><small>Time to your first IT job</small></div></div>")


def driver_card(kind: str, title: str, items: list[str], empty: str = "") -> None:
    """Soft tile listing what helps ('helping', mint, up arrow) or what holds back ('holding', peach, down arrow).
    Items are trusted HTML from the API's own wording, as elsewhere in the app. The arrow keeps it readable without colour."""
    tone, arrow = ("mint", "up") if kind == "helping" else ("peach", "down")
    body = "".join(f"<li>{i}</li>" for i in items) if items else f"<li class='muted'>{esc(empty)}</li>"
    st.markdown(f"<div class='tile t-{tone}'><div class='head'>{icon(arrow)}{esc(title)}</div><ul>{body}</ul></div>",
                unsafe_allow_html=True)


def progress(labels: list[str], current: int) -> None:
    """Step bar: labels in order, current is 1-based (earlier steps show as done)."""
    cells = "".join(f"<div class='{'done' if i < current else 'now' if i == current else ''}'>{esc(t)}</div>"
                    for i, t in enumerate(labels, 1))
    st.markdown(f"<div class='progress' role='img' aria-label='Step {current} of {len(labels)}'>{cells}</div>",
                unsafe_allow_html=True)


def estimate_card(median: str, low: str, high: str, confidence: str, headline: str) -> None:
    """Show the most-likely time separately from its range, confidence and explanation."""
    st.markdown(
        f"<div class='result-grid'>"
        f"<div class='result-months'><div class='lbl'>Most likely time to a first IT job</div>"
        f"<div class='big'>{esc(median)} <small>months</small></div></div>"
        f"<div class='result-details'><div class='range-label'>Typical range</div>"
        f"<div class='rng'>{esc(low)} - {esc(high)} months</div>"
        f"<div class='confidence'>{esc(confidence)}</div>"
        f"<div class='hl'>{headline}</div></div></div>", unsafe_allow_html=True)


GLOSSARY = {
    "c-index": ("C-index", "How well the model puts graduates in the right order of who finds work sooner. "
                           "0.5 is no better than guessing; 1.0 is perfect."),
    "ibs": ("Integrated Brier score", "The average squared error of the predicted chance of having a job, over time. "
                                      "Lower is better; 0.25 means the prediction tells you nothing."),
    "calibration": ("Calibration", "Whether predicted chances match what actually happened. If the model says 60%, "
                                   "about 60 in 100 such graduates should have found work."),
    "hazard ratio": ("Hazard ratio", "How much a factor changes the rate of finding a job. Above 1 means sooner, "
                                     "below 1 means later."),
    "confidence interval": ("Confidence interval", "A range that probably contains the true value. A wide range "
                                                   "means less certainty."),
    "censored": ("Censored", "A graduate who had not yet found an IT job when the survey was taken. We know they "
                             "waited at least that long, but not how long in total."),
    "bootstrap": ("Bootstrap", "Re-running the analysis on many resampled copies of the data to see how much the "
                               "result moves."),
    "kaplan-meier": ("Kaplan-Meier curve", "A standard way to draw the share who have found a job by each month, "
                                           "counting those still looking."),
}


def term(key: str, label: str | None = None) -> str:
    """A glossary word with its plain-language definition as a hover tooltip (HTML, for use inside trusted text)."""
    name, definition = GLOSSARY[key]
    return f"<abbr class='term' title='{html.escape(definition, quote=True)}'>{esc(label or name)}</abbr>"


def hero_art() -> str:
    """Home banner illustration (own artwork, inline SVG): a CV found by a magnifying glass, a green tick for success,
    and a rising bar chart - job search leading to success."""
    return (
        "<svg class='hero-art' viewBox='0 0 340 300' role='img' aria-label='A CV found by a magnifying glass, "
        "with a tick and a rising chart: job search leading to success'>"
        "<circle cx='170' cy='150' r='132' fill='#ffffff' opacity='.10'/>"
        "<g transform='rotate(-4 130 146)'>"
        "<rect x='58' y='50' width='152' height='194' rx='16' fill='#ffffff'/>"
        "<circle cx='102' cy='96' r='22' fill='#cfe0ff'/><path d='M80 134 a22 18 0 0 1 44 0 z' fill='#cfe0ff'/>"
        "<rect x='136' y='82' width='56' height='9' rx='4.5' fill='#1d52cf'/>"
        "<rect x='136' y='99' width='38' height='7' rx='3.5' fill='#cfd9f5'/>"
        "<rect x='136' y='113' width='46' height='7' rx='3.5' fill='#cfd9f5'/>"
        "<rect x='76' y='146' width='116' height='2' fill='#e3e9f8'/>"
        "<rect x='76' y='158' width='46' height='18' rx='9' fill='#19b394'/>"
        "<rect x='128' y='158' width='38' height='18' rx='9' fill='#ffa62b'/>"
        "<rect x='172' y='158' width='22' height='18' rx='9' fill='#7c5cff'/>"
        "<rect x='76' y='190' width='116' height='7' rx='3.5' fill='#dbe4fa'/>"
        "<rect x='76' y='205' width='86' height='7' rx='3.5' fill='#dbe4fa'/>"
        "<rect x='76' y='220' width='100' height='7' rx='3.5' fill='#dbe4fa'/></g>"
        "<g class='float'>"
        "<circle cx='214' cy='178' r='46' fill='#ffffff' fill-opacity='.22' stroke='#ffa62b' stroke-width='10'/>"
        "<path d='M190 160 a30 30 0 0 1 22 -12' fill='none' stroke='#ffffff' stroke-width='5' stroke-linecap='round' opacity='.8'/>"
        "<line x1='247' y1='211' x2='292' y2='256' stroke='#ffa62b' stroke-width='14' stroke-linecap='round'/>"
        "<circle cx='214' cy='178' r='20' fill='#19b394'/>"
        "<path d='M204 178 l8 8 l14 -16' fill='none' stroke='#ffffff' stroke-width='5' stroke-linecap='round' "
        "stroke-linejoin='round'/></g>"
        "<g><rect x='252' y='112' width='15' height='34' rx='4' fill='#ffffff' opacity='.9'/>"
        "<rect x='273' y='92' width='15' height='54' rx='4' fill='#ffc86b'/>"
        "<rect x='294' y='68' width='15' height='78' rx='4' fill='#3ee0c0'/>"
        "<path d='M250 84 L274 62 L298 44' fill='none' stroke='#ffffff' stroke-width='3.5' stroke-linecap='round' "
        "stroke-linejoin='round'/><path d='M286 42 L300 43 L296 57' fill='none' stroke='#ffffff' stroke-width='3.5' "
        "stroke-linecap='round' stroke-linejoin='round'/></g>"
        "<circle cx='36' cy='72' r='5' fill='#ffc86b'/><circle cx='318' cy='186' r='5' fill='#3ee0c0'/>"
        "<circle cx='54' cy='262' r='4' fill='#ffffff' opacity='.8'/>"
        "<path d='M44 150 l3 7 l7 3 l-7 3 l-3 7 l-3 -7 l-7 -3 l7 -3 z' fill='#ffe29a'/>"
        "<path d='M322 100 l2 5 l5 2 l-5 2 l-2 5 l-2 -5 l-5 -2 l5 -2 z' fill='#ffffff' opacity='.9'/>"
        "</svg>")


def side_summary(title: str, items: list[tuple[str, str]]) -> None:
    """Soft panel with label/value chips, e.g. the profile so far. Values are plain text and are escaped."""
    chips = "".join(f"<span class='chip'>{esc(k)} <b>{esc(v)}</b></span>" for k, v in items)
    st.markdown(f"<div class='sidepanel'><h4>{esc(title)}</h4><div class='chiprow'>{chips}</div></div>",
                unsafe_allow_html=True)


def tip(text: str) -> None:
    """A short helpful hint (plain text, escaped)."""
    st.markdown(f"<div class='tip'><b>Tip.</b> {esc(text)}</div>", unsafe_allow_html=True)


def in_short(text: str) -> None:
    """One plain-English line that says what a tab shows. Text is trusted HTML written in the page code."""
    st.markdown(f"<div class='inshort'><b>In short:</b> {text}</div>", unsafe_allow_html=True)


def fact_tile(tone: str, value: str, label: str, sub: str = "") -> None:
    """Soft tile with one big figure and a plain-language line. tone: blue | mint | peach | lav."""
    st.markdown(f"<div class='tile t-{tone} fact'><div class='v'>{esc(value)}</div><div class='l'>{label}</div>"
                + (f"<div class='s'>{sub}</div>" if sub else "") + "</div>", unsafe_allow_html=True)


def rating(label: str, key: str, default: int = 3, help: str | None = None) -> int:
    """1-5 pill buttons (easier to tap than a slider). Always has a value selected."""
    st.session_state.setdefault(key, default)      # value lives in session state so it survives step changes
    return st.segmented_control(label, [1, 2, 3, 4, 5], required=True, key=key, help=help)


def nav_tile(page: str, title: str, desc: str, icon_name: str) -> None:
    """Bordered tile that links to another page."""
    with st.container(border=True):
        st.markdown(f"<div class='navtile'>{icon(icon_name)}<b>{esc(title)}</b><span>{esc(desc)}</span></div>",
                    unsafe_allow_html=True)
        st.page_link(page, label="Open", icon=":material/arrow_forward:")


def error_card(message: str, hint: str = "") -> None:
    st.markdown(f"<div class='errorcard'><b>Something went wrong.</b> {esc(message)}"
                + (f"<div class='muted'>{esc(hint)}</div>" if hint else "") + "</div>", unsafe_allow_html=True)


def synthetic_notice(info: dict) -> None:
    if info.get("is_synthetic"):
        st.markdown("<div class='notice'>⚠️ <div><b>Demo model.</b> Predictions on this page are trained on "
                    "<b>synthetic (computer-generated) graduate survey data</b>, not real graduates. "
                    "The job-market figures come from real job ads.</div></div>", unsafe_allow_html=True)


def q_text(q: float) -> str:
    return "q < 0.001" if q < 0.001 else f"q = {q:.3f}"


PLOT = {"displayModeBar": False}
