"""Job market: the study of real entry-level IT job ads (core/market_study.py, served by GET /job-market)."""
from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from frontend import charts_study as cs
from frontend import ui

m = ui.fetch("/job-market")
if not m:
    st.stop()

ui.hero("Job market · real data",
        "What Sri Lankan employers ask of entry-level IT graduates",
        f"{m['n_ads']} real internship, trainee, junior and associate IT vacancies from {m['n_companies']} companies, "
        f"posted {pd.Timestamp(m['date_from']):%d %b} - {pd.Timestamp(m['date_to']):%d %b %Y}, coded against the same "
        "26-skill vocabulary the predictor uses.", compact=True)

prev = m["prevalence"]
tech = [p for p in prev if p["kind"] == "technical"]
soft = [p for p in prev if p["kind"] == "soft"]
top_pair = max(m["pairs"], key=lambda p: p["lift"]) if m["pairs"] else None
sig = [t for t in m["seniority_tests"] if t["significant"]]

# -- Key takeaways ---------------------------------------------------------------
ui.section("Key takeaways", "The four things most worth knowing.", "real")
row1 = st.columns(2, gap="medium")
with row1[0]:
    ui.fact_tile("blue", f"{soft[0]['pct']:.0f}%",
                 f"of ads ask for <b>{ui.esc(soft[0]['label'].lower())}</b>, the most requested soft skill.",
                 f"95% CI {soft[0]['ci_low']:.0f}-{soft[0]['ci_high']:.0f}%. Then {ui.esc(soft[1]['label'].lower())} "
                 f"({soft[1]['pct']:.0f}%) and {ui.esc(soft[2]['label'].lower())} ({soft[2]['pct']:.0f}%).")
with row1[1]:
    ui.fact_tile("mint", f"{tech[0]['pct']:.0f}%",
                 f"of ads ask for <b>{ui.esc(tech[0]['label'].lower())}</b>, the top technical skill.",
                 f"Then {ui.esc(tech[1]['label'].lower())} ({tech[1]['pct']:.0f}%) and "
                 f"{ui.esc(tech[2]['label'].lower())} ({tech[2]['pct']:.0f}%). Most-named technology: "
                 f"{ui.esc(m['technologies'][0]['name'])} ({m['technologies'][0]['pct']:.0f}% of ads).")
row2 = st.columns(2, gap="medium")
with row2[0]:
    if top_pair:
        ui.fact_tile("lav", f"{top_pair['lift']:.1f}×",
                     f"more often than chance: <b>{ui.esc(top_pair['a'])} + {ui.esc(top_pair['b'])}</b> appear together.",
                     f"Seen in {top_pair['ads']} ads (phi {top_pair['phi']:.2f}), the strongest skill bundle.")
with row2[1]:
    if sig:
        t = sig[0]
        ui.fact_tile("peach", f"{t['intern_pct']:.0f}%",
                     f"of internship ads ask for <b>{ui.esc(t['label'].lower())}</b>, against {t['junior_pct']:.0f}% of "
                     "junior roles.",
                     f"Significant after correcting for {len(m['seniority_tests'])} comparisons ({ui.q_text(t['q'])}).")


def wilson(k: float, n: int) -> tuple[float, float]:
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


# -- Employer survey ---------------------------------------------------------------
def employer_survey(qs: dict) -> None:
    """The employer / recruiter questionnaire (GET /questionnaire), shown next to the job-ad study."""
    tag = "synthetic" if qs["is_synthetic"] else "real"
    ui.section("What employers told us", "From the employer / recruiter questionnaire, not the job ads."
               + (" These answers are <b>synthetic</b> until the real survey is collected." if tag == "synthetic" else ""),
               tag)
    ev = qs["employer_views"]
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**How important is each hiring factor?** <span class='muted'>(mean 1-5, E7)</span>",
                    unsafe_allow_html=True)
        st.plotly_chart(cs.mean_bars([(f["label"], f["mean"], f["ci"]) for f in qs["employer_factors"]]),
                        config=ui.PLOT, key="emp_factors")
        st.markdown("**Technologies expected to grow** <span class='muted'>(% of employers, E10)</span>",
                    unsafe_allow_html=True)
        st.plotly_chart(cs.pct_bars([(r["label"], r["pct"]) for r in ev["future_tech"][:8]]), config=ui.PLOT,
                        key="emp_future")
    with c2:
        st.markdown("**What makes one graduate more employable** <span class='muted'>(% ranking it in their top 3, "
                    "E20)</span>", unsafe_allow_html=True)
        st.plotly_chart(cs.pct_bars([(r["factor"], r["pct_top3"]) for r in qs["employer_ranking"]], color=cs.ORANGE),
                        config=ui.PLOT, key="emp_rank")
        st.markdown("**Weaknesses employers see in fresh graduates** <span class='muted'>(% of employers, "
                    "E12/E13)</span>", unsafe_allow_html=True)
        weak = sorted(ev["tech_weaknesses"] + ev["soft_weaknesses"], key=lambda r: -(r["pct"] or 0))[:8]
        st.plotly_chart(cs.pct_bars([(r["label"], r["pct"]) for r in weak], color=cs.VIOLET), config=ui.PLOT,
                        key="emp_weak")
    upd = {r["key"]: r["pct"] for r in ev["skill_update_freq"]}
    often = (upd.get("more_than_yearly") or 0) + (upd.get("1-2_years") or 0)
    st.markdown(f"<div class='item'><b>{often:.0f}%</b> of employers update the technical skills they require at "
                f"least every 1-2 years (E11) - skill demand changes quickly.</div>", unsafe_allow_html=True)
    st.markdown("**Entry-level roles employers recruit for** <span class='muted'>(% of employers, E5)</span>",
                unsafe_allow_html=True)
    st.plotly_chart(cs.pct_bars([(r["label"], r["pct"]) for r in ev["roles_recruited"][:8]], color=cs.AQUA),
                    config=ui.PLOT, key="emp_roles")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**What universities should improve** (E15)")
        st.markdown("\n".join(f"- {r['label']} - {r['pct']:.0f}%" for r in ev["uni_improvements"][:5]))
    with c2:
        st.markdown("**Biggest hiring challenges** (E19)")
        st.markdown("\n".join(f"- {r['label']} - {r['pct']:.0f}%" for r in ev["hiring_challenges"][:5]))


questionnaire = ui.fetch("/questionnaire")
has_employers = bool(questionnaire and questionnaire.get("employer_views"))
names = ["Skills in demand", "Role profiles", "Skills together", "Interns vs junior", "Technologies"]
icons = [":material/trending_up:", ":material/badge:", ":material/hub:", ":material/school:", ":material/code:"]
if has_employers:
    names.append("Employers")
    icons.append(":material/business_center:")
tabs = st.tabs([f"{i} {n}" for i, n in zip(icons, names)])

# -- Skill demand ----------------------------------------------------------------
with tabs[0]:
    ui.in_short(f"<b>{ui.esc(prev[0]['label'])}</b> is the most requested skill ({prev[0]['pct']:.0f}% of ads). "
                "Pick a role family to see what that role asks for.")
    ui.section("Skill demand", "Share of ads that mention each skill, with 95% confidence intervals.", "real")
    roles = ["All roles"] + list(m["roles"])
    role = st.segmented_control("Role family", roles, default="All roles", label_visibility="collapsed")
    if role and role != "All roles":
        n = m["roles"][role]
        rows = []
        for p in prev:
            pct = m["by_role"][role][p["key"]]
            lo, hi = wilson(pct * n / 100, n)
            rows.append({**p, "pct": pct, "ads": round(pct * n / 100), "ci_low": lo, "ci_high": hi})
        rows.sort(key=lambda r: -r["pct"])
        st.caption(f"{role}: {n} ads. Small groups give wide intervals.")
    else:
        rows = prev
    c1, c2 = st.columns([1.6, 1], gap="large")
    with c1:
        st.plotly_chart(cs.demand_bars(rows, 15), config=ui.PLOT, key="demand")
    with c2:
        st.markdown("**Ads by role family**")
        st.plotly_chart(cs.count_bars(list(m["roles"].items())), config=ui.PLOT, key="roles")
        st.markdown("**Intern / trainee vs junior / associate**")
        st.plotly_chart(cs.count_bars(list(m["seniority"].items()), color=cs.VIOLET), config=ui.PLOT, key="seniority")

# -- Role profiles -----------------------------------------------------------------
with tabs[1]:
    ui.in_short("Different roles ask for different things. Find the column for the role you want and read down it: "
                "darker cells are skills that role's ads name more often.")
    ui.section("Role profiles", "Each column is a role family; each cell is the share of that role's ads naming the "
                                "skill. QA ads are defined by testing; business-analysis ads by communication.", "real")
    st.plotly_chart(cs.role_heatmap(m["by_role"], m["roles"], prev[:14]), config=ui.PLOT, key="heat")

# -- Skill bundles -------------------------------------------------------------------
with tabs[2]:
    if top_pair:
        ui.in_short(f"<b>{ui.esc(top_pair['a'])}</b> and <b>{ui.esc(top_pair['b'])}</b> are asked for together "
                    f"{top_pair['lift']:.1f}× more often than chance. Learning related skills together can pay off.")
    ui.section("Skills that go together", "Lines join skills that appear in the same ads more often than chance "
                                          "(lift ≥ 1.3); the 18 strongest links are drawn; thicker = stronger "
                                          "association (phi). Hover a line for details.", "real")
    c1, c2 = st.columns([1.5, 1], gap="large")
    with c1:
        st.plotly_chart(cs.skill_network(prev, m["pairs"]), config=ui.PLOT, key="network")
    with c2:
        st.markdown("**Strongest bundles**")
        st.dataframe(pd.DataFrame(m["pairs"][:12])[["a", "b", "ads", "lift", "phi"]]
                     .rename(columns={"a": "Skill", "b": "With", "ads": "Ads", "lift": "Lift", "phi": "Phi"}).round(2),
                     hide_index=True, height=460)

# -- Interns vs junior -------------------------------------------------------------------
with tabs[3]:
    if sig:
        t = sig[0]
        ui.in_short(f"Internship ads stress <b>{ui.esc(t['label'].lower())}</b> more than junior roles do "
                    f"({t['intern_pct']:.0f}% vs {t['junior_pct']:.0f}%). If you are applying for an internship, show it.")
    else:
        ui.in_short("Internship and junior ads ask for similar things; no skill differs clearly between the two.")
    ui.section("Interns vs junior roles", "The ten skills whose share differs most between the two groups. "
                                          "★ = significant after Benjamini-Hochberg correction (q < 0.05).", "real")
    st.plotly_chart(cs.dumbbell(m["seniority_tests"]), config=ui.PLOT, key="dumbbell")

# -- Technologies -------------------------------------------------------------------------
with tabs[4]:
    ui.in_short(f"<b>{ui.esc(m['technologies'][0]['name'])}</b> is the most named tool or language "
                f"({m['technologies'][0]['pct']:.0f}% of ads). Ads were collected over "
                f"{len(m['monthly'])} months.")
    ui.section("Named technologies", "Specific tools and languages mentioned in the ads (beyond the 26-skill "
                                     "vocabulary).", "real")
    c1, c2 = st.columns([1.6, 1], gap="large")
    with c1:
        st.plotly_chart(cs.count_bars([(t["name"], t["ads"]) for t in m["technologies"][:15]]), config=ui.PLOT,
                        key="tech")
    with c2:
        st.markdown("**Ads collected by month posted**")
        st.plotly_chart(cs.count_bars(list(m["monthly"].items())[::-1], color=cs.BLUE), config=ui.PLOT, key="monthly")

# -- Employers ----------------------------------------------------------------------------------
if has_employers:
    with tabs[5]:
        ui.in_short("What recruiters themselves said in the employer survey: which skills they value, where graduates "
                    "fall short, and what they expect to grow.")
        employer_survey(questionnaire)

with st.expander("About these ads, method and limitations", icon=":material/info:"):
    ui.kpis([
        (m["n_ads"], "Job ads", "topjobs.lk · ITPro.lk · LinkedIn", "real"),
        (m["n_companies"], "Hiring companies", "distinct employers"),
        (len(m["roles"]), "Role families", ", ".join(list(m["roles"])[:3]) + "…"),
        (f"{m['emphasis']['pct_any_soft']:.0f}%", "Ask for soft skills",
         f"{m['emphasis']['mean_soft']:.1f} soft vs {m['emphasis']['mean_technical']:.1f} technical per ad"),
    ])
    st.markdown("\n".join(f"- {n}" for n in m["notes"]) + "\n- Every ad's source link, the coding method and the "
                "excluded ads (with reasons) are listed in `data/raw/job_ads_provenance.csv`.\n- A snapshot of "
                "three months of advertised vacancies, not of all hiring (many roles are filled without an ad).")
