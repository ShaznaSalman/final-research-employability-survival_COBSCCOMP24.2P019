"""Home: what the system does, headline numbers, and where each kind of data comes from."""
from __future__ import annotations

import streamlit as st

from frontend import ui
from frontend.api_client import APIError

try:
    info = ui.model_info()
except APIError as e:
    st.error(str(e))
    st.stop()
market = ui.fetch("/job-market") or {}

ui.hero("Home · Sri Lankan IT graduates",
        "How long until your first IT job - and what moves it?",
        "A survival-analysis decision-support system for Sri Lankan IT graduates. It estimates time to a first IT job, "
        "explains what helps and what holds a profile back, and compares a graduate's skills with what employers "
        "are actually advertising for.")

ui.kpis([
    (market.get("n_ads", "-"), "Real job ads analysed", f"{market.get('n_companies', '-')} companies · "
                                                        f"{market.get('date_from', '')[:7]} to {market.get('date_to', '')[:7]}", "real"),
    (len(info["cv_metrics"]), "Survival models compared", "fixed train / validation / test split, no data reuse"),
    (f"{(info.get('test_metrics') or {}).get('uno_c', 0):.2f}", "Uno's C-index (locked test set)",
     f"selected: {info['selected_model_label']}"),
    (info["training_n"], "Graduates in training data", f"{info['events']} found an IT job · {info['censored']} still looking",
     "synthetic" if info["is_synthetic"] else "real"),
])

c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    with st.container(border=True):
        st.markdown("#### :material/insights: My prediction")
        st.markdown("<p class='muted'>Enter a profile and get a time-to-job estimate with its uncertainty, what helps, "
                    "what holds it back, and what-if scenarios.</p>", unsafe_allow_html=True)
        st.page_link("views/predict.py", label="Open the predictor", icon=":material/arrow_forward:")
with c2:
    with st.container(border=True):
        st.markdown("#### :material/work: Job market")
        st.markdown("<p class='muted'>What 300+ real entry-level IT ads ask for: skill demand, role profiles, skill "
                    "bundles, interns vs junior roles and named technologies.</p>", unsafe_allow_html=True)
        st.page_link("views/market.py", label="Explore the job market", icon=":material/arrow_forward:")
with c3:
    with st.container(border=True):
        st.markdown("#### :material/science: Research lab")
        st.markdown("<p class='muted'>How well the model works and why it can be trusted: model comparison, "
                    "calibration, effects, fairness, robustness and a simulation study.</p>", unsafe_allow_html=True)
        st.page_link("views/research.py", label="See the evidence", icon=":material/arrow_forward:")

if market:
    ui.section("What employers ask for", "From the real job ads - see the Job market page for the full analysis.", "real")
    top = market["prevalence"][:3]
    e = market["emphasis"]
    f1, f2, f3 = st.columns(3, gap="medium")
    with f1:
        ui.finding(f"{top[0]['label']}: {top[0]['pct']:.0f}% of ads",
                   f"The most requested skill, ahead of {top[1]['label'].lower()} ({top[1]['pct']:.0f}%) "
                   f"and {top[2]['label'].lower()} ({top[2]['pct']:.0f}%).")
    with f2:
        ui.finding(f"{e['pct_any_soft']:.0f}% ask for soft skills",
                   f"An ad names on average {e['mean_technical']:.1f} technical and {e['mean_soft']:.1f} soft skills.")
    with f3:
        sig = [t for t in market["seniority_tests"] if t["significant"]]
        if sig:
            t = sig[0]
            ui.finding(f"Interns: {t['label'].lower()} {t['intern_pct']:.0f}% vs {t['junior_pct']:.0f}%",
                       "Internship ads ask for this much more often than junior / associate roles "
                       f"({ui.q_text(t['q'])}).")

ui.section("How it works")
s1, s2, s3, s4 = st.columns(4, gap="medium")
for col, (n, t, d) in zip((s1, s2, s3, s4), [
        ("1", "Survey data", "Graduate questionnaire: GPA, internships, projects, certifications and 26 skill ratings, "
                             "with the month of the first IT job (or still looking)."),
        ("2", "Survival models", "Seven models that handle graduates still looking (censoring): fitted on a training "
                                 "set, compared on a validation set, and the chosen one tested once on a locked test "
                                 "set; 200 bootstrap refits give the uncertainty."),
        ("3", "Market evidence", "Real job ads coded against the same 26-skill vocabulary, so a profile can be compared "
                                 "with real demand."),
        ("4", "Checks", "Calibration, fairness by subgroup, sensitivity, temporal validation and a simulation study.")]):
    col.markdown(f"<div class='card'><h4>{n} · {t}</h4><p>{d}</p></div>", unsafe_allow_html=True)

ui.section("Where the data comes from")
d1, d2 = st.columns(2, gap="medium")
d1.markdown(f"<div class='card'><h4>{ui.tag_html('real')} Job ads</h4><p>{market.get('n_ads', '-')} entry-level IT "
            "vacancies from topjobs.lk, ITPro.lk and LinkedIn, each with its source link. Skills coded by hand from "
            "flyers or by fixed keyword rules.</p></div>", unsafe_allow_html=True)
if info["is_synthetic"]:
    survey_text = ("Generated by a documented simulator calibrated to published Sri Lankan figures, so the "
                   "pipeline can be validated against known true effects before real survey data is used.")
else:
    survey_text = (f"Main-study questionnaire responses: {info.get('modelling_n', '-')} graduates used for modelling "
                   f"(plus {info.get('final_year_n', '-')} final-year students, used descriptively only) and "
                   f"{info.get('employers_n', '-')} employers, collected August-October 2026. Pilot and test "
                   "responses are excluded.")
d2.markdown(f"<div class='card'><h4>{ui.tag_html('synthetic' if info['is_synthetic'] else 'real')} Graduate and employer "
            f"surveys</h4><p>{survey_text}</p></div>", unsafe_allow_html=True)
st.caption(info["disclaimer"])
