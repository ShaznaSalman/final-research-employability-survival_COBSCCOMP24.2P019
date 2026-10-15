"""Home: what JobSpark does, one clear way in, what employers ask for, and (folded away) the numbers behind it."""
from __future__ import annotations

import streamlit as st

from frontend import profile as prof, ui
from frontend.api_client import APIError

try:
    info = ui.model_info()
except APIError as e:
    ui.error_card(str(e), "Check that the API is running (uvicorn api.main:app --port 8000), then reload.")
    st.stop()
market = ui.fetch("/job-market") or {}

with st.container(key="homehero"):
    text, art = st.columns([1.5, 1], gap="large", vertical_alignment="center")
    with text:
        st.markdown("<div class='kicker'>Sri Lanka · IT graduates</div>"
                    "<h1>How long until your first IT job?</h1>"
                    "<p>Answer a few questions about your degree, experience and skills. You get a plain-language "
                    "estimate and the changes most likely to bring it forward.</p>", unsafe_allow_html=True)
        go, explore = st.columns(2, gap="small")
        if go.button("Get my estimate", type="primary", icon=":material/arrow_forward:", use_container_width=True):
            st.switch_page("views/predict.py")
        explore.page_link("views/market.py", label="Explore the job market", icon=":material/work:")
    with art:
        st.markdown(ui.hero_art(), unsafe_allow_html=True)

saved = st.session_state.get("device_stored")
if saved and not st.session_state.get("profile_done") and prof.clean_answers(saved):
    st.markdown("<div class='tile t-lav' style='margin-top:1rem'><div class='head'>Welcome back</div>"
                "<div class='sub'>Your profile is saved on this device only. Continue with it, or forget it.</div></div>",
                unsafe_allow_html=True)
    keep, forget, _ = st.columns([0.28, 0.28, 0.44], gap="small")
    if keep.button("Continue", type="primary", icon=":material/arrow_forward:", use_container_width=True,
                   key="home_continue"):
        prof.restore_from_device(saved)
        st.switch_page("views/dashboard.py")
    forget.button("Forget me", icon=":material/delete:", use_container_width=True, key="home_forget",
                  on_click=prof.forget_device)

if market:
    ui.section("What employers ask for", "From real entry-level IT job ads in Sri Lanka.", "real")
    top = market["prevalence"][:3]
    e = market["emphasis"]
    f1, f2, f3 = st.columns(3, gap="medium")
    with f1:
        ui.fact_tile("blue", f"{top[0]['pct']:.0f}%",
                     f"of ads ask for <b>{ui.esc(top[0]['label'].lower())}</b>, the most requested skill.",
                     f"Next: {ui.esc(top[1]['label'].lower())} ({top[1]['pct']:.0f}%) and "
                     f"{ui.esc(top[2]['label'].lower())} ({top[2]['pct']:.0f}%).")
    with f2:
        ui.fact_tile("mint", f"{e['pct_any_soft']:.0f}%", "of ads ask for soft skills such as teamwork and communication.",
                     f"An ad names {e['mean_technical']:.1f} technical and {e['mean_soft']:.1f} soft skills on average.")
    with f3:
        sig = [t for t in market["seniority_tests"] if t["significant"]]
        if sig:
            t = sig[0]
            ui.fact_tile("peach", f"{t['intern_pct']:.0f}%",
                         f"of internship ads ask for <b>{ui.esc(t['label'].lower())}</b>, against "
                         f"{t['junior_pct']:.0f}% of junior roles.",
                         "Internship ads lean on this more than junior roles do.")

ui.section("How it works")
s1, s2, s3, s4 = st.columns(4, gap="medium")
for col, (n, tone, t, d) in zip((s1, s2, s3, s4), [
        ("1", "blue", "Tell us about you", "Your degree, experience and skills."),
        ("2", "mint", "Get your estimate", "A time range with its uncertainty, not a promise."),
        ("3", "peach", "See what helps", "What is helping you and what is holding you back."),
        ("4", "lav", "Try changes", "Preview how small changes could shift the estimate.")]):
    col.markdown(f"<div class='tile t-{tone} step'><div class='badge b-{tone}'>{n}</div><h4 style='margin:0 0 .3rem'>{t}</h4>"
                 f"<div class='sub'>{d}</div></div>", unsafe_allow_html=True)

with st.expander("Load a saved profile file", icon=":material/upload_file:"):
    uploaded = st.file_uploader("JobSpark profile (.json)", type="json", key="home_upload",
                                label_visibility="collapsed")
    if uploaded is not None:
        raw = prof.read_profile_file(uploaded.getvalue().decode("utf-8", errors="replace"))
        if raw is not None and prof.clean_answers(raw):
            if st.button("Use this profile", type="primary", key="home_use_file"):
                prof.apply_answers(raw)
                st.switch_page("views/dashboard.py")
        else:
            ui.error_card("That file is not a JobSpark profile, or it has no usable answers.")

with st.expander("For researchers: the numbers behind the model and the data sources", icon=":material/science:"):
    ui.kpis([
        (market.get("n_ads", "-"), "Real job ads analysed", f"{market.get('n_companies', '-')} companies · "
                                                            f"{market.get('date_from', '')[:7]} to {market.get('date_to', '')[:7]}", "real"),
        (len(info["cv_metrics"]), "Survival models compared", "fixed train / validation / test split, no data reuse"),
        (f"{(info.get('test_metrics') or {}).get('uno_c', 0):.2f}", "Uno's C-index (locked test set)",
         f"selected: {info['selected_model_label']}"),
        (info["training_n"], "Graduates in training data", f"{info['events']} found an IT job · {info['censored']} still looking",
         "synthetic" if info["is_synthetic"] else "real"),
    ])
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
