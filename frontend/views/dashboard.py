"""My dashboard: the estimate and the few things a graduate needs first. Detail lives on the Improve, Compare and Ask pages."""
from __future__ import annotations

import streamlit as st

from core import wording  # labels only - no model logic
from frontend import charts, profile as prof, ui
from frontend.api_client import APIError

S = st.session_state
try:
    info = ui.model_info()
except APIError as e:
    ui.error_card(str(e), "Check that the API is running (uvicorn api.main:app --port 8000), then reload.")
    st.stop()
horizon = info["horizon_months"]
profile = prof.require()

with st.spinner("Calculating your estimate…"):
    pred, err = prof.call("/predict", profile)
    if err:
        ui.error_card(err)
        st.stop()
    expl, _ = prof.call("/explain", profile)
    sg, _ = prof.call("/suggestions", profile)
    mk, _ = prof.call("/market-alignment", profile)

title, edit = st.columns([3, 1], vertical_alignment="center")
title.header("Your results", anchor=False)
if edit.button("Edit profile", icon=":material/edit:", use_container_width=True, key="dash_edit"):
    S["flow_step"] = 1
    st.switch_page("views/predict.py")
ui.synthetic_notice(info)

hi = f"{horizon}+" if pred["high"]["more_than"] else pred["high"]["display"]
med = f"{horizon}+" if pred["median"]["more_than"] else pred["median"]["display"]
now = pred["months_since_graduation"]

main, side = st.columns([1.15, 1], gap="large")
with main:
    ui.estimate_card(med, pred["low"]["display"], hi, wording.CONFIDENCE_LABEL[pred["confidence"]], pred["headline"])
    st.caption(f"{pred['confidence_reason']} Source: survey answers of {info['training_n']} Sri Lankan IT graduates.")
    for c in pred["cautions"]:
        st.warning(c, icon=":material/warning:")
    st.plotly_chart(charts.range_bar([{"label": "You", "pred": pred, "color": charts.BLUE}], horizon, now=now),
                    config=ui.PLOT, key="range_main")
with side:
    if pred.get("milestones"):
        with st.container(border=True):
            st.markdown("**Chance of having found an IT job by…**")
            st.plotly_chart(charts.milestones(pred["milestones"]), config=ui.PLOT, key="milestones")
            with st.expander("What these numbers mean"):
                st.markdown("\n".join(f"- {m['text']}" for m in pred["milestones"]))

glance = bool(pred.get("milestones")) or bool(mk)
if glance:
    ui.section("Your results at a glance", "Hover or tap a chart for the exact numbers.")
    g1, g2, g3 = st.columns(3, gap="medium")
    with g1:
        if pred.get("milestones"):
            with st.container(border=True):
                st.markdown("**Chance of a job by…**")
                ms = {m["months"]: m for m in pred["milestones"]}
                guess = pred["median"].get("months") or max(ms)
                st.session_state.setdefault("dash_month", min(ms, key=lambda k: abs(k - guess)))
                chosen = st.segmented_control("Months after graduating", list(ms), format_func=lambda k: f"{k} mo",
                                              key="dash_month", required=True, label_visibility="collapsed")
                m = ms[chosen]
                st.plotly_chart(charts.job_donut(m), config=ui.PLOT, key="donut_job")
                st.caption(f"About {m['employed_pct']:.0f}% of graduates with profiles like yours had found an IT job "
                           f"by {m['months']} months (likely {m['low_pct']:.0f}-{m['high_pct']:.0f}%).")
    if mk:
        with g2:
            with st.container(border=True):
                st.markdown("**Your skills vs what employers rate**")
                st.plotly_chart(charts.skill_radar(mk["skills"]), config=ui.PLOT, key="radar_skills")
                st.caption("Ratings out of 5 for the 8 most requested skills.")
        with g3:
            with st.container(border=True):
                st.markdown("**Where employers expect more**")
                gaps = charts.gap_bars(mk["skills"])
                if gaps is None:
                    st.markdown("<div class='muted'>Your ratings are at or above what employers rate for every skill.</div>",
                                unsafe_allow_html=True)
                else:
                    st.plotly_chart(gaps, config=ui.PLOT, key="bars_gaps")
                    st.caption("How far each skill is above your own rating, on the 1-5 scale.")

if expl:
    h, b = st.columns(2, gap="medium")
    with h:
        ui.driver_card("helping", "What's helping you", [i["text"] for i in expl["helping"][:3]],
                       "Nothing stands out above the typical graduate.")
    with b:
        ui.driver_card("holding", "What's holding you back", [i["text"] for i in expl["hurting"][:3]],
                       "Nothing stands out below the typical graduate.")
    if len(expl["helping"]) > 3 or len(expl["hurting"]) > 3:
        with st.expander("See every factor"):
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                st.markdown("**Helping**")
                prof.show_items(expl["helping"], "Nothing stands out above the typical graduate.")
            with c2:
                st.markdown("**Holding you back**")
                prof.show_items(expl["hurting"], "Nothing stands out below the typical graduate.")
    st.caption(expl["note"])

if sg and sg["suggestions"]:
    ui.section("Your next steps", "Small changes linked with finding work sooner in profiles like yours.")
    with st.container(border=True):
        for i, sug in enumerate(sg["suggestions"][:3]):
            t, btn = st.columns([4, 1], vertical_alignment="center")
            t.markdown(f"<div class='item'>{sug['text']}</div>", unsafe_allow_html=True)
            if btn.button("Try it", key=f"dash_try_{i}", use_container_width=True,
                          help="Opens Improve with this change applied."):
                S["pending_changes"] = sug["changes"]
                st.switch_page("views/improve.py")
        st.caption(sg["note"])

ui.section("Go further")
t1, t2, t3, t4 = st.columns(4, gap="medium")
with t1:
    ui.nav_tile("views/improve.py", "Improve", "Preview what-if changes", "target")
with t2:
    ui.nav_tile("views/compare.py", "Compare", "Employers and peers", "scale")
with t3:
    ui.nav_tile("views/market.py", "Job market", "What ads ask for", "case")
with t4:
    ui.nav_tile("views/ask.py", "Ask", "Questions about your result", "chat")

S.setdefault("remember_device", False)
with st.expander("Save my profile", icon=":material/save:"):
    st.toggle("Remember me on this device", key="remember_device", on_change=prof.remember_changed,
              help="Keeps your answers in this browser only, so you can come back without re-entering them. "
                   "Switch it off, or use 'Forget me' on Home, to remove them.")
    st.download_button("Download my profile", prof.profile_file(), file_name="jobspark-profile.json",
                       mime="application/json", icon=":material/download:")
    st.caption("Your answers never go to a server: they stay in this browser or in the file you download. "
               "Load the file again from the Home page, on any device.")

with st.expander("How to read this result", icon=":material/help:"):
    st.markdown("\n".join(f"- {line}" for line in wording.READING_GUIDE))
st.caption(info["disclaimer"] + " " + info["associational_note"])
