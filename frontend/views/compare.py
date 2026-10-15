"""Compare: how the profile lines up with employer demand and with graduates in similar groups."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import wording  # labels only - no model logic
from frontend import charts, charts_study as cs, profile as prof, ui
from frontend.api_client import APIError

try:
    info = ui.model_info()
except APIError as e:
    ui.error_card(str(e), "Check that the API is running (uvicorn api.main:app --port 8000), then reload.")
    st.stop()
horizon = info["horizon_months"]
profile = prof.require()
call, show_items = prof.call, prof.show_items

st.header("How your profile compares", anchor=False)
ui.synthetic_notice(info)
t_emp, t_peer, t_say = st.tabs([":material/business_center: Employers", ":material/groups: Peers",
                                 ":material/forum: What people say"])

with t_emp:
    with st.spinner("Comparing with employer demand…"):
        mk, err = call("/market-alignment", profile)
    if err:
        ui.error_card(err)
    else:
        c1, c2 = st.columns([1.5, 1], gap="large")
        with c1:
            st.markdown("**The 10 most in-demand skills**")
            st.plotly_chart(charts.skill_gap(mk["skills"]), config=ui.PLOT, key="gap")
        with c2:
            st.markdown("**Biggest gaps for you**")
            if mk["top_gaps"]:
                for g in mk["top_gaps"]:
                    links = " · ".join(f"<a href='{r['url']}' target='_blank'>{r['title']}</a> "
                                       f"<span class='muted'>({r['provider']})</span>" for r in g.get("resources", []))
                    st.markdown(f"<div class='item'>{g['text']}" + (f"<br><span class='muted'>Free resources:</span> "
                                f"{links}" if links else "") + "</div>", unsafe_allow_html=True)
            else:
                st.markdown(mk["message"])
            st.caption(mk["note"])
        with st.expander("All skills (table)"):
            st.dataframe(pd.DataFrame(mk["skills"])[["label", "your_rating", "employer_importance", "ads_pct"]]
                         .rename(columns={"label": "Skill", "your_rating": "You (1-5)",
                                          "employer_importance": "Employers (1-5)", "ads_pct": "% of job ads"}),
                         hide_index=True)

with t_peer:
    with st.spinner("Finding similar graduates…"):
        pb, err = call("/peer-benchmark", profile)
    if err:
        ui.error_card(err)
    else:
        c1, c2 = st.columns([1.3, 1], gap="large")
        with c1:
            st.markdown("**How long it took graduates in your groups**")
            st.plotly_chart(charts.peer_medians(pb["groups"], horizon), config=ui.PLOT, key="peer")
        with c2:
            for g in pb["groups"]:
                st.markdown(f"<div class='item'><b>{g['group_label']} - {g['level_label']}</b> "
                            f"<span class='muted'>(n = {g['n']})</span><br>{g['text']}</div>", unsafe_allow_html=True)
        st.caption(pb["note"] + " " + wording.PEERS_UNADJUSTED_NOTE)

with t_say:
    with st.spinner("Reading the survey results…"):
        ins, err = call("/insights", {"profile": profile, "extras": prof.survey_extras(profile)})
    if err:
        ui.error_card(err)
    else:
        if ins["is_synthetic"]:
            st.markdown("<div class='notice'>⚠️ <div>The questionnaire answers behind this tab are "
                        "<b>synthetic (computer-generated)</b> until the real survey is collected.</div></div>",
                        unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**Weak spots employers often see - and you rated low**")
            show_items([{"text": w["text"]} for w in ins["watch_outs"]],
                       "None of your weaker skills is a common employer complaint.")
        with c2:
            st.markdown("**What graduates who found a job quickly had done**")
            for t in ins["fast_finders"]:
                flag = (" <b>- worth considering</b>" if t["suggest"] else
                        (" <span class='muted'>(you: yes)</span>" if t["you"] else ""))
                star = "" if t["significant"] else " <span class='muted'>(difference not significant)</span>"
                st.markdown(f"<div class='item'>{ui.esc(t['text'])}{star}{flag}</div>", unsafe_allow_html=True)

        with st.expander("More from the graduate and employer surveys"):
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown("**What employers expect to grow**")
                show_items([{"text": g["text"]} for g in ins["growing"]], "")
            with c2:
                st.markdown("**How employers rate each hiring factor** <span class='muted'>(1-5)</span>",
                            unsafe_allow_html=True)
                st.dataframe(pd.DataFrame([{"Factor": v["label"], "Employers (mean)": v["mean"], "You": v["you"] or ""}
                                           for v in ins["employers_value"]]), hide_index=True)
            c1, c2, c3 = st.columns(3, gap="medium")
            with c1:
                st.markdown("**Where employed graduates searched**")
                if ins["where_to_search"]:
                    st.plotly_chart(cs.pct_bars([(r["label"], r["pct"]) for r in ins["where_to_search"]], color=cs.AQUA),
                                    config=ui.PLOT, key="say_search")
            with c2:
                st.markdown("**Where graduates learn**")
                if ins["where_to_learn"]:
                    st.plotly_chart(cs.pct_bars([(r["label"], r["pct"]) for r in ins["where_to_learn"]], color=cs.BLUE),
                                    config=ui.PLOT, key="say_learn")
            with c3:
                st.markdown("**Most common challenges**")
                if ins["common_challenges"]:
                    st.plotly_chart(cs.pct_bars([(r["label"], r["pct"]) for r in ins["common_challenges"]], color=cs.ORANGE),
                                    config=ui.PLOT, key="say_challenges")
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown("**Gaps graduates underestimate**")
                show_items([{"text": u["text"]} for u in ins["underestimated_gaps"]],
                           "Graduates and employers broadly agree on the main skill gaps.")
                st.markdown("**Certifications - what graduates hold and where demand is growing**")
                show_items([{"text": c["text"]} for c in ins["certifications"]], "")
            with c2:
                st.markdown("**What interviewers told graduates they lacked**")
                show_items([{"text": f"{r['label']} - {r['pct']:.0f}% of graduates who had interviews"}
                            for r in ins["interviewers_said_lacking"]], "")
                if ins["interview_feedback"]:
                    st.caption("Areas most often raised in interview feedback: " + ", ".join(
                        f"{r['label'].lower()} ({r['pct']:.0f}%)" for r in ins["interview_feedback"]) + ".")
                st.markdown("**Choosing a project**")
                roles = ", ".join(f"{r['label']} ({r['pct']:.0f}%)" for r in ins["roles_hired"])
                areas = ", ".join(f"{r['label'].lower()} ({r['pct']:.0f}%)" for r in ins["project_areas"])
                show_items([{"text": f"Roles employers hire for most: {roles}."},
                            {"text": f"Where graduates have built projects: {areas}."}], "")
        st.caption(ins["note"])

st.page_link("views/dashboard.py", label="Back to my dashboard", icon=":material/arrow_back:")
st.caption(info["disclaimer"] + " " + info["associational_note"])
