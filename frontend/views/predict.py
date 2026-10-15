"""My prediction: profile in, time-to-first-IT-job estimate out (via the API), with explanations and what-ifs."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import schema, wording  # option lists and labels only - no model logic
from frontend import charts, ui
from frontend.api_client import APIError, post

try:
    info = ui.model_info()
except APIError as e:
    st.error(str(e))
    st.stop()
horizon = info["horizon_months"]
AREAS = {**schema.SKILL_AREAS, **schema.SOFT_AREA}
AREA_DEFAULT = {"soft": 3.5}


def call(path: str, payload: dict):
    """Profile-based calls are NOT cached, so no profile is kept in server memory."""
    try:
        return post(path, payload), None
    except APIError as e:
        return None, str(e)


def area_items(area: str) -> list[str]:
    return [schema.SKILLS_BY_KEY[k].column for k in AREAS[area]["skills"]]


ui.hero("My prediction", "Your estimated time to a first IT job",
        "Describe your degree, experience and skills. The estimate updates as you type, with its uncertainty, what is "
        "helping or holding you back, and what could change it.", compact=True)
ui.synthetic_notice(info)

left, right = st.columns([1, 1.75], gap="large")

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
with left:
    st.markdown("##### Your profile")
    status = st.segmented_control("Status", list(wording.STATUS_LABELS), default="graduated", required=True,
                                  format_func=wording.STATUS_LABELS.get, label_visibility="collapsed")
    profile: dict = {"respondent_status": status}
    if status == "graduated":
        looking = st.toggle("I'm still looking for my first IT job",
                            help="Your estimate will then be based on graduates who were also still looking at this point.")
        if looking:
            profile["months_since_graduation"] = st.number_input("Months since you graduated", 1, horizon - 1, 3, 1)

    with st.container(border=True):
        st.markdown("**:material/school: Academic**")
        uses_class = st.toggle("My degree uses a class instead of a GPA")
        if uses_class:
            profile["degree_class"] = st.selectbox("Degree class", list(wording.DEGREE_CLASS_LABELS), index=1,
                                                   format_func=wording.DEGREE_CLASS_LABELS.get)
        else:
            profile["gpa"] = st.slider("GPA" if status == "graduated" else "Current GPA", 0.0, 4.0, 3.0, 0.01)
        profile["degree_type"] = st.selectbox("Degree programme", list(wording.DEGREE_TYPE_LABELS), index=1,
                                              format_func=wording.DEGREE_TYPE_LABELS.get)
        profile["university_type"] = st.selectbox("Type of institution", list(wording.INSTITUTION_LABELS),
                                                  format_func=wording.INSTITUTION_LABELS.get)
        # Name is for the user's reference only: the estimate depends on the type, so it is not sent to the API.
        institution = st.selectbox("Institution", [*wording.INSTITUTIONS_BY_TYPE[profile["university_type"]],
                                                   wording.INSTITUTION_OTHER],
                                   key=f"institution_{profile['university_type']}")
        if institution == wording.INSTITUTION_OTHER:
            st.text_input("Institution name", key=f"institution_other_{profile['university_type']}")

    with st.container(border=True):
        st.markdown("**:material/work_history: Experience**")
        c1, c2 = st.columns(2)
        profile["internship_months"] = c1.number_input("Internship months (total)", 0, 36, 6, 1)
        profile["project_count"] = c2.number_input("Projects outside coursework", 0, 30, 3, 1)
        profile["certifications_count"] = c1.number_input("Professional certifications", 0, 20, 1, 1)

    with st.container(border=True):
        h1, h2 = st.columns([1, 1])
        h1.markdown("**:material/psychology: Skills** <span class='muted'>(1 = none, 5 = advanced)</span>",
                    unsafe_allow_html=True)
        detailed = h2.toggle("Detailed mode", help="Rate all 17 technical and 9 soft skills individually.")
        store = st.session_state.setdefault("area_store", {a: AREA_DEFAULT.get(a, 3.0) for a in AREAS})
        if not detailed:
            for a, spec in AREAS.items():
                store[a] = st.slider(spec["label"], 1.0, 5.0, float(store[a]), 0.5, key=f"area_{a}")
                profile[f"area_{a}"] = store[a]
        else:
            st.caption("Items start from your Quick-mode area ratings.")
            for a, spec in AREAS.items():
                with st.expander(spec["label"]):
                    for col in area_items(a):
                        key = f"item_{col}"
                        if key not in st.session_state:
                            st.session_state[key] = int(round(store[a]))
                        skill = next(s for s in schema.SKILL_VOCABULARY if s.column == col)
                        profile[col] = st.slider(skill.label, 1, 5, key=key)

    with st.container(border=True):
        when = "before you completed your degree" if status == "graduated" else "so far"
        st.markdown(f"**:material/checklist: Preparation** <span class='muted'>({when})</span>",
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        profile["has_portfolio"] = "yes" if c1.toggle("Portfolio (GitHub / website)") else "no"
        profile["open_source"] = "yes" if c2.toggle("Contributed to open source") else "no"
        profile["hackathon_count"] = c1.number_input("Hackathons / competitions", 0, 30, 0, 1)
        profile["prep_start"] = c2.selectbox("Started preparing for IT jobs", list(wording.PREP_START_LABELS), index=2,
                                             format_func=wording.PREP_START_LABELS.get)
        profile["self_learning_freq"] = c1.slider("Self-learning outside coursework", 1, 5, 3,
                                                  help="1 = never, 5 = very frequently")
        profile["sdlc_familiarity"] = c2.slider("Familiarity with SDLC practices", 1, 5, 3,
                                                help="1 = no knowledge, 5 = advanced")
    # The survey insights use the same answers.
    extras = {"has_portfolio": profile["has_portfolio"] == "yes", "open_source": profile["open_source"] == "yes",
              "hackathons": profile["hackathon_count"] > 0, "self_learning": profile["self_learning_freq"] >= 4,
              "early_prep": profile["prep_start"] in ("before_university", "first_year", "second_third_year")}


def current_area(a: str) -> float:
    if not detailed:
        return float(profile[f"area_{a}"])
    vals = [profile[c] for c in area_items(a)]
    return round(2 * sum(vals) / len(vals)) / 2          # nearest 0.5 for the what-if slider


def show_items(items: list[dict], empty: str):
    if not items:
        st.markdown(f"<div class='muted'>{empty}</div>", unsafe_allow_html=True)
    for i in items:
        st.markdown(f"<div class='item'>{i['text']}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
with right:
    pred, err = call("/predict", profile)
    if err:
        st.error(err)
        st.stop()
    expl, _ = call("/explain", profile)

    hi = f"{horizon}+" if pred["high"]["more_than"] else pred["high"]["display"]
    med = f"{horizon}+" if pred["median"]["more_than"] else pred["median"]["display"]
    st.markdown(f"<div class='result'><div class='lbl'>Most likely time to a first IT job</div>"
                f"<div class='big'>{ui.esc(med)} <small>months</small></div>"
                f"<div class='rng'>Typical range {ui.esc(pred['low']['display'])} - {ui.esc(hi)} months · "
                f"{ui.esc(wording.CONFIDENCE_LABEL[pred['confidence']])}</div>"
                f"<div class='hl'>{pred['headline']}</div></div>", unsafe_allow_html=True)
    st.caption(f"{pred['confidence_reason']} Source: survey answers of {info['training_n']} Sri Lankan IT graduates.")
    with st.expander("How to read this result", icon=":material/help:"):
        st.markdown("\n".join(f"- {line}" for line in wording.READING_GUIDE))
    for c in pred["cautions"]:
        st.warning(c, icon=":material/warning:")
    now = pred["months_since_graduation"]
    st.plotly_chart(charts.range_bar([{"label": "You", "pred": pred, "color": charts.BLUE}], horizon, now=now),
                    config=ui.PLOT, key="range_main")

    if expl:
        h, b = st.columns(2, gap="medium")
        with h:
            with st.container(border=True):
                st.markdown("**:material/trending_up: What's helping**")
                show_items(expl["helping"], "Nothing stands out above the typical graduate.")
        with b:
            with st.container(border=True):
                st.markdown("**:material/trending_down: What's holding you back**")
                show_items(expl["hurting"], "Nothing stands out below the typical graduate.")
        st.caption(expl["note"])

    if pred.get("milestones"):
        st.markdown("**Chance of having found an IT job by…**")
        st.plotly_chart(charts.milestones(pred["milestones"]), config=ui.PLOT, key="milestones")
        st.caption(" ".join(m["text"] for m in pred["milestones"]))

    # -- What-if -----------------------------------------------------------------
    current = {"internship_months": profile["internship_months"], "project_count": profile["project_count"],
               "certifications_count": profile["certifications_count"],
               **{f"area_{a}": current_area(a) for a in AREAS}}
    if status == "final_year" and "gpa" in profile:
        current["gpa"] = profile["gpa"]

    def slider_key(feat: str) -> str:
        return f"wi_{feat}_{current[feat]}"                      # resets when your profile changes

    def try_suggestion(changes: dict):
        for feat in current:                                     # start from your profile, then apply this one
            st.session_state.pop(slider_key(feat), None)
        for feat, v in changes.items():
            if feat in current:
                st.session_state[slider_key(feat)] = type(current[feat])(v) if feat != "gpa" else float(v)

    sg, _ = call("/suggestions", profile)
    if sg:
        with st.container(border=True):
            st.markdown("**:material/lightbulb: Combinations to explore** <span class='muted'>(small changes linked "
                        "with finding work sooner in profiles like yours)</span>", unsafe_allow_html=True)
            if not sg["suggestions"]:
                st.caption(sg["message"])
            for i, sug in enumerate(sg["suggestions"]):
                t, b = st.columns([5, 1], vertical_alignment="center")
                t.markdown(f"<div class='item'>{sug['text']}</div>", unsafe_allow_html=True)
                b.button("Try this", key=f"try_{i}", on_click=try_suggestion, args=(sug["changes"],),
                         help="Sets the what-if sliders below to this combination.")
            st.caption(sg["note"])

    with st.container(border=True):
        st.markdown("**:material/tune: What if…** <span class='muted'>(things you can change)</span>",
                    unsafe_allow_html=True)
        new, skipped = {}, []
        cols = st.columns(3)
        for i, (feat, value) in enumerate(current.items()):
            spec = schema.ACTIONABLE_FEATURES[feat]
            lo, hi_ = spec.min, spec.max
            if spec.max_change is not None:
                lo, hi_ = max(lo, value - spec.max_change), min(hi_, value + spec.max_change)
            if not (lo <= value <= hi_):
                skipped.append(spec.label)
                continue
            step = spec.step
            cast = int if float(step).is_integer() and feat != "gpa" and not feat.startswith("area_") else float
            st.session_state.setdefault(slider_key(feat), cast(value))   # "Try this" may have set it already
            new[feat] = cols[i % 3].slider(spec.label, cast(lo), cast(hi_), step=cast(step), key=slider_key(feat))
        changes = {k: v for k, v in new.items() if abs(float(v) - float(current[k])) > 1e-9}
        if skipped:
            st.caption("Outside the what-if range: " + ", ".join(skipped))
        if changes:
            wi, err = call("/whatif", {"profile": profile, "changes": changes})
            if err:
                st.warning(err)
            else:
                st.plotly_chart(charts.range_bar([{"label": "Now", "pred": wi["original"], "color": charts.BLUE},
                                                  {"label": "With changes", "pred": wi["changed"],
                                                   "color": charts.ORANGE}], horizon, now=now),
                                config=ui.PLOT, key="range_whatif")
                st.markdown(f"**{wi['message']}**")
                for c in wi["cautions"]:
                    st.warning(c, icon=":material/warning:")
        else:
            st.caption(wording.WHATIF_PROMPT)


# ---------------------------------------------------------------------------
# Ask about your result (grounded assistant - no language model, nothing stored)
# ---------------------------------------------------------------------------
CHAT_STARTERS = ["Why is my estimate like this?", "What if I do a 6-month internship?",
                 "Which skills should I learn?", "Can I trust this estimate?"]
history = st.session_state.setdefault("chat_history", [])
# Collapsed until used, so the estimate stays the focus of the page (usability study feedback).
with st.expander("Ask about your result", icon=":material/chat:", expanded=bool(history)):
    st.caption("Answers come only from this system's own results, using your current profile - it is not a general "
               "chatbot. Nothing you type is stored.")
    for role, text in history[-12:]:
        with st.chat_message(role):
            st.markdown(text)
    asked = None
    for col, q in zip(st.columns(len(CHAT_STARTERS)), CHAT_STARTERS):
        if col.button(q, key=f"chat_starter_{q}", use_container_width=True):
            asked = q
    typed = st.chat_input("Ask about your estimate, e.g. \"What if I get two certifications?\"", key="chat_input")
    asked = typed or asked
    if asked:
        answer, err = call("/chat", {"profile": profile, "message": asked})
        history += [("user", asked), ("assistant", err or answer["reply"])]
        st.rerun()
    if history and st.button("Clear chat", key="chat_clear"):
        st.session_state["chat_history"] = []
        st.rerun()


# ---------------------------------------------------------------------------
# Compared to employers / peers
# ---------------------------------------------------------------------------
ui.section("How your profile compares")
t_emp, t_peer, t_say = st.tabs([":material/business_center: Compared to employers", ":material/groups: Compared to peers",
                                 ":material/forum: What employers and graduates say"])

with t_emp:
    mk, err = call("/market-alignment", profile)
    if err:
        st.error(err)
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
    pb, err = call("/peer-benchmark", profile)
    if err:
        st.error(err)
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
    ins, err = call("/insights", {"profile": profile, "extras": extras})
    if err:
        st.error(err)
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
            st.markdown("**What employers expect to grow**")
            show_items([{"text": g["text"]} for g in ins["growing"]], "")
        with c2:
            st.markdown("**What graduates who found a job quickly had done**")
            for t in ins["fast_finders"]:
                flag = (" <b>- worth considering</b>" if t["suggest"] else
                        (" <span class='muted'>(you: yes)</span>" if t["you"] else ""))
                star = "" if t["significant"] else " <span class='muted'>(difference not significant)</span>"
                st.markdown(f"<div class='item'>{ui.esc(t['text'])}{star}{flag}</div>", unsafe_allow_html=True)
            st.markdown("**How employers rate each hiring factor** <span class='muted'>(1-5)</span>",
                        unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([{"Factor": v["label"], "Employers (mean)": v["mean"], "You": v["you"] or ""}
                                       for v in ins["employers_value"]]), hide_index=True)
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            st.markdown("**Where employed graduates searched**")
            show_items([{"text": f"{r['label']} - {r['pct']:.0f}%"} for r in ins["where_to_search"]], "")
        with c2:
            st.markdown("**Where graduates learn**")
            show_items([{"text": f"{r['label']} - {r['pct']:.0f}%"} for r in ins["where_to_learn"]], "")
        with c3:
            st.markdown("**Most common challenges**")
            show_items([{"text": f"{r['label']} - {r['pct']:.0f}%"} for r in ins["common_challenges"]], "")
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

st.caption(info["disclaimer"] + " " + info["associational_note"])
