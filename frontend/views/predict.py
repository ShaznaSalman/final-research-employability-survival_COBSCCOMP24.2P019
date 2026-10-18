"""My profile: four short steps (about you, experience, skills, preparation), then on to the dashboard."""
from __future__ import annotations

import streamlit as st

from core import schema, wording  # option lists and labels only - no model logic
from frontend import profile as prof, ui
from frontend.api_client import APIError, post

try:
    info = ui.model_info()
except APIError as e:
    ui.error_card(str(e), "Check that the API is running (uvicorn api.main:app --port 8000), then reload.")
    st.stop()
horizon = info["horizon_months"]
S = st.session_state
STEPS = ["About you", "Experience", "Skills", "Preparation"]
prof.keep_state()
ready, val, AREAS, DEFAULTS, area_items = prof.ready, prof.val, prof.AREAS, prof.DEFAULTS, prof.area_items


def go(step: int) -> None:
    S["flow_step"] = step


# ---------------------------------------------------------------------------
# The four steps
# ---------------------------------------------------------------------------
def step_about() -> None:
    st.segmented_control("I am", list(wording.STATUS_LABELS), required=True, format_func=wording.STATUS_LABELS.get,
                         key=ready("p_status"))
    status = S["p_status"]
    if status == "graduated":
        st.toggle("I'm still looking for my first IT job", key=ready("p_looking"),
                  help="Your estimate will then be based on graduates who were also still looking at this point.")
        if S["p_looking"]:
            st.number_input("Months since you graduated", min_value=1, max_value=horizon - 1, step=1, key=ready("p_months"))
    st.divider()
    st.toggle("My degree uses a class instead of a GPA", key=ready("p_uses_class"))
    if S["p_uses_class"]:
        st.selectbox("Degree class", list(wording.DEGREE_CLASS_LABELS), format_func=wording.DEGREE_CLASS_LABELS.get,
                     key=ready("p_degree_class"))
    else:
        st.slider("GPA" if status == "graduated" else "Current GPA", 0.0, 4.0, step=0.01, key=ready("p_gpa"))
    st.selectbox("Degree programme", list(wording.DEGREE_TYPE_LABELS), format_func=wording.DEGREE_TYPE_LABELS.get,
                 key=ready("p_degree_type"))
    st.selectbox("Type of institution", list(wording.INSTITUTION_LABELS), format_func=wording.INSTITUTION_LABELS.get,
                 key=ready("p_university_type"))
    # Name is for the user's reference only: the estimate depends on the type, so it is not sent to the API.
    utype = S["p_university_type"]
    institution = st.selectbox("Institution", [*wording.INSTITUTIONS_BY_TYPE[utype], wording.INSTITUTION_OTHER],
                               key=f"p_institution_{utype}")
    if institution == wording.INSTITUTION_OTHER:
        st.text_input("Institution name", key=f"p_institution_other_{utype}")


def step_experience() -> None:
    c1, c2 = st.columns(2)
    c1.number_input("Internship months (total)", min_value=0, max_value=36, step=1, key=ready("p_internship"))
    c2.number_input("Projects outside coursework", min_value=0, max_value=30, step=1, key=ready("p_projects"))
    c1.number_input("Professional certifications", min_value=0, max_value=20, step=1, key=ready("p_certs"))


def step_skills() -> None:
    st.caption("Rate yourself from 1 (none) to 5 (advanced).")
    st.toggle("Rate each skill individually", key=ready("p_detailed"),
              help="Rate all 17 technical and 9 soft skills one by one instead of by area.")
    if not S["p_detailed"]:
        for a, spec in AREAS.items():
            ui.rating(spec["label"], key=f"p_area_{a}", default=DEFAULTS[f"p_area_{a}"])
    else:
        st.caption("Each skill starts from your rating for its area.")
        for a, spec in AREAS.items():
            with st.expander(spec["label"]):
                for col in area_items(a):
                    skill = next(s for s in schema.SKILL_VOCABULARY if s.column == col)
                    ui.rating(skill.label, key=f"p_item_{col}", default=int(round(val(f"p_area_{a}"))))


def step_preparation() -> None:
    when = "before you completed your degree" if val("p_status") == "graduated" else "so far"
    st.caption(f"What you had done {when}.")
    c1, c2 = st.columns(2)
    c1.toggle("Portfolio (GitHub / website)", key=ready("p_portfolio"))
    c2.toggle("Contributed to open source", key=ready("p_oss"))
    c1.number_input("Hackathons / competitions", min_value=0, max_value=30, step=1, key=ready("p_hack"))
    c2.selectbox("Started preparing for IT jobs", list(wording.PREP_START_LABELS),
                 format_func=wording.PREP_START_LABELS.get, key=ready("p_prep"))
    ui.rating("Self-learning outside coursework", key="p_selflearn", default=DEFAULTS["p_selflearn"],
              help="1 = never, 5 = very frequently")
    ui.rating("Familiarity with SDLC practices", key="p_sdlc", default=DEFAULTS["p_sdlc"],
              help="1 = no knowledge, 5 = advanced")


STEP_VIEWS = [step_about, step_experience, step_skills, step_preparation]
STEP_TITLES = ["Tell us about you", "What have you done so far?", "How would you rate your skills?",
               "How did you prepare?"]


TIPS = [
    "Pick the status that fits you today. Final-year students get a prediction too, based on graduates in a similar position.",
    "Count internships and projects you have finished, not ones you plan to do. Small, completed projects count.",
    "Rate yourself honestly. Your ratings are compared with what real job ads ask for on the dashboard.",
    "These habits are linked with finding work sooner in the survey data. They are patterns, not guarantees.",
]


def profile_so_far() -> list[tuple[str, str]]:
    """Short label/value pairs for the answers given up to the current step (read live from the saved answers)."""
    items = [("Status", wording.STATUS_LABELS[val("p_status")])]
    items.append(("Degree class", wording.DEGREE_CLASS_LABELS[val("p_degree_class")]) if val("p_uses_class")
                 else ("GPA", f"{val('p_gpa'):.2f}"))
    items += [("Degree", wording.DEGREE_TYPE_LABELS[val("p_degree_type")]),
              ("Institution", wording.INSTITUTION_LABELS[val("p_university_type")])]
    if step >= 2:
        items += [("Internship", f"{val('p_internship')} mo"), ("Projects", str(val("p_projects"))),
                  ("Certifications", str(val("p_certs")))]
    if step >= 3:
        if val("p_detailed"):
            ratings = [S.get(f"{prof.ITEM_PREFIX}{c}", 3) for c in prof.ITEM_COLUMNS]
        else:
            ratings = [val(f"p_area_{a}") for a in AREAS]
        items.append(("Skills average", f"{sum(ratings) / len(ratings):.1f} / 5"))
    if step >= 4:
        items += [("Portfolio", "yes" if val("p_portfolio") else "no"), ("Open source", "yes" if val("p_oss") else "no"),
                  ("Hackathons", str(val("p_hack")))]
    return items

try:                                          # ?step=1..4 opens a given step; handy for links and checks
    S.setdefault("flow_step", min(max(int(st.query_params.get("step", 1)), 1), len(STEPS)))
except ValueError:
    S.setdefault("flow_step", 1)
step = min(S["flow_step"], len(STEPS))

main_col, side_col = st.columns([1.7, 1], gap="large")
with main_col:
    ui.progress(STEPS, step)
    st.caption(f"Step {step} of {len(STEPS)}")
    st.header(STEP_TITLES[step - 1], anchor=False)
    ui.synthetic_notice(info)
    with st.container(border=True):
        STEP_VIEWS[step - 1]()
    if step == len(STEPS):
        st.info(
            "Your profile is ready. Next, JobSpark will calculate your estimated time to a first IT job, "
            "explain what is influencing it, and show career paths and practical next steps.",
            icon=":material/analytics:",
        )
    back, nxt = st.columns(2)
    back.button("Back", icon=":material/arrow_back:", on_click=go, args=(step - 1,), disabled=step == 1,
                use_container_width=True, key=f"flow_back_{step}")
    if step < len(STEPS):
        nxt.button(f"Next: {STEPS[step]}", type="primary", icon=":material/arrow_forward:", on_click=go,
                   args=(step + 1,), use_container_width=True, key=f"flow_next_{step}")
    elif nxt.button("Calculate my results", type="primary", icon=":material/arrow_forward:", use_container_width=True,
                    key=f"flow_next_{step}"):
        S["profile_done"] = True
        st.switch_page("views/dashboard.py")
    st.caption("Your answers stay in this browser session only. " + info["disclaimer"])

with side_col:
    with st.container(key="sidepanel"):
        ui.side_summary("Your profile so far", profile_so_far())
        ui.tip(TIPS[step - 1])
