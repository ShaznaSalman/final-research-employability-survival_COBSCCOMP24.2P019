"""Career path exploration: compare self-rated skills with the role profiles in the local job-ad study."""
from __future__ import annotations

import streamlit as st

from core import wording
from core.pathways import role_pathways
from frontend import profile as prof, ui
from frontend.api_client import APIError


def _select_career_role(role: str) -> None:
    st.session_state["career_target_role"] = role


ui.hero("Career path", "Explore roles that fit your skills",
        "Connect your overall time-to-first-job estimate and personalized guidance with entry-level role profiles "
        "that overlap with your self-rated skills.", compact=True, art=ui.career_path_art())

try:
    info = ui.model_info()
except APIError as e:
    ui.error_card(str(e), "Check that the API is running (uvicorn api.main:app --port 8000), then reload.")
    st.stop()

profile = prof.require()
job_market = ui.fetch("/job-market")
if not job_market:
    st.stop()

with st.spinner("Connecting your estimate and career guidance…"):
    prediction, err = prof.call("/predict", profile)
    if err:
        ui.error_card(err)
        st.stop()
    explanation, err = prof.call("/explain", profile)
    if err:
        ui.error_card(err)
        st.stop()
    suggestions, err = prof.call("/suggestions", profile)
    if err:
        ui.error_card(err)
        st.stop()

ui.synthetic_notice(info)

try:
    pathways = role_pathways(profile, job_market)
except ValueError as e:
    ui.error_card(str(e))
    st.stop()

recommendations = pathways["recommendations"]
roles = pathways["roles"]
if not roles:
    st.info("There is not enough skill information in your profile to compare roles.")
    st.stop()

if recommendations:
    ui.section("Role profiles closest to your current ratings",
               "Ranked by your weighted average self-rating in skills mentioned in each role's job ads. "
               "This is separate from your overall first-job estimate and is not a hiring probability.")
    cards = st.columns(len(recommendations), gap="medium")
    for index, (card, recommendation) in enumerate(zip(cards, recommendations)):
        with card:
            with st.container(border=True):
                if index == 0:
                    st.markdown("<span class='best-match-badge'><span class='best-match-star'>★</span>"
                                "Best match</span>", unsafe_allow_html=True)
                st.markdown(f"**{recommendation['role']}**")
                st.metric("Role-skill rating", f"{recommendation['match_rating']:.1f} / 5")
                st.caption(f"Based on {recommendation['ads']} collected ads.")
                st.button("Explore this role", key=f"career_recommend_{index}", use_container_width=True,
                          on_click=_select_career_role, args=(recommendation["role"],))

ui.section("Explore role skills", "Choose a recommendation above to see its skill gaps here, or select another role.")
role_names = [item["role"] for item in roles]
default_role = st.session_state.get("career_target_role", recommendations[0]["role"] if recommendations else role_names[0])
if default_role not in role_names:
    default_role = role_names[0]
target = st.selectbox("Target role family", role_names, index=role_names.index(default_role),
                      key="career_target_role")
selected = next(item for item in roles if item["role"] == target)
ui.section(f"Skills for {selected['role']}")
st.markdown(f"**{selected['role']}** · role-skill rating **{selected['match_rating']:.1f} / 5** · "
            f"based on **{selected['ads']}** collected ads")
if selected["ads"] < 20:
    st.warning("This role has a small ad sample, so its skill profile may be less representative.")
if selected["gaps"]:
    st.markdown("**Skills to consider strengthening**")
    st.caption("These are skills mentioned in this role's ads where your self-rating is below 4/5. "
               "Higher ad share and lower ratings appear first.")
    st.dataframe(
        [{"Skill": item["label"], "Your rating (1-5)": item["your_rating"],
          "Ads mentioning skill": f"{item['ads_pct']:.0f}%"} for item in selected["gaps"]],
        hide_index=True,
        use_container_width=True,
    )
else:
    st.success("No role-specific skill gaps stood out from the ratings you provided.")

horizon = prediction["horizon_months"]
median = f"{horizon}+" if prediction["median"]["more_than"] else prediction["median"]["display"]
high = f"{horizon}+" if prediction["high"]["more_than"] else prediction["high"]["display"]
ui.section("Your overall first-job estimate",
           "This estimate is for finding a first IT job across roles generally. It is not a separate prediction "
           "for any role shown below.")
ui.estimate_card(median, prediction["low"]["display"], high,
                 wording.CONFIDENCE_LABEL[prediction["confidence"]], prediction["headline"])
st.caption(prediction["confidence_reason"]
           + " Source: survey answers from Sri Lankan IT graduates.")
for caution in prediction["cautions"]:
    st.warning(caution, icon=":material/warning:")

explain_cols = st.columns(2, gap="medium")
with explain_cols[0]:
    ui.driver_card("helping", "What's helping your overall estimate",
                   [item["text"] for item in explanation["helping"][:3]],
                   "Nothing stands out above the typical graduate.")
with explain_cols[1]:
    ui.driver_card("holding", "What's holding your overall estimate back",
                   [item["text"] for item in explanation["hurting"][:3]],
                   wording.LOCAL_NO_HURTING)
st.caption(explanation["note"])

if suggestions["suggestions"]:
    ui.section("Personalized ways to improve your overall estimate",
               "These model-based suggestions are for the general first-job estimate; they are not role-specific.")
    for index, suggestion in enumerate(suggestions["suggestions"]):
        with st.container(border=True):
            text_col, action_col = st.columns([4, 1], vertical_alignment="center")
            text_col.markdown(f"<div class='item'>{suggestion['text']}</div>", unsafe_allow_html=True)
            if action_col.button("Try it", type="primary", key=f"career_try_suggestion_{index}", use_container_width=True,
                                 help="Preview this change on Improve my chances."):
                st.session_state["pending_changes"] = suggestion["changes"]
                st.switch_page("views/improve.py")
    st.caption(suggestions["note"])
elif suggestions.get("message"):
    st.info(suggestions["message"])

st.markdown(
    "Role fit reflects self-ratings and job advertisements collected for this research prototype. It does not "
    "measure your full qualifications, predict role-specific hiring outcomes, or guarantee that a role is right "
    "for you. Job ads are a limited snapshot; requirements vary by employer."
)
st.page_link("views/market.py", label="Explore all role profiles in Job Market", icon=":material/work:")
