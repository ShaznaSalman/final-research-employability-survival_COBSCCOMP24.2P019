"""Optional profile history and progress tracking through the isolated profile service."""
from __future__ import annotations

import streamlit as st

from core.pathways import role_pathways
from frontend import device_store, profile as prof, profile_store_client as saved, ui

S = st.session_state
TOKEN_KEY = "profile_store_token"
PROFILE_AVATARS = ("🙂", "😀", "😎", "🤓", "🧑‍💻", "👩‍💻", "👨‍💻", "🚀", "🌱", "⭐", "🐼", "🦊")

st.header("Saved profile & progress", anchor=False)
st.markdown("Save dated profile snapshots, choose a role goal, and track your next steps across devices.")
logout_notice = S.pop("account_logout_notice", None)
if logout_notice:
    st.warning(logout_notice)

try:
    health = saved.get("/health")
except saved.ProfileStoreError as e:
    ui.error_card(str(e))
    st.stop()

if not health["phone_sign_in_enabled"]:
    st.warning(
        "Phone sign-in is not configured yet. Add your SMSAPI.lk credentials to the profile service's local "
        ".env file, then restart the service."
    )
    st.info(
        "Your existing profile and device-only save continue to work. Profile snapshots and progress are stored "
        "in the separate database only after you verify your phone."
    )
    with st.expander("Why is phone sign-in disabled?"):
        st.markdown(
            "The separate profile service is isolated from the prediction API. It enables OTP only when an HTTPS "
            "SMS endpoint, API key, and sender ID are configured. Never paste your API key into chat or commit it."
        )
    st.stop()

token = S.get(TOKEN_KEY)
if not token:
    with st.form("profile_store_phone"):
        phone = st.text_input("Sri Lankan mobile number", placeholder="+94 77 123 4567")
        send_code = st.form_submit_button("Send sign-in code")
    if send_code:
        try:
            saved.post("/auth/request-code", {"phone": phone})
            S["profile_store_pending_phone"] = phone
            st.success("A verification code was sent. It expires in five minutes.")
        except saved.ProfileStoreError as e:
            st.error(str(e))

    pending_phone = S.get("profile_store_pending_phone")
    if pending_phone:
        with st.form("profile_store_verify"):
            code = st.text_input("Six-digit verification code", max_chars=6)
            verify = st.form_submit_button("Verify and sign in")
        if verify:
            try:
                response = saved.post("/auth/verify", {"phone": pending_phone, "code": code})
                S[TOKEN_KEY] = response["access_token"]
                device_store.sync_auth_token(response["access_token"])
                S.pop("profile_store_pending_phone", None)
                st.rerun()
            except saved.ProfileStoreError as e:
                st.error(str(e))
    st.stop()

try:
    account = saved.get("/me", token)
except saved.ProfileStoreError as e:
    S.pop(TOKEN_KEY, None)
    S.pop("account_display_name", None)
    S.pop("account_avatar_emoji", None)
    device_store.sync_auth_token(None)
    st.error(f"Your sign-in is no longer valid: {e}")
    st.stop()

if (S.get("account_display_name") != account["display_name"]
        or S.get("account_avatar_emoji") != account["avatar_emoji"]):
    S["account_display_name"] = account["display_name"]
    S["account_avatar_emoji"] = account["avatar_emoji"]
    st.rerun()

st.subheader("Profile settings", anchor="profile-settings")
with st.form("profile_identity_settings"):
    display_name = st.text_input(
        "Name or username",
        value=account["display_name"],
        max_chars=40,
        help="This name is shown on your profile and is not used for predictions.",
    )
    avatar_emoji = st.selectbox(
        "Choose an avatar",
        PROFILE_AVATARS,
        index=PROFILE_AVATARS.index(account["avatar_emoji"]),
    )
    save_identity = st.form_submit_button("Save profile appearance")
if save_identity:
    try:
        updated_account = saved.patch(
            "/me",
            {"display_name": display_name, "avatar_emoji": avatar_emoji},
            token,
        )
        S["account_display_name"] = updated_account["display_name"]
        S["account_avatar_emoji"] = updated_account["avatar_emoji"]
        st.success("Profile appearance saved to your account.")
        st.rerun()
    except saved.ProfileStoreError as e:
        st.error(str(e))

heading, logout = st.columns([4, 1], vertical_alignment="center")
heading.caption(f"Signed in as {account['phone']}. Saved records expire after one year.")
if logout.button("Sign out", key="profile_store_logout"):
    try:
        saved.post("/auth/logout", {}, token)
    except saved.ProfileStoreError as e:
        S["account_logout_notice"] = f"Signed out on this browser, but the service could not revoke the session: {e}"
    else:
        S["account_logout_notice"] = "You have signed out."
    S.pop(TOKEN_KEY, None)
    S.pop("account_display_name", None)
    S.pop("account_avatar_emoji", None)
    device_store.sync_auth_token(None)
    st.rerun()

has_profile = bool(S.get("profile_done"))
profile = prof.build_profile() if has_profile else None
if profile:
    st.subheader("Save a profile snapshot")
    st.caption(
        "Saves your profile answers, prediction, personalized suggestions, career-path recommendations, "
        "and any active Improve my chances comparison."
    )
    if st.button("Save current profile and estimate", key="save_profile_snapshot"):
        prediction, error = prof.call("/predict", profile)
        if error:
            st.error(f"Could not calculate the prediction: {error}")
        else:
            explanation, error = prof.call("/explain", profile)
            if error:
                st.error(f"Could not calculate the explanation: {error}")
            else:
                suggestions, error = prof.call("/suggestions", profile)
                if error:
                    st.error(f"Could not calculate personalized suggestions: {error}")
                else:
                    market_alignment, error = prof.call("/market-alignment", profile)
                    if error:
                        st.error(f"Could not calculate employer skill alignment: {error}")
                    else:
                        job_market = ui.fetch("/job-market")
                        if job_market is None:
                            st.error("Could not load job-market data for role recommendations.")
                        else:
                            try:
                                pathways = role_pathways(profile, job_market)
                            except ValueError as e:
                                st.error(f"Could not calculate role recommendations: {e}")
                            else:
                                selected_role = S.get("career_target_role") or S.get("account_target_role")
                                selected_role_output = next(
                                    (role for role in pathways["roles"] if role["role"] == selected_role), None
                                )
                                current = {
                                    "internship_months": profile["internship_months"],
                                    "project_count": profile["project_count"],
                                    "certifications_count": profile["certifications_count"],
                                    **{f"area_{area}": prof.current_area(profile, area) for area in prof.AREAS},
                                }
                                if profile["respondent_status"] == "final_year" and "gpa" in profile:
                                    current["gpa"] = profile["gpa"]
                                active_changes = {}
                                for feature, value in current.items():
                                    key = f"wi_{feature}_{value}"
                                    if key in S and abs(float(S[key]) - float(value)) > 1e-9:
                                        active_changes[feature] = S[key]

                                what_if = None
                                if active_changes:
                                    what_if, error = prof.call("/whatif", {
                                        "profile": profile,
                                        "changes": active_changes,
                                    })
                                    if error:
                                        st.error(f"Could not calculate the active Improve my chances scenario: {error}")
                                        st.stop()

                                try:
                                    snapshot = saved.post("/snapshots", {
                                        "answers": prof.answers(),
                                        "profile": profile,
                                        "estimate": {
                                            "median": prediction["median"],
                                            "low": prediction["low"],
                                            "high": prediction["high"],
                                            "confidence": prediction["confidence"],
                                            "headline": prediction["headline"],
                                        },
                                        "outputs": {
                                            "prediction": prediction,
                                            "explanation": explanation,
                                            "suggestions": suggestions,
                                            "market_alignment": market_alignment,
                                            "career_pathways": pathways,
                                            "career_path": {
                                                "selected_role": selected_role_output,
                                            },
                                            "improve_chances": {
                                                "suggestions": suggestions,
                                                "what_if": {
                                                    "changes": active_changes,
                                                    "result": what_if,
                                                } if what_if else None,
                                            },
                                        },
                                    }, token)
                                    st.success(f"Profile snapshot saved on {snapshot['created_at']}.")
                                except saved.ProfileStoreError as e:
                                    st.error(str(e))
else:
    st.info("Build your profile to save a profile snapshot.")

try:
    snapshots = saved.get("/snapshots", token)
except saved.ProfileStoreError as e:
    ui.error_card(str(e))
    st.stop()

st.subheader("Profile history")
if snapshots:
    st.dataframe([{
        "Saved": item["created_at"],
        "Typical months": item["estimate"]["median"]["display"],
        "Range": f'{item["estimate"]["low"]["display"]}–{item["estimate"]["high"]["display"]}',
        "Confidence": item["estimate"]["confidence"],
    } for item in snapshots], hide_index=True, width="stretch")
    for item in snapshots:
        outputs = item.get("outputs", {})
        if outputs:
            with st.expander(f"Prediction outputs and recommendations · {item['created_at']}", expanded=False):
                prediction = outputs.get("prediction", {})
                if prediction:
                    st.markdown(f"**Prediction:** {prediction.get('headline', 'Estimate saved')}")
                    st.caption(
                        f"Confidence: {prediction.get('confidence', 'Unavailable')}. "
                        f"{prediction.get('confidence_reason', '')}"
                    )
                    for caution in prediction.get("cautions", []):
                        st.warning(caution)
                explanation = outputs.get("explanation", {})
                if explanation:
                    st.markdown("**What helped**")
                    prof.show_items(explanation.get("helping", []), "No standout factors were saved.")
                    st.markdown("**What held the estimate back**")
                    prof.show_items(explanation.get("hurting", []), "No standout factors were saved.")
                suggestions = outputs.get("suggestions", {}).get("suggestions", [])
                if suggestions:
                    st.markdown("**Personalized next steps**")
                    for suggestion in suggestions:
                        st.markdown(f"- {suggestion['text']}")
                alignment = outputs.get("market_alignment", {}).get("skills", [])
                if alignment:
                    st.markdown("**Employer skill alignment**")
                    st.dataframe(alignment, hide_index=True, width="stretch")
                recommendations = outputs.get("career_pathways", {}).get("recommendations", [])
                if recommendations:
                    st.markdown("**Recommended role profiles**")
                    st.dataframe([{
                        "Role": role["role"],
                        "Skill match (1-5)": role["match_rating"],
                        "Ads": role["ads"],
                        "Skill gaps": ", ".join(gap["label"] for gap in role["gaps"]),
                    } for role in recommendations], hide_index=True, width="stretch")
                career_path = outputs.get("career_path", {})
                selected_role = career_path.get("selected_role")
                if selected_role:
                    st.markdown(f"**Selected career path: {selected_role['role']}**")
                    st.caption(
                        f"Skill match: {selected_role['match_rating']:.1f}/5 · "
                        f"{selected_role['ads']} collected ads"
                    )
                    if selected_role["gaps"]:
                        st.markdown("Skills to strengthen: " + ", ".join(
                            gap["label"] for gap in selected_role["gaps"]
                        ))
                improve = outputs.get("improve_chances", {})
                what_if = improve.get("what_if")
                if what_if:
                    st.markdown("**Saved Improve my chances scenario**")
                    st.markdown(f"Changes tested: `{what_if['changes']}`")
                    result = what_if["result"]
                    if result:
                        st.markdown(result.get("message", "What-if comparison saved."))
                        for caution in result.get("cautions", []):
                            st.warning(caution)
                with st.expander("View all saved output data"):
                    st.json(outputs)
    restore_col, delete_col = st.columns(2)
    if restore_col.button("Restore latest profile to this browser", key="restore_latest_profile"):
        if prof.apply_answers(snapshots[0]["answers"]):
            st.session_state["_session_restored"] = True
            st.success("Latest saved profile restored in this browser.")
            st.rerun()
        else:
            st.error("The latest saved profile did not contain usable answers.")
    snapshot_ids = [item["id"] for item in snapshots]
    selected_snapshot = delete_col.selectbox("Snapshot to delete", snapshot_ids, key="delete_snapshot_id")
    if delete_col.button("Delete selected snapshot", key="delete_profile_snapshot"):
        try:
            saved.delete(f"/snapshots/{selected_snapshot}", token)
            st.rerun()
        except saved.ProfileStoreError as e:
            st.error(str(e))
else:
    st.caption("No profile snapshots yet.")

st.subheader("Career goal")
if profile:
    job_market = ui.fetch("/job-market")
    if job_market:
        try:
            pathways = role_pathways(profile, job_market)
            roles = pathways["roles"]
        except ValueError as e:
            roles = []
            st.error(str(e))
        if roles:
            role_names = [item["role"] for item in roles]
            selected_role = st.selectbox("Target role family", role_names, key="account_target_role")
            role_item = next(item for item in roles if item["role"] == selected_role)
            if st.button("Save this career goal", key="save_career_goal"):
                try:
                    saved.post("/goals", {"role": selected_role, "gaps": role_item["gaps"]}, token)
                    st.success(f"{selected_role} saved as your career goal.")
                except saved.ProfileStoreError as e:
                    st.error(str(e))
else:
    st.caption("Build your profile to set a goal using your current skill-gap comparison.")

try:
    goals = saved.get("/goals", token)
except saved.ProfileStoreError as e:
    ui.error_card(str(e))
    goals = []
for goal in goals:
    with st.container(border=True):
        st.markdown(f"**{goal['role']}** · {goal['status'].replace('_', ' ').title()}")
        if goal["gaps"]:
            st.caption("Current focus: " + ", ".join(gap["label"] for gap in goal["gaps"][:5]))
        controls = st.columns([2, 1, 1])
        new_status = controls[0].selectbox(
            "Goal status", ["active", "paused", "completed"], index=["active", "paused", "completed"].index(goal["status"]),
            key=f"goal_status_{goal['id']}",
        )
        if controls[1].button("Update", key=f"update_goal_{goal['id']}"):
            try:
                saved.patch(f"/goals/{goal['id']}", {"status": new_status}, token)
                st.rerun()
            except saved.ProfileStoreError as e:
                st.error(str(e))
        if controls[2].button("Delete", key=f"delete_goal_{goal['id']}"):
            try:
                saved.delete(f"/goals/{goal['id']}", token)
                st.rerun()
            except saved.ProfileStoreError as e:
                st.error(str(e))

st.subheader("Action checklist")
if profile:
    suggestions, error = prof.call("/suggestions", profile)
    if error:
        st.error(error)
    elif suggestions["suggestions"]:
        suggestion_options = {item["text"]: item["text"] for item in suggestions["suggestions"]}
        selected_actions = st.multiselect("Add personalized suggestions", list(suggestion_options), key="saved_actions")
        if st.button("Add selected suggestions to checklist", key="add_saved_actions") and selected_actions:
            try:
                for title in selected_actions:
                    saved.post("/actions", {"title": title}, token)
                st.success(f"Added {len(selected_actions)} action(s).")
                st.rerun()
            except saved.ProfileStoreError as e:
                st.error(str(e))

with st.form("new_progress_action"):
    action_title = st.text_input("Add your own action", max_chars=240)
    action_detail = st.text_input("Details (optional)", max_chars=2000)
    add_action = st.form_submit_button("Add action")
if add_action and action_title.strip():
    try:
        saved.post("/actions", {"title": action_title, "detail": action_detail}, token)
        st.rerun()
    except saved.ProfileStoreError as e:
        st.error(str(e))

try:
    actions = saved.get("/actions", token)
except saved.ProfileStoreError as e:
    ui.error_card(str(e))
    actions = []
for action in actions:
    with st.container(border=True):
        title_col, state_col, button_col = st.columns([3, 2, 1], vertical_alignment="center")
        title_col.markdown(f"**{action['title']}**")
        if action["detail"]:
            title_col.caption(action["detail"])
        states = ["not_started", "in_progress", "completed"]
        state = state_col.selectbox("Progress", states, index=states.index(action["state"]),
                                    format_func=lambda value: value.replace("_", " ").title(),
                                    key=f"action_state_{action['id']}")
        if button_col.button("Update", key=f"update_action_{action['id']}"):
            try:
                saved.patch(f"/actions/{action['id']}", {"state": state}, token)
                st.rerun()
            except saved.ProfileStoreError as e:
                st.error(str(e))
        if button_col.button("Delete", key=f"delete_action_{action['id']}"):
            try:
                saved.delete(f"/actions/{action['id']}", token)
                st.rerun()
            except saved.ProfileStoreError as e:
                st.error(str(e))

st.divider()
st.subheader("Delete your account and saved progress")
st.warning("This permanently deletes your account, profile history, career goals, and checklist.")
if st.button("Delete my account and data", type="primary", key="delete_profile_account"):
    try:
        saved.delete("/me", token)
        S.pop(TOKEN_KEY, None)
        S.pop("account_display_name", None)
        S.pop("account_avatar_emoji", None)
        device_store.sync_auth_token(None)
        st.success("Your account and saved data were deleted.")
        st.rerun()
    except saved.ProfileStoreError as e:
        st.error(str(e))
