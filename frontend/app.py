"""Streamlit frontend (JobSpark): sidebar navigation with icons (a menu button on phones).
All data comes from the FastAPI backend (API_URL in .env).

Run:  streamlit run frontend/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # so `core` and `frontend` import from the root

import streamlit as st  # noqa: E402

from frontend import chat_widget, device_store, profile, profile_store_client as saved, ui  # noqa: E402

st.set_page_config(page_title="JobSpark · Time to your first IT job", page_icon=":material/auto_awesome:", layout="wide")
profile.keep_state()
profile.sync_device()                                              # keep the browser copy current while 'remember me' is on
_browser = device_store.mount(session=profile.session_payload(),   # what this browser reports: saved answers, the real
                              profile_done=bool(st.session_state.get("profile_done")),
                              display_name=st.session_state.get("account_display_name", ""),
                              avatar_emoji=st.session_state.get("account_avatar_emoji", "🙂"),
                              signed_in=bool(st.session_state.get("profile_store_token")),
                              auth_sync=st.session_state.get("_auth_sync"))  # profile, theme, and tab answers
st.session_state["device_stored"] = _browser.stored
auth_sync = st.session_state.get("_auth_sync")
if (_browser.auth_token and not st.session_state.get("profile_store_token")
        and not (auth_sync and not auth_sync["token"])):
    st.session_state["profile_store_token"] = _browser.auth_token
    st.rerun()
identity_update = _browser.identity_update
if identity_update:
    nonce = identity_update.get("nonce")
    if nonce and nonce != st.session_state.get("_avatar_identity_nonce"):
        st.session_state["_avatar_identity_nonce"] = nonce
        token = st.session_state.get("profile_store_token")
        if not token:
            st.session_state["_avatar_identity_error"] = "Sign in before saving profile appearance."
        else:
            try:
                account = saved.patch(
                    "/me",
                    {
                        "display_name": identity_update.get("display_name", ""),
                        "avatar_emoji": identity_update.get("avatar_emoji", ""),
                    },
                    token,
                )
                st.session_state["account_display_name"] = account["display_name"]
                st.session_state["account_avatar_emoji"] = account["avatar_emoji"]
                st.session_state["_avatar_identity_success"] = True
            except saved.ProfileStoreError as error:
                st.session_state["_avatar_identity_error"] = str(error)
if st.session_state.pop("_avatar_identity_success", False):
    st.success("Profile appearance saved to your account.")
identity_error = st.session_state.pop("_avatar_identity_error", None)
if identity_error:
    st.error(identity_error)
if _browser.theme:
    st.session_state["ui_theme"] = _browser.theme
if _browser.session and not st.session_state.get("_session_restored"):   # after a reload (e.g. the theme switch)
    st.session_state["_session_restored"] = True
    if profile.restore_session(_browser.session):
        st.rerun()                                                 # once, so the top bar and pages see the restored profile
ui.inject()

HERE = Path(__file__).resolve().parent
st.logo(HERE / "assets" / f"logo_{'dark' if ui.is_dark() else 'light'}.svg", size="large")

VIEWS = HERE / "views"
pages = {
    "Start": [
        st.Page(VIEWS / "home.py", title="Home", icon=":material/home:", default=True),
        st.Page(VIEWS / "predict.py", title="My profile", icon=":material/person:"),
    ],
    "My results": [
        st.Page(VIEWS / "dashboard.py", title="My dashboard", icon=":material/dashboard:"),
        st.Page(VIEWS / "compare.py", title="Compare", icon=":material/balance:"),
    ],
    "My plan": [
        st.Page(VIEWS / "career_path.py", title="Career path", icon=":material/route:"),
        st.Page(VIEWS / "account.py", title="Saved profile & progress", icon=":material/track_changes:"),
        st.Page(VIEWS / "improve.py", title="Improve my chances", icon=":material/tune:"),
    ],
    "More": [
        st.Page(VIEWS / "market.py", title="Job market", icon=":material/work:"),
    ],
}
page = st.navigation(pages, position="sidebar")
chat_widget.render(hidden=page.title == "Ask")                    # Spark, the floating robot assistant (the Ask page is the full chat)
page.run()
