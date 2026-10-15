"""Ask: a grounded assistant that answers only from this system's own results (no language model, nothing stored)."""
from __future__ import annotations

import streamlit as st

from frontend import profile as prof, ui
from frontend.api_client import APIError

try:
    info = ui.model_info()
except APIError as e:
    ui.error_card(str(e), "Check that the API is running (uvicorn api.main:app --port 8000), then reload.")
    st.stop()
profile = prof.require()

CHAT_STARTERS = ["Why is my estimate like this?", "What if I do a 6-month internship?",
                 "Which skills should I learn?", "Can I trust this estimate?"]

st.header("Ask about your result", anchor=False)
st.markdown("<p class='muted'>Answers come only from this system's own results, using your current profile - it is "
            "not a general chatbot. Nothing you type is stored.</p>", unsafe_allow_html=True)

history = st.session_state.setdefault("chat_history", [])
for role, text in history[-12:]:
    with st.chat_message(role):
        st.markdown(text)
asked = None
for col, q in zip(st.columns(2), CHAT_STARTERS[:2]):
    if col.button(q, key=f"chat_starter_{q}", use_container_width=True):
        asked = q
for col, q in zip(st.columns(2), CHAT_STARTERS[2:]):
    if col.button(q, key=f"chat_starter_{q}", use_container_width=True):
        asked = q
typed = st.chat_input("Ask about your estimate, e.g. \"What if I get two certifications?\"", key="chat_input")
asked = typed or asked
if asked:
    answer, err = prof.call("/chat", {"profile": profile, "message": asked})
    history += [("user", asked), ("assistant", err or answer["reply"])]
    st.rerun()
if history and st.button("Clear chat", key="chat_clear"):
    st.session_state["chat_history"] = []
    st.rerun()

st.page_link("views/dashboard.py", label="Back to my dashboard", icon=":material/arrow_back:")
st.caption(info["disclaimer"])
