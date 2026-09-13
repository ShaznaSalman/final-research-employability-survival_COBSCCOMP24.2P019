"""Streamlit frontend: four pages with top navigation. All data comes from the FastAPI backend (API_URL in .env).

Run:  streamlit run frontend/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # so `core` and `frontend` import from the root

import streamlit as st  # noqa: E402

from frontend import ui  # noqa: E402

st.set_page_config(page_title="Time to First IT Job", page_icon=":material/work:", layout="wide")
ui.inject()

HERE = Path(__file__).resolve().parent / "views"
pages = [
    st.Page(HERE / "home.py", title="Home", icon=":material/home:", default=True),
    st.Page(HERE / "predict.py", title="My prediction", icon=":material/insights:"),
    st.Page(HERE / "market.py", title="Job market", icon=":material/work:"),
    st.Page(HERE / "research.py", title="Research lab", icon=":material/science:"),
]
st.navigation(pages, position="top").run()
