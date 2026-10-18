"""Improve my chances: suggested combinations and what-if sliders (things the graduate can change)."""
from __future__ import annotations

import streamlit as st

from core import schema, wording  # option lists and labels only - no model logic
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
status = profile["respondent_status"]
call = prof.call

pred, err = call("/predict", profile)
if err:
    ui.error_card(err)
    st.stop()
now = pred["months_since_graduation"]

st.header("Improve my chances", anchor=False)
st.markdown("<p class='muted'>Try changes you could realistically make and see how the estimate moves. "
            "These are associations in survey data, not guarantees.</p>", unsafe_allow_html=True)
ui.synthetic_notice(info)

current = {"internship_months": profile["internship_months"], "project_count": profile["project_count"],
           "certifications_count": profile["certifications_count"],
           **{f"area_{a}": prof.current_area(profile, a) for a in prof.AREAS}}
if status == "final_year" and "gpa" in profile:
    current["gpa"] = profile["gpa"]


def slider_key(feat: str) -> str:
    return f"wi_{feat}_{current[feat]}"                      # resets when your profile changes


def try_suggestion(changes: dict):
    for feat in current:                                     # start from your profile, then apply this one
        S.pop(slider_key(feat), None)
    for feat, v in changes.items():
        if feat in current:
            S[slider_key(feat)] = type(current[feat])(v) if feat != "gpa" else float(v)


pending = S.pop("pending_changes", None)                     # set by "Try it" on the dashboard
if pending:
    try_suggestion(pending)

sg, _ = call("/suggestions", profile)
if sg:
    st.markdown("**:material/lightbulb: Combinations to explore** <span class='muted'>(small changes linked "
                "with finding work sooner in profiles like yours)</span>", unsafe_allow_html=True)
    if not sg["suggestions"]:
        st.caption(sg["message"])
    for i, sug in enumerate(sg["suggestions"]):
        with st.container(border=True):
            t, b = st.columns([5, 1], vertical_alignment="center")
            t.markdown(f"<div class='item'>{sug['text']}</div>", unsafe_allow_html=True)
            b.button("Try this", type="primary", key=f"try_{i}", on_click=try_suggestion, args=(sug["changes"],),
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
        S.setdefault(slider_key(feat), cast(value))   # "Try this" may have set it already
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
                                               "color": charts.AQUA}], horizon, now=now),
                            config=ui.PLOT, key="range_whatif")
            st.markdown(f"**{wi['message']}**")
            for c in wi["cautions"]:
                st.warning(c, icon=":material/warning:")
    else:
        st.caption(wording.WHATIF_PROMPT)

st.page_link("views/dashboard.py", label="Back to my dashboard", icon=":material/arrow_back:")
st.caption(info["disclaimer"] + " " + info["associational_note"])
