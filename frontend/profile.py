"""The graduate's profile answers, shared by every page.

Answers live in session state under p_* keys (this browser session only, nothing is stored on the server), so they
survive moving between steps and pages. This module has the defaults, the API payload and small derived values.
"""
from __future__ import annotations

import json

import streamlit as st

from core import schema, wording  # option lists and labels only - no model logic
from frontend import ui
from frontend.api_client import APIError, post

S = st.session_state
AREAS = {**schema.SKILL_AREAS, **schema.SOFT_AREA}
AREA_DEFAULT = {"soft": 4}                    # whole-number pills, so the soft-skills start moves from 3.5 to 4

DEFAULTS = {
    "p_status": "graduated", "p_looking": False, "p_months": 3, "p_uses_class": False,
    "p_degree_class": list(wording.DEGREE_CLASS_LABELS)[1], "p_gpa": 3.0,
    "p_degree_type": list(wording.DEGREE_TYPE_LABELS)[1], "p_university_type": list(wording.INSTITUTION_LABELS)[0],
    "p_internship": 6, "p_projects": 3, "p_certs": 1, "p_detailed": False,
    "p_portfolio": False, "p_oss": False, "p_hack": 0, "p_prep": list(wording.PREP_START_LABELS)[2],
    "p_selflearn": 3, "p_sdlc": 3, **{f"p_area_{a}": AREA_DEFAULT.get(a, 3) for a in AREAS},
}


def keep_state() -> None:
    """Keep values of widgets that are not on screen on this page (Streamlit drops them otherwise)."""
    for k in [k for k in S if k.startswith("p_") or k == "remember_device"]:
        S[k] = S[k]


def val(key: str):
    return S.get(key, DEFAULTS[key])


def ready(key: str) -> str:
    """Seed a widget's value in session state (once) and return its key."""
    S.setdefault(key, DEFAULTS[key])
    return key


def area_items(area: str) -> list[str]:
    return [schema.SKILLS_BY_KEY[k].column for k in AREAS[area]["skills"]]


def build_profile() -> dict:
    """The profile sent to the API, built from the saved answers."""
    status = val("p_status")
    profile: dict = {"respondent_status": status}
    if status == "graduated" and val("p_looking"):
        profile["months_since_graduation"] = val("p_months")
    if val("p_uses_class"):
        profile["degree_class"] = val("p_degree_class")
    else:
        profile["gpa"] = val("p_gpa")
    profile["degree_type"] = val("p_degree_type")
    profile["university_type"] = val("p_university_type")
    profile["internship_months"] = val("p_internship")
    profile["project_count"] = val("p_projects")
    profile["certifications_count"] = val("p_certs")
    if not val("p_detailed"):
        for a in AREAS:
            profile[f"area_{a}"] = float(val(f"p_area_{a}"))
    else:
        for a in AREAS:
            for col in area_items(a):
                profile[col] = S.get(f"p_item_{col}", int(round(val(f"p_area_{a}"))))
    profile["has_portfolio"] = "yes" if val("p_portfolio") else "no"
    profile["open_source"] = "yes" if val("p_oss") else "no"
    profile["hackathon_count"] = val("p_hack")
    profile["prep_start"] = val("p_prep")
    profile["self_learning_freq"] = val("p_selflearn")
    profile["sdlc_familiarity"] = val("p_sdlc")
    return profile


def survey_extras(profile: dict) -> dict:
    """The survey insights use the same answers."""
    return {"has_portfolio": profile["has_portfolio"] == "yes", "open_source": profile["open_source"] == "yes",
            "hackathons": profile["hackathon_count"] > 0, "self_learning": profile["self_learning_freq"] >= 4,
            "early_prep": profile["prep_start"] in ("before_university", "first_year", "second_third_year")}


def current_area(profile: dict, area: str) -> float:
    if not val("p_detailed"):
        return float(profile[f"area_{area}"])
    vals = [profile[c] for c in area_items(area)]
    return round(2 * sum(vals) / len(vals)) / 2          # nearest 0.5 for the what-if slider


def call(path: str, payload: dict):
    """Profile-based calls are NOT cached, so no profile is kept in server memory."""
    try:
        return post(path, payload), None
    except APIError as e:
        return None, str(e)


def require() -> dict:
    """The profile for a results page. Without a finished profile, show a prompt to build one and stop."""
    keep_state()
    if not S.get("profile_done"):
        st.header("Build your profile first", anchor=False)
        st.markdown("<p class='muted'>Four short steps about your degree, experience and skills give you a personal "
                    "estimate.</p>", unsafe_allow_html=True)
        st.page_link("views/predict.py", label="Start my profile", icon=":material/arrow_forward:")
        st.stop()
    return build_profile()


def show_items(items: list[dict], empty: str) -> None:
    if not items:
        st.markdown(f"<div class='muted'>{empty}</div>", unsafe_allow_html=True)
    for i in items:
        st.markdown(f"<div class='item'>{i['text']}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Saving the profile on the user's own device (browser storage) or in a downloaded file.
# Nothing is sent to the server's storage: answers go to the browser or the user's disk only.
# ---------------------------------------------------------------------------
ITEM_PREFIX = "p_item_"
BOUNDS = {"p_months": (1, 36), "p_internship": (0, 36), "p_projects": (0, 30), "p_certs": (0, 20), "p_hack": (0, 30),
          "p_selflearn": (1, 5), "p_sdlc": (1, 5), **{f"p_area_{a}": (1, 5) for a in AREAS}}
OPTIONS = {"p_status": wording.STATUS_LABELS, "p_degree_class": wording.DEGREE_CLASS_LABELS,
           "p_degree_type": wording.DEGREE_TYPE_LABELS, "p_university_type": wording.INSTITUTION_LABELS,
           "p_prep": wording.PREP_START_LABELS}
ITEM_COLUMNS = {col for a in AREAS for col in area_items(a)}


def answers() -> dict:
    """The answers the user has given so far (only keys that exist, so untouched defaults are not saved)."""
    return {k: S[k] for k in S if k in DEFAULTS or (k.startswith(ITEM_PREFIX) and k[len(ITEM_PREFIX):] in ITEM_COLUMNS)}


def clean_answers(raw) -> dict:
    """Keep only known keys whose values have the right type and range. Anything else is dropped, so a damaged or
    hand-edited saved profile cannot put the app into a state it cannot handle."""
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    for k, v in raw.items():
        if k in OPTIONS:
            if isinstance(v, str) and v in OPTIONS[k]:
                out[k] = v
        elif k in BOUNDS:
            lo, hi = BOUNDS[k]
            if isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi:
                out[k] = v
        elif k == "p_gpa":
            if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 4:
                out[k] = float(v)
        elif k in DEFAULTS and isinstance(DEFAULTS[k], bool):
            if isinstance(v, bool):
                out[k] = v
        elif k.startswith(ITEM_PREFIX) and k[len(ITEM_PREFIX):] in ITEM_COLUMNS:
            if isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 5:
                out[k] = v
    return out


def apply_answers(raw) -> bool:
    """Load saved answers into this session. Returns False when nothing usable was found."""
    clean = clean_answers(raw)
    if not clean:
        return False
    S.update(clean)
    S["profile_done"] = True
    return True


def restore_from_device(raw) -> bool:
    """Continue with the answers saved in this browser; later changes are saved back while 'remember me' stays on."""
    if not apply_answers(raw):
        return False
    S["remember_device"] = True
    S["_saved_sig"] = json.dumps(answers(), sort_keys=True)
    return True


def profile_file() -> str:
    return json.dumps({"jobspark_profile": 1, "answers": answers()}, indent=2)


def read_profile_file(text: str):
    """The answers inside an uploaded profile file, or None when it is not a JobSpark profile."""
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("jobspark_profile") != 1:
        return None
    return data.get("answers")


def _send(command: dict) -> None:
    S["_store_n"] = S.get("_store_n", 0) + 1
    S["_store_cmd"] = {**command, "n": S["_store_n"]}


def remember_changed() -> None:
    """Callback for the 'Remember me on this device' switch."""
    if S.get("remember_device"):
        S["_saved_sig"] = json.dumps(answers(), sort_keys=True)
        _send({"op": "save", "answers": answers()})
    else:
        forget_device()


def forget_device() -> None:
    S["remember_device"] = False
    S.pop("_saved_sig", None)
    _send({"op": "clear"})


def session_payload() -> dict | None:
    """What the browser keeps for this tab (sessionStorage) so a reload does not lose the answers; None when nothing
    has been entered yet, so a fresh page load never overwrites what the tab already kept."""
    given = answers()
    return {"answers": given, "done": bool(S.get("profile_done"))} if given else None


def restore_session(raw) -> bool:
    """Put back the answers this tab kept before a reload. Same validation as a saved profile."""
    if not isinstance(raw, dict):
        return False
    clean = clean_answers(raw.get("answers"))
    if not clean:
        return False
    S.update(clean)
    if raw.get("done") is True:
        S["profile_done"] = True
    return True


def sync_device() -> None:
    """While 'remember me' is on, keep the browser copy in step with the answers (runs at the top of every page)."""
    if S.get("remember_device"):
        sig = json.dumps(answers(), sort_keys=True)
        if S.get("_saved_sig") != sig:
            S["_saved_sig"] = sig
            _send({"op": "save", "answers": answers()})
