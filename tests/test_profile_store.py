"""Saved-profile handling (frontend/profile.py): only known, correctly typed, in-range answers are accepted."""
from __future__ import annotations

import json

import pytest

from frontend import profile as prof


def test_clean_keeps_valid_answers():
    raw = {"p_status": "graduated", "p_gpa": 3, "p_internship": 6, "p_looking": True, "p_area_soft": 4,
           "p_degree_type": list(prof.OPTIONS["p_degree_type"])[0]}
    clean = prof.clean_answers(raw)
    assert clean == {**raw, "p_gpa": 3.0}
    assert isinstance(clean["p_gpa"], float)


@pytest.mark.parametrize("key,value", [
    ("p_status", "astronaut"), ("p_gpa", 4.5), ("p_gpa", -1), ("p_gpa", "3.2"), ("p_gpa", True),
    ("p_internship", 99), ("p_internship", 2.5), ("p_internship", True), ("p_looking", "yes"),
    ("p_area_soft", 0), ("p_selflearn", 6), ("p_unknown", 1), ("not_a_key", "x"),
])
def test_clean_drops_invalid_or_unknown(key, value):
    assert prof.clean_answers({key: value}) == {}


def test_clean_item_ratings():
    col = sorted(prof.ITEM_COLUMNS)[0]
    assert prof.clean_answers({f"p_item_{col}": 4}) == {f"p_item_{col}": 4}
    assert prof.clean_answers({f"p_item_{col}": 9}) == {}
    assert prof.clean_answers({"p_item_not_a_skill": 3}) == {}


def test_clean_rejects_non_dicts():
    for bad in (None, [], "text", 5):
        assert prof.clean_answers(bad) == {}


def test_profile_file_round_trip():
    text = json.dumps({"jobspark_profile": 1, "answers": {"p_gpa": 3.4, "p_status": "final_year"}})
    assert prof.read_profile_file(text) == {"p_gpa": 3.4, "p_status": "final_year"}


@pytest.mark.parametrize("text", ["not json", "[]", "{}", json.dumps({"jobspark_profile": 2, "answers": {}}),
                                  json.dumps({"answers": {"p_gpa": 3}})])
def test_read_profile_file_rejects_other_files(text):
    assert prof.read_profile_file(text) is None


def test_defaults_are_all_accepted_by_clean():
    """Every default must survive validation, otherwise a freshly saved profile could not be loaded back."""
    assert set(prof.clean_answers(prof.DEFAULTS)) == set(prof.DEFAULTS)


# --- tab handoff (answers kept in sessionStorage across a reload, e.g. the theme switch) -------------------------------
@pytest.fixture
def state(monkeypatch):
    fake: dict = {}
    monkeypatch.setattr(prof, "S", fake)
    return fake


def test_session_payload_is_none_until_something_is_entered(state):
    assert prof.session_payload() is None                       # a fresh load must not overwrite what the tab kept
    state["p_gpa"] = 3.4
    state["profile_done"] = True
    state["unrelated"] = "x"
    assert prof.session_payload() == {"answers": {"p_gpa": 3.4}, "done": True}


def test_restore_session_applies_valid_answers_and_done_flag(state):
    assert prof.restore_session({"answers": {"p_gpa": 3.1, "p_internship": 8, "p_status": "bogus"}, "done": True})
    assert state == {"p_gpa": 3.1, "p_internship": 8, "profile_done": True}


def test_restore_session_without_done_keeps_profile_incomplete(state):
    assert prof.restore_session({"answers": {"p_projects": 2}, "done": False})
    assert "profile_done" not in state and state["p_projects"] == 2


@pytest.mark.parametrize("raw", [None, "x", [], {}, {"answers": {}}, {"answers": {"p_unknown": 1}}, {"answers": "x"}])
def test_restore_session_ignores_unusable_data(state, raw):
    assert prof.restore_session(raw) is False and state == {}
