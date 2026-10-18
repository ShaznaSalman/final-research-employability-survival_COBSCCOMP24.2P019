"""Floating assistant (frontend/chat_widget.py): what it answers, and what it does when it cannot."""
from __future__ import annotations

import pytest

from frontend import chat_widget as cw


@pytest.fixture(autouse=True)
def fixed_profile(monkeypatch):
    monkeypatch.setattr(cw.prof, "build_profile", lambda: {"respondent_status": "graduated", "gpa": 3.2})


def test_without_a_finished_profile_it_asks_for_one_and_calls_nothing():
    def boom(*a, **k):
        raise AssertionError("the API must not be called before the profile is complete")
    reply, follow = cw.reply_for("Why is my estimate like this?", False, boom)
    assert reply == cw.NOT_READY and follow == []
    assert "My profile" in reply


@pytest.mark.parametrize("message", ["hi", "hii", "hio", "What can I ask?", "How does the model work?",
                                     "Do you store my data?", "Is GPA important?"])
def test_general_questions_are_answered_without_a_profile(message):
    seen = {}

    def call(path, payload):
        seen.update(path=path, payload=payload)
        return {"reply": "A general answer.", "follow_ups": ["How does the model work?"]}, None

    reply, follow = cw.reply_for(message, False, call)
    assert reply == "A general answer."
    assert follow == ["How does the model work?"]
    assert seen == {"path": "/chat", "payload": {"profile": None, "message": message}}


def test_a_normal_answer_returns_the_reply_and_follow_ups_from_the_api():
    seen = {}

    def call(path, payload):
        seen.update(path=path, payload=payload)
        return {"reply": "About 5 months.", "follow_ups": ["What if I do a 6-month internship?", "Which skills should I learn?"]}, None

    reply, follow = cw.reply_for("How long will it take me?", True, call)
    assert reply == "About 5 months." and follow == ["What if I do a 6-month internship?", "Which skills should I learn?"]
    assert seen["path"] == "/chat" and seen["payload"] == {"profile": {"respondent_status": "graduated", "gpa": 3.2},
                                                           "message": "How long will it take me?"}


@pytest.mark.parametrize("result", [(None, "API not reachable"), (None, None)])
def test_an_unreachable_service_gives_a_friendly_message(result):
    reply, follow = cw.reply_for("hello", True, lambda path, payload: result)
    assert reply == cw.UNREACHABLE and follow == []


def test_missing_follow_ups_become_an_empty_list():
    reply, follow = cw.reply_for("hi", True, lambda p, q: ({"reply": "ok"}, None))
    assert reply == "ok" and follow == []


def test_starters_are_questions_the_assistant_understands():
    from core.chat import FOLLOW_UPS
    assert all(q.endswith("?") for q in cw.STARTERS) and len(cw.STARTERS) == 4
    assert set(cw.STARTERS) <= set(FOLLOW_UPS) | {"What if I do a 6-month internship?"}
    assert all(q.endswith("?") for q in cw.BASIC_STARTERS)


def test_personal_questions_still_require_a_finished_profile():
    def boom(*a, **k):
        raise AssertionError("profile-dependent questions must not call the API without a profile")
    reply, follow = cw.reply_for("How long will it take me?", False, boom)
    assert reply == cw.NOT_READY and follow == []
