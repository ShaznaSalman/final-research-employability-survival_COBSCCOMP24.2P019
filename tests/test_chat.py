"""Tests for the grounded assistant (core/chat.py) and the /chat endpoint. No language model is involved."""
import pytest

from core import wording
from core.chat import Assistant, classify, parse_changes
from core.explain import Explainer
from core.suggest import current_values
from tests.conftest import QUICK_PROFILE


@pytest.fixture(scope="module")
def assistant(cox_predictor):
    return Assistant(cox_predictor, Explainer(cox_predictor), reference=None)


@pytest.mark.parametrize("message,intent", [
    ("How long will it take me?", "estimate"), ("why is my estimate like this", "explain"),
    ("What if I do a 6-month internship?", "whatif"), ("which skills should I learn?", "skills"),
    ("how can I find work sooner", "suggestions"), ("can I trust this?", "trust"),
    ("Is GPA important?", "factor"), ("how do I compare with similar graduates", "peers"),
    ("how does it work", "method"), ("do you store my data", "privacy"), ("hello", "greeting"),
    ("hi", "greeting"), ("hii", "greeting"), ("hio", "greeting"),
    ("good evening my friend", "greeting"), ("thankyuo", "greeting"), ("thanku", "greeting"),
    ("thankyou", "greeting"),
    ("what is the capital of France", "unknown"),
    ("why is my estmate like this", "explain"), ("which skils should i lern", "skills"),
    ("What if I do a 6-month internshp?", "whatif"), ("can i trust this esimate?", "trust"),
    ("Who will win the cricket world cup?", "unknown"),
])
def test_classify(message, intent):
    assert classify(message) == intent


def test_misspelled_what_if_question_is_corrected_before_parsing(assistant):
    response = assistant.respond("What if I do a 6-month internshp?", QUICK_PROFILE)
    assert response["intent"] == "whatif"
    assert "Change tested" in response["reply"]
    assert "internship_months" not in response["reply"]


def test_parse_changes_adds_and_sets():
    cur = {"internship_months": 3.0, "certifications_count": 1.0, "project_count": 2.0, "area_cloud_devops": 2.0}
    ch, _ = parse_changes("what if I do a 6 month internship and get two certifications", cur, "graduated")
    assert ch == {"internship_months": 9.0, "certifications_count": 3.0}
    ch, _ = parse_changes("what if I improve my cloud skills to 4", cur, "graduated")
    assert ch["area_cloud_devops"] == 4.0
    ch, notes = parse_changes("what if I do an internship", cur, "graduated")
    assert ch["internship_months"] == 9.0 and notes            # assumed 6 months, and says so


def test_parse_changes_respects_ranges():
    ch, _ = parse_changes("what if I get 50 certifications", {"certifications_count": 1.0}, "graduated")
    assert ch["certifications_count"] == 10.0                  # clipped to the allowed maximum


def test_estimate_uses_the_model_output(assistant, cox_predictor):
    r = assistant.respond("How long will it take me?", QUICK_PROFILE)
    assert cox_predictor.predict(QUICK_PROFILE)["headline"] in r["reply"]
    assert r["sources"] == ["prediction"]


@pytest.mark.parametrize("message,intent", [
    ("hi", "greeting"), ("What can I ask?", "help"), ("How does the model work?", "method"),
    ("Do you store my data?", "privacy"), ("Can I trust this estimate?", "trust"),
    ("Is GPA important?", "factor"), ("Who will win the cricket world cup?", "unknown"),
])
def test_general_questions_work_without_a_profile(assistant, message, intent):
    response = assistant.respond(message, None)
    assert response["intent"] == intent
    assert response["reply"]
    assert response["follow_ups"]


@pytest.mark.parametrize("message,expected", [
    ("good evening my friend", "Good evening!"),
    ("good morning", "Good morning!"),
    ("good afternoon", "Good afternoon!"),
    ("thanks", "You're welcome!"),
    ("thankyuo", "You're welcome!"),
    ("thanku", "You're welcome!"),
    ("thankyou", "You're welcome!"),
])
@pytest.mark.parametrize("profile", [QUICK_PROFILE, None])
def test_greetings_get_a_contextual_acknowledgement(assistant, message, expected, profile):
    response = assistant.respond(message, profile)
    assert response["intent"] == "greeting"
    assert response["reply"].startswith(expected)


def test_explanation_says_when_no_strong_factor_holds_profile_back(assistant, monkeypatch):
    monkeypatch.setattr(assistant.explainer, "explain", lambda _: {
        "helping": [{"text": "A factor is helping."}], "hurting": [], "note": wording.LOCAL_NOTE,
    })
    response = assistant.respond("Why is my estimate like this?", QUICK_PROFILE)
    assert wording.LOCAL_NO_HURTING in response["reply"]


@pytest.mark.parametrize("message", [
    "hi", "What can I ask?", "How does the model work?", "Do you store my data?",
    "Can I trust this estimate?", "Who will win the cricket world cup?",
])
def test_basic_answers_without_a_profile_use_simple_language(assistant, message):
    reply = assistant.respond(message, None)["reply"]
    assert len(reply.split()) <= 45
    assert "survival-analysis" not in reply


def test_personal_questions_require_a_profile(assistant):
    response = assistant.respond("How long will it take me?", None)
    assert response["intent"] == "estimate"
    assert "Complete your profile" in response["reply"]
    assert response["sources"] == []


def test_whatif_matches_the_whatif_module(assistant, cox_predictor):
    from core.whatif import what_if
    cur = current_values(QUICK_PROFILE)
    expected = what_if(cox_predictor, QUICK_PROFILE, {"internship_months": cur["internship_months"] + 6})["message"]
    r = assistant.respond("What if I do a 6-month internship?", QUICK_PROFILE)
    assert expected in r["reply"] and "Change tested" in r["reply"]
    assert wording.ASSOCIATIONAL_NOTE in r["reply"]


def test_gpa_change_refused_for_graduates(assistant):
    r = assistant.respond("what if my gpa was 3.8", QUICK_PROFILE)
    assert "fixed once you have graduated" in r["reply"]


def test_out_of_scope_is_refused(assistant):
    r = assistant.respond("Who will win the cricket world cup?", QUICK_PROFILE)
    assert r["intent"] == "unknown" and "not a general chatbot" in r["reply"]


def test_every_intent_answers_without_error(assistant):
    for q in ("hi", "help", "why", "Is GPA important?", "can I trust this", "how does it work", "do you store data",
              "how can I find work sooner", "which skills should I learn", "compare with peers"):
        r = assistant.respond(q, QUICK_PROFILE)
        assert r["reply"] and not r["reply"].startswith("I couldn't")


def test_message_is_truncated(assistant):
    r = assistant.respond("why " * 400, QUICK_PROFILE)
    assert r["intent"] == "explain"


def test_chat_never_stores_or_calls_out():
    """The assistant writes nothing, logs nothing and calls no external service or language model."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "core" / "chat.py").read_text(encoding="utf-8")
    for forbidden in ("open(", ".to_csv", ".write_text", "sqlite3", "logging", "requests", "openai", "anthropic",
                      "http"):
        assert forbidden not in src, forbidden
