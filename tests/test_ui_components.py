"""Shared UI components (frontend/ui.py): they build the expected HTML and escape plain text."""
from __future__ import annotations

import pytest

from frontend import ui


@pytest.fixture
def html(monkeypatch):
    captured: list[str] = []
    monkeypatch.setattr(ui.st, "markdown", lambda body, **kw: captured.append(body))
    return captured


def test_driver_card_uses_tone_and_arrow_per_kind(html):
    ui.driver_card("helping", "Helping you", ["6 months of internship"])
    ui.driver_card("holding", "Holding you back", ["No portfolio"])
    assert "t-mint" in html[0] and "M12 19V5" in html[0]          # up arrow
    assert "t-peach" in html[1] and "M12 5v14" in html[1]         # down arrow


def test_driver_card_empty_message_and_title_escaped(html):
    ui.driver_card("helping", "<b>x</b>", [], empty="Nothing stands out.")
    assert "Nothing stands out." in html[0] and "&lt;b&gt;x&lt;/b&gt;" in html[0]


def test_progress_marks_done_current_and_upcoming(html):
    ui.progress(["About you", "Experience", "Skills", "Preparation"], 2)
    body = html[0]
    assert "class='done'>About you" in body and "class='now'>Experience" in body and "class=''>Skills" in body
    assert "Step 2 of 4" in body


def test_estimate_card_escapes_numbers_but_keeps_headline_html(html):
    ui.estimate_card("5", "3", "9", "Moderate confidence", "<b>Faster</b> than typical")
    body = html[0]
    assert "5 <small>months</small>" in body and "Typical range 3 - 9 months" in body
    assert "<b>Faster</b> than typical" in body
    ui.estimate_card("<i>5</i>", "3", "9", "c", "h")
    assert "&lt;i&gt;5&lt;/i&gt;" in html[1]


def test_error_card_escapes_message(html):
    ui.error_card("API <down>", hint="Try again")
    assert "API &lt;down&gt;" in html[0] and "Try again" in html[0]


def test_brand_and_icons():
    assert "JobSpark" in ui.brand_html() and "Time to your first IT job" in ui.brand_html()
    assert all(f"<svg" in ui.icon(name) for name in ui.ICONS)


def test_dark_tokens_override_every_light_token():
    names = lambda css: {line.split(":")[0].strip() for line in css.replace(";", ";\n").splitlines()   # noqa: E731
                         if line.strip().startswith("--")}
    assert names(ui.TOKENS_DARK) <= names(ui.TOKENS_LIGHT)
    assert {"--ink", "--panel", "--brand", "--blue-soft", "--mint-soft", "--peach-soft"} <= names(ui.TOKENS_DARK)


def test_term_has_hover_definition_and_escapes():
    html_ = ui.term("c-index")
    assert "<abbr class='term'" in html_ and "title='How well the model" in html_ and ">C-index</abbr>" in html_
    assert "title='" in ui.term("censored", "it's <b>")            # definition quoted safely
    assert "&lt;b&gt;" in ui.term("censored", "it's <b>")          # label escaped


def test_every_glossary_entry_has_a_definition():
    for key, (name, definition) in ui.GLOSSARY.items():
        assert name and len(definition) > 40, key
