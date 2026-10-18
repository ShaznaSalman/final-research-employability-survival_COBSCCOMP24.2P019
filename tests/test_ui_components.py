"""Shared UI components (frontend/ui.py): they build the expected HTML and escape plain text."""
from __future__ import annotations

import base64

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


def test_estimate_card_splits_months_from_details_and_escapes_numbers(html):
    ui.estimate_card("5", "3", "9", "Moderate confidence", "<b>Faster</b> than typical")
    body = html[0]
    assert "class='result-months'" in body and "class='result-details'" in body
    assert "5 <small>months</small>" in body
    assert "Typical range" in body and "3 - 9 months" in body and "Moderate confidence" in body
    assert "<b>Faster</b> than typical" in body
    ui.estimate_card("<i>5</i>", "3", "9", "c", "h")
    assert "&lt;i&gt;5&lt;/i&gt;" in html[1]


def test_error_card_escapes_message(html):
    ui.error_card("API <down>", hint="Try again")
    assert "API &lt;down&gt;" in html[0] and "Try again" in html[0]


def test_brand_and_icons():
    assert "JobSpark" in ui.brand_html() and "Time to your first IT job" in ui.brand_html()
    assert all(f"<svg" in ui.icon(name) for name in ui.ICONS)


def test_hero_can_show_accessible_career_art(html):
    art = ui.career_path_art()
    ui.hero("Career path", "Explore roles", "Find roles that fit.", compact=True, art=art)
    assert "has-art" in html[0]
    assert "data:image/svg+xml;base64," in html[0]
    assert "choosing a career path among IT roles" in html[0]
    svg = base64.b64decode(art.split(",", 1)[1]).decode("utf-8")
    assert "career-orbit" in svg and "careerOrbit 36s linear infinite" in svg
    assert "prefers-reduced-motion:reduce" in svg
    assert "career-person" in svg and "ffc6a4" in svg and "302a3f" in svg
    assert ".hero.has-art .hero-art { display:block; grid-row:1; width:180px;" in ui.CSS
    assert ".hero.has-art > div:first-child { grid-row:2; }" in ui.CSS


def test_dark_tokens_override_every_light_token():
    names = lambda css: {line.split(":")[0].strip() for line in css.replace(";", ";\n").splitlines()   # noqa: E731
                         if line.strip().startswith("--")}
    assert names(ui.TOKENS_DARK) <= names(ui.TOKENS_LIGHT)
    assert {"--ink", "--panel", "--brand", "--blue-soft", "--mint-soft", "--peach-soft"} <= names(ui.TOKENS_DARK)
    assert ".best-match-badge" in ui.CSS and "border:1.5px solid #f5bd16" in ui.CSS
    assert ".best-match-star { color:#e0a800;" in ui.CSS


def test_term_has_hover_definition_and_escapes():
    html_ = ui.term("c-index")
    assert "<abbr class='term'" in html_ and "title='How well the model" in html_ and ">C-index</abbr>" in html_
    assert "title='" in ui.term("censored", "it's <b>")            # definition quoted safely
    assert "&lt;b&gt;" in ui.term("censored", "it's <b>")          # label escaped


def test_every_glossary_entry_has_a_definition():
    for key, (name, definition) in ui.GLOSSARY.items():
        assert name and len(definition) > 40, key
