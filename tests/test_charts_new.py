"""Dashboard charts (frontend/charts.py): donut gauge, skill radar and gap bars."""
from __future__ import annotations

import pytest

from frontend import charts, ui

MILESTONE = {"months": 12, "employed_pct": 62.0, "low_pct": 55.0, "high_pct": 70.0}


def skill(label, you, emp, demand):
    return {"label": label, "your_rating": you, "employer_importance": emp, "demand": demand, "ads_pct": demand}


SKILLS = [skill("Problem Solving", 3.0, 4.5, 69), skill("Teamwork", 4.0, 4.0, 61), skill("SQL / Database Management", 2.0, 4.0, 37),
          skill("Software Testing", 3.5, 3.0, 44), skill("REST APIs", 2.5, 3.5, 33), skill("Git / GitHub", 3.0, 3.8, 26),
          skill("JavaScript", 3.0, 3.2, 28), skill("Docker", 1.0, 3.0, 12), skill("Cloud", 1.5, 3.4, 20),
          skill("Unrated", None, 3.0, 50)]


def test_donut_shows_the_share_and_the_remainder():
    fig = charts.job_donut(MILESTONE)
    pie = fig.data[0]
    assert list(pie.values) == [62.0, 38.0] and pie.hole == 0.72
    assert any("62%" in a.text for a in fig.layout.annotations) and any("12 months" in a.text for a in fig.layout.annotations)


@pytest.mark.parametrize("pct", [0.0, 100.0])
def test_donut_handles_extremes(pct):
    pie = charts.job_donut({**MILESTONE, "employed_pct": pct}).data[0]
    assert sum(pie.values) == pytest.approx(100.0)


def test_radar_compares_you_with_employers_on_the_most_requested_skills():
    fig = charts.skill_radar(SKILLS)
    assert [t.name for t in fig.data] == ["Employers", "You"]
    assert len(fig.data[0].r) == 9                       # 8 skills, with the first repeated to close the shape
    assert "Unrated" not in " ".join(fig.data[0].theta)  # skills without a rating are left out
    assert fig.layout.polar.radialaxis.range == (0, 5)


def test_gap_bars_list_only_skills_where_employers_expect_more_largest_first():
    fig = charts.gap_bars(SKILLS)
    labels = list(fig.data[0].y)[::-1]                   # bars are drawn bottom-up, so reverse for reading order
    assert labels[0] == "SQL / Database Management" and "Teamwork" not in labels and "Software Testing" not in labels
    assert len(labels) == 5 and list(fig.data[0].x)[::-1] == sorted(fig.data[0].x, reverse=True)


def test_gap_bars_none_when_there_are_no_gaps():
    assert charts.gap_bars([skill("A", 4.0, 3.0, 10), skill("B", 3.0, 3.0, 5)]) is None


def test_new_charts_follow_the_dark_theme(monkeypatch):
    monkeypatch.setattr(ui, "theme_type", lambda: "dark")
    for fig in (charts.job_donut(MILESTONE), charts.skill_radar(SKILLS), charts.gap_bars(SKILLS)):
        text = fig.to_json()
        for light in (charts.INK, charts.GRID, charts.TRACK, charts.SURFACE):
            assert f'"{light}"' not in text, light
