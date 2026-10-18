"""Role-fit comparisons derived from the local job-ad study."""
import pytest

from core.pathways import role_gaps, role_pathways


def test_role_pathways_rank_by_ad_weighted_skill_ratings():
    profile = {"tech_python": 4, "soft_teamwork": 2}
    job_market = {
        "roles": {"Software engineering": 30, "Data & AI": 15, "Other": 20},
        "by_role": {
            "Software engineering": {"python": 100, "teamwork": 50},
            "Data & AI": {"python": 50},
            "Other": {"teamwork": 100},
        },
    }

    result = role_pathways(profile, job_market)

    assert [item["role"] for item in result["recommendations"]] == ["Data & AI", "Software engineering"]
    software = next(item for item in result["roles"] if item["role"] == "Software engineering")
    assert software["match_rating"] == pytest.approx(10 / 3, abs=0.01)
    assert software["gaps"][0]["skill"] == "teamwork"
    assert role_gaps(result, "Software engineering") == software["gaps"]


def test_role_pathways_skips_unrated_roles_and_validates_target():
    result = role_pathways({}, {"roles": {"Other": 5}, "by_role": {"Other": {"python": 100}}})
    assert result == {"recommendations": [], "roles": []}
    with pytest.raises(ValueError, match="Unknown role family"):
        role_gaps(result, "Software engineering")
    with pytest.raises(ValueError, match="at least 1"):
        role_pathways({}, {}, limit=0)
