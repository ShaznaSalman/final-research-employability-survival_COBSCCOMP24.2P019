"""Thesis Table 5.2 (functional test scenarios) from the saved final model.

  python docs/thesis_figures/table5_2_scenarios.py   -> docs/thesis_figures/table5_2_scenarios.md

The profiles are the ones used for the explanation checks (proof/08_model_development/explanation_checks.md).
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from core.explain import Explainer  # noqa: E402
from core.predict import Predictor  # noqa: E402
from core.suggest import suggest  # noqa: E402

AREAS = ("area_programming", "area_web_apis", "area_data_ai", "area_cloud_devops", "area_quality_security", "area_soft")
PROFILES = {
    "Strong": dict(respondent_status="graduated", gpa=3.6, degree_type="computer_science", university_type="state",
                   certifications_count=3, internship_months=12, project_count=4,
                   **dict(zip(AREAS, (4.5, 4, 3.5, 3.5, 4, 4.5)))),
    "Average": dict(respondent_status="graduated", gpa=3.0, degree_type="information_technology",
                    university_type="state", certifications_count=1, internship_months=6, project_count=2,
                    **dict.fromkeys(AREAS, 3)),
    "Weak": dict(respondent_status="graduated", gpa=2.6, degree_type="information_technology",
                 university_type="non_state", certifications_count=0, internship_months=0, project_count=0,
                 **dict.fromkeys(AREAS, 2)),
}
LABEL = {"internship_months": "internship", "technical_mean": "technical skills", "soft_mean": "soft skills",
         "certifications_count": "certifications", "gpa": "GPA", "project_count": "projects"}


def main():
    pr = Predictor.load()
    ex = Explainer(pr)
    rows = ["| Profile | Estimate | Helping | Holding back | First suggestion |", "|---|---|---|---|---|"]
    for name, prof in PROFILES.items():
        r, e, s = pr.predict(prof), ex.explain(prof), suggest(pr, prof)
        first = s["suggestions"][0]["text"] if s["suggestions"] else "None"
        rows.append(f"| {name} | {r['headline']} ({r['confidence']}) | "
                    f"{', '.join(LABEL.get(f['feature'], f['feature']) for f in e['helping']) or '-'} | "
                    f"{', '.join(LABEL.get(f['feature'], f['feature']) for f in e['hurting']) or '-'} | {first} |")
    searching = pr.predict(dict(PROFILES["Weak"], months_since_graduation=6))
    rows += ["", f"Weak profile still searching after 6 months: {searching['headline']} ({searching['confidence']})"]
    out = Path(__file__).with_name("table5_2_scenarios.md")
    out.write_text("# Table 5.2 - functional test scenarios (saved final model)\n\n" + "\n".join(rows) + "\n",
                   encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
