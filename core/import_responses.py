"""Convert the main-study response workbooks in "New Data/Main data/" into the CSV inputs in data/raw/.

  python -m core.import_responses

Only the MAIN study is imported. The pilot and technical-test workbooks ("New Data/Pilot Test/",
"New Data/Technical test/") are never used for the model or the questionnaire study.

Survey date: the responses were collected on a shared Excel sheet, and the Timestamp column was added
afterwards, so it is not the date each person answered. Every response is therefore given the same
reference survey date, the end of data collection (SURVEY_DATE). The original workbooks are not changed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from core import schema

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "New Data" / "Main data"
RAW_DIR = ROOT / "data" / "raw"
SURVEY_DATE = "2026-10-06 00:00:00"   # latest Timestamp in the main-study workbooks = end of collection

FILES = {  # workbook name prefix -> output CSV
    "Graduate Employability Survey": "graduate_responses.csv",
    "Employer _ Recruiter Survey": "employer_responses.csv",
}


def convert(source_dir: Path = SOURCE_DIR, raw_dir: Path = RAW_DIR, survey_date: str = SURVEY_DATE) -> dict:
    out = {}
    for prefix, name in FILES.items():
        matches = sorted(source_dir.glob(f"{prefix}*.xlsx"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one '{prefix}*.xlsx' in {source_dir}, found {len(matches)}")
        df = pd.read_excel(matches[0], sheet_name=0, keep_default_na=False, na_values=schema.SURVEY_NA_VALUES)
        df["Timestamp"] = survey_date
        df.to_csv(raw_dir / name, index=False, encoding="utf-8")
        out[name] = {"source": matches[0].name, "rows": len(df)}
    return out


if __name__ == "__main__":
    for name, info in convert().items():
        print(f"{name}: {info['rows']} rows from {info['source']}")
