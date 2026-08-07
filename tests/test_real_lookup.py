"""The real institution lookup (data/reference/real/university_types.csv), staged for real survey data."""
from pathlib import Path

import pandas as pd

from core import clean, forms_adapter, schema, wording

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "reference" / "real" / "university_types.csv"


def test_every_dropdown_institution_is_in_the_lookup_with_its_type():
    lut = clean.load_university_lookup(REAL)
    for utype, names in wording.INSTITUTIONS_BY_TYPE.items():
        for name in names:
            assert lut.get(forms_adapter.norm(name)) == utype, name


def test_lookup_is_consistent():
    df = pd.read_csv(REAL)
    assert set(df["university_type"]) <= set(schema.UNIVERSITY_TYPES)
    norm = df["university_name"].map(forms_adapter.norm)
    # a name variant must never map to two different types or institutions
    assert not df.assign(n=norm).groupby("n")[["university_type", "institution"]].nunique().gt(1).any().any()
    # every institution has a single type
    assert df.groupby("institution")["university_type"].nunique().eq(1).all()


def test_staged_lookup_does_not_replace_the_synthetic_one_until_moved():
    path, synthetic = clean.find_input("university_types")
    assert REAL.parent != path.parent            # the staging folder is never searched automatically
