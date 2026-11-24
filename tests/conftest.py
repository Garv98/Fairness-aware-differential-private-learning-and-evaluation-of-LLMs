import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fairdp.data.loaders import TabularDataset  # noqa: E402


@pytest.fixture
def synthetic_credit() -> TabularDataset:
    """Small synthetic dataset with the credit schema (no real data needed)."""
    rng = np.random.default_rng(0)
    n = 3000
    df = pd.DataFrame({
        "person_age": rng.integers(20, 70, n),
        "person_income": rng.lognormal(10.8, 0.5, n).round(),
        "person_emp_length": rng.integers(0, 20, n).astype(float),
        "loan_amnt": rng.integers(500, 30000, n),
        "loan_int_rate": rng.uniform(5, 20, n),
        "loan_percent_income": rng.uniform(0.01, 0.6, n),
        "cb_person_cred_hist_length": rng.integers(2, 30, n),
        "person_home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN", "OTHER"], n, p=[.5, .4, .09, .01]),
        "loan_intent": rng.choice(["EDUCATION", "MEDICAL", "VENTURE"], n),
        "loan_grade": rng.choice(list("ABCDE"), n),
        "cb_person_default_on_file": rng.choice(["Y", "N"], n),
    })
    logit = -2 + 3 * df["loan_percent_income"] + 0.1 * (df["loan_int_rate"] - 12) \
        + 0.8 * (df["person_home_ownership"] == "RENT")
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    from fairdp.data.loaders import CREDIT_NUMERICAL, CREDIT_CATEGORICAL, CREDIT_SENSITIVE
    return TabularDataset("credit", df, y, list(CREDIT_NUMERICAL), list(CREDIT_CATEGORICAL),
                          df[CREDIT_SENSITIVE].copy())
