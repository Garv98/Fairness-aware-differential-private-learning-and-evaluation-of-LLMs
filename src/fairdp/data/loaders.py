"""Dataset loaders returning a common :class:`TabularDataset` structure.

Supported datasets
------------------
* ``credit``  - Kaggle "Credit Risk Dataset" (laotse). Place the CSV at
  ``data/credit_risk_dataset.csv`` (not redistributed, see data/README.md).
* ``adult``   - UCI Adult via OpenML (downloaded and cached in ``data/``).
* ``acs_income`` / ``acs_employment`` - folktables ACS tasks (downloaded and
  cached in ``data/acs``). States configurable.

Raw sensitive columns are kept in ``sensitive`` so that :mod:`fairdp.groups`
can derive marginal and intersectional groups. Whether sensitive columns are
also used as model features is controlled by ``drop_sensitive_features``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class TabularDataset:
    name: str
    X: pd.DataFrame                 # raw (unprocessed) feature frame
    y: np.ndarray                   # binary target, int {0,1}
    numerical: list[str]
    categorical: list[str]
    sensitive: pd.DataFrame         # raw columns used to build groups
    positive_label: str = "1"
    notes: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)


# --------------------------------------------------------------------------
# Credit risk (Kaggle, laotse)
# --------------------------------------------------------------------------
CREDIT_TARGET = "loan_status"
CREDIT_NUMERICAL = [
    "person_age", "person_income", "person_emp_length", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
]
CREDIT_CATEGORICAL = [
    "person_home_ownership", "loan_intent", "loan_grade",
    "cb_person_default_on_file",
]
CREDIT_SENSITIVE = ["person_age", "person_income", "person_home_ownership"]


def load_credit(path: str | Path | None = None, clean_outliers: bool = True,
                drop_sensitive_features: bool = False) -> TabularDataset:
    path = Path(path) if path else DATA_DIR / "credit_risk_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the Kaggle 'laotse/credit-risk-dataset' "
            "CSV and place it there (see data/README.md)."
        )
    df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    notes = {"rows_after_dedup": len(df)}
    if clean_outliers:
        # Implausible values (e.g. age 144, employment length 123 years) are
        # treated as data-entry errors: rows with impossible ages are dropped,
        # impossible employment lengths are set to missing (imputed later).
        bad_age = df["person_age"] > 100
        notes["dropped_age_gt_100"] = int(bad_age.sum())
        df = df[~bad_age].copy()
        bad_emp = df["person_emp_length"] > (df["person_age"] - 14)
        notes["emp_length_set_nan"] = int(bad_emp.sum())
        df.loc[bad_emp, "person_emp_length"] = np.nan
        df = df.reset_index(drop=True)

    y = df[CREDIT_TARGET].astype(int).to_numpy()
    X = df.drop(columns=[CREDIT_TARGET])
    num, cat = list(CREDIT_NUMERICAL), list(CREDIT_CATEGORICAL)
    if drop_sensitive_features:
        X = X.drop(columns=CREDIT_SENSITIVE)
        num = [c for c in num if c not in CREDIT_SENSITIVE]
        cat = [c for c in cat if c not in CREDIT_SENSITIVE]
    sensitive = df[CREDIT_SENSITIVE].copy()
    return TabularDataset("credit", X, y, num, cat, sensitive,
                          positive_label="default", notes=notes)


# --------------------------------------------------------------------------
# UCI Adult (OpenML id 1590, version 2)
# --------------------------------------------------------------------------
ADULT_NUMERICAL = ["age", "fnlwgt", "education-num", "capital-gain",
                   "capital-loss", "hours-per-week"]
ADULT_CATEGORICAL = ["workclass", "education", "marital-status", "occupation",
                     "relationship", "race", "sex", "native-country"]
ADULT_SENSITIVE = ["age", "sex", "race"]


def load_adult(drop_sensitive_features: bool = False) -> TabularDataset:
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml("adult", version=2, as_frame=True,
                         data_home=str(DATA_DIR / "openml"))
    df = bunch.frame.copy()
    df = df.drop_duplicates().reset_index(drop=True)
    y = (df["class"].astype(str).str.strip() == ">50K").astype(int).to_numpy()
    X = df.drop(columns=["class"])
    for c in ADULT_CATEGORICAL:
        X[c] = X[c].astype("object")
    num, cat = list(ADULT_NUMERICAL), list(ADULT_CATEGORICAL)
    if drop_sensitive_features:
        X = X.drop(columns=["sex", "race"])
        cat = [c for c in cat if c not in ("sex", "race")]
    sensitive = df[ADULT_SENSITIVE].copy()
    sensitive["sex"] = sensitive["sex"].astype(str)
    sensitive["race"] = sensitive["race"].astype(str)
    return TabularDataset("adult", X, y, num, cat, sensitive,
                          positive_label=">50K")


# --------------------------------------------------------------------------
# folktables ACS tasks
# --------------------------------------------------------------------------
ACS_CATEGORICAL = {
    # folktables codes are integers but semantically categorical
    "income": ["COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P"],
    "employment": ["SCHL", "MAR", "RELP", "DIS", "ESP", "CIT", "MIG", "MIL",
                   "ANC", "NATIVITY", "DEAR", "DEYE", "DREM", "SEX", "RAC1P"],
}
ACS_SENSITIVE = ["AGEP", "SEX", "RAC1P"]


def load_acs(task: str = "income", states: list[str] | None = None,
             year: str = "2018", drop_sensitive_features: bool = False,
             max_rows: int | None = None, seed: int = 0) -> TabularDataset:
    import folktables

    states = states or ["CA"]
    source = folktables.ACSDataSource(survey_year=year, horizon="1-Year",
                                      survey="person",
                                      root_dir=str(DATA_DIR / "acs"))
    raw = source.get_data(states=states, download=True)
    problem = {"income": folktables.ACSIncome,
               "employment": folktables.ACSEmployment}[task]
    feats, labels, _ = problem.df_to_pandas(raw)
    X = feats.reset_index(drop=True)
    y = labels.iloc[:, 0].astype(int).to_numpy()
    if max_rows and len(X) > max_rows:
        idx = np.random.default_rng(seed).choice(len(X), max_rows, replace=False)
        idx.sort()
        X, y = X.iloc[idx].reset_index(drop=True), y[idx]
    cat = [c for c in ACS_CATEGORICAL[task] if c in X.columns]
    num = [c for c in X.columns if c not in cat]
    for c in cat:
        X[c] = X[c].astype(int).astype(str)
    sensitive = pd.DataFrame({
        "AGEP": X["AGEP"].astype(float),
        "SEX": X["SEX"],
        "RAC1P": X["RAC1P"],
    })
    if drop_sensitive_features:
        X = X.drop(columns=["SEX", "RAC1P"])
        cat = [c for c in cat if c not in ("SEX", "RAC1P")]
    return TabularDataset(f"acs_{task}", X, y, num, cat, sensitive,
                          notes={"states": states, "year": year})


def load_dataset(name: str, **kwargs) -> TabularDataset:
    if name == "credit":
        return load_credit(**kwargs)
    if name == "adult":
        return load_adult(**kwargs)
    if name.startswith("acs_"):
        return load_acs(task=name.split("_", 1)[1], **kwargs)
    raise ValueError(f"unknown dataset {name!r}")
