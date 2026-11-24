"""Demographic group construction (marginal + intersectional).

Each dataset has a :class:`GroupSpec`. ``spec.fit(sensitive_train)`` learns any
data-dependent thresholds (e.g. income quartiles) on the *training* split and
returns a binner that maps raw sensitive columns to categorical group labels.

The credit-risk binning reproduces the original notebook
(``notebooks/04_fairness_mitigation.ipynb`` cell 2): age bins (0,25], (25,35],
(35,inf) and income split at train Q1/Q3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

Binner = Callable[[pd.DataFrame], pd.DataFrame]
SEP = "|"


@dataclass
class GroupSpec:
    default_attrs: list[str]
    fit: Callable[[pd.DataFrame], Binner]
    stratify_key: Callable[[pd.DataFrame], pd.Series]


def intersect(groups: pd.DataFrame, attrs: list[str]) -> pd.Series:
    """Intersectional cell label, e.g. ``"Young|Low|RENT"``."""
    out = groups[attrs[0]].astype(str)
    for a in attrs[1:]:
        out = out + SEP + groups[a].astype(str)
    return out


def _age_bins(age: pd.Series, edges, labels) -> pd.Series:
    return pd.cut(age.astype(float), bins=edges, labels=labels, right=True).astype(str)


# ---------------------------------------------------------------- credit ----
def _credit_fit(train: pd.DataFrame) -> Binner:
    q1 = float(train["person_income"].quantile(0.25))
    q3 = float(train["person_income"].quantile(0.75))

    def binner(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "age": _age_bins(df["person_age"], [0, 25, 35, np.inf],
                             ["Young", "Middle", "Older"]),
            "income": pd.cut(df["person_income"].astype(float),
                             bins=[-np.inf, q1, q3, np.inf],
                             labels=["Low", "Middle", "High"],
                             include_lowest=True).astype(str),
            "home": df["person_home_ownership"].astype(str),
        }, index=df.index)

    binner.thresholds = {"income_q1": q1, "income_q3": q3}
    return binner


CREDIT = GroupSpec(
    default_attrs=["age", "income", "home"],
    fit=_credit_fit,
    stratify_key=lambda s: s["person_home_ownership"].astype(str),
)


# ----------------------------------------------------------------- adult ----
def _adult_race(r: pd.Series) -> pd.Series:
    r = r.astype(str).str.strip()
    return r.where(r.isin(["White", "Black"]), "Other")


def _adult_fit(train: pd.DataFrame) -> Binner:
    def binner(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "sex": df["sex"].astype(str).str.strip(),
            "race": _adult_race(df["race"]),
            "age": _age_bins(df["age"], [0, 30, 50, np.inf],
                             ["<=30", "31-50", ">50"]),
        }, index=df.index)
    return binner


ADULT = GroupSpec(
    default_attrs=["sex", "race", "age"],
    fit=_adult_fit,
    stratify_key=lambda s: s["sex"].astype(str) + "_" + _adult_race(s["race"]),
)


# ------------------------------------------------------------------- ACS ----
_ACS_RACE = {"1": "White", "2": "Black", "6": "Asian"}


def _acs_race(r: pd.Series) -> pd.Series:
    r = r.astype(str).str.replace(r"\.0$", "", regex=True)
    return r.map(_ACS_RACE).fillna("Other")


def _acs_sex(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(r"\.0$", "", regex=True)
    return s.map({"1": "Male", "2": "Female"}).fillna("Unknown")


def _acs_fit(train: pd.DataFrame) -> Binner:
    def binner(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "sex": _acs_sex(df["SEX"]),
            "race": _acs_race(df["RAC1P"]),
            "age": _age_bins(df["AGEP"], [0, 30, 50, np.inf],
                             ["<=30", "31-50", ">50"]),
        }, index=df.index)
    return binner


ACS = GroupSpec(
    default_attrs=["sex", "race", "age"],
    fit=_acs_fit,
    stratify_key=lambda s: _acs_sex(s["SEX"]) + "_" + _acs_race(s["RAC1P"]),
)


def get_spec(dataset_name: str) -> GroupSpec:
    if dataset_name == "credit":
        return CREDIT
    if dataset_name == "adult":
        return ADULT
    if dataset_name.startswith("acs_"):
        return ACS
    raise ValueError(f"no group spec for {dataset_name!r}")


class CellIndex:
    """Integer encoding of intersectional cells + their marginal parents.

    Used by IG-DPSGD to aggregate per-cell statistics and to derive marginal
    statistics from cell statistics (post-processing, no extra privacy cost).
    The set of cells is the cartesian product of the attribute levels observed
    in *training* data; this list is treated as public schema information.
    """

    def __init__(self, train_groups: pd.DataFrame, attrs: list[str]):
        self.attrs = list(attrs)
        self.levels = {a: sorted(train_groups[a].astype(str).unique()) for a in attrs}
        grids = np.meshgrid(*[np.arange(len(self.levels[a])) for a in attrs], indexing="ij")
        combos = np.stack([g.ravel() for g in grids], axis=1)  # [n_cells, n_attrs]
        self.combos = combos
        self.names = [SEP.join(self.levels[a][i] for a, i in zip(attrs, row)) for row in combos]
        self._lookup = {n: i for i, n in enumerate(self.names)}
        # membership matrices: for each attr, [n_levels, n_cells]
        self.marginal_membership = {
            a: np.stack([(combos[:, k] == lv).astype(np.float64)
                         for lv in range(len(self.levels[a]))])
            for k, a in enumerate(attrs)
        }

    @property
    def n_cells(self) -> int:
        return len(self.names)

    def encode(self, groups: pd.DataFrame) -> np.ndarray:
        labels = intersect(groups, self.attrs)
        return labels.map(self._lookup).fillna(-1).astype(int).to_numpy()
